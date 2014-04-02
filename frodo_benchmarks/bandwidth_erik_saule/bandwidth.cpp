#include <iostream>
#include <stdio.h>
#include <omp.h>
#include "timestamp.hpp"
#include <string.h>
#include <malloc.h>
#include <xmmintrin.h>
#include <immintrin.h>

//#define VECTSUMBASED

int main(int argc, char* argv[])
{
  char** buffers;
  int nbcore = 61;
  int nbthread = 244;
  long int SIZE = 16*1024*1024;

  std::cout<<"make sure by setting KMP_AFFINITY to verbose that the first "<<nbcore<<" openmp threads map to different cores."<<std::endl;

  buffers = new char*[nbcore];
  util::timestamp* timeof = new util::timestamp[nbthread];

  int NBCOREACTIVE = 1;
  int NBTHREADACTIVE = 1;

  if (argc < 3)
    {
      std::cerr<<"usage : "<<argv[0]<<" <core> <threadpercore>"<<std::endl;
      return -1;
    }

  NBCOREACTIVE = atoi(argv[1]);
  NBTHREADACTIVE = atoi(argv[2]);

#if defined MEMCOPYBASED || defined MEMCOPYVECTBASED || defined MEMSETBASED || defined MEMSETNOREAD || defined MEMSETNOREADNOORDER
  char** dest_location = new char*[nbthread];
#endif

    int max_threads = omp_get_max_threads();
    std::cout << "max number threads: " << max_threads << std::endl;
    std::cout << "nb core active: " << NBCOREACTIVE << std::endl;
    std::cout << "nb threads active per core: " << NBTHREADACTIVE << std::endl;

#pragma omp parallel
{
    //memory allocs

    int tid = omp_get_thread_num();

    timeof[tid] = util::timestamp(0,0);

    // buffers is a shared variable (array of pointers)
    // size of each buffer[i]: SIZE bytes (16 Mbytes)
    if (tid < nbcore) {
      buffers[tid] = (char*) _mm_malloc( SIZE, 64); //let's allocate on the 64 bytes boundary
      for (long int i=0; i <SIZE; ++i)
	    buffers[tid][i] = 0;
    }


#pragma omp barrier

#if defined MEMCOPYBASED || defined MEMCOPYVECTBASED || defined MEMSETBASED || defined MEMSETNOREAD || defined MEMSETNOREADNOORDER
    dest_location[tid] = (char*) _mm_malloc(SIZE, 64);
    memset (dest_location[tid], 0, SIZE);
#endif

#pragma omp barrier

    if (tid == 0)
      std::cerr<<"allocated"<<std::endl;
    //    if (tid%nbcore == 1)

    int t = tid/61;
    int c = tid%61;
    if (c < NBCOREACTIVE && t < NBTHREADACTIVE) {

	//for (int doit=0; doit<10; ++doit) {
	for (int doit=0; doit<10; ++doit) {
	    //      char* destination = new char[SIZE];
	    util::timestamp beg;
#ifndef CHARBASED
	    int sum = 0;
#else
	    char sum = 0;
#endif
	    for (int i=0; i< nbcore; ++i) {
#if defined MEMCOPYBASED || defined MEMCOPYVECTBASED || defined MEMSETBASED || defined MEMSETNOREAD || defined MEMSETNOREADNOORDER
	      char* destination = dest_location[tid];
#endif
          //fprintf(stderr, "c= %d, t= %d, i= %d, buf offset: %d\n", c, t, i, (c+i+16*t)%nbcore);
          // t = 0,1,2,3
          // c = 0,1,..,60
          // loop i = 0,..,60
#if 1
	      char* in = buffers[(c + i + 16*t)%nbcore];
#else
// seg fault
	      char* in = buffers[tid];
#endif
	      
#if defined MEMCOPYBASED || defined MEMCOPYVECTBASED || defined MEMSETBASED || defined MEMSETNOREAD || defined MEMSETNOREADNOORDER

	      char* d = destination;
#endif
	      
#ifdef SIMPLESUMBASED
#ifdef INTBASED
	      int* ini = (int*) in;
#endif
#ifdef CHARBASED
	      char* ini = (char*) in;
#endif
	      long int S = SIZE/sizeof(*ini);
	      for (long int j = 0; j<S; ++j) {
		    sum += ini[j];
	      }
#endif
	      
#ifdef VECTSUMBASED
	      char* end = in+SIZE;
	      __m512i sump= _mm512_setzero_epi32();
#pragma unroll(4)
	      for (; in < end; in+=64) {
#ifdef USEPREFETCH
		_mm_prefetch (in + 4096, _MM_HINT_T1);
#endif
		__m512i temp = _mm512_load_epi32(in);
		sump = _mm512_add_epi32(sump,temp);
	      }
	      sum = _mm512_reduce_add_epi32(sump);
#endif
	      
#ifdef MEMCOPYBASED
	      memcpy(destination, in, SIZE);
#endif

#ifdef MEMCOPYVECTBASED
	      char* end = destination+SIZE;
	      for (; destination < end; in += 64, destination += 64) {
		__m512 foo = _mm512_load_ps(in);
		_mm512_storenr_ps (destination, foo);
	      }
#endif
	      
#ifdef MEMSETBASED
	      memset (destination, 1, SIZE);
#endif

#ifdef MEMSETNOREAD
	      char* end = destination+SIZE;
	      __m512 foo = _mm512_setzero();
	      for (; destination < end; destination += 64) {
		_mm512_storenr_ps (destination, foo);
	      }
#endif

#ifdef MEMSETNOREADNOORDER
	      char* end = destination+SIZE;
	      __m512 foo = _mm512_setzero();
	      for (; destination < end; destination += 64) {
		_mm512_storenrngo_ps (destination, foo);
	      }
#endif
	    }
	    
	    
	    util::timestamp end;
	    
	    if (doit == 4)
	      timeof[tid] = end-beg;

	    //std::cerr<<"sum is "<<(int) sum<<std::endl;
	  }


    }
} // end omp parallel

  double aggregatedBW = 0.;
  for (int i=0; i<nbthread;++i) {
    int t = i/61;
    int c = i%61;
    double BW = SIZE*nbcore/(timeof[i]);
    if (timeof[i] == util::timestamp(0,0)) BW = 0;
    aggregatedBW += BW;
    if (BW > 1) {
        std::cout<<"(tid: "<<t<<", core: "<<c<<"), timeof[i] sec to download "<< SIZE*nbcore <<" bytes. BW: "<<BW<<" bytes/s"<<std::endl;
    }
    //std::cout<<"time,id=%d: "<< (tid %d, core %d) timeof[i] <<" sec to download "<< SIZE*nbcore <<" bytes. BW: "<<BW<<" bytes/s"<<std::endl;
  }

  std::cout<<"aggregated:" <<aggregatedBW<<std::endl;

  return 0;
}
