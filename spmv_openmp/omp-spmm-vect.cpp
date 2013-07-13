#include "main-spmm.hpp"
#include <immintrin.h>
#include "memutils.hpp"

#define THROW_AWAY 10


template<typename VertexType, typename EdgeType, typename Scalar, int nbvector>
void my_spmm(VertexType nVtx, EdgeType*xadj, VertexType *adj, Scalar* val, Scalar *in, Scalar* out) {
#pragma omp parallel 
  {
#pragma omp for schedule(runtime)
    for (VertexType i = 0; i < nVtx; ++i)
      {
	Scalar output = 0.;
	
	VertexType* beg = adj+xadj[i];
	VertexType* end = adj+xadj[i+1];
	Scalar* val2 = val+xadj[i];
	
	Scalar outtemp[nbvector];
	for (int k=0; k<nbvector;++k)
	  outtemp[k] = 0.;


	for (auto p = beg; p < end; ++p)
	  {
	    auto j = *p;
	    auto value = *val2;
	    Scalar* inbase = in+j*nbvector;
	    
	    for (int k=0; k<nbvector; ++k) {
	      outtemp[k] += inbase[k] * value;
	    }
	    
	    ++val2;
	  }

	Scalar* outbase = out+i*nbvector;
	memcpy(outbase, outtemp, nbvector*sizeof(Scalar));
      }
  }
}

#ifdef MIC

#ifdef REGULAR_STORE
#define STORE_INST _mm512_store_pd
#endif
#ifdef NR_STORE
#define STORE_INST _mm512_storenr_pd
#endif
#ifdef NRNGO_STORE
#define STORE_INST _mm512_storenrngo_pd
#endif

template<>
void my_spmm<int,int,double,4>(int nVtx, int*xadj, int *adj, double* val, double *in, double* out) {
  typedef int VertexType;
  typedef int EdgeType;
  typedef double Scalar;
  const int nbvector=4; // changed from original of 8 (fill MIC vector register completely)
#pragma omp parallel 
  {
#pragma omp for schedule(runtime)
    for (VertexType i = 0; i < nVtx; ++i)
      {
	//do output = 0.;
	
	VertexType* beg = adj+xadj[i];
	VertexType* end = adj+xadj[i+1];
	Scalar* val2 = val+xadj[i];
	
	//Scalar outtemp[nbvector];
	
	//for (int k=0; k<nbvector;++k)
	//outtemp[k] = 0.;
	__m512d outtemp = _mm512_setzero_pd();


	for (auto p = beg; p < end; ++p)
	  {
	    EdgeType j = *p;
	    Scalar value = *val2;
	    Scalar* inbase = in+j*nbvector;

	    __m512d multiplier = _mm512_set1_pd(value);
	    __m512d intemp = _mm512_load_pd(inbase);
	    
	    outtemp = _mm512_fmadd_pd(multiplier, intemp, outtemp);
	    // for (int k=0; k<nbvector; ++k) {
	    //   outtemp[k] += inbase[k] * value;
	    // }
	    
	    ++val2;
	  }

	Scalar* outbase = out+i*nbvector;
	//	std::cout<<outbase<<std::endl;
	STORE_INST(outbase, outtemp);
	//memcpy(outbase, outtemp, nbvector*sizeof(Scalar));
      }
  }
}


template<>
void my_spmm<int,int,double,8>(int nVtx, int*xadj, int *adj, double* val, double *in, double* out) {
  typedef int VertexType;
  typedef int EdgeType;
  typedef double Scalar;
  const int nbvector=8; // changed from original of 8 (fill MIC vector register completely)
#pragma omp parallel 
  {
#pragma omp for schedule(runtime)
    for (VertexType i = 0; i < nVtx; ++i)
      {
	//do output = 0.;
	
	VertexType* beg = adj+xadj[i];
	VertexType* end = adj+xadj[i+1];
	Scalar* val2 = val+xadj[i];
	
	//Scalar outtemp[nbvector];
	
	//for (int k=0; k<nbvector;++k)
	//outtemp[k] = 0.;
	__m512d outtemp = _mm512_setzero_pd();


	for (auto p = beg; p < end; ++p)
	  {
	    EdgeType j = *p;
	    Scalar value = *val2;
	    Scalar* inbase = in+j*nbvector;

	    __m512d multiplier = _mm512_set1_pd(value);
	    __m512d intemp = _mm512_load_pd(inbase);
	    
	    outtemp = _mm512_fmadd_pd(multiplier, intemp, outtemp);
	    // for (int k=0; k<nbvector; ++k) {
	    //   outtemp[k] += inbase[k] * value;
	    // }
	    
	    ++val2;
	  }

	Scalar* outbase = out+i*nbvector;
	//	std::cout<<outbase<<std::endl;
	STORE_INST(outbase, outtemp);
	//memcpy(outbase, outtemp, nbvector*sizeof(Scalar));
      }
  }
}

template<>
void my_spmm<int,int,double,16>(int nVtx, int*xadj, int *adj, double* val, double *in, double* out) {
  typedef int VertexType;
  typedef int EdgeType;
  typedef double Scalar;
  const int nbvector=16;
#pragma omp parallel 
  {
#pragma omp for schedule(runtime)
    for (VertexType i = 0; i < nVtx; ++i)
      {
	//do output = 0.;
	
	VertexType* beg = adj+xadj[i];
	VertexType* end = adj+xadj[i+1];
	Scalar* val2 = val+xadj[i];
	
	//Scalar outtemp[nbvector];
	
	//for (int k=0; k<nbvector;++k)
	//outtemp[k] = 0.;
	__m512d outtemp1 = _mm512_setzero_pd();
	__m512d outtemp2 = _mm512_setzero_pd();
	

	for (auto p = beg; p < end; ++p)
	  {
	    EdgeType j = *p;
	    Scalar value = *val2;
	    Scalar* inbase = in+j*nbvector;

	    __m512d multiplier = _mm512_set1_pd(value);
	    __m512d intemp1 = _mm512_load_pd(inbase);
	    __m512d intemp2 = _mm512_load_pd(inbase+8);
	    
	    outtemp1 = _mm512_fmadd_pd(multiplier, intemp1, outtemp1);
	    outtemp2 = _mm512_fmadd_pd(multiplier, intemp2, outtemp2);
	    // for (int k=0; k<nbvector; ++k) {
	    //   outtemp[k] += inbase[k] * value;
	    // }
	    
	    ++val2;
	  }

	Scalar* outbase = out+i*nbvector;
	//	std::cout<<outbase<<std::endl;
	STORE_INST(outbase, outtemp1);
	STORE_INST(outbase+8, outtemp2);
	//_mm512_storenrngo_pd(outbase, outtemp1);
	//_mm512_storenrngo_pd(outbase+8, outtemp2);

	//memcpy(outbase, outtemp, nbvector*sizeof(Scalar));
      }
  }
}


template<>
void my_spmm<int,int,double,32>(int nVtx, int*xadj, int *adj, double* val, double *in, double* out) {
  typedef int VertexType;
  typedef int EdgeType;
  typedef double Scalar;
  const int nbvector=32;
#pragma omp parallel 
  {
#pragma omp for schedule(runtime)
    for (VertexType i = 0; i < nVtx; ++i)
      {
	//do output = 0.;
	
	VertexType* beg = adj+xadj[i];
	VertexType* end = adj+xadj[i+1];
	Scalar* val2 = val+xadj[i];
	
	//Scalar outtemp[nbvector];
	
	//for (int k=0; k<nbvector;++k)
	//outtemp[k] = 0.;
	__m512d outtemp1 = _mm512_setzero_pd();
	__m512d outtemp2 = _mm512_setzero_pd();
	__m512d outtemp3 = _mm512_setzero_pd();
	__m512d outtemp4 = _mm512_setzero_pd();
	

	for (auto p = beg; p < end; ++p)
	  {
	    EdgeType j = *p;
	    Scalar value = *val2;
	    Scalar* inbase = in+j*nbvector;

	    __m512d multiplier = _mm512_set1_pd(value);
	    __m512d intemp1 = _mm512_load_pd(inbase);
	    __m512d intemp2 = _mm512_load_pd(inbase+8);
	    __m512d intemp3 = _mm512_load_pd(inbase+16);
	    __m512d intemp4 = _mm512_load_pd(inbase+24);
	    
	    outtemp1 = _mm512_fmadd_pd(multiplier, intemp1, outtemp1);
	    outtemp2 = _mm512_fmadd_pd(multiplier, intemp2, outtemp2);
	    outtemp3 = _mm512_fmadd_pd(multiplier, intemp3, outtemp3);
	    outtemp4 = _mm512_fmadd_pd(multiplier, intemp4, outtemp4);
	    // for (int k=0; k<nbvector; ++k) {
	    //   outtemp[k] += inbase[k] * value;
	    // }
	    
	    ++val2;
	  }

	Scalar* outbase = out+i*nbvector;
	//	std::cout<<outbase<<std::endl;
	STORE_INST(outbase, outtemp1);
	STORE_INST(outbase+8, outtemp2);
	STORE_INST(outbase+16, outtemp3);
	STORE_INST(outbase+24, outtemp4);
	//memcpy(outbase, outtemp, nbvector*sizeof(Scalar));
      }
  }
}

#endif

#ifndef MIC
template<>
void my_spmm<int,int,double,32>(int nVtx, int*xadj, int *adj, double* val, double *in, double* out) {
  typedef int VertexType;
  typedef int EdgeType;
  typedef double Scalar;
  const int nbvector=32;
#pragma omp parallel 
  {
#pragma omp for schedule(runtime)
    for (VertexType i = 0; i < nVtx; ++i)
      {
	//do output = 0.;
	
	VertexType* beg = adj+xadj[i];
	VertexType* end = adj+xadj[i+1];
	Scalar* val2 = val+xadj[i];
	
	//Scalar outtemp[nbvector];
	
	//for (int k=0; k<nbvector;++k)
	//outtemp[k] = 0.;
	__m128d outtemp [nbvector/2];
	for (int x=0; x<nbvector/2; ++x)
	  outtemp[x] = _mm_setzero_pd();
	

	for (auto p = beg; p < end; ++p)
	  {
	    EdgeType j = *p;
	    Scalar value = *val2;
	    Scalar* inbase = in+j*nbvector;

	    __m128d multiplier = _mm_set1_pd(value);
	    for (int x=0; x<nbvector/2; ++x) {
	      __m128d intemp1 = _mm_load_pd(inbase+2*x);
	      //	      outtemp[x] = _mm_fmadd_pd(multiplier, intemp1, outtemp[x]);//only avx2 support FMA but neither westmere or sandy bridge supports it
	      intemp1 = _mm_mul_pd( intemp1, multiplier);
	      outtemp[x] = _mm_add_pd( intemp1, outtemp[x]);	      
	    }
	    // for (int k=0; k<nbvector; ++k) {
	    //   outtemp[k] += inbase[k] * value;
	    // }
	    
	    ++val2;
	  }

	Scalar* outbase = out+i*nbvector;
	//	std::cout<<outbase<<std::endl;
	for (int x=0; x<nbvector/2; ++x) {
	  _mm_store_pd(outbase+2*x, outtemp[x]);
	}
	//memcpy(outbase, outtemp, nbvector*sizeof(Scalar));
      }
  }
}

template<>
void my_spmm<int,int,double,16>(int nVtx, int*xadj, int *adj, double* val, double *in, double* out) {
  typedef int VertexType;
  typedef int EdgeType;
  typedef double Scalar;
  const int nbvector=16;
#pragma omp parallel 
  {
#pragma omp for schedule(runtime)
    for (VertexType i = 0; i < nVtx; ++i)
      {
	//do output = 0.;
	
	VertexType* beg = adj+xadj[i];
	VertexType* end = adj+xadj[i+1];
	Scalar* val2 = val+xadj[i];
	
	//Scalar outtemp[nbvector];
	
	//for (int k=0; k<nbvector;++k)
	//outtemp[k] = 0.;
	__m128d outtemp [nbvector/2];
	for (int x=0; x<nbvector/2; ++x)
	  outtemp[x] = _mm_setzero_pd();
	

	for (auto p = beg; p < end; ++p)
	  {
	    EdgeType j = *p;
	    Scalar value = *val2;
	    Scalar* inbase = in+j*nbvector;

	    __m128d multiplier = _mm_set1_pd(value);
	    for (int x=0; x<nbvector/2; ++x) {
	      __m128d intemp1 = _mm_load_pd(inbase+2*x);
	      //	      outtemp[x] = _mm_fmadd_pd(multiplier, intemp1, outtemp[x]);//only avx2 support FMA but neither westmere or sandy bridge supports it
	      intemp1 = _mm_mul_pd( intemp1, multiplier);
	      outtemp[x] = _mm_add_pd( intemp1, outtemp[x]);	      
	    }
	    // for (int k=0; k<nbvector; ++k) {
	    //   outtemp[k] += inbase[k] * value;
	    // }
	    
	    ++val2;
	  }

	Scalar* outbase = out+i*nbvector;
	//	std::cout<<outbase<<std::endl;
	for (int x=0; x<nbvector/2; ++x) {
	  _mm_store_pd(outbase+2*x, outtemp[x]);
	}
	//memcpy(outbase, outtemp, nbvector*sizeof(Scalar));
      }
  }
}
#endif

template<typename VertexType, typename EdgeType, typename Scalar>
void my_spmm(VertexType nVtx, EdgeType*xadj, VertexType *adj, Scalar* val, Scalar *in, Scalar* out, int nbvector) {
#pragma omp parallel for schedule(runtime)
      for (VertexType i = 0; i < nVtx; ++i)
	{
	  Scalar output = 0.;

	  VertexType* beg = adj+xadj[i];
	  VertexType* end = adj+xadj[i+1];
	  Scalar* val2 = val+xadj[i];
	  Scalar* outbase = out+i*nbvector;
	  for (auto p = beg; p < end; ++p)
	    {
	      auto j = *p;
	      auto value = *val2;
	      Scalar* inbase = in+j*nbvector;

	      
	      for (int k=0; k<nbvector; ++k) {
		outbase[k] += inbase[k] * value;
	      }
	
	      ++val2;
	    }
	}
}

template<typename VertexType, typename EdgeType, typename Scalar>
int main_spmm(VertexType nVtx, EdgeType*xadj, VertexType *adj, Scalar* val, Scalar *in, Scalar* out, int nbvector,
	      int nTry, //algo parameter
	      util::timestamp& totaltime, std::string& algo_out
	      )
{
  totaltime = util::timestamp(0,0);
  util::timestamp start(0,0);
  util::timestamp stop(0,0);
  bool coldcache = true;

  for (int TRY=0; TRY<THROW_AWAY+nTry; ++TRY)
    {
      if (TRY >= THROW_AWAY)
	start = util::timestamp();

#pragma omp parallel for schedule(static)
      for (VertexType i = 0; i< nbvector*nVtx; ++i)
	out[i] = 0.;      

      if (nbvector == 8)
	my_spmm<VertexType, EdgeType, Scalar, 8> (nVtx, xadj, adj, val, in, out);
      else if (nbvector == 4)
	my_spmm<VertexType, EdgeType, Scalar, 4> (nVtx, xadj, adj, val, in, out); // seg fault
      else if (nbvector == 16)
	my_spmm<VertexType, EdgeType, Scalar, 16> (nVtx, xadj, adj, val, in, out);
      else if (nbvector == 32)
	my_spmm<VertexType, EdgeType, Scalar, 32> (nVtx, xadj, adj, val, in, out);
      else if (nbvector == 64)
	my_spmm<VertexType, EdgeType, Scalar, 64> (nVtx, xadj, adj, val, in, out);
      else
	my_spmm<VertexType, EdgeType, Scalar>(nVtx, xadj, adj, val, in, out, nbvector);


      if (TRY >= THROW_AWAY) {
	stop = util::timestamp();
	totaltime += stop - start;
      }


      if (coldcache) {
#pragma omp parallel
	{
	  evict_array_from_cache(adj, xadj[nVtx]*sizeof(*adj));
	  evict_array_from_cache(xadj, (nVtx+1)*sizeof(*xadj));
	  evict_array_from_cache(val, xadj[nVtx]*sizeof(*val));
	  evict_array_from_cache(in, nVtx*sizeof(*in)*nbvector);
	  evict_array_from_cache(out, nVtx*sizeof(*out)*nbvector);

#pragma omp barrier
	}
      }

    }

  return 0;
}



