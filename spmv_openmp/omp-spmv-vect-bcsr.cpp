#include <algorithm>
#include "main-spmv.hpp"
#include <map>
#include <immintrin.h>

#include <omp.h>

#include "Padded2DArray.hpp"
#include "memutils.hpp"

#define THROW_AWAY 10

//#define BR 8
//#define BC 8
//#define RMJR
//#define RMJR2
//#define CMJR
//#define SWI
#define DEBUG

#ifndef BC
#error BC undefined
#endif

#ifndef BR
#error BR undefined
#endif

//#define SHOWLOADBALANCE

#define _mm512_loadnr_pd(block) _mm512_extload_pd(block, _MM_UPCONV_PD_NONE, _MM_BROADCAST64_NONE, _MM_HINT_NT)

using namespace std;

template<typename VertexType, typename EdgeType, typename Scalar>
int main_spmv(VertexType nVtx, EdgeType* xadj, VertexType *adj, Scalar* val, Scalar *in, Scalar* out,
	      int nTry, //algo parameter
	      util::timestamp& totaltime, std::string& algo_out
	      )
{
  util::timestamp start(0,0);

  util::timestamp cacheflush(0,0);

  bool coldcache = true;
  
  typedef long int blockid;

  VertexType m = nVtx;
  EdgeType nz = xadj[m];
  int perbnnz = BR*BC;

  VertexType nbr = ceil((Scalar)m / BR); 
  VertexType nbc = ceil((Scalar)m / BC); 
 
  std::cerr<<"nbr "<<nbr<<" nbc "<<nbc<<std::endl;

  blockid* bids = (blockid*) malloc(sizeof(blockid) * nz);
  for(VertexType i = 0; i < m; i++) {    
    for (EdgeType p = xadj[i]; p < xadj[i+1]; p++) {
      bids[p] = ((blockid)nbc) * (i/BR) + (adj[p]/BC);                  
    }
  }
  std::sort (bids,bids+nz); 


  if ( bids[0] < 0) {
    std::cerr<<"something dead wrong happened on "<<__FILE__<<":"<<__LINE__<<" (certainly integer overflow)"<<std::endl;
    std::cerr<<"quitting"<<std::endl;
    exit (-1);
  }
 
#ifdef DEBUG
  for(EdgeType i = 1; i < nz; i++) {
    if(bids[i] < bids[i-1]) {
      printf("is not sorted");
      exit(1);
    }    
  }
#endif

  VertexType* bptrs = (VertexType*) malloc(sizeof(VertexType) * (nbr + 1));
  memset (bptrs, 0, sizeof(VertexType) * (nbr + 1));

  std::map<blockid, EdgeType> border;
  
  bptrs[0] = 0;
  VertexType rcount = 0;
  EdgeType bcount = 0;    

  {
    blockid prev = -1;    
    for(EdgeType i = 0 ; i < nz; i++) {            
      blockid bid = bids[i];
      VertexType rid = bid / nbc;
      if(bid != prev){
	prev = bid;
	while(rcount < rid) {
	  bptrs[rcount++] = bcount;
	}
	
	border[bid] = bcount;  
	bids[bcount++] = (bid % nbc)*BC; 
      }
    }

    //taking care of potential empty rows and last end pointer
    while(rcount <= nbr) {
      bptrs[rcount++] = bcount;
    }
  }
  std::cerr<<"there are "<<bcount<<" blocks "<<nz<<" nzs: density is "<<(double)nz / (bcount*BC*BR)<<std::endl;

  //  Scalar* bnnz = (Scalar*) malloc(sizeof(Scalar) * bcount * perbnnz);
  Scalar* bnnz = (Scalar*) _mm_malloc(sizeof(Scalar) * bcount * perbnnz, 64);
  memset(bnnz, 0, sizeof(Scalar) * bcount * perbnnz);
  
  for(VertexType i = 0; i < m; i++) {    
    for (EdgeType p = xadj[i]; p < xadj[i+1]; p++) {
      VertexType j = adj[p];
      blockid bid = (blockid) nbc * (i/BR) + (j/BC);             
      VertexType rid = bid / nbc;

      EdgeType loc = border[bid] * perbnnz; //blockposition
      VertexType reli = i % BR;
      VertexType relj = j % BC;
      
      EdgeType relloc; //offset in the block

#ifdef RMJR 
      relloc = reli * BC + BR;
#endif
#ifdef RMJR2
      relloc = reli * BC + BR;
#endif
#ifdef CMJR
      relloc = relj * BR + BC;
#endif
#ifdef SWI
      //warning computes the wrong results
      relloc = relj * BR + BC; 
#endif

      bnnz[loc + relloc] = val[p];      
    }
  }

  {
    VertexType maxdegree = 0;
    for (VertexType blrow = 0; blrow < nbr; ++blrow) {
      VertexType degree = bptrs[blrow+1] - bptrs[blrow];
      if ( degree < 0) {
	std::cerr<<"something dead wrong happened on "<<__FILE__<<":"<<__LINE__<<" (certainly integer overflow)"<<std::endl;
	std::cerr<<"quitting"<<std::endl;
	exit (-1);
      }

      maxdegree = std::max(degree,maxdegree);
    }
    std::cerr<<"max block degree "<<maxdegree<<std::endl;
    std::cerr<<"bptrs[nbr] "<<bptrs[nbr]<<std::endl;
  }


#ifdef SHOWLOADBALANCE
  Padded1DArray<int> count (244);

  for (int i=0; i<244; ++i)
    count[i] = 0;
#endif

  printf ("starting multiply\n");
    
  for (int TRY=0; TRY<THROW_AWAY+nTry; ++TRY)
    {
      if (TRY == THROW_AWAY) {
	start = util::timestamp();
	cacheflush = util::timestamp (0,0);
      }

#ifndef MIC
#pragma omp parallel
      {
	int tid = omp_get_thread_num();
#ifdef SHOWLOADBALANCE
	count[tid] = 0;
#endif
	
#pragma omp for schedule(runtime)
	for (VertexType blrow = 0; blrow < nbr; ++blrow) {
#ifdef SHOWLOADBALANCE
	  ++count[tid];
#endif

	  Scalar output[BR];
	  memset (output, 0, sizeof(Scalar)*BR);
	  
	  //for each block on a row
	  for (EdgeType p = bptrs[blrow]; p != bptrs[blrow+1]; ++p) {
	    Scalar* input = in+bids[p];
	    
	    Scalar* block = bnnz+p*perbnnz;
#ifdef RMJR
	    for (int br=0; br<BR;++br) {
	      for (int bc=0; bc<BC;++bc) {
		output[br] += input[bc] * (*block);
		++block;
	      }
	    }
#endif
#ifdef CMJR
	    //column major
	    for (int bc=0; bc<BC;++bc) {
	      for (int br=0; br<BR;++br) {
		output[br] += input[bc] * (*block);
		++block;
	      }
	    }
#endif
#ifdef SWI
#error swizzle unsupported on CPU
#endif
	  }
	  //copy local computatin to global memory
	  memcpy (&(out[blrow*BR]), output, sizeof(Scalar)*BR);
	}
      }
#endif

#ifdef MIC
      assert (sizeof(Scalar) == 8);
#ifdef CMJR
      assert (BR == 8);
      //assume column major
#pragma omp parallel 
      {
	int tid = omp_get_thread_num();
#ifdef SHOWLOADBALANCE
	count[tid] = 0;
#endif
#pragma omp for schedule(runtime)
	for (VertexType blrow = 0; blrow < nbr; ++blrow) {
#ifdef SHOWLOADBALANCE
	  ++count[tid];
#endif
	  __m512d outtemp = _mm512_setzero_pd();
	  
	  //for each block on a row
	  auto endptr = bptrs[blrow+1];
	  for (EdgeType p = bptrs[blrow]; p != endptr; ++p) {
	    
	    Scalar* input = in+bids[p];
	    //_mm_prefetch ((char*)(bids+p)+256 , _MM_HINT_T0);
	    Scalar* block = bnnz+p*perbnnz;
	    //_mm_prefetch ((char*)(bnnz+p*perbnnz)+256 , _MM_HINT_T0);
	    
	    for (int bc=0; bc<BC;++bc) {
	      __m512d multiplier = _mm512_set1_pd(input[bc]);
	      __m512d intemp = _mm512_loadnr_pd(block); //no need to keep the block in the cache
	      //	      _mm_prefetch ((char*)(block)+512 , _MM_HINT_T0);
	      block += BR;
	      outtemp = _mm512_fmadd_pd(multiplier, intemp, outtemp);
	    }
	  }  
	  
	  //	  _mm512_storenr_pd(&(out[blrow*BR]), outtemp);
	  _mm512_storenrngo_pd(&(out[blrow*BR]), outtemp);
	}
      }
#endif

#ifdef RMJR
      //assume row major
      assert (BC == 8);
#pragma omp parallel 
      {
	int tid = omp_get_thread_num();
#ifdef SHOWLOADBALANCE
	count[tid] = 0;
#endif
#pragma omp for schedule(runtime)
	for (VertexType blrow = 0; blrow < nbr; ++blrow) {
#ifdef SHOWLOADBALANCE
	  ++count[tid];
#endif

	  Scalar output[BR];
	  memset (output, 0, sizeof(Scalar)*BR);
	  
	  //for each block on a row
	  for (EdgeType p = bptrs[blrow]; p != bptrs[blrow+1]; ++p) {
	    
	    Scalar* input = in+bids[p];
	    Scalar* block = bnnz+p*perbnnz;
	    __m512d intemp = _mm512_load_pd(input);
	    
	    
	    for (int br=0; br<BR;++br) {
	      __m512d nze = _mm512_loadnr_pd(block); //no need to keep the block in the cache
	      block += BC;
	      
	      __m512d inter = _mm512_mul_pd(nze, intemp);
	      output[br] += _mm512_reduce_add_pd(inter);
	    }
	  }  
	  
	  memcpy (&(out[blrow*BR]), output, sizeof(Scalar)*BR);
	}
      }
#endif


#ifdef RMJR2
      //assume row major
      assert (BC == 8);
#pragma omp parallel
      {
	int tid = omp_get_thread_num();
#ifdef SHOWLOADBALANCE
	count[tid] = 0;
#endif
#pragma omp for schedule(runtime)
	for (VertexType blrow = 0; blrow < nbr; ++blrow) {
#ifdef SHOWLOADBALANCE
	  ++count[tid];
#endif

	  Scalar output[BR];
	  memset (output, 0, sizeof(Scalar)*BR);
	  
	  __m512d outtemp[BR];
	  for (int br = 0; br< BR; ++br)
	    outtemp[br] = _mm512_setzero_pd();
	  
	  //for each block on a row
	  for (EdgeType p = bptrs[blrow]; p != bptrs[blrow+1]; ++p) {
	    
	    Scalar* input = in+bids[p];
	    Scalar* block = bnnz+p*perbnnz;
	    __m512d intemp = _mm512_load_pd(input);
	    
	    
	    for (int br=0; br<BR;++br) {
	      __m512d nze = _mm512_loadnr_pd(block); //no need to keep the block in the cache
	      block += BC;
	      
	      outtemp[br] = _mm512_fmadd_pd(nze, intemp, outtemp[br]);
	    }
	  }  
	  
	  for (int br = 0; br< BR; ++br)
	    output[br] = _mm512_reduce_add_pd(outtemp[br]);
	  
	  memcpy (&(out[blrow*BR]), output, sizeof(Scalar)*BR);
	}
      }
#endif



#ifdef SWI
      assert (BC == 8);
      assert (BR == 8);

#pragma omp parallel 
      {
	int tid = omp_get_thread_num();
#ifdef SHOWLOADBALANCE
	count[tid] = 0;
#endif
#pragma omp for schedule(runtime)
	for (VertexType blrow = 0; blrow < nbr; ++blrow) {
#ifdef SHOWLOADBALANCE
	  ++count[tid];
#endif
	  __m512d outtemp = _mm512_setzero_pd();
	  
	  //for each block on a row
	  for (EdgeType p = bptrs[blrow]; p != bptrs[blrow+1]; ++p) {
	    
	    Scalar* input = in+bids[p];
	    Scalar* block = bnnz+p*perbnnz;
	    
	    
	    __m512d multiplier = _mm512_load_pd(input);
	    //	  for (int bc=0; bc<BC/2;++bc) {
	    //1
	    {
	      __m512d intemp1 = _mm512_loadnr_pd(block); //no need to keep the block in the cache
	      block += BR;
	      outtemp = _mm512_fmadd_pd(multiplier, intemp1, outtemp);
	      
	      __m512d intemp2 = _mm512_loadnr_pd(block); //no need to keep the block in the cache
	      block += BR;
	      outtemp =_mm512_fmadd_pd(_mm512_swizzle_pd(multiplier, _MM_SWIZ_REG_BADC), intemp2, outtemp);
	      
	      multiplier = (__m512d) _mm512_permute4f128_ps ( (__m512) multiplier, _MM_PERM_ADCB);
	    }
	    //2
	    {
	      __m512d intemp1 = _mm512_loadnr_pd(block); //no need to keep the block in the cache
	      block += BR;
	      outtemp = _mm512_fmadd_pd(multiplier, intemp1, outtemp);
	      
	      __m512d intemp2 = _mm512_loadnr_pd(block); //no need to keep the block in the cache
	      block += BR;
	      outtemp =_mm512_fmadd_pd(_mm512_swizzle_pd(multiplier, _MM_SWIZ_REG_BADC), intemp2, outtemp);
	      
	      multiplier = (__m512d) _mm512_permute4f128_ps ( (__m512) multiplier, _MM_PERM_ADCB);
	    }
	    //3
	    {
	      __m512d intemp1 = _mm512_loadnr_pd(block); //no need to keep the block in the cache
	      block += BR;
	      outtemp = _mm512_fmadd_pd(multiplier, intemp1, outtemp);
	      
	      __m512d intemp2 = _mm512_loadnr_pd(block); //no need to keep the block in the cache
	      block += BR;
	      outtemp =_mm512_fmadd_pd(_mm512_swizzle_pd(multiplier, _MM_SWIZ_REG_BADC), intemp2, outtemp);
	      
	      multiplier = (__m512d) _mm512_permute4f128_ps ( (__m512) multiplier, _MM_PERM_ADCB);
	    }
	    //4
	    {
	      __m512d intemp1 = _mm512_loadnr_pd(block); //no need to keep the block in the cache
	      block += BR;
	      outtemp = _mm512_fmadd_pd(multiplier, intemp1, outtemp);
	      
	      __m512d intemp2 = _mm512_loadnr_pd(block); //no need to keep the block in the cache
	      //block += BR;
	      outtemp =_mm512_fmadd_pd(_mm512_swizzle_pd(multiplier, _MM_SWIZ_REG_BADC), intemp2, outtemp);
	      
	      
	    }
	    
	  }  
	  
	  //	  _mm512_storenr_pd(&(out[blrow*BR]), outtemp);
	  _mm512_storenrngo_pd(&(out[blrow*BR]), outtemp);
	}

      }
#endif

 
#endif

      if (coldcache) {
#pragma omp parallel
	{
	  util::timestamp beg(0,0);
#pragma omp master
	  beg = util::timestamp();

	  evict_array_from_cache(bids, bptrs[nbr]*sizeof(*bids));
	  evict_array_from_cache(bptrs, (nbr+1)*sizeof(*bptrs));
	  evict_array_from_cache(bnnz, sizeof(*bnnz) * bcount * perbnnz);
	  evict_array_from_cache(in, nVtx*sizeof(*in));
	  evict_array_from_cache(out, nVtx*sizeof(*out));
	  util::timestamp end(0,0);
#pragma omp barrier
#pragma omp master
	  {
	    end = util::timestamp();
	    cacheflush += end-beg;
	  }
	}
      }

    }
  util::timestamp stop;  


#ifdef SHOWLOADBALANCE
  std::cout<<"load balance"<<std::endl;
  for (int i=0; i< 244; ++i)
    std::cout<<count[i]<<std::endl;
#endif

  totaltime += stop - start - cacheflush;

  return 0;
}



