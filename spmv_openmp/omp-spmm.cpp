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



