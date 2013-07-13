#include "timestamp.hpp"
#include <string>
#include "graph.hpp"
#include <malloc.h>
#include <fstream>
#include <xmmintrin.h>


template<typename VertexType, typename EdgeType, typename Scalar>
int main_spmm(VertexType nVtx, EdgeType*xadj, VertexType *adj, Scalar* val, Scalar *in, Scalar* out, int nbvector,
	      int nTry, //algo parameter
	      util::timestamp& totaltime, std::string& algo_out
	      );

int main(int argc, char *argv[])
{
  int nVtx, nCol, nEdge, *xadj, *adj, nTry = 1;
  char *filename = argv[1];
  char timestr[20];
  
  if (argc < 2) {
      fprintf(stderr, "usage: %s <filename> [nTry]\n", argv[0]);
      exit(1);
  }
    
  if (argc >= 3) {
      nTry = atoi(argv[2]);
      if (!nTry) {
          fprintf(stderr, "nTry was 0, setting it to 1\n");
          nTry = 1;
      } 
  } 

  double* val;
  ReadGraph<int,int,double>(argv[1], &nVtx, &nCol, &xadj, &adj, &val, NULL);
  int nbvector = 8; // changed from original of 16
  nEdge = xadj[nVtx];

  //double* out = new double[nVtx*nbvector];
  //double* in = new double[nVtx*nbvector];
  
  double* in = (double*) _mm_malloc(nVtx*nbvector*sizeof(double), 64);
  double* out = (double*) _mm_malloc(nVtx*nbvector*sizeof(double), 64);

  // double* out = (double*) ((long)out_alloc + 64-(long)out_alloc%64);
  // double* in = (double*) ((long)in_alloc + 64-(long)in_alloc%64);


  // if (in == NULL || out == NULL) {
  //   std::cerr<<"WTF"<<std::endl;
  // }


  // posix_memalign((void**)&in, 64, nVtx*nbvector);
  // posix_memalign((void**)&out, 64, nVtx*nbvector);


  for (int i=0; i<nVtx*nbvector; ++i)
    in[i] = (double) (i%100);

  //alarm(2400); //most likely our run will not take that long. When the alarm is triggered, there will be no callback to process it and th eprocess will die.

  if (strrchr(argv[1], '/'))
      filename = strrchr(argv[1], '/') + 1;
  
  util::timestamp totaltime(0,0);  
  
  std::string algo_out;

  std::cout<<"graph read"<<std::endl;

  main_spmm<int,int,double>(nVtx, xadj, adj, val, in, out, nbvector, nTry, totaltime, algo_out);
  
  totaltime /= nTry;
  totaltime.to_c_str(timestr, 20);

  long int totalsize = nVtx*nbvector*sizeof(double)*2 //in and out
                     +(nVtx+1)*sizeof(int)  //xadj
                     +xadj[nVtx]*(sizeof(int)+sizeof(double)); //adj and val

  std::cout<<"filename: "<<filename
	   <<" nVtx: "<<nVtx
	   <<" nonzero: "<<nEdge
	   <<" AvgTime: "<<(double)totaltime
	   <<" Gflops: "<<2.*nbvector*((double)nEdge)/((double)totaltime)/1000/1000/1000
           <<" Bandwidth: "<<totalsize/((double)totaltime)/1000/1000/1000
	   <<" "<<algo_out<<std::endl;
  



  // std::ofstream outfile ("a");
  // for (int i=0; i< nVtx*nbvector; ++i) {
  //   outfile<<out[i]<<'\n';
  // }
  // outfile<<std::flush;
  // outfile.close();
    


  //  delete[] in;
  //  delete[] out;

  // free(in);
  // free(out);

  //_mm_free(in);
  //_mm_free(out);

  free(adj);
  free(xadj);
  free(val);
  
  return 0;
}
