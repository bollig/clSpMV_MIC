/*
  File  : graph.h
  Author: UVC
  Date  : Oct 1st, 2004
  Descr : chaco/metis format graph reader
*/
#ifndef _GRAPH_H_
#define _GRAPH_H_

#include "graph_impl.hpp"

//void ReadGraph(char *filename, int *numofvertex, int **pxadj, int **padjncy, int **padjncyw, int **ppvw);


///generates a dense matrix of size columns and rows
template<typename VertexType, typename EdgeType, typename Scalar>
void generateDense (VertexType *nVtx, VertexType *nCol,
		    EdgeType** xadj, VertexType **adj, Scalar** val, 
		    VertexType size) {
  std::cerr<<"generating dense "<<size<<std::endl;

  *nCol = *nVtx = size;

  *xadj = (EdgeType* ) malloc(sizeof(EdgeType)*(size+1));

  *adj = (VertexType*) malloc(sizeof(VertexType)*size*size);
  *val = (Scalar*) malloc(sizeof(Scalar)*size*size);


  for (VertexType i=0; i<=size; ++i)
    (*xadj)[i] = i*size;

#pragma omp parallel for
  for (EdgeType i=0; i<size*size; ++i) {
    (*adj)[i] = i%size;
    (*val)[i] = 1.;
  }

}

///generate a banded matrix of size columns and rows and a bandwidth
///of bandsize
template<typename VertexType, typename EdgeType, typename Scalar>
void generateBanded (VertexType *nVtx, VertexType *nCol,
		     EdgeType** xadj, VertexType **adj, Scalar** val,
		     VertexType size, VertexType bandsize) {
  std::cerr<<"generating banded "<<size<<" "<<bandsize<<std::endl;

  *nCol = *nVtx = size;

  *xadj = (EdgeType* ) malloc(sizeof(EdgeType)*(size+1));

  *adj = (VertexType*) malloc(sizeof(VertexType)*size*bandsize);
  *val = (Scalar*) malloc(sizeof(Scalar)*size*bandsize);


  for (VertexType i=0; i<=size; ++i)
    (*xadj)[i] = i*bandsize;

#pragma omp parallel for
  for (EdgeType i=0; i<bandsize*size; ++i) {
    (*adj)[i] = (i/bandsize+i%bandsize)%size;

    (*val)[i] = 1.;
  }
}

///generate a complete binary tree of a given depth
template<typename VertexType, typename EdgeType, typename Scalar>
void generateBinaryTree (VertexType *nVtx, VertexType *nCol,
		     EdgeType** xadj, VertexType **adj, Scalar** val,
		     VertexType depth) {
  VertexType size = (1<<depth) - 1;

  //VertexType nbinternode = (1<<(depth-1))-1;

  EdgeType nbedge = 2*(size-1); //symetric graph (so x2)

  std::cerr<<"generating tree "<<size<<std::endl;

  *nCol = *nVtx = size;

  *xadj = (EdgeType* ) malloc(sizeof(EdgeType)*(size+1));

  *adj = (VertexType*) malloc(sizeof(VertexType)*nbedge);
  *val = (Scalar*) malloc(sizeof(Scalar)*nbedge);


  (*xadj)[0] = 0;
  for (VertexType i=0; i<size; ++i) {
    VertexType btreeid = i+1;
    (*xadj)[btreeid] = (*xadj)[btreeid-1];

    //parent edge
    VertexType btreepid = btreeid/2;
    if (btreeid != 1) {
      (*adj)[(*xadj)[btreeid]] = btreepid -1;
      (*val)[(*xadj)[btreeid]] = 1.;
      (*xadj)[btreeid]++;
    }
    
    //left child
    VertexType btreelcid = 2*btreeid;
    if (btreelcid <= size) {
      (*adj)[(*xadj)[btreeid]] = btreelcid -1;
      (*val)[(*xadj)[btreeid]] = 1.;
      (*xadj)[btreeid]++;
    }

    //right child
    VertexType btreercid = 2*btreeid+1;
    if (btreercid <= size) {
      (*adj)[(*xadj)[btreeid]] = btreercid -1;
      (*val)[(*xadj)[btreeid]] = 1.;
      (*xadj)[btreeid]++;    
    }
  }
  
  std::cout<<"generated "<<(*xadj)[size]<<" edges (should be :"<<nbedge<<")"<<std::endl;
}

///generates a sizeXsize matrix where there are degree vertices on
///each row which are regularly spaced by spacing columns starting
///from the diagonal
template <typename VertexType, typename EdgeType, typename Scalar>
void generateRegularSpaced (VertexType *nVtx, VertexType *nCol,
			    EdgeType** xadj, VertexType **adj, Scalar** val,
			    VertexType size, VertexType degree, VertexType spacing) {
  assert (degree>0);
  *nVtx = *nCol = size;
  *xadj = (EdgeType*) malloc(sizeof(EdgeType)*(size+1));
  *adj = (VertexType*) malloc(sizeof(VertexType)*size*degree);
  *val = (Scalar*) malloc(sizeof(Scalar)*size*degree);
  
#pragma omp parallel for schedule(runtime)
  for (VertexType u = 0; u<size; ++u) {
    (*xadj)[u] = u*degree;
    for (VertexType off = 0; off<degree; ++off) {
      EdgeType edgeposition = (*xadj)[u]+off;
      (*adj)[edgeposition] = (u+off*spacing)%size;
      (*val)[edgeposition] = 1.;
    }
  } 
  (*xadj)[size] = degree*size;
}


///generates a sizeXsize matrix where there are degree vertices on
///each row which are regularly spaced by spacing columns starting
///where the previous row left.
template <typename VertexType, typename EdgeType, typename Scalar>
void generateRegularSpacedNoCache (VertexType *nVtx, VertexType *nCol,
				   EdgeType** xadj, VertexType **adj, Scalar** val,
				   VertexType size, VertexType degree, VertexType spacing) {
  assert (degree>0);
  *nVtx = *nCol = size;
  *xadj = (EdgeType*) malloc(sizeof(EdgeType)*(size+1));
  *adj = (VertexType*) malloc(sizeof(VertexType)*size*degree);
  *val = (Scalar*) malloc(sizeof(Scalar)*size*degree);
  
#pragma omp parallel for schedule(runtime)
  for (VertexType u = 0; u<size; ++u) {
    (*xadj)[u] = u*degree;
    for (VertexType off = 0; off<degree; ++off) {
      EdgeType edgeposition = (*xadj)[u]+off;
      (*adj)[edgeposition] = ((edgeposition)%size)*spacing%size;
      (*val)[edgeposition] = 1.;
    }
  } 
  (*xadj)[size] = degree*size;
}

///generates a sizeXsize matrix where all the diagonal is full and
///there are degree vertices on each row which are randomly
///distributed to the columns of the matrix
template <typename VertexType, typename EdgeType, typename Scalar>
void generateRegularRandom (VertexType *nVtx, VertexType *nCol,
			    EdgeType** xadj, VertexType **adj, Scalar** val,
			    VertexType size, VertexType degree) {
  assert (degree>0);
  *nVtx = *nCol = size;
  *xadj = (EdgeType*) malloc(sizeof(EdgeType)*(size+1));
  *adj = (VertexType*) malloc(sizeof(VertexType)*size*degree);
  *val = (Scalar*) malloc(sizeof(Scalar)*size*degree);

#pragma omp parallel for schedule(runtime)
  for (VertexType u = 0; u<size; ++u) {
    (*xadj)[u] = u*degree;
    for (VertexType off = 0; off<degree; ++off) {
      EdgeType edgeposition = (*xadj)[u]+off;
      (*adj)[edgeposition] = u+rand();
      if ((*adj)[edgeposition] < 0)
	(*adj)[edgeposition] = -(*adj)[edgeposition];
      (*adj)[edgeposition] %= size;
      (*val)[edgeposition] = 1.;
    }
  } 
  (*xadj)[size] = degree*size;
}


///generates a sizeXsize matrix where all the diagonal is full and for
///every <every> vertices there are degree vertices on each row which
///are regularly spaced by spacing columns
template <typename VertexType, typename EdgeType, typename Scalar>
void generateHotRegularSpaced (VertexType *nVtx, VertexType *nCol,
			       EdgeType** xadj, VertexType **adj, Scalar** val,
			       VertexType size, VertexType every, VertexType degree, VertexType spacing) {
  assert (degree>0);
  *nVtx = *nCol = size;
  *xadj = (EdgeType*) malloc(sizeof(EdgeType)*(size+1));
  EdgeType approxsize = size+(EdgeType)(degree*(float)(size)/every);

  *adj = (VertexType*) malloc(sizeof(VertexType)*approxsize);
  *val = (Scalar*) malloc(sizeof(Scalar)*approxsize);
  
  (*xadj)[0] = 0;

  for (VertexType u = 0; u<size; ++u) {
    (*xadj)[u+1] = (*xadj)[u];

    if (u % every) {
      //cold vertex
      EdgeType edgeposition = (*xadj)[u+1];

      (*adj)[edgeposition] = u;
      (*val)[edgeposition] = 1.;

      (*xadj)[u+1] ++;
    }
    else {
      //hot vertex
      for (VertexType off = 0; off<degree; ++off) {
	EdgeType edgeposition = (*xadj)[u+1];

	(*adj)[edgeposition] = (u+off*spacing)%size;
	(*val)[edgeposition] = 1.;

	(*xadj)[u+1] ++;
      }
    }    
  } 
}



///generates a sizeXsize matrix every <every> vertices there are
///degree vertices on each row which are regularly spaced by <spacing>
///columns and start where the previous finished
template <typename VertexType, typename EdgeType, typename Scalar>
void generateHotRegularSpacedNoCache (VertexType *nVtx, VertexType *nCol,
				      EdgeType** xadj, VertexType **adj, Scalar** val,
				      VertexType size, VertexType every, VertexType degree, VertexType spacing) {
  assert (degree>0);
  *nVtx = *nCol = size;
  *xadj = (EdgeType*) malloc(sizeof(EdgeType)*(size+1));
  EdgeType approxsize = size+(EdgeType)(degree*(float)(size)/every);

  *adj = (VertexType*) malloc(sizeof(VertexType)*approxsize);
  *val = (Scalar*) malloc(sizeof(Scalar)*approxsize);
  
  (*xadj)[0] = 0;

  for (VertexType u = 0; u<size; ++u) {
    (*xadj)[u+1] = (*xadj)[u];

    if (u % every) {
      //cold vertex
    }
    else {
      //hot vertex
      for (VertexType off = 0; off<degree; ++off) {
	EdgeType edgeposition = (*xadj)[u]+off;
	(*adj)[edgeposition] = ((edgeposition)%size)*spacing%size;
	(*val)[edgeposition] = 1.;

	(*xadj)[u+1] ++;
      }
    }    
  } 
}




template<typename VertexType, typename EdgeType, typename Scalar>
void generateCycle (VertexType *nVtx, VertexType *nCol,
		    EdgeType** xadj, VertexType **adj, Scalar** val,
		    VertexType size) {
  EdgeType nbedge = 2*(size); 

  std::cerr<<"generating cycle "<<size<<std::endl;

  *nCol = *nVtx = size;

  *xadj = (EdgeType* ) malloc(sizeof(EdgeType)*(size+1));

  *adj = (VertexType*) malloc(sizeof(VertexType)*nbedge);
  *val = (Scalar*) malloc(sizeof(Scalar)*nbedge);


  (*xadj)[0] = 0;
  for (VertexType i=0; i<size; ++i) {

    std::cerr<<"i = "<<i<<std::endl;

    VertexType btreeid = i;

    (*xadj)[i+1] = (*xadj)[i];

    
    //left child
    VertexType lcid = btreeid-1;
    {
      (*adj)[(*xadj)[btreeid+1]] = (lcid +size)%size;
      (*val)[(*xadj)[btreeid+1]] = 1.;
      (*xadj)[btreeid+1]++;
    }

    //right child
    VertexType rcid = btreeid+1;
    {
      (*adj)[(*xadj)[btreeid+1]] = (rcid + size) % size;
      (*val)[(*xadj)[btreeid+1]] = 1.;
      (*xadj)[btreeid+1]++;    
    }
  }
  
  std::cout<<"generated "<<(*xadj)[size]<<" edges (should be :"<<nbedge<<")"<<std::endl;
}




#endif
