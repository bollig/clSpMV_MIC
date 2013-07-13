#include <algorithm>
#include <deque>

#ifdef HAVE_PATOH_

#include <patoh.h>

void hypergraph(int nVtx, int * adj, int* xadj, int* permut)
{
  PaToH_Parameters args;
  PaToH_Initialize_Parameters(&args, PATOH_CONPART/*PATOH_CUTPART*/, PATOH_SUGPARAM_QUALITY);

  args._k = 50;
  args.MemMul_CellNet += 5;
  args.MemMul_Pins += 5;
  args.MemMul_General += 5;
  //  args.balance = 0;
  int* partvec = new int[nVtx];
  int* partweights = new int[args._k];

  
  int *cwghts = new int[nVtx];
  int *nwghts = new int[nVtx];
  int cut = 0;


  for (int i=0; i<nVtx; i++)
    cwghts[i] = nwghts[i] = 1;

  int err = PaToH_Alloc(&args, nVtx, nVtx, 1, cwghts, nwghts, xadj, adj);
  if (err != 0)
    std::cerr<<"patoh error alloc"<<std::endl;

  PaToH_Part(&args, nVtx, nVtx, 1, 0, cwghts, nwghts,
	     xadj, adj, NULL, partvec, partweights, &cut);


  // for (int i=0; i<nVtx; i++)
  //   std::cout<<partvec[i]<<std::endl;

  int *prefixsum = new int[args._k+1];
  prefixsum[0] = 0;
  for (int i=1; i<=args._k; i++)
    prefixsum[i] = prefixsum[i-1] + partweights[i-1];


  for (int u=0; u<nVtx; u++)
    permut[u] = prefixsum[partvec[u]]++;

  delete[] cwghts;
  delete[] nwghts;
  delete[] partvec;
  delete[] partweights;
  PaToH_Free();
}

#endif

template<typename vertex, typename edge>
void BreadthFirstSearch(vertex nVtx, const edge* adj, const vertex* xadj, vertex* permut, vertex* nb_connected_component=NULL, vertex* maxlevel_found=NULL, vertex* largest_level=NULL)
{
  typedef std::deque<int> NODEQUEUE;

  for (vertex i=0; i<nVtx; i++)
    permut[i] = -1;

  assert (adj != NULL);
  assert (xadj != NULL);

  NODEQUEUE q;
  int nbfound = 0;
  int connected_component = 0;

  vertex* level = NULL;
  vertex ml=0;


  if (maxlevel_found != NULL)
    level = new vertex[nVtx];

  vertex largestlevel;
  vertex currentlevel;

  if (largest_level)
    *largest_level = 0;

  while (nbfound != nVtx)
    {
      connected_component++;
      //find an unseen node and push it
      {
	assert (q.empty());
	for (vertex i=0; i< nVtx; i++)
	  {
	    if (permut[i] < 0)
	      {
		q.push_back(i);
		permut[i] = nbfound;
		if (level)
		  level[i] = 0;
		nbfound++;
		assert (nbfound <= nVtx);
		break;
	      }
	  }
	assert (! q.empty());
	
      }  

      //traverse the graph
      largestlevel = 0;
      currentlevel = 0;
      {
	while (! q.empty())
	  {
	    vertex u = q.front();
	    q.pop_front();
	    
	    if (level) {
	      if (level[u] == currentlevel)
		{
		  largestlevel++;
		}
	      else
		{
		  if (largest_level)
		    *largest_level = std::max(*largest_level, largestlevel);
		  currentlevel = level[u];
		  largestlevel = 0;
		}
	    }

	    for (edge j = xadj[u]; j < xadj[u+1]; j++)
	      {
		vertex v = adj[j];
		if (permut[v] < 0)
		  {
		    q.push_back(v);
		    permut[v] = nbfound;
		    if (level)
		      {
			level[v] = level[u] + 1;
			ml = level[v];
		      }
		    nbfound++;
		    assert (nbfound <= nVtx);
		  }
	      }	    
	  }
      }
      if (largest_level)
	*largest_level = std::max(*largest_level, largestlevel);

      if (maxlevel_found)
	*maxlevel_found = std::max(ml, *maxlevel_found);
    }

  if (nb_connected_component)
    *nb_connected_component = connected_component;

  if (level)
    delete[] level;
}

template<typename T>
void shuffle(T* array, int size)
{
  T r;
  T i;
  T tmp;
  for(i = size-1; i>=0; i--)
    {
      r = (T)(drand48()*i);
      assert (r>=0);
      assert (r<=i);
      /*swap i and r*/
      tmp = array[i];
      array[i] = array[r];
      array[r] = tmp;
    }
}




template<typename T>
class reversepermut
{
  const T* perm;
public:
  reversepermut(const T* p)
      :perm(p)
  {}
  
  bool operator() (const T& a, const T& b)
  {
    return perm[a]<perm[b];
  }
  
};


template<typename vertex, typename edge>
void permutegraph (vertex nVtx, const edge* xadj, const vertex* adj, const vertex* permut, edge* new_xadj, vertex* new_adj, bool should_sort_adj=true)
{
  vertex * sortedpermut = new vertex [nVtx]; //sortedpermut[i] is the old id of the new i

  for (int i=0; i< nVtx; i++)
    {
      sortedpermut[i] = i;
    }
  
  reversepermut<vertex> comp(permut);

  std::sort<vertex*, reversepermut<vertex> > ((vertex*)sortedpermut, sortedpermut+nVtx, comp);


  new_xadj[0] = 0;
  
  for (vertex i=0; i<nVtx; i++)
    {
      vertex old_u = sortedpermut[i];
      assert(old_u < nVtx);
      
      new_xadj[i+1] = new_xadj[i] + xadj[old_u+1] - xadj[old_u];

      for (edge j=xadj[old_u]; j<xadj[old_u+1]; j++)
	{
	  vertex old_v = adj[j];
	  assert(old_v < nVtx);
	  vertex new_v = permut[old_v];
	  assert(new_v < nVtx);

	  new_adj[j-xadj[old_u]+new_xadj[i]] = new_v;
	}
    }

  //sorting the adjacency list if needed

  if (should_sort_adj)
    for (vertex i=0; i<nVtx; i++)
      {
	std::sort(new_adj+new_xadj[i], new_adj+new_xadj[i+1]);
      }

  delete[] sortedpermut;
}
