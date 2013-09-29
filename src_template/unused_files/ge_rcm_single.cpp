//=======================================================================
// Copyright 1997, 1998, 1999, 2000 University of Notre Dame.
// Authors: Andrew Lumsdaine, Lie-Quan Lee, Jeremy G. Siek
//          Doug Gregor, D. Kevin McGrath
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//=======================================================================

#include <boost/config.hpp>
#include <vector>
#include <iostream>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/cuthill_mckee_ordering.hpp>
#include <boost/graph/properties.hpp>
#include <boost/graph/bandwidth.hpp>

// Details description of Boost Graph Library
// http://www.boost.org/doc/libs/1_38_0/libs/graph/doc/using_adjacency_list.html#sec:choosing-graph-type

/*
  Sample Output
  original bandwidth: 8
  Reverse Cuthill-McKee ordering starting at: 6
    8 3 0 9 2 5 1 4 7 6 
    bandwidth: 4
  Reverse Cuthill-McKee ordering starting at: 0
    9 1 4 6 7 2 8 5 3 0 
    bandwidth: 4
  Reverse Cuthill-McKee ordering:
    0 8 5 7 3 6 4 2 1 9 
    bandwidth: 4
 */
//----------------------------------------------------------------------
void ge_rcm(std::vector<int>& col_id, int nb_rows, int nb_zeros_per_row, std::vector<int>& perm, std::vector<int>& perm_inv)
{
  using namespace boost;
  using namespace std;

  if (col_id.size() != (nb_zeros_per_row * nb_rows) ) {
    printf("col_id.size() : %d\n", col_id.size());
    printf"nb_zeros_per row: %d\n", nb_zeros_per_row);
    printf("nb_rows= %d\n", nb_rows);
  }

  typedef adjacency_list<vecS, vecS, undirectedS, 
     property<vertex_color_t, default_color_type,
       property<vertex_degree_t,int> > > Graph;
  typedef graph_traits<Graph>::vertex_descriptor Vertex;
  typedef graph_traits<Graph>::vertices_size_type size_type;

  typedef std::pair<std::size_t, std::size_t> Pair;
  //Pair edges[14] = { Pair(0,3), //a-d
  std::vector<Pair> edges(col_id.size());
  Pair p;

  for (int r=0; r < nb_rows; r++) {
    p.first = r;
    for (int i=0; i < nb_nonzeros_per_row; i++) {
        p.second = col_id[i+nb_nonzeros_per_row*r];
        edges.push_back(p);
    }
  }

  Graph G(nb_rows);

  // I kept diagonal elements (which reference themselves)
    for (int i = 0; i < nb_rows; ++i)
        add_edge(edges[i].first, edges[i].second, G);
    }

  graph_traits<Graph>::vertex_iterator ui, ui_end;

  property_map<Graph,vertex_degree_t>::type deg = get(vertex_degree, G);
  for (boost::tie(ui, ui_end) = vertices(G); ui != ui_end; ++ui)
    deg[*ui] = degree(*ui, G);

  property_map<Graph, vertex_index_t>::type
    index_map = get(vertex_index, G);

  std::cout << "original bandwidth: " << bandwidth(G) << std::endl;

  std::vector<Vertex> inv_perm(num_vertices(G));
  std::vector<size_type> perm(num_vertices(G));

  {
    //reverse cuthill_mckee_ordering
    cuthill_mckee_ordering(G, inv_perm.rbegin(), get(vertex_color, G),
                           make_degree_map(G));
    
#if 0
    cout << "Reverse Cuthill-McKee ordering:" << endl;
    cout << "  ";
    for (std::vector<Vertex>::const_iterator i=inv_perm.begin();
       i != inv_perm.end(); ++i)
      cout << index_map[*i] << " ";
    cout << endl;
#endif

    for (size_type c = 0; c != inv_perm.size(); ++c)
       perm[index_map[inv_perm[c]]] = c;

    std::cout << "  bandwidth: " 
              << bandwidth(G, make_iterator_property_map(&perm[0], index_map, perm[0]))
              << std::endl;
  }
}
//----------------------------------------------------------------------
