#ifndef _VCL_BANDWIDTH_REDUCITON_H_
#define _VCL_BANDWIDTH_REDUCITON_H_
/* =========================================================================
   Copyright (c) 2010-2013, Institute for Microelectronics,
                            Institute for Analysis and Scientific Computing,
                            TU Wien.
   Portions of this software are copyright by UChicago Argonne, LLC.

                            -----------------
                  ViennaCL - The Vienna Computing Library
                            -----------------

   Project Head:    Karl Rupp                   rupp@iue.tuwien.ac.at
               
   (A list of authors and contributors can be found in the PDF manual)

   License:         MIT (X11), see file LICENSE in the base directory
============================================================================= */

/*
*   Tutorial: Matrix bandwidth reduction algorithms
*/


#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <map>
#include <vector>
#include <deque>
#include <cmath>

#include "viennacl/misc/bandwidth_reduction.hpp"

namespace vcl {

//----------------------------------------------------------------------

class RCM_VCL
{
    public:

//
// Part 1: Helper functions
//

// Reorders a matrix according to a previously generated node
// number permutation vector r
    //----------------------------------------------------------------------
    //----------------------------------------------------------------------
    std::vector< std::map<int, double> > reorder_matrix(std::vector< std::map<int, double> > const & matrix, std::vector<int> const & r)
    {
        std::vector< std::map<int, double> > matrix2(r.size());
        std::vector<std::size_t> r2(r.size());
        
        for (std::size_t i = 0; i < r.size(); i++)
            r2[r[i]] = i;

        for (std::size_t i = 0; i < r.size(); i++)
            for (std::map<int, double>::const_iterator it = matrix[r[i]].begin();  it != matrix[r[i]].end(); it++)
                matrix2[i][r2[it->first]] = it->second;
        
        return matrix2;
    }
    //----------------------------------------------------------------------

// Calculates the bandwidth of a matrix
    int calc_bw(std::vector< std::map<int, double> > const & matrix)
    {
        int bw = 0;
        
        for (std::size_t i = 0; i < matrix.size(); i++) {
            int col_min = matrix.size() + 1;
            int col_max = -1;
            for (std::map<int, double>::const_iterator it = matrix[i].begin();  it != matrix[i].end(); it++) {
                col_min = (it->first < col_min) ? it->first : col_min;
                col_max = (it->first > col_max) ? it->first : col_max;
           }
           bw = std::max(bw, col_max-col_min+1);
        }

        printf("bw= %d\n", bw);
        
        return bw;
    }
    //----------------------------------------------------------------------
    // Calculate the bandwidth of a reordered matrix
    int calc_reordered_bw(std::vector< std::map<int, double> > const & matrix,  std::vector<int> const & r)
    {
        std::vector<int> r2(r.size());
        int bw = 0;

        // evaluate the number of nodes outside a specified bandwidth
        //int spec_bw = 4000;  // specified bandwidth
        int count_outside; // number of nodes outside a bandwidth of 4000
        
        for (std::size_t i = 0; i < r.size(); i++)
            r2[r[i]] = i;

        for (std::size_t i = 0; i < r.size(); i++) {
            int col_min = matrix.size() + 1;
            int col_max = -1;
            int row_bw = 0.0;
            for (std::map<int, double>::const_iterator it = matrix[r[i]].begin();  it != matrix[r[i]].end(); it++) {
                int rfirst = r2[it->first];
                col_min = (rfirst < col_min) ? rfirst : col_min;
                col_max = (rfirst > col_max) ? rfirst : col_max;
                //int width = static_cast<int>(r2[it->first]);
                //if (width > spec_bw) count_outside++;
                //row_bw = std::max(bw, std::abs(static_cast<int>(i - r2[it->first])));
                //bw = std::max(bw, std::abs(static_cast<int>(i - r2[it->first])));
            }
           bw = std::max(bw, col_max-col_min+1);
           // printf("row %d, bw: %d\n", i, row_bw);
       }

        //printf("nb points ouside bandwidth of %d : %d\n", spec_bw, count_outside);
        printf("reordered bw= %d\n", bw);
        
        return bw;
    }
    //----------------------------------------------------------------------
    // Generates a random permutation by Knuth shuffle algorithm
    // reference: http://en.wikipedia.org/wiki/Knuth_shuffle 
    //  (URL taken on July 2nd, 2011)
    std::vector<int> generate_random_reordering(int n)
    {
        std::vector<int> r(n);
        int tmp;
        int j;
        
        for (int i = 0; i < n; i++)
            r[i] = i;
        
        for (int i = 0; i < n - 1; i++)
        {
            j = i + static_cast<std::size_t>((static_cast<double>(rand()) / static_cast<double>(RAND_MAX)) * (n - 1 - i));
            if (j != i)
            {
                tmp = r[i];
                r[i] = r[j];
                r[j] = tmp;
            }
        }
        
        return r;
    }
    //----------------------------------------------------------------------
    // function for the generation of a three-dimensional mesh incidence matrix
    //  l:  x dimension
    //  m:  y dimension
    //  n:  z dimension
    //  tri: true for tetrahedral mesh, false for cubic mesh
    //  return value: matrix of size l * m * n
    std::vector< std::map<int, double> > gen_3d_mesh_matrix(int l, int m, int n, bool tri)
    {
        std::vector< std::map<int, double> > matrix;
        int s;
        int ind;
        int ind1;
        int ind2;
        
        s = l * m * n;
        matrix.resize(s);
        for (int i = 0; i < l; i++)
        {
            for (int j = 0; j < m; j++)
            {
                for (int k = 0; k < n; k++)
                {
                    ind = i + l * j + l * m * k;
                    
                    matrix[ind][ind] = 1.0;
                    
                    if (i > 0)
                    {
                        ind2 = ind - 1;
                        matrix[ind][ind2] = 1.0;
                        matrix[ind2][ind] = 1.0;
                    }
                    if (j > 0)
                    {
                        ind2 = ind - l;
                        matrix[ind][ind2] = 1.0;
                        matrix[ind2][ind] = 1.0;
                    }
                    if (k > 0)
                    {
                        ind2 = ind - l * m;
                        matrix[ind][ind2] = 1.0;
                        matrix[ind2][ind] = 1.0;
                    }
                    
                    if (tri)
                    {
                        if (i < l - 1 && j < m - 1)
                        {
                            if ((i + j + k) % 2 == 0)
                            {
                                ind1 = ind;
                                ind2 = ind + 1 + l;
                            }
                            else
                            {
                                ind1 = ind + 1;
                                ind2 = ind + l;
                            }
                            matrix[ind1][ind2] = 1.0;
                            matrix[ind2][ind1] = 1.0;
                        }
                        if (i < l - 1 && k < n - 1)
                        {
                            if ((i + j + k) % 2 == 0)
                            {
                                ind1 = ind;
                                ind2 = ind + 1 + l * m;
                            }
                            else
                            {
                                ind1 = ind + 1;
                                ind2 = ind + l * m;
                            }
                            matrix[ind1][ind2] = 1.0;
                            matrix[ind2][ind1] = 1.0;
                        }
                        if (j < m - 1 && k < n - 1)
                        {
                            if ((i + j + k) % 2 == 0)
                            {
                                ind1 = ind;
                                ind2 = ind + l + l * m;
                            }
                            else
                            {
                                ind1 = ind + l;
                                ind2 = ind + l * m;
                            }
                            matrix[ind1][ind2] = 1.0;
                            matrix[ind2][ind1] = 1.0;
                        }
                    }
                }
            }
        }
    
        return matrix;
    }
}; // class

//----------------------------------------------------------------------
class ConvertMatrix
{
public:
    typedef std::vector< std::map<int, double> > MAP;
    typedef std::map<int, double>::const_iterator MapIter;
    RCM_VCL rvcl;
    int nb_rows;
    std::vector<int>& col_ind;
    std::vector<int>& row_ptr;
    std::vector<int>& new_col_ind;
    std::vector<int>& new_row_ptr;
    std::vector< std::map<int, double> > matrix2; // = vicl.reorder_matrix(matrix, r);
    std::vector< std::map<int, double> > matrix; 
    std::vector<int> order;
    std::vector<int> inv_order;

public:

    ConvertMatrix(int nb_rows_, std::vector<int>& col_ind_, std::vector<int>& row_ptr_,
                                std::vector<int>& new_col_ind_, std::vector<int>& new_row_ptr_) :
        col_ind(col_ind_), row_ptr(row_ptr_),
        new_col_ind(new_col_ind_), new_row_ptr(new_row_ptr_)
    {
        this->nb_rows = nb_rows_;
        matrix.resize(nb_rows);
        int sz = col_ind.size();

        for (int i=0; i < nb_rows; i++) {
            int b = row_ptr[i];
            int e = row_ptr[i+1];
            for (int j=b; j < e; j++) {
                int col = col_ind[j];
                matrix[i][col] = 1.0;
                matrix[col][i] = 1.0;  // symmetrize for ViennaCL to work
            }
            matrix[i][i] = 1.0; // diagonal elment
        }

#if 0
        for (int i=0; i < nb_rows; i++) {
            int sz = matrix[i].size();
            for (MapIter it = matrix[i].begin();  it != matrix[i].end(); it++) {
                printf("mat(%d,%d) : %f\n", i, it->first, it->second);
            }
        }
#endif
    }
    //---------------------------------------------------------
    void reorderMatrix()
    {
        matrix2.resize(order.size());
        std::vector<std::size_t> inv_order(order.size());
        
        for (std::size_t i = 0; i < order.size(); i++)
            inv_order[order[i]] = i;

        for (std::size_t i = 0; i < order.size(); i++) {
            for (std::map<int, double>::const_iterator it = matrix[order[i]].begin();  it != matrix[order[i]].end(); it++) {
                matrix2[i][inv_order[it->first]] = it->second;
            }
        }

        printf("\n\n");

#if 0
        for (int i=0; i < nb_rows; i++) {
            int sz = matrix2[i].size();
            for (MapIter it = matrix2[i].begin();  it != matrix2[i].end(); it++) {
                printf("mat2(%d,%d) : %f\n", i, it->first, it->second);
            }
        }
#endif


        printf("\n\n");

        int bw = rvcl.calc_bw(matrix2);
        printf("bandwidth of reordered matrix: %d\n", bw);

#if 0
        for (int i=0; i < nb_rows; i++) {
            for (int j=0; j < matrix2[i].size(); j++) {
                printf("mat2(%d,%d) = %f\n", i, j, matrix2[i][j]);
           }
        }
#endif

        printf("\n\n");
    }
    //----------------------------------------------------------------------
    void computeReorderedEllMatrix()
    {
        printf("coputer reordered\n");
        printf("order size: %d\n", order.size());
        inv_order.resize(order.size());

        for (std::size_t i = 0; i < order.size(); i++) {
            inv_order[order[i]] = i;
        }

        printf("nb_rows = %d\n", nb_rows);
        new_col_ind.resize(col_ind.size());

        // first reorder row_ptr
        for (int i=0; i < row_ptr.size()-1; i++) {
            printf("row_ptr[%d] = %d\n", i, row_ptr[i]);
            int new_row = inv_order[i];
        }

#if 0
        // nnz not defined
        //
            for (int i=0; i < nb_rows; i++) {
                for (int j=0; j < nnz; j++) {
                    int row_index = inv_order[i];
                    int col_index = inv_order[col_ind[j+i*nnz]];
                    new_col_ind[j+row_index*nnz] = col_index;
                }
                // Placing the sort at this location does not work for reasons unknown
                // It is because new_col_ind is not filled one of ITS rows at a time
            }
#endif

    }
    //---------------------------------------------------------
    int calcOrigBandwidth() 
    {
        return rvcl.calc_bw(matrix);
    }
    //---------------------------------------------------------
    int calcReorderedBandwidth()
    {
        //return rvcl.calc_...(matrix);
    }
    //---------------------------------------------------------
    void reduceBandwidthRCM() 
    {
        //r = rvcl.generate_random_reordering(n);
        //std::vector<int> r = rvcl.generate_random_reordering(n);
        
        // Reorder using Cuthill-McKee algorithm
        
        std::cout << "-- Cuthill-McKee algorithm --" << std::endl;
        std::cout << " * Original bandwidth: " << rvcl.calc_bw(matrix) << std::endl;
        printf("matrix size: %d\n", matrix.size());
        order = viennacl::reorder(matrix, viennacl::cuthill_mckee_tag());
        std::cout << " * Reordered bandwidth(matrix): " << rvcl.calc_reordered_bw(matrix, order) << std::endl;
        printf("reorder matrix\n");
        reorderMatrix(); // matrix -> matrix2
        std::cout << " * Bandwidth(matrix2): " << rvcl.calc_bw(matrix2) << std::endl;

        //
        // Reorder using advanced Cuthill-McKee algorithm
        //
        //std::cout << "-- Advanced Cuthill-McKee algorithm --" << std::endl;
        //double a = 0.0;
        //std::size_t gmax = 1;
        //r = viennacl::reorder(matrix2, viennacl::advanced_cuthill_mckee_tag(a, gmax));
        //std::cout << " * Reordered bandwidth: " << rvcl.calc_reordered_bw(matrix2, r) << std::endl;
    }
    //---------------------------------------------------------
    //---------------------------------------------------------
    //---------------------------------------------------------
    //---------------------------------------------------------
    //---------------------------------------------------------
}; // class ConvertMatrix


}; // namespace


#endif
