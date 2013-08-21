#include <iostream>
#include <vector>

#include "reorder.h"
#include "vcl_bandwidth_reduction.h"

using namespace std;

void setupMatrix(std::vector<int>& col_ind, std::vector<int>& row_ptr)
{
    int nb_rows;

#if 0
    // Symmetric matrix, non-Ellpack
	row_ptr.push_back(0); 
	// 1 0 0 0 1 0 0 0
	row_ptr.push_back(2); 
	// 0 1 1 0 0 1 0 1
	row_ptr.push_back(6); 
	// 0 1 1 0 1 0 0 0
	row_ptr.push_back(9); 
	// 0 0 0 1 0 0 1 0
    row_ptr.push_back(11);
	// 1 0 1 0 1 0 0 0
	row_ptr.push_back(14); 
	// 0 1 0 0 0 1 0 1
	row_ptr.push_back(17); 
	// 0 0 0 1 0 0 1 0
	row_ptr.push_back(19); 
	// 0 1 0 0 0 1 0 1
	row_ptr.push_back(22); 

	col_ind.push_back(0); 
	col_ind.push_back(4); 

	col_ind.push_back(1); 
	col_ind.push_back(2); 
	col_ind.push_back(5); 
	col_ind.push_back(7); 

	col_ind.push_back(1); 
	col_ind.push_back(2); 
	col_ind.push_back(4); 

	col_ind.push_back(3); 
	col_ind.push_back(6); 

	col_ind.push_back(0); 
	col_ind.push_back(2); 
	col_ind.push_back(4); 

	col_ind.push_back(1); 
	col_ind.push_back(5); 
	col_ind.push_back(7); 

	col_ind.push_back(3); 
	col_ind.push_back(6); 

	col_ind.push_back(1); 
	col_ind.push_back(5); 
	col_ind.push_back(7); 

    return;
#endif

#if 0 
    // nonsymmetric
    nb_rows = 4;
	row_ptr.push_back(0); 
	// 0 1 0 1
	row_ptr.push_back(2); 
    col_ind.push_back(1);
    col_ind.push_back(3);
	// 1 0 1 0
	row_ptr.push_back(4); 
    col_ind.push_back(0);
    col_ind.push_back(2);
	// 0 1 1 0
	row_ptr.push_back(6); 
    col_ind.push_back(1);
    col_ind.push_back(2);
	// 0 1 1 0
	row_ptr.push_back(8); 
    col_ind.push_back(1);
    col_ind.push_back(2);
#endif

#if 1
    // matrix with 1 on the two diagonals
    //nb_rows = 125000; // very slow with 20000, let alone 262000 or 10^6
    nb_rows = 8; // very slow with 20000, let alone 262000 or 10^6
    row_ptr.resize(nb_rows+1);
    row_ptr[0] = 0;
    int nnz = 2;

    for (int i=0; i < nb_rows; i++) {
        row_ptr[i+1] = row_ptr[i] + 2;
        col_ind.push_back(i);
        col_ind.push_back(nb_rows-i-1);
        //std::sort(&col_ind[0]+nnz*i, &col_ind[0]+nnz*(i+1));
    }
#endif

#if 0
    // matrix with nonzeros on first 2 colummns (i.e., unsymmetric) (no bandwidth reduction)
    // matrix with nonzeros on main diagonal on first lower subdiagonal
    // nonsymmetric matrix
    nb_rows = 20;
    row_ptr.resize(nb_rows+1);
    row_ptr[0] = 0;
    for (int i=0; i < nb_rows; i++) {
        if (i == 0) {
            col_ind.push_back(0);
            col_ind.push_back(nb_rows-1);
        } else {
            col_ind.push_back(i-1);
            col_ind.push_back(i);
        }
        row_ptr[i+1] = row_ptr[i] + 2;
    }
#endif

}
//-----------------------------------------------
int main (int argc, char** argv)
{
	std::vector<int> row_ptr; 
	std::vector<int> col_ind; 
	std::vector<int> new_row_ptr; 
	std::vector<int> new_col_ind; 
	
    setupMatrix(col_ind, row_ptr);

    int nb_rows = row_ptr.size()-1;

#if 0
    printf("using Ublas\n");
    bollig::ConvertMatrix c(nb_rows, col_ind, row_ptr, new_col_ind, new_row_ptr);
    c.convertToCSR();
    c.reduceBandwidthRCM();
    //c.printOutputMatrix(); // Not working properly
    exit(0);
#endif

    printf("Bandwidth reduction using ViennaCL\n");
    new_col_ind.resize(0);
    new_row_ptr.resize(0);
    vcl::ConvertMatrix d(nb_rows, col_ind, row_ptr, new_col_ind, new_row_ptr);
    int bw = d.calcOrigBandwidth();
    printf("initial bw (VCL): %d\n", bw);
    d.reduceBandwidthRCM();
    //d.reorderMatrix(); // done in reduceBandwidthwithRCM
    //d.computeReorderedEllMatrix();
    // only makes sense if r->get_reordered_system(A, V, A_reordered, V_reordered) call first;
    // (not really needed)

	return 0; 
}
//----------------------------------------------------------------------
