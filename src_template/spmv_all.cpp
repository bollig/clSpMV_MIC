#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "print_utils.h"

#include "matrix_storage.h"
#include "spmv_serial.h"
#include "fileio.h"
#if 0
#include "spmv_bdia.h"
#include "spmv_dia.h"
#include "spmv_ell.h"
#include "spmv_bell.h"
#include "spmv_sell.h"
#include "spmv_sbell.h"
#include "spmv_coo.h"
#include "spmv_csr_vector.h"
#include "spmv_csr_scalar.h"
#include "spmv_bcsr.h"
#include "mem_bandwidth.h"
#endif

#include "class_ell.h" // renable when bell works
#include "class_bell.h"
#include "class_sell.h"
#include "class_sbell.h"



//using namespace spmv;

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
		printf("\nUsage: spmv_all input_matrix.mtx method execution_times");
		printf("\nThe matrix needs to be in the matrix market format");
		printf("\nThe method is the format you want to use:");
		printf("\n\tMethod 0: mesure the memory bandwidth and kernel launch overhead only");
		printf("\n\tMethod 1: use the csr matrix format, using the scalar implementations");
		printf("\n\tMethod 2: use the csr matrix format, using the vector implementations");
		printf("\n\tMethod 3: use the bdia matrix format");
		printf("\n\tMethod 4: use the dia matrix format");
		printf("\n\tMethod 5: use the ell matrix format");
		printf("\n\tMethod 6: use the coo matrix format");
		printf("\n\tMethod 7: use the bell matrix format");
		printf("\n\tMethod 8: use the bcsr matrix format");
		printf("\n\tMethod 9: use the sell matrix format");
		printf("\n\tMethod 10: use the sbell matrix format");
		printf("\nThe execution_times refers to how many times of SpMV you want to do to benchmark the execution time\n");
		return 0;
    }

    char* filename = argv[1];
    int choice = 1;
    if (argc > 2)
	choice = atoi(argv[2]);
    int dim2Size = 1;
    int ntimes = 20;
    if (argc > 3)
	ntimes = atoi(argv[3]);

    coo_matrix<int, float> mat;
    //coo_matrix<int, double> mat_d;
    //init_coo_matrix(mat);
    //init_coo_matrix(mat_d);
    spmv::ReadMMF(filename, &mat);
    //spmv::ReadMMF(filename, &mat_d);
	printf("READ INPUT FILE: \n");
	mat.print();

    char* clspmvpath = getenv("CLSPMVPATH");
    char clfilename[1000];
    
    if (choice == 0)
    {
	//sprintf(clfilename, "%s%s", clspmvpath, "/kernels/mem_bandwidth.cl");
	//bandwidth_test(clfilename, CONTEXTTYPE, dim2Size);
    }
	#if 0
    else if (choice == 1)
    {
	sprintf(clfilename, "%s%s", clspmvpath, "/kernels/spmv_csr_scalar.cl");
	spmv_csr_scalar(clfilename, &mat, dim2Size, ntimes, CONTEXTTYPE);
    }
    else if (choice == 2)
    {
	sprintf(clfilename, "%s%s", clspmvpath, "/kernels/spmv_csr_vector.cl");
	spmv_csr_vector(clfilename, &mat, dim2Size, ntimes, CONTEXTTYPE);
    }
    else if (choice == 3)
    {
	sprintf(clfilename, "%s%s", clspmvpath, "/kernels/spmv_bdia.cl");
	spmv_bdia(clfilename, &mat, dim2Size, ntimes, CONTEXTTYPE);
    }
    else if (choice == 4)
    {
	sprintf(clfilename, "%s%s", clspmvpath, "/kernels/spmv_dia.cl");
	spmv_dia(clfilename, &mat, dim2Size, ntimes, CONTEXTTYPE);
    }
	#endif
    else if (choice == 5)
    {
	#if 1
	sprintf(clfilename, "%s%s", clspmvpath, "/kernels/spmv_ell.cl");
	printf("befpre spmv_ell\n");
	spmv::spmv_ell("spmv_ell.cl", &mat, dim2Size, ntimes, CONTEXTTYPE);
	//spmv::spmv_ell(clfilename, &mat, dim2Size, ntimes, CONTEXTTYPE);
	#endif
	//spmv::spmv_ell(clfilename, &mat_d, dim2Size, ntimes, CONTEXTTYPE);
    }
	#if 0
    else if (choice == 6)
    {
	sprintf(clfilename, "%s%s", clspmvpath, "/kernels/spmv_coo.cl");
	spmv_coo(clfilename, &mat, dim2Size, ntimes, CONTEXTTYPE);
    }
	#endif
    else if (choice == 7)
    {
	//printf("**** FIRST GET SINGLE PRECISION WORKING! ***\n");
	sprintf(clfilename, "%s%s", clspmvpath, "/kernels/spmv_bell.cl");
	//spmv::spmv_bell(clfilename, &mat, dim2Size, ntimes, CONTEXTTYPE);
	// directory name not required. Read in at lower level using CL_KERNEL ENV variable
	spmv::spmv_bell("spmv_bell.cl", &mat, dim2Size, ntimes, CONTEXTTYPE);
	//sprintf(clfilename, "%s%s", clspmvpath, "/kernels/spmv_bell_d.cl");
	//spmv::spmv_bell(clfilename, &mat_d, dim2Size, ntimes, CONTEXTTYPE);
    }
	#if 0
    else if (choice == 8)
    {
	sprintf(clfilename, "%s%s", clspmvpath, "/kernels/spmv_bcsr.cl");
	spmv_bcsr(clfilename, &mat, dim2Size, ntimes, CONTEXTTYPE);
    }
	#endif
    else if (choice == 9)
    {
	sprintf(clfilename, "%s%s", clspmvpath, "/kernels/spmv_sell.cl");
	//spmv_sell(clfilename, &mat, dim2Size, ntimes, CONTEXTTYPE);
	spmv::spmv_sell("spmv_sell.cl", &mat, dim2Size, ntimes, CONTEXTTYPE);
    }
	#if 1
    else if (choice == 10)
    {
	sprintf(clfilename, "%s%s", clspmvpath, "/kernels/spmv_sbell.cl");
	spmv::spmv_sbell("spmv_sbell.cl", &mat, dim2Size, ntimes, CONTEXTTYPE);
    }
	#endif

    //free_coo_matrix(mat);

    return 0;
}


