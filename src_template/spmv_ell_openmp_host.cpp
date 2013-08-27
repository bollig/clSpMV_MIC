#include <stdlib.h>
#include <stdio.h>
#include <string.h>

//#include "print_utils.h"

#include "projectsettings.h"

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

// I do not like the ViennaCL interface
//#include "cuthill_mckee.hpp"

//#include "globals.h"

//#include "class_ell.h" // renable when bell works
//#include "class_bell.h"
//#include "class_sell.h"
//#include "class_sbell.h"
#include "ell_openmp_host.h"

// comes from rbffd (Bollig) and inserted in include_template manually
#include "rbffd_io.h"


//using namespace spmv;

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
		printf("\nUsage: spmv_all input_matrix.mtx method execution_times");
		printf("\nThe matrix needs to be in the matrix market format");
		printf("\nThe method is the format you want to use:");
		printf("\n\tMethod 1: Use the ell format and openmp in native mode");
		printf("\nThe execution_times refers to how many times of SpMV you want to do to benchmark the execution time\n");
		return 0;
    }

	for (int i=0; i < argc; i++) {
		printf("argv[%d]= %s\n", i, argv[i]);
	}

// REWRITE argument list by using argv++ every time argv is used, and argc++ when argc is used
    std::string filename = argv[1]; argc--; argv++;
    printf("filename= %s\n", filename.c_str());
    int choice = 1;
    if (argc > 1) {
		//choice = atoi(argv[2]); 
		choice = atoi(argv[1]); argc--; argv++;
        printf("choice= %d\n", choice);
	}
    int dim2Size = 1;
    int ntimes = 20;
    if (argc > 1) {
		//ntimes = atoi(argv[3]); 
		ntimes = atoi(argv[1]); argc--; argv++;
        printf("ntimes= %d\n", ntimes);
	}

	for (int i=0; i < argc; i++) {
		printf("argv[%d]= %s\n", i, argv[i]);
	}

	ProjectSettings pj("test.conf");
	//pj.ParseFile("in_file.txt"); // parameters change run to run

	std::string asci_binary = REQUIRED<std::string>("asci_binary");
    printf("asci_binary= %s\n", asci_binary.c_str());
	//filename = OPTIONAL<std::string>("data_filename", filename);
	//int c = OPTIONAL<int>("case", "10"); // ERROR, see next line
	//src_template/spmv_all.cpp:85: error: no matching function for call to ‘ProjectSettingsSingleton::getOptional(const char [5], int)’
//	printf("c= %d\n", c);
//
    std::string in_format = REQUIRED<std::string>("in_format");

	printf("filename = %s\n", filename.c_str());
	printf("choice= %d\n", choice);


    char clfilename[1000];
    
    #if 1
    if (choice > 0)
    {
        printf("spmv in ell format using OpenNP in Native mode\n");
	    //spmv::spmv_ell_openmp(&mat, dim2Size, ntimes);
        spmv::spmv_ell_openmp_host<float>(filename);
    }
#endif

    return 0;
}


