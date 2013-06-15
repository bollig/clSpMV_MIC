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

#ifndef NDEBUG
 #define NDEBUG
#endif

//
// *** System
//
#include <iostream>

//
// *** Boost
//
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/triangular.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/operation_sparse.hpp>

//
// *** ViennaCL
//
//#define VIENNACL_DEBUG_ALL
#define VIENNACL_WITH_UBLAS 1
#include "viennacl/scalar.hpp"
#include "viennacl/compressed_matrix.hpp"
#include "viennacl/coordinate_matrix.hpp"
#include "viennacl/ell_matrix.hpp"
#include "viennacl/hyb_matrix.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/linalg/prod.hpp"
#include "viennacl/linalg/norm_2.hpp"
#include "viennacl/linalg/ilu.hpp"
#include "viennacl/linalg/detail/ilu/common.hpp"
#include "viennacl/io/matrix_market.hpp"
#include "examples/tutorial/Random.hpp"
#include "examples/tutorial/vector-io.hpp"


#include <vector>
#include <string>
#include "rbffd/rbffd_io.h"
#include "settings/projectsettings.h"

ProjectSettings ps;

//
// -------------------------------------------------------------
//
using namespace boost::numeric;
//
// -------------------------------------------------------------
//
//template <typename ScalarType>
//if (!viennacl::io::read_matrix_market_file(ublas_matrix, "../../examples/testdata/one_million.mtx"))
//----------------------------------------------------------------------
template <typename ScalarType>
ScalarType diff(ScalarType & s1, viennacl::scalar<ScalarType> & s2) 
{
   if (s1 != s2)
      return (s1 - s2) / std::max(fabs(s1), std::fabs(s2));
   return 0;
}

//----------------------------------------------------------------------
template <typename ScalarType>
ScalarType diff(ublas::vector<ScalarType> & v1, viennacl::vector<ScalarType> & v2)
{
   ublas::vector<ScalarType> v2_cpu(v2.size());
   viennacl::backend::finish();
   viennacl::copy(v2.begin(), v2.end(), v2_cpu.begin());

   for (unsigned int i=0;i<v1.size(); ++i)
   {
      if ( std::max( std::fabs(v2_cpu[i]), std::fabs(v1[i]) ) > 0 )
      {
        //if (std::max( std::fabs(v2_cpu[i]), std::fabs(v1[i]) ) < 1e-10 )  //absolute tolerance (avoid round-off issues)
        //  v2_cpu[i] = 0;
        //else
          v2_cpu[i] = std::fabs(v2_cpu[i] - v1[i]) / std::max( std::fabs(v2_cpu[i]), std::fabs(v1[i]) );
      }
      else
         v2_cpu[i] = 0.0;
      
      if (v2_cpu[i] > 0.0001)
      {
        //std::cout << "Neighbor: "      << i-1 << ": " << v1[i-1] << " vs. " << v2_cpu[i-1] << std::endl;
        std::cout << "Error at entry " << i   << ": " << v1[i]   << " vs. " << v2_cpu[i]   << std::endl;
        //std::cout << "Neighbor: "      << i+1 << ": " << v1[i+1] << " vs. " << v2_cpu[i+1] << std::endl;
        exit(0);
      }
   }

   return norm_inf(v2_cpu);
}


//----------------------------------------------------------------------
template <typename ScalarType, typename VCL_MATRIX>
ScalarType diff(ublas::compressed_matrix<ScalarType> & cpu_matrix, VCL_MATRIX & gpu_matrix)
{
  typedef ublas::compressed_matrix<ScalarType>  CPU_MATRIX;
  CPU_MATRIX from_gpu;
   
  viennacl::backend::finish();
  viennacl::copy(gpu_matrix, from_gpu);

  ScalarType error = 0;
   
  //step 1: compare all entries from cpu_matrix with gpu_matrix:
  //std::cout << "Ublas matrix: " << std::endl;
  for (typename CPU_MATRIX::const_iterator1 row_it = cpu_matrix.begin1();
        row_it != cpu_matrix.end1();
        ++row_it)
  {
    //std::cout << "Row " << row_it.index1() << ": " << std::endl;
    for (typename CPU_MATRIX::const_iterator2 col_it = row_it.begin();
          col_it != row_it.end();
          ++col_it)
    {
      //std::cout << "(" << col_it.index2() << ", " << *col_it << std::endl;
      ScalarType current_error = 0;
      
      if ( std::max( std::fabs(cpu_matrix(col_it.index1(), col_it.index2())), 
                      std::fabs(from_gpu(col_it.index1(), col_it.index2()))   ) > 0 )
        current_error = std::fabs(cpu_matrix(col_it.index1(), col_it.index2()) - from_gpu(col_it.index1(), col_it.index2())) 
                          / std::max( std::fabs(cpu_matrix(col_it.index1(), col_it.index2())), 
                                      std::fabs(from_gpu(col_it.index1(), col_it.index2()))   );
      if (current_error > error)
        error = current_error;
    }
  }

  //step 2: compare all entries from gpu_matrix with cpu_matrix (sparsity pattern might differ):
  //std::cout << "ViennaCL matrix: " << std::endl;
  for (typename CPU_MATRIX::const_iterator1 row_it = from_gpu.begin1();
        row_it != from_gpu.end1();
        ++row_it)
  {
    //std::cout << "Row " << row_it.index1() << ": " << std::endl;
    for (typename CPU_MATRIX::const_iterator2 col_it = row_it.begin();
          col_it != row_it.end();
          ++col_it)
    {
      //std::cout << "(" << col_it.index2() << ", " << *col_it << std::endl;
      ScalarType current_error = 0;
      
      if ( std::max( std::fabs(cpu_matrix(col_it.index1(), col_it.index2())), 
                      std::fabs(from_gpu(col_it.index1(), col_it.index2()))   ) > 0 )
        current_error = std::fabs(cpu_matrix(col_it.index1(), col_it.index2()) - from_gpu(col_it.index1(), col_it.index2())) 
                          / std::max( std::fabs(cpu_matrix(col_it.index1(), col_it.index2())), 
                                      std::fabs(from_gpu(col_it.index1(), col_it.index2()))   );
      if (current_error > error)
        error = current_error;
    }
  }

  return error;
}


//----------------------------------------------------------------------
// -------------------------------------------------------------
//
template< typename NumericT, typename Epsilon >
int test(Epsilon const& epsilon)
{
	int retval;

  ublas::vector<NumericT> rhs;
  ublas::vector<NumericT> result;

  std::vector<int> rows;
  std::vector<int> cols;
  std::vector<NumericT> vals;

#if 1
	RBFFD_IO<NumericT> io;
	int width, height; 


	std::string file_binary = ps.getRequired<std::string>("filename");
	int err = io.loadFromBinaryMMFile(rows, cols, vals, width, height, file_binary);
	if (err != 0) exit(1);
	printf("height, width, nonzeros= %d, %d, %d\n", width, height, rows.size());

  	ublas::compressed_matrix<NumericT> ublas_matrix(height, width, rows.size());

	for (int i=0; i < rows.size(); i++) {
		//ublas_matrix(rows[i], cols[i]) = vals[i];
		ublas_matrix(rows[i], cols[i]) = random<NumericT>();
	}
	printf("******** SIZE *****************\n");
	printf("ublas size: %d, %d\n", ublas_matrix.size1(), ublas_matrix.size2());
#else
  	ublas::compressed_matrix<NumericT> ublas_matrix;
  	if (!viennacl::io::read_matrix_market_file(ublas_matrix, "../../examples/testdata/mat65k.mtx"))
#endif

	std::cout << "done reading matrix" << std::endl;
  
	rhs.resize(ublas_matrix.size2());
	printf("\n*** rhs.size = %d\n", rhs.size());
	for (std::size_t i=0; i<rhs.size(); ++i)
	{
		//ublas_matrix(i,i) = NumericT(0.5);   // Get rid of round-off errors by making row-sums unequal to zero:
    	rhs[i] = NumericT(1) + random<NumericT>();
		//if (i < 10) printf("rhs= %f\n", rhs[i]);
	}

	result = rhs;


  viennacl::vector<NumericT> vcl_rhs(rhs.size());
  viennacl::vector<NumericT> vcl_result(result.size()); 
  //viennacl::vector<NumericT> vcl_result2(result.size()); 
  // 3rd argument is exepected number of nonzeros. 
  viennacl::compressed_matrix<NumericT> vcl_compressed_matrix;
  viennacl::coordinate_matrix<NumericT> vcl_coordinate_matrix;
  viennacl::ell_matrix<NumericT> vcl_ell_matrix;
  //viennacl::hyb_matrix<NumericT> vcl_hyb_matrix;

  viennacl::copy(rhs.begin(), rhs.end(), vcl_rhs.begin());
  viennacl::copy(ublas_matrix, vcl_compressed_matrix);
  viennacl::copy(ublas_matrix, vcl_coordinate_matrix);
  viennacl::copy(ublas_matrix, vcl_ell_matrix);

  // --------------------------------------------------------------------------          
  std::cout << "Testing products: ublas" << std::endl;
  result     = viennacl::linalg::prod(ublas_matrix, rhs);

  
  std::cout << "Testing products: compressed_matrix" << std::endl;
  	for (int i=0; i < 5; i++) {
  		vcl_result = viennacl::linalg::prod(vcl_compressed_matrix, vcl_rhs);
  	}

	std::cout << "Testing products: coordinate_matrix" << std::endl;
	for (int i=0; i< 5; i++) {
  		vcl_result = viennacl::linalg::prod(vcl_compressed_matrix, vcl_rhs);
	}

	std::cout << "Testing products: ell_matrix" << std::endl;
	for (int i=0; i < 5; i++) {
  		vcl_result = viennacl::linalg::prod(vcl_compressed_matrix, vcl_rhs);
	}

  return retval;
}
//
// -------------------------------------------------------------
//
int main()
{
	ps.ParseFile("test.conf");
	std::string coprocessor = ps.getRequired<std::string>("coprocessor");

#ifdef VIENNACL_WITH_OPENCL
  // Choose the first Phi device (WORKS)
  if (coprocessor == "MIC") {
  	viennacl::ocl::set_context_device_type(0, viennacl::ocl::accelerator_tag());
  } 
  //verify that it is selected:
  std::cout << viennacl::ocl::current_context().current_device().info() << "\n";
#endif

  std::cout << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << "## Test :: Sparse Matrices" << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << std::endl;

  printf("***** 1 ***** \n");

  int retval = EXIT_SUCCESS;

  std::cout << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << std::endl;
  #if 1
  {
    typedef float NumericT;
    NumericT epsilon = static_cast<NumericT>(1E-4);
    std::cout << "# Testing setup:" << std::endl;
    std::cout << "  eps:     " << epsilon << std::endl;
    std::cout << "  numeric: float" << std::endl;
    retval = test<NumericT>(epsilon);
    if( retval == EXIT_SUCCESS )
        std::cout << "# Test passed" << std::endl;
    else
        return retval;
  }
  #endif
  std::cout << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << std::endl;
  
#ifdef VIENNACL_WITH_OPENCL
  // Choose the first Phi device (WORKS)
  printf("*** 1 ***\n");
  if( viennacl::ocl::current_device().double_support() )
#endif
  {
    {
  printf("*** has double support ***\n");
      typedef double NumericT;
      NumericT epsilon = 1.0E-13;
      std::cout << "# Testing setup:" << std::endl;
      std::cout << "  eps:     " << epsilon << std::endl;
      std::cout << "  numeric: double" << std::endl;
      retval = test<NumericT>(epsilon);
      if( retval == EXIT_SUCCESS )
        std::cout << "# Test passed" << std::endl;
      else
        return retval;
    }
    std::cout << std::endl;
    std::cout << "----------------------------------------------" << std::endl;
    std::cout << std::endl;
  }
#ifdef VIENNACL_WITH_OPENCL
  else
    std::cout << "No double precision support, skipping test..." << std::endl;
#endif
  
  
  std::cout << std::endl;
  std::cout << "------- Test completed --------" << std::endl;
  std::cout << std::endl;
  
  return retval;
}
