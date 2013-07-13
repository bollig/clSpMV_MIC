//TODO (disable from Makefile)
//FIX	two_vec_compare_T(coores, tmpresult, rownum);

#ifndef __CLASS_SPMV_ELL_OPENMP_H__
#define __CLASS_SPMV_ELL_OPENMP_H__

#include "util.h"
//#include "CL/cl.h"
#include "matrix_storage.h"
#include <vector>
#include <algorithm>
#include <string>

#include <omp.h>

//#include "oclcommon.h"
#include "openmp_base.h"

#include "timer_eb.h"

namespace spmv {

#define USE(x) using OPENMP_BASE<T>::x
//#define USECL(x) using CLBaseClass::x

template <typename T>
class ELL_OPENMP : public OPENMP_BASE<T>
{
public:
     EB::TimerList tm;
    //USE(devices);
    //USE(context);
	//USE(cmdQueue);
    //USE(program);
	//USE(errorCode);

    ////Create device memory objects
    //USE(devColid);
    //USE(devData);
    //USE(devVec);
    //USE(devRes);
    //USE(devTexVec);

	USE(ntimes);
	USE(filename);

    USE(aligned_length);
    USE(nnz);
    USE(rownum);
	USE(vecsize);
    USE(ellnum);
	USE(coo_mat);

	USE(opttime);
	USE(optmethod);

    double overallopttime;
	std::vector<T> paddedvec_v;
	std::vector<T> vec_v;
	std::vector<T> result_v;
	std::vector<T> coores_v;

    USE(dim2); // relates to workgroups

    //USE(vec);
    //USE(result);
    USE(coores);

	USE(getKernelName);
	//USECL(loadKernel);
	//USECL(enqueueKernel);

    ell_matrix<int, T> mat;

public:
	ELL_OPENMP(coo_matrix<int, T>* mat, int dim2Size, int ntimes);
	~ELL_OPENMP<T>() {
    	//free_ell_matrix(mat);
    //	free(vec);
    	//free(result);
    	//free(coores);
	}

	virtual void run();

protected:
	virtual void method_0(int nb=0);
	virtual void method_1(int nb=0);
};

//----------------------------------------------------------------------
template <typename T>
void ELL_OPENMP<T>::run()
{
    method_0();
	method_1(4);
}
//----------------------------------------------------------------------
template <typename T>
ELL_OPENMP<T>::ELL_OPENMP(coo_matrix<int, T>* coo_mat, int dim2Size, int ntimes) : 
   OPENMP_BASE<T>(coo_mat, dim2Size, ntimes)
{
    int offset = 3;
    tm["spmv"] = new EB::Timer("[method_0] Matrix-Vector multiply", offset);

	// Create matrices
//printf("inside ell constructor\n");
    printMatInfo_T(coo_mat);
    //ell_matrix<int, T> mat;
    coo2ell<int, T>(coo_mat, &mat, GPU_ALIGNMENT, 0);
    //vec = (T*)malloc(sizeof(T)*coo_mat->matinfo.width);
	vec_v.resize(coo_mat->matinfo.width);
    //result = (T*)malloc(sizeof(T)*coo_mat->matinfo.height);
    result_v.resize(coo_mat->matinfo.height);
	std::fill(vec_v.begin(), vec_v.end(), 1.);
	std::fill(result_v.begin(), result_v.end(), 0.);
    coores_v.resize(coo_mat->matinfo.height);
	// CHECKING Supposedly on CPU, but execution is on GPU!!
    //spmv_only_T<T>(coo_mat, vec_v, coores_v);

    //Initialize values
    aligned_length = mat.ell_height_aligned;
    printf("aligned_length= %d\n", aligned_length);
    nnz = mat.matinfo.nnz;
    rownum = mat.matinfo.height;
    vecsize = mat.matinfo.width;
    ellnum = mat.ell_num;

	printf("nnz= %d\n", nnz);
	printf("rownum= %d\n", rownum);
	printf("vecsize= %d\n", vecsize);
	printf("ellnum= %d\n", ellnum);
}
//----------------------------------------------------------------------
template <typename T>
void ELL_OPENMP<T>::method_0(int nbit)
{
	printf("============== METHOD 0 ===================\n");
    // implementation on the CPU, using OpenMP, and ELL_OPENMP format

    int nz = mat.ell_num;
    std::vector<int>& col_id = mat.ell_col_id;
    std::vector<T>& data = mat.ell_data;

    printf("nz in row: %d\n", nz);
    printf("nb rows: %d\n", vec_v.size());
    printf("aligned_length= %d\n", aligned_length);
    printf("vector length= %d\n", vec_v.size());
    printf("result_length= %d\n", result_v.size());

    int n = 128*128*128;
    std::vector<float> a(n, 1.);
    std::vector<float> b(n, 1.);
    std::vector<float> c(n, 1.);
    float gflops;
    float elapsed; 

    printf("maximum nb threads: %d\n", omp_get_max_threads());
    //omp_set_num_threads(10);


    // The time depends on th the input file size, which is not possible since what 
    //I am timing are arrays defined immediately above.
 
    //int tid = omp_get_thread_num();
#if 0
    for (int it=0; it < 10; it++) {
        tm["spmv"]->start();
#pragma omp parallel
{
#pragma omp for
        for (int i=0; i < n; i++) {
            //printf("*** thread id: %d\n", omp_get_thread_num());
            c[i] = 0.0;
            int base = 128;
            for (int j=base; j < base+128; j++) {
                c[i] += a[j] + cos(b[j]*c[j]);
            }
        }
}
//#pragma omp barrier
        elapsed = tm["spmv"]->end();
        gflops = n*128*3*1.e-9 / (1e-3*elapsed);
        printf("Gflops: %f, time: %f (ms)\n", gflops, elapsed);
    }
    //elapsed  = tm["smpv"]->getTime();
#endif

        //----------------------------
// Probably wrong value, whci might explain poor performance.
 printf("aligned_length= %d\n", aligned_length);
 //for (int i=0; i < 100; i++) {
        //printf("col_id[%d]= %d\n", i, col_id[i]);
 //}
#if 1
    float matrixelem;
    float vecelem;
    float accumulant;
    int vecid;
    //const int aligned = aligned_length; // Gflop goes from 25 to 0.5 (Cannot make it private)
    int aligned = aligned_length; 

    for (int it=0; it < 10; it++) {
        tm["spmv"]->start();
#pragma omp parallel private(vecid, matrixelem, vecelem, accumulant, aligned )
{
#pragma omp for 
        for (int row=0; row < vec_v.size(); row++) {
            int matoffset = row;
            accumulant = 0.;
// force vectorization of loop (I doubt it is appropriate)
// went from 22 to 30Gfops  (static,1)
// went from 27 to 30Gfops  (guided,16)
#pragma simd
            for (int i = 0; i < nz; i++) {
	            vecid = col_id[matoffset];
                //printf("row %d, sten %d, vecid= %d\n", row, i, vecid);
	            //matrixelem = data[matoffset];
	            //vecelem = vec_v[vecid];
	            //accumulant += matrixelem * vecelem;
	            accumulant += data[matoffset] * vec_v[vecid];
	            //matoffset += aligned_length;
	            matoffset += aligned;
            }
            //printf("*** thread id: %d\n", omp_get_thread_num());
            result_v[row] = accumulant;
        }
}
    elapsed = tm["spmv"]->end();
    gflops = 2.*nz*vec_v.size()*1e-9 / (1e-3*tm["spmv"]->getTime()); // assumes count of 1
    printf("Gflops: %f, time: %f (ms)\n", gflops, elapsed);
   }
#endif

// Less efficient than above. Hard to believe, since it looks identical. 
#if 0
    int nb_rows = vec_v.size();
    tm["spmv"]->start();
#pragma omp parallel
#pragma omp for 
    for (int row=0; row < nb_rows; row++) {
        float accumulant = 0.0;
        int matoffset = row;
        for (int i = 0; i < nz; i++) {
            matoffset = nb_rows * i + row;
	        int vecid = col_id[matoffset];
	        float matrixelem = data[matoffset];
	        float vecelem = vec_v[vecid];
	        accumulant += matrixelem * vecelem;
	        //matoffset += aligned_length;
        }
        result_v[row] = accumulant;
    }
    printf("*** after loop thread id: %d\n", omp_get_thread_num());
#pragma omp barrier
#pragma omp master
    printf("*** thread id: %d\n", omp_get_thread_num());
    tm["spmv"]->end();
#endif

    tm.printAll(stdout, 80);
    printf("time = %f ms\n", tm["spmv"]->getTime());

    printf("Gflops: %f, time: %f (ms)\n", gflops, elapsed);
}
//----------------------------------------------------------------------
template <typename T>
void ELL_OPENMP<T>::method_1(int nb_vectors)
{
	printf("============== METHOD 1 ===================\n");
    // implementation on the CPU, using OpenMP, and ELL_OPENMP format

    int nz = mat.ell_num;
    std::vector<int>& col_id = mat.ell_col_id;
    std::vector<T>& data = mat.ell_data;

    printf("nz in row: %d\n", nz);
    printf("nb rows: %d\n", vec_v.size());
    printf("aligned_length= %d\n", aligned_length);
    printf("vector length= %d\n", vec_v.size());
    printf("result_length= %d\n", result_v.size());
    printf("nb_vectors= %d\n", nb_vectors);

    int nb_rows = vec_v.size();
    std::vector<float> vec_in(nb_rows*nb_vectors, 1.);
    std::vector<float> vec_out(nb_rows*nb_vectors, 1.);
    float gflops;
    float elapsed; 

#if 1
    float matrixelem;
    float vecelem;
    printf("after def of accumulant\n");
    int vecid;
    //const int aligned = aligned_length; // Gflop goes from 25 to 0.5 (Cannot make it private)
    int aligned = aligned_length; 
    printf("size of vec_in/out= %d, %d\n", vec_in.size(), vec_out.size());
    int nb_vec = nb_vectors;

    for (int it=0; it < 10; it++) {
        tm["spmv"]->start();
#pragma omp parallel private(vecid, matrixelem, vecelem, aligned, nb_vec )
{
        std::vector<float> accumulant(nb_vec);
#pragma omp for 
        for (int row=0; row < vec_v.size(); row++) {
            int matoffset = row;
            std::fill(accumulant.begin(), accumulant.end(), 0.);
// force vectorization of loop (I doubt it is appropriate)
            for (int i = 0; i < nz; i++) {
	            vecid = col_id[matoffset];
                float data_offset = data[matoffset];
#pragma simd
                for (int v = 0; v < nb_vec; v++) {
                    //printf("row= %d, i= %d, v= %d\n", row, i, v);
                    //printf("indx = %d, matoffset= %d\n", vecid+v*nb_rows, matoffset);
                    //printf("size accumulant= %d\n", accumulant.size());
                    //printf("indx= %d\n", v+vecid*nz);
	                accumulant[v] += data_offset * vec_in[v+vecid*nz];
                    //printf("1\n");
                }
	            matoffset += aligned;
            }
#pragma simd
            for (int v=0; v < nb_vec; v++) {
                //vec_out[vecid+v*nb_rows] = accumulant[vecid+v*nb_rows];
                vec_out[v+vecid*nz] = accumulant[v];
            }
        }
}
    elapsed = tm["spmv"]->end();
    gflops = 2.*nz*vec_v.size()*1e-9 / (1e-3*tm["spmv"]->getTime()); // assumes count of 1
    printf("Gflops: %f, time: %f (ms)\n", gflops, elapsed);
   }
#endif

    printf("col_id.size()= %d\n", col_id.size());
    printf("vec_in.size()= %d\n", vec_in.size());
    printf("data.size() = %d\n", data.size());

    std::vector<float> tmp(vec_out.size());
    std::copy(vec_out.begin(), vec_out.end(), tmp.begin());

// Less efficient than above. Hard to believe, since it looks identical. 

    tm.printAll(stdout, 80);
    printf("time = %f ms\n", tm["spmv"]->getTime());

    printf("Gflops: %f, time: %f (ms)\n", gflops, elapsed);
    printf("vec_out[3] = %f\n", vec_out[3]);
}
//----------------------------------------------------------------------

#if 1
template <typename  T>
void spmv_ell_openmp(coo_matrix<int, T>* coo_mat, int dim2Size, int ntimes)
{

	printf("** GORDON, spmv_ell\n");
	ELL_OPENMP<T> ell_ocl(coo_mat, dim2Size, ntimes);

	ell_ocl.run();

	printf("GORDON: after ell_ocl.run\n");

	double opttime = ell_ocl.getOptTime();
	int optmethod = ell_ocl.getOptMethod();

	printf("\n------------------------------------------------------------------------\n");
	printf("ELL_OPENMP best time %f ms best method %d", opttime*1000.0, optmethod);
	printf("\n------------------------------------------------------------------------\n");
}
#endif
//----------------------------------------------------------------------

}; // namespace

#endif
