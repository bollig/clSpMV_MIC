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


// load 8 double precision floats converted from block (which presumably can be float or double.)
#include <immintrin.h>
#define _mm512_loadnr_pd(block) _mm512_extload_pd(block, _MM_UPCONV_PD_NONE, _MM_BROADCAST64_NONE, _MM_HINT_NT)
#define _mm512_loadnr_ps(block) _mm512_extload_ps(block, _MM_UPCONV_PD_NONE, _MM_BROADCAST64_NONE, _MM_HINT_NT)

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

    void spmv_serial(ell_matrix<int, T>& mat, std::vector<T>& v, std::vector<T>& result);
    void spmv_serial_row(ell_matrix<int, T>& mat, std::vector<T>& v, std::vector<T>& result);
    void spmv_serial_row(ell_matrix<int, T>& mat, std::vector<T>& data, std::vector<T>& v, std::vector<T>& result);
    void fill_random(ell_matrix<int, T>& mat, std::vector<T>& v);
    T l2norm(std::vector<T>& v);
    T l2norm(T*, int n);

	virtual void run();
	void method_3(int nb=0);
	void method_4(int nb=0);
	void method_5(int nb=0);
	void method_6(int nb=0);
	void method_7(int nb=0); // 4 matrices, 4 vectors
	void method_8(int nb=0); // 4 matrices, 4 vectors

    inline __m512 permute(__m512 v1, _MM_PERM_ENUM perm);
    inline __m512 read_aaaa(float* a);
    inline __m512i read_aaaa(int* a);
    inline __m512 read_abcd(float* a);
    inline __m512 tensor_product(float* a, float* b);
    void print_f(float* res, const std::string msg="");
    void print_i(int* res, const std::string msg="");
    void print_ps(const __m512 v1, const std::string msg="");
    void print_epi32(const __m512i v1, const std::string msg="");

protected:
	virtual void method_0(int nb=0);
	virtual void method_1(int nb=0);
	virtual void method_2(int nb=0);
};

//----------------------------------------------------------------------
template <typename T>
void ELL_OPENMP<T>::run()
{
    //spmv_serial(mat, vec_v, result_v);
    //printf("l2norm of serial version: %f\n", l2norm(result_v));
    //spmv_serial_row(mat, vec_v, result_v);
    //printf("l2norm of serial row version: %f\n", l2norm(result_v));
    //exit(0);
    //method_0();
	//method_1(4);
	//method_2(1);
	//method_3(4);
	//method_4(4);
	//method_5(4); // correct results
	//method_6(4);
	//method_7(4); // correct results
	method_8(4);
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
    //coo_mat->print(1024);
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

    float gflops;
    float elapsed; 

    //printf("maximum nb threads: %d\n", omp_get_max_threads());

    int nb_rows = vec_v.size();
    //nb_rows = 262144; // hardcode // did not affect Gflops
    printf("aligned_length= %d\n", aligned_length);
    //for (int i=0; i < nb_rows; i++) {
        //printf("vec_v[%i] = %f\n", i, vec_v[i]);
    //}
    printf("aligned_length= %d\n", aligned_length);
    printf("nz*aligned_length= %d\n", nz*aligned_length);
    printf("size of col_id: %d\n", col_id.size());
    printf("size of data: %d\n", data.size());
    //exit(0);
    //for (int i=0; i < nz*aligned_length; i++) {
        //printf("col_id[%i] = %d, data[%d]= %f\n", i, col_id[i], i, data[i]);
    //}
    printf("aligned_length= %d\n", aligned_length);


    //int tid = omp_get_thread_num();

        //----------------------------
// Probably wrong value, whci might explain poor performance.
 printf("aligned_length= %d\n", aligned_length);
#if 1
    int matoffset;
    float accumulant;
    int vecid;
    const int aligned = aligned_length; 
    //const int aligned = aligned_length; // Gflop goes from 25 to 0.5 (Cannot make it private)

    for (int it=0; it < 10; it++) {
        tm["spmv"]->start();
#if 1
//#pragma omp parallel private(vecid, accumulant, aligned )
//#pragma omp parallel private(vecid, matrixelem, accumulant, matoffset) firstprivate(aligned, nb_rows)
#pragma omp parallel firstprivate(nb_rows)
{
#pragma omp for 
// simd slows it down
#pragma simd   
        for (int row=0; row < nb_rows; row++) {
            int matoffset = row;
            float accumulant = 0.;
// force vectorization of loop (I doubt it is appropriate)
// went from 22 to 30Gfops  (static,1)
// went from 27 to 30Gfops  (guided,16)
#pragma simd
            for (int i = 0; i < nz; i++) {
	            const int vecid = col_id[matoffset];
	            accumulant += data[matoffset] * vec_v[vecid];
                //float vv = vec_v[vecid];
                //float dd = data[matoffset];
	            //accumulant += dd*vv; // slows code down by 10%
	            matoffset += aligned;
            }
            //printf("*** thread id: %d\n", omp_get_thread_num());
            result_v[row] = accumulant;
        }
        //printf("aligned= %d\n", aligned);
}
    elapsed = tm["spmv"]->end();
    gflops = 2.*nz*vec_v.size()*1e-9 / (1e-3*tm["spmv"]->getTime()); // assumes count of 1
    printf("Gflops: %f, time: %f (ms)\n", gflops, elapsed);
#else 
    for (int row=0; row < vec_v.size(); row++) {
        int matoffset = row;
        accumulant = 0.;

        for (int i = 0; i < nz; i++) {
            vecid = col_id[matoffset];
            accumulant += data[matoffset] * vec_v[vecid];
            matoffset += aligned;
        }
        result_v[row] = accumulant;
    }
#endif
   }
#endif

    tm.printAll(stdout, 80);
    printf("time = %f ms\n", tm["spmv"]->getTime());

    printf("Gflops: %f, time: %f (ms)\n", gflops, elapsed);
    //for (int i=0; i < 10; i++) {
        //printf("omp result: %f\n", result_v[i]);
    //}


    printf("l2norm of omp version: %f\n", l2norm(result_v));
    spmv_serial(mat, vec_v, result_v);
    printf("l2norm of serial version: %f\n", l2norm(result_v));
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
#pragma simd
            for (int i = 0; i < nz; i++) {
	            vecid = col_id[matoffset];
                float data_offset = data[matoffset];
#pragma simd
                for (int v = 0; v < nb_vec; v++) {
	                accumulant[v] += data_offset * vec_in[v+vecid*nz];
                }
	            matoffset += aligned;
            }
#pragma simd
            for (int v=0; v < nb_vec; v++) {
                // vec_out[vecid+v*nb_rows] = accumulant[vecid+v*nb_rows];   // 145 Gflop with vec=4
                vec_out[v+vecid*nz] = accumulant[v];  // 170 Gflop with vec=4
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
template <typename T>
void ELL_OPENMP<T>::method_2(int nb_vectors)
{
	printf("============== METHOD 2 ===================\n");
    printf("sizeof(T) = %d\n", sizeof(T));
    // implementation on the CPU, using OpenMP, and ELL_OPENMP format

    int nz = mat.ell_num;
    std::vector<int>& col_id = mat.ell_col_id;
    std::vector<T>& data = mat.ell_data;

    std::vector<T> v(mat.matinfo.height);
    std::vector<T> result(mat.matinfo.height, 0.);
    printf("fill random\n");
    fill_random(mat, v);
    printf("spmv_serial\n");
    spmv_serial(mat, v, result);
#if 1
    for (int i=0; i < 10; i++) {printf("result: %f\n", result[i]); }
    exit(0);
#endif

    assert(nb_vectors == 1);

    // Create arrays that duplicate the weight matrix four times. Different weights, but the same column pointers
    std::vector<T> data4 = std::vector<T>(data.size()*4,0.);
    T dat;
    int nb_mat = 4; // number derivatives to use
#pragma omp parallel private(dat)
{
#pragma omp for
    for (int i=0; i < data.size(); i++) {
        dat = data[i];
        for (int j=0; j < nb_mat; j++) {
            data4[nb_mat*i+j] = dat;
       }
    }
}

    printf("nz in row: %d\n", nz);
    printf("vec_v size: %d\n", vec_v.size());
    printf("aligned_length= %d\n", aligned_length);
    printf("vector length= %d\n", vec_v.size());
    printf("result_length= %d\n", result_v.size());
    printf("nb_vectors= %d\n", nb_vectors);
    printf("col_id.size = %d\n", col_id.size());

    int nb_rows = vec_v.size();
    std::vector<float> vec_in(nb_rows*nb_vectors, 1.);
    // 4 derivatives of nb_vectors functions
    std::vector<float> vec_out(4*nb_rows*nb_vectors, 1.);
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
#pragma omp parallel private(vecid, matrixelem, vecelem, aligned, nb_vec, nb_rows )
{
        std::vector<float> accumulant(4*nb_vec);
        std::vector<float> data_offset(4);  // one per derivative
#pragma omp for 
        for (int row=0; row < nb_rows; row++) {
            int matoffset = row;
            std::fill(accumulant.begin(), accumulant.end(), 0.);
// force vectorization of loop (I doubt it is appropriate)
#pragma simd
            for (int i = 0; i < nz; i++) {
	            vecid = col_id[matoffset];
#pragma simd
                for (int j=0; j < nb_mat; j++) {
                    data_offset[j] = data[nb_mat*matoffset+j];
#pragma simd
                    for (int v = 0; v < nb_vec; v++) {
                        // formula for single vector  [v=0]
                        // vector: (vx,vy,vz,vl)_1, (vx,vy,vz,vl)_2
	                    accumulant[j] += data_offset[j] * vec_in[j+nb_mat*vecid];
	                   // accumulant[v+j*nb_vec] += data_offset[j] * vec_in[v+j*nb_vec+vecid*nz];
	                    //accumulant[j+v*4] += data_offset[j] * vec_in[4*v+j+vecid*nz];
                    }
               }
	           matoffset += aligned;
            }
#pragma simd
            for (int j=0; j < nb_mat; j++) {
                for (int v=0; v < nb_vec; v++) {
                    // single vector
                    vec_out[j+vecid*nb_mat] = accumulant[j];  // 170 Gflop with vec=4
                    // single vector
                    //vec_out[vecid+j*nb_rows] = accumulant[j];  // 170 Gflop with vec=4

                    // vec_out[vecid+v*nb_rows] = accumulant[vecid+v*nb_rows];   // 145 Gflop with vec=4
                    //vec_out[v+vecid*nz] = accumulant[v];  // 170 Gflop with vec=4
                }
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
#endif
//----------------------------------------------------------------------
template <typename T>
void ELL_OPENMP<T>::method_3(int nbit)
{
	printf("============== METHOD 3 ===================\n");
    printf("Implement streaming\n");
    // implementation on the CPU, using OpenMP, and ELL_OPENMP format

    int nz = mat.ell_num;
    std::vector<int>& col_id = mat.ell_col_id;
    std::vector<T>& data = mat.ell_data;
    int nb_rows = vec_v.size();

    // transpose col_id array 
    // Current version, from ell_matrix:  [nz][nrow]
    // Transform to [nrow][nz]
    std::vector<int> col_id_t(col_id.size());
    for (int n=0; n < nz; n++) {
        for (int row=0; row < nb_rows; row++) {
            col_id_t[n+nz*row] = col_id[row+n*nb_rows];
        }
    }
 
    printf("nz in row: %d\n", nz);
    printf("nb rows: %d\n", vec_v.size());
    printf("aligned_length= %d\n", aligned_length);
    printf("vector length= %d\n", vec_v.size());
    printf("result_length= %d\n", result_v.size());

    float gflops;
    float elapsed; 

    //printf("maximum nb threads: %d\n", omp_get_max_threads());

    printf("aligned_length= %d\n", aligned_length);
    printf("aligned_length= %d\n", aligned_length);
    printf("nz*aligned_length= %d\n", nz*aligned_length);
    printf("size of col_id: %d\n", col_id.size());
    printf("size of data: %d\n", data.size());
    printf("aligned_length= %d\n", aligned_length);


    //int tid = omp_get_thread_num();

        //----------------------------
#if 1
    int matoffset;
    int vecid;
    const int aligned = aligned_length; 
    //const int aligned = aligned_length; // Gflop goes from 25 to 0.5 (Cannot make it private)

    // Must now work on alignmentf vectors. 
    for (int it=0; it < 10; it++) {
        tm["spmv"]->start();
//#pragma omp parallel private(vecid, accumulant, aligned )
//#pragma omp parallel private(vecid, matrixelem, accumulant, matoffset) firstprivate(aligned, nb_rows, nz)
#pragma omp parallel firstprivate(nb_rows, nz)
{
        //std::vector<float> datv(32);
        //std::vector<float> vecidv(32);
    float accumulant;
#pragma omp for 
// simd slows it down
#pragma simd   
//#pragma ivdep
// segmentation faul because vectors are not aligned
//#pragma vector aligned
        for (int row=0; row < nb_rows; row++) {
            // should have a stride of 32 in order to define
            // line result_v = accumulant
            int matoffset = row*nz;
            float accumulant = 0.;
            //for (int i = 0; i < nz; i++) {
//#pragma simd
            //for (int i = 0; i < nz; i++) {
                //datv[i] = data[matoffset+i];
                //vecidv[i] = vec_v[col_id_t[matoffset+i]];
            //}
#pragma simd
// segmentation faul because vectors are not aligned
//#pragma vector aligned
// slows code
//#pragma ivdep 
            for (int i = 0; i < nz; i++) {
	            int vecid = col_id_t[matoffset+i];
                float d = data[matoffset+i];
	            accumulant   += d * vec_v[vecid];  // most efficient, 12 Gflops
	            //accumulant += datv[i] * vec_v[vecid];
	            //accumulant += datv[i] * vec_v[vecidv[i]];
	            //accumulant += datv[i] * vecidv[i];
	            //matoffset += aligned;
            }
            result_v[row] = accumulant;
        }
}
    elapsed = tm["spmv"]->end();
    gflops = 2.*nz*vec_v.size()*1e-9 / (1e-3*tm["spmv"]->getTime()); // assumes count of 1
    printf("Gflops: %f, time: %f (ms)\n", gflops, elapsed);
   }
#endif

    tm.printAll(stdout, 80);
    printf("time = %f ms\n", tm["spmv"]->getTime());

    printf("Gflops: %f, time: %f (ms)\n", gflops, elapsed);

    printf("l2norm of omp version: %f\n", l2norm(result_v));

    spmv_serial(mat, vec_v, result_v);
    printf("l2norm of serial version: %f\n", l2norm(result_v));

}
//----------------------------------------------------------------------
template <typename T>
void ELL_OPENMP<T>::method_4(int nbit)
{
    // Align the vectors so I can use vector instrincis and other technique.s 
    // Repalce std::vector by pointers to floats and ints. 
	printf("============== METHOD 4 ===================\n");
    printf("Implement streaming\n");
    // implementation on the CPU, using OpenMP, and ELL_OPENMP format

    int nz = mat.ell_num;
    std::vector<int>& col_id = mat.ell_col_id;
    std::vector<T>& data = mat.ell_data;
    int nb_rows = vec_v.size();

    // transpose col_id array 
    // Current version, from ell_matrix:  [nz][nrow]
    // Transform to [nrow][nz]
    std::vector<int> col_id_t(col_id.size());
    std::vector<float> data_t(data.size());
    for (int row=0; row < nb_rows; row++) {
        for (int n=0; n < nz; n++) {
            col_id_t[n+nz*row] = col_id[row+n*nb_rows];
              data_t[n+nz*row]   = data[row+n*nb_rows];
        }
    }

    //for (int i=0; i < 128; i++) {
        //printf("col_id_t[%d] = %d\n", i, col_id_t[i]);
   //}
 
    printf("nz in row: %d\n", nz);
    printf("nb rows: %d\n", vec_v.size());
    printf("aligned_length= %d\n", aligned_length);
    printf("vector length= %d\n", vec_v.size());
    printf("result_length= %d\n", result_v.size());

    float gflops;
    float elapsed; 

    //printf("maximum nb threads: %d\n", omp_get_max_threads());

    printf("aligned_length= %d\n", aligned_length);
    printf("aligned_length= %d\n", aligned_length);
    printf("nz*aligned_length= %d\n", nz*aligned_length);
    printf("size of col_id: %d\n", col_id.size());
    printf("size of data: %d\n", data.size());
    printf("aligned_length= %d\n", aligned_length);


    //int tid = omp_get_thread_num();
    //
    // arrays required: data, cl_id_t, vec_v, result_v

    float* result_va = (float*) _mm_malloc(result_v.size()*sizeof(float), 64);
    float* vec_va = (float*) _mm_malloc(vec_v.size()*sizeof(float), 64);
    float* data_a = (float*) _mm_malloc(data.size()*sizeof(float), 64);
    int* col_id_ta = (int*) _mm_malloc(data.size()*sizeof(int), 64);

    for (int i=0; i < result_v.size(); i++) { result_va[i] = result_v[i]; }
    for (int i=0; i < vec_v.size(); i++) { vec_va[i] = vec_v[i]; }
    for (int i=0; i < data.size(); i++) { data_a[i] = data_t[i]; }
    for (int i=0; i < col_id_t.size(); i++) { col_id_ta[i] = col_id_t[i]; }

        //----------------------------
#if 1
    int matoffset;
    int vecid;
    const int aligned = aligned_length; 
    //const int aligned = aligned_length; // Gflop goes from 25 to 0.5 (Cannot make it private)

    // Must now work on alignmentf vectors. 
    for (int it=0; it < 10; it++) {
        tm["spmv"]->start();
//#pragma omp parallel private(vecid, accumulant, aligned )
//#pragma omp parallel private(vecid, matrixelem, accumulant, matoffset) firstprivate(aligned, nb_rows, nz)
#pragma omp parallel firstprivate(nb_rows, nz)
{
        //std::vector<float> datv(32);
        //std::vector<float> vecidv(32);
#pragma omp for 
// simd slows it down (12 Gflops), 22Gflops (compact case), 2 Gflops (random case)
//#pragma simd   
#pragma ivdep
// segmentation fault because vectors are not aligned
//#pragma vector aligned
        for (int row=0; row < nb_rows; row++) {
            float accumulant = 0;
            // should have a stride of 32 in order to define
            // line result_v = accumulant
#pragma simd
// segmentation faul because vectors are not aligned
#pragma vector aligned
// slows code
#pragma ivdep 
            for (int i = 0; i < nz; i++) {
                int matoffset = row*nz+i;
	            int vecid = col_id_ta[matoffset];
                float d = data_a[matoffset];
	            accumulant   += d * vec_va[vecid];  // most efficient, 12 Gflops
	            //accumulant += datv[i] * vec_v[vecid];
	            //accumulant += datv[i] * vec_v[vecidv[i]];
	            //accumulant += datv[i] * vecidv[i];
	            //matoffset += aligned;
            }
            result_va[row] = accumulant;
        }
}
    elapsed = tm["spmv"]->end();
    gflops = 2.*nz*vec_v.size()*1e-9 / (1e-3*tm["spmv"]->getTime()); // assumes count of 1
    printf("Gflops: %f, time: %f (ms)\n", gflops, elapsed);
   }
#endif

    tm.printAll(stdout, 80);
    printf("time = %f ms\n", tm["spmv"]->getTime());

    printf("Gflops: %f, time: %f (ms)\n", gflops, elapsed);

    printf("l2norm of omp version: %f\n", l2norm(result_va, result_v.size()));

    spmv_serial(mat, vec_v, result_v);
    printf("l2norm of serial version: %f\n", l2norm(result_v));

    _mm_free(result_va);
    _mm_free(vec_va);
    _mm_free(data_a);
    _mm_free(col_id_ta);

}
//----------------------------------------------------------------------
//----------------------------------------------------------------------
template <typename T>
void ELL_OPENMP<T>::method_5(int nbit)
{
    // Align the vectors so I can use vector instrincis and other technique.s 
    // Repalce std::vector by pointers to floats and ints. 
	printf("============== METHOD 5 ===================\n");
    printf("Implement streaming\n");
    // vectors are aligned. Start using vector _mm_ constructs. 

    fill_random(mat, vec_v);

    int nz = mat.ell_num;
    std::vector<int>& col_id = mat.ell_col_id;
    std::vector<T>& data = mat.ell_data;
    int nb_rows = vec_v.size();

    // transpose col_id array 
    // Current version, from ell_matrix:  [nz][nrow]
    // Transform to [nrow][nz]
    std::vector<int> col_id_t(col_id.size());
    std::vector<float> data_t(data.size());
    for (int row=0; row < nb_rows; row++) {
        for (int n=0; n < nz; n++) {
            col_id_t[n+nz*row] = col_id[row+n*nb_rows];
              data_t[n+nz*row]   = data[row+n*nb_rows];
        }
    }

    //for (int i=0; i < 128; i++) {
        //printf("col_id_t[%d] = %d\n", i, col_id_t[i]);
    //}
 
    printf("nz in row: %d\n", nz);
    printf("nb rows: %d\n", vec_v.size());
    printf("aligned_length= %d\n", aligned_length);
    printf("vector length= %d\n", vec_v.size());
    printf("result_length= %d\n", result_v.size());
    printf("nz*aligned_length= %d\n", nz*aligned_length);
    //printf("maximum nb threads: %d\n", omp_get_max_threads());
    printf("size of col_id: %d\n", col_id.size());
    printf("size of data: %d\n", data.size());

    float gflops;
    float elapsed; 



    float* result_va = (float*) _mm_malloc(result_v.size()*sizeof(float), 64);
    float* vec_va    = (float*) _mm_malloc(vec_v.size()*sizeof(float), 64);
    float* data_a    = (float*) _mm_malloc(data.size()*sizeof(float), 64);
    int* col_id_ta   = (int*)   _mm_malloc(col_id.size()*sizeof(int), 64);

    for (int i=0; i < result_v.size(); i++) { result_va[i] = result_v[i]; }
    for (int i=0; i < vec_v.size(); i++)    { vec_va[i]    = vec_v[i];    }
    for (int i=0; i < data.size(); i++)     { data_a[i]    = data_t[i];   }
    for (int i=0; i < col_id.size(); i++)   { col_id_ta[i] = col_id_t[i]; }

        //----------------------------
#if 1
    int matoffset;
    int vecid;
    const int aligned = aligned_length; 
    //const int aligned = aligned_length; // Gflop goes from 25 to 0.5 (Cannot make it private)

    // Must now work on alignmentf vectors. 
    // Produces the correct serial result
    for (int it=0; it < 10; it++) {
        tm["spmv"]->start();
#pragma omp parallel firstprivate(nb_rows, nz)
{
    const int scale = 4; // in bytes
    const int skip = 16;
// Perhaps I should organize loops differently? Vectorize of rows? 
#pragma omp for 
        for (int row=0; row < nb_rows; row++) {
            __m512 accu = _mm512_setzero_ps();
            // should have a stride of 32 in order to define
            // line result_v = accumulant
            float a = 0.0;
            // skip 16 if floats, skip 8 if doubles
            for (int i = 0; i < nz; i+=skip) {  // nz is multiple of 32 (for now)
                // 16 ints
                int matoffset = row*nz + i;
                __m512i vecidv = _mm512_load_epi32(col_id_ta+matoffset);  // only need 1/2 registers if dealing with doubles
                __m512  dv     = _mm512_load_ps(data_a+matoffset);
                __m512  vv     = _mm512_i32gather_ps(vecidv, vec_va, scale); // scale = 4 (floats)
                //accu = _mm512_fmadd_ps(dv, vv, accu);
                accu = _mm512_mul_ps(dv, vv);
                a += _mm512_reduce_add_ps(accu); 
            }
            result_va[row] = a;
        }
}
    tm["spmv"]->end();
    elapsed = tm["spmv"]->getTime();
    gflops = 2.*nz*vec_v.size()*1e-9 / (1e-3*elapsed); // assumes count of 1
    printf("Gflops: %f, time: %f (ms)\n", gflops, elapsed);
   }
#endif

#if 0
    std::vector<float> one_res(nb_rows);  // single result
    for (int w=0; w < 4; w++) {
        for (int i=0; i < nb_rows; i++) {
            one_res[i] = result_va[4*i+w];
        }
        printf("method_5, l2norm[%d]=of omp version: %f\n", w, l2norm(one_res));
    }

    spmv_serial(mat, vec_v, result_v);
    printf("method_5, l2norm of serial version: %f\n", l2norm(result_v));
#endif


    printf("method_5, l2norm of omp version: %f\n", l2norm(result_va, result_v.size()));
    spmv_serial(mat, vec_v, result_v);
    printf("method_5, l2norm of serial version: %f\n", l2norm(result_v));
    spmv_serial_row(mat, vec_v, result_v);
    printf("method_5, l2norm of serial row version: %f\n", l2norm(result_v));

    _mm_free(result_va);
    _mm_free(vec_va);
    _mm_free(data_a);
    _mm_free(col_id_ta);

}
//----------------------------------------------------------------------
template <typename T>
void ELL_OPENMP<T>::method_6(int nbit)
{
    // Align the vectors so I can use vector instrincis and other technique.s 
    // Repalce std::vector by pointers to floats and ints. 
	printf("============== METHOD 6 ===================\n");
    printf("Implement streaming\n");
    printf("INCORRECT REULTS\n");
    // vectors are aligned. Start using vector _mm_ constructs. 

    int nz = mat.ell_num;
    std::vector<int>& col_id = mat.ell_col_id;
    std::vector<T>& data = mat.ell_data;
    int nb_rows = vec_v.size();

    // transpose col_id array 
    // Current version, from ell_matrix:  [nz][nrow]
    // Transform to [nrow][nz]
    std::vector<int> col_id_t(col_id.size());
    std::vector<float> data_t(data.size());
    for (int row=0; row < nb_rows; row++) {
        for (int n=0; n < nz; n++) {
            col_id_t[n+nz*row] = col_id[row+n*nb_rows];
              data_t[n+nz*row]   = data[row+n*nb_rows];
        }
    }

    //for (int i=0; i < 128; i++) {
        //printf("col_id_t[%d] = %d\n", i, col_id_t[i]);
    //}
 
    printf("nz in row: %d\n", nz);
    printf("nb rows: %d\n", vec_v.size());
    printf("aligned_length= %d\n", aligned_length);
    printf("vector length= %d\n", vec_v.size());
    printf("result_length= %d\n", result_v.size());
    printf("nz*aligned_length= %d\n", nz*aligned_length);
    //printf("maximum nb threads: %d\n", omp_get_max_threads());
    printf("size of col_id: %d\n", col_id.size());
    printf("size of data: %d\n", data.size());

    float gflops;
    float elapsed; 



    float* result_va = (float*) _mm_malloc(result_v.size()*sizeof(float), 64);
    float* vec_va    = (float*) _mm_malloc(vec_v.size()*sizeof(float), 64);
    float* data_a    = (float*) _mm_malloc(data.size()*sizeof(float), 64);
    int* col_id_ta   = (int*)   _mm_malloc(col_id.size()*sizeof(int), 64);

    for (int i=0; i < result_v.size(); i++) { result_va[i] = result_v[i]; }
    for (int i=0; i < vec_v.size(); i++)    { vec_va[i]    = vec_v[i];    }
    for (int i=0; i < data.size(); i++)     { data_a[i]    = data_t[i];   }
    for (int i=0; i < col_id.size(); i++)   { col_id_ta[i] = col_id_t[i]; }

        //----------------------------
#if 1
    int matoffset;
    int vecid;
    const int aligned = aligned_length; 
    //const int aligned = aligned_length; // Gflop goes from 25 to 0.5 (Cannot make it private)

   tm["spmv"]->reset();

    // Must now work on alignmentf vectors. 
    // Produces the correct serial result
    for (int it=0; it < 1; it++) { // check for correctness
    //for (int it=0; it < 10; it++) {
        tm["spmv"]->start();
#pragma omp parallel firstprivate(nb_rows, nz)
{
    const int scale = 4; // in bytes
    const int skip = 16;
    float row_accum[skip];
// Perhaps I should organize loops differently? Vectorize of rows? 
#pragma omp for 
        for (int row=0; row < nb_rows; row+=skip) {
            __m512 row_accu = _mm512_setzero_ps();
         for (int ir=0; ir < skip; ir++) {
#pragma simd
            __m512    accu = _mm512_setzero_ps();
            __mmask16 mask = _mm512_int2mask(ir); // has an effect on results. 
            // should have a stride of 32 in order to define
            // line result_v = accumulant
            float a = 0.0;  // How can make sure it stays in a vector register or in cache? 
            // skip 16 if floats, skip 8 if doubles
            for (int i=0; i < nz; i+=skip) {  // nz is multiple of 32 (for now)
                // 16 ints
                int matoffset = (ir+row)*nz + i;
                __m512i vecidv = _mm512_load_epi32(col_id_ta+matoffset);  // only need 1/2 registers if dealing with doubles
                __m512  dv     = _mm512_load_ps(data_a+matoffset);
                __m512  vv     = _mm512_i32gather_ps(vecidv, vec_va, scale); // scale = 4 (floats)
                accu = _mm512_fmadd_ps(dv, vv, accu);
            }
             __m512 reduce = _mm512_set1_ps(_mm512_reduce_add_ps(accu)); // float -> 16 floats
             row_accu = _mm512_mask_add_ps(row_accu, mask, reduce, row_accu);
          }
          _mm512_store_ps(result_va+row, row_accu); // PRODUCING WRONG RESULTS
        }
}
    tm["spmv"]->end();
    elapsed = tm["spmv"]->getTime();
    gflops = 2.*nz*vec_v.size()*1e-9 / (1e-3*elapsed); // assumes count of 1
    printf("Gflops: %f, time: %f (ms)\n", gflops, elapsed);
   }
#endif

    printf("l2norm of omp version: %f\n", l2norm(result_va, result_v.size()));

    spmv_serial(mat, vec_v, result_v);
    printf("l2norm of serial version: %f\n", l2norm(result_v));
    spmv_serial_row(mat, vec_v, result_v);
    printf("l2norm of serial row version: %f\n", l2norm(result_v));

    _mm_free(result_va);
    _mm_free(vec_va);
    _mm_free(data_a);
    _mm_free(col_id_ta);

}
//----------------------------------------------------------------------
template <typename T>
void ELL_OPENMP<T>::method_7(int nbit)
{
    // Align the vectors so I can use vector instrincis and other technique.s 
    // Replace std::vector by pointers to floats and ints. 
    // Process 4 matrices and 4 vectors. Make 4 matrices identical. 
	printf("============== METHOD 7 ===================\n");
    printf("Implement streaming\n");
    // vectors are aligned. Start using vector _mm_ constructs. 

    fill_random(mat, vec_v);

    int nb_mat = 4; // must be 4
    int nb_vec = 4; // must be 4
    int nb_rows = vec_v.size();
    int nz = mat.ell_num;

    std::vector<int>& col_id = mat.ell_col_id;
    std::vector<float>& data = mat.ell_data;

    //std::vector<float> vec_vt(nb_vec*vec_v.size());
    //std::vector<float> result_vt(nb_vec*nb_mat*vec_v.size());
    //std::vector<int> col_id_t(col_id.size()*4);
    //std::vector<float> data_t(data.size()*4);

    float* vec_vt    = (float*) _mm_malloc(sizeof(float) * nb_vec * vec_v.size(), 64);
    float* result_vt = (float*) _mm_malloc(sizeof(float) * nb_vec * nb_mat * vec_v.size(), 64);
    int*   col_id_t  = (int*)   _mm_malloc(sizeof(int)   * col_id.size(), 16);
    float* data_t    = (float*) _mm_malloc(sizeof(float) * nb_mat * data.size(), 64);

    if (vec_vt == 0 || result_vt == 0 || col_id_t == 0 || data_t == 0) {
        printf("1. memory allocation failed\n");
        exit(0);
    }

    // transpose col_id array 
    // Current version, from ell_matrix:  [nz][nrow]
    // Transform to [nrow][nz]
    #define COL(c,r)   col_id_t[(c) + nz*(r)]
    //#define DATA(m,c,r)    data_t[(m) + nb_mat*((c) + nz*(r))]
    #define VEC(m,r)       vec_vt[(m) + nb_mat*(r)]
    #define RES(m,v,r)  result_vt[(m) + nb_mat*((v) + nb_vec*(r))]

    // Create 4 identical vectors, 4 identical matrices (easier to check results at first)
    for (int m=0; m < nb_mat; m++) {
        for (int r=0; r < nb_rows; r++) {
            VEC(m,r) = vec_v[r];
        }
    }

    for (int r=0; r < nb_rows; r++) {
        for (int n=0; n < nz; n++) {
            COL(n,r) = col_id[r + nb_rows*n];
        }
    }
 
    // Restructure data array
    // order:[r=0,m=0,n=(0,1,2,3)],[r=0,m=1,n=(0,1,2,3)],...,[r=0,m=3,n=(0,1,2,3)]
    //       [r=1,m=0,n=(0,1,2,3)],....
    int n_skip = 4;
    int nz4 = nz / n_skip; // check that nz is multiple of n_skip
    for (int r=0; r < nb_rows; r++) {
    for (int n=0; n < nz; n += n_skip) {
        int n4 = n / 4; // if n_skip == 4
        for (int m=0; m < nb_mat; m++) {
            for (int in=0; in < n_skip; in++) {
                data_t[in+n_skip*(m+nb_mat*(n4+nz4*r))] = data[r+(n+in)*nb_rows];
            }
        }
    }}

    //for (int i=0; i< nb_rows*nz*nb_mat; i++) {
        //printf("data_t[%d]= %f\n", i, data_t[i]);
    //}
    //exit(0);

    print_f(&data[0], "data");
    print_f(data_t, "data_t");
    print_i(&col_id[0], "col_id");
    print_i(col_id_t, "col_id_t");
    print_f(&vec_v[0], "vec_v");
    print_f(vec_vt, "vec_vt");

#if 0
    printf("after initialization\n");

    //for (int i=0; i < 128; i++) {
        //printf("col_id_t[%d] = %d\n", i, col_id_t[i]);
    //}
 
    printf("nz in row: %d\n", nz);
    printf("nb rows: %d\n", vec_v.size());
    printf("aligned_length= %d\n", aligned_length);
    printf("vector length= %d\n", vec_v.size());
    printf("result_length= %d\n", result_v.size());
    printf("nz*aligned_length= %d\n", nz*aligned_length);
    printf("maximum nb threads: %d\n", omp_get_max_threads());
    printf("size of col_id: %d\n", col_id.size());
    printf("size of data: %d\n", data.size());
#endif

    float gflops;
    float elapsed; 

#if 0
    float* result_va = (float*) _mm_malloc(result_v.size()*sizeof(float), 64);
    float* vec_va    = (float*) _mm_malloc(vec_v.size()*sizeof(float), 64);
    float* data_a    = (float*) _mm_malloc(data.size()*sizeof(float), 64);
    int* col_id_ta   = (int*)   _mm_malloc(col_id.size()*sizeof(int), 64);

    for (int i=0; i < result_v.size(); i++) { result_va[i] = result_v[i]; }
    for (int i=0; i < vec_v.size(); i++)    { vec_va[i]    = vec_v[i];    }
    for (int i=0; i < data.size(); i++)     { data_a[i]    = data_t[i];   }
    for (int i=0; i < col_id.size(); i++)   { col_id_ta[i] = col_id_t[i]; }
#endif

// work on unpacking 
    //exit(0);

        //----------------------------
#if 1
    int matoffset;
    int vecid;
    const int aligned = aligned_length; 
    //const int aligned = aligned_length; // Gflop goes from 25 to 0.5 (Cannot make it private)
 
#if 0
    // Time pure loads
    for (int it=0; it < 10; it++) {
        tm["spmv"]->start();
#pragma omp parallel
{
#pragma omp for
       for (int r=0; r  < nb_rows; r += 16) {
            const __m512 v1_old = _mm512_load_ps(vec_vt + r);
             _mm512_store_ps(result_vt + r, v1_old);
        }
}
       tm["spmv"]->end();
       float elapsed_mem = tm["spmv"]->getTime();
       float gbytes = (nb_rows/16)*sizeof(float) * (16+16) * 1.e-9;
       //printf("gbytes= %f\n", gbytes);
       float bandwidth = gbytes / (elapsed_mem*1.e-3);
       printf("bandwidth= %f (gbytes/sec), time: %f (ms)\n", bandwidth, elapsed_mem);
    }
#endif

//..................
// Time read 4 floats at a time with masking
//
#if 0
    // Time pure loads
    for (int it=0; it < 10; it++) {
        tm["spmv"]->start();

#pragma omp parallel
{
    const int int_mask_lo = (1 << 0) + (1 << 4) + (1 << 8) + (1 << 12);
    const int nb_mat = 4;
    const int nb_vec = 4;
#pragma omp for
       for (int r=0; r  < nb_rows; r += 4) {
            //const __m512 v1_old = _mm512_load_ps(vec_vt + r);
            //  reading 4 floats
             __m512 v1_old = _mm512_mask_loadunpacklo_ps(v1_old, _mm512_int2mask(int_mask_lo), vec_vt+r);
             _mm512_store_ps(result_vt+4*r, v1_old);  // writing 16 floats
       }
}
       tm["spmv"]->end();
       float elapsed_mem = tm["spmv"]->getTime();
       float gbytes = (nb_rows/4)*sizeof(float) * (16.+4.) * 1.e-9;
       float bandwidth = gbytes / (elapsed_mem*1.e-3);
       printf("bandwidth unpack= %f (gbytes/sec), time: %f (ms)\n", bandwidth, elapsed_mem);
    }
#endif
//.......................................................
    // Must now work on alignmentf vectors. 
    // Produces the correct serial result
    for (int it=0; it < 10; it++) {
        tm["spmv"]->start();
#pragma omp parallel firstprivate(nb_rows, nz)
{
    //const int skip = 1;
    const int int_mask_lo = (1 << 0) + (1 << 4) + (1 << 8) + (1 << 12);
    const int nb_mat = 4;
    const int nb_vec = 4;

    __m512  v1_old  = _mm512_setzero_ps();
    __m512  v2_old  = _mm512_setzero_ps();
    __m512  v3_old  = _mm512_setzero_ps();
    __m512i i3_old  = _mm512_setzero_ps();

#pragma omp for 
        for (int r=0; r < nb_rows; r++) {
//#pragma simd
            //printf("***** row %d\n", r);
            if (r > 1) exit(0);
            __m512 accu = _mm512_setzero_ps(); // 16 floats for 16 matrices
            float* addr_vector;
            int    icol;

#pragma simd
            for (int n=0; n < nz; n+=4) {  // nz is multiple of 32 (for now)
                    //printf("==== n= %d\n", n);

                // (m=[0,..,3],c=0),(m=[0,..,3],c=1),..,(m=[0,..,3],c=3) == 16 elements
                // Left is least significant (in Intel vectorization charts, right is least significant
                // Left here, is right in the Intel documents.
                // m0c0,m0c1,m0c2,m0c3,  m1c0,m1c1,m1c2,m2c3,  ...., m3c0,m3c1,m3c2,m3c3
                v1_old = _mm512_load_ps(data_t + nb_mat*(n + r*nz)); // load 16 at a time
                //printf("v1_old offset: %d, nb_mat= %d\n", n*nb_mat, nb_mat);
                //print_ps(v1_old, "v1_old: data_t");

                // icol is the same for all matrices
                icol         = col_id_t[n+nz*r+0];   // single element (but next 4 in cache)
                addr_vector  = vec_vt + nb_mat*icol; 
                // f0v0 means 0th element of function 0
                // read 4 vectors (f0v0,f1v0,f2v0,f3v0) and create vector (f0v0,f1v0,f2v0,f3v0, f0v0,f1v0,f2v0,f3v0,...)
                v3_old = _mm512_extload_ps(addr_vector, _MM_UPCONV_PS_NONE, _MM_BROADCAST_4X16, _MM_HINT_NT);
                // m0c0,m0c0,m0c0,m0c0,   m1c0,m1c0,m1c0,m1c0,   m2c0,m2c0,m2c0,m2c0,   m3c0,m3c0,m3c0,m3c0
                v2_old = _mm512_swizzle_ps(v1_old, _MM_SWIZ_REG_AAAA);
                //printf("1st icol= %d\n", icol);
                print_ps(v3_old, "1st v3old, extload_ps");
                print_ps(v2_old, "1st swizzle");
                accu = _mm512_fmadd_ps(v3_old, v2_old, accu);

                //-----
                icol         = col_id_t[n+nz*r+1];
                addr_vector  = vec_vt + nb_mat*icol;
                // read 4 vectors (m0v0,m0v1,m0v2,m0v3) and create vector (m0v0,m0v1,m0v2,m0v3,m0v0,m0v1,m0v2,m0v3,...)
                v3_old = _mm512_extload_ps(addr_vector, _MM_UPCONV_PS_NONE, _MM_BROADCAST_4X16, _MM_HINT_NT);
                // m0c0,m0c0,m0c0,m0c0, m1c0,m1c0,m1c0,m1c0, m2c0,m2c0,m2c0,m2c0,   m3c0,m3c0,m3c0,m3c0
                v2_old = _mm512_swizzle_ps(v1_old, _MM_SWIZ_REG_BBBB);
                //printf("2nd icol= %d\n", icol);
                print_ps(v3_old, "2nd v3old, extload_ps");
                print_ps(v2_old, "2nd swizzle");
                //exit(0);
                accu = _mm512_fmadd_ps(v3_old, v2_old, accu);

                //-----
                icol         = col_id_t[n+nz*r+2];
                addr_vector  = vec_vt + nb_mat*icol;
                v3_old = _mm512_extload_ps(addr_vector, _MM_UPCONV_PS_NONE, _MM_BROADCAST_4X16, _MM_HINT_NT);
                // m0c2,m0c2,m0c2,m0c2, m1c2,m1c2,m1c2,m1c2, m2c2,m2c2,m2c2,m2c2,   m3c2,m3c2,m3c2,m3c2
                v2_old = _mm512_swizzle_ps(v1_old, _MM_SWIZ_REG_CCCC);
                accu = _mm512_fmadd_ps(v3_old, v2_old, accu);
                //printf("3rd icol= %d\n", icol);
                //print_ps(v3_old, "3rd v3old, extload_ps");
                //print_ps(v2_old, "3rd swizzle");

                //-----
                icol         = col_id_t[n+nz*r+3];
                addr_vector  = vec_vt + nb_mat*icol;
                v3_old = _mm512_extload_ps(addr_vector, _MM_UPCONV_PS_NONE, _MM_BROADCAST_4X16, _MM_HINT_NT);
                // m0c3,m0c3,m0c3,m0c3, m1c3,m1c3,m1c3,m1c3, m2c3,m2c3,m2c3,m2c3,   m3c3,m3c3,m3c3,m3c3
                v2_old = _mm512_swizzle_ps(v1_old, _MM_SWIZ_REG_DDDD);
                accu = _mm512_fmadd_ps(v3_old, v2_old, accu);
                //printf("4th icol= %d\n", icol);
                //print_ps(v3_old, "4th v3old, extload_ps");
                //print_ps(v2_old, "4th swizzle");
                //print_ps(accu, "accu");
                //if (n >= 8) exit(0);
            }
            //if (r % skip == 0)
            _mm512_store_ps(result_vt+nb_mat*nb_vec*r, accu);
            // nr: no register
            //_mm512_storenr_ps(result_vt+nb_mat*nb_vec*r, accu);
            // no ordering and no read of cache lines from memory
            //_mm512_storenrngo_ps(result_vt+nb_mat*nb_vec*r, accu);
        } 
}
    tm["spmv"]->end();  // time for each matrix/vector multiply
    elapsed = tm["spmv"]->getTime();
    gflops = nb_mat * nb_vec * 2.*nz*vec_v.size()*1e-9 / (1e-3*elapsed); // assumes count of 1
    printf("Gflops: %f, time: %f (ms)\n", gflops, elapsed);
   }
#endif

#if 1
    std::vector<float> one_res(nb_rows);  // single result
    for (int w=0; w < 16; w++) {
        for (int i=0; i < nb_rows; i++) {
            one_res[i] = result_vt[16*i+w];
        }
        printf("method_7, l2norm[%d]=of omp version: %f\n", w, l2norm(one_res));
    }

    spmv_serial(mat, vec_v, result_v);
    printf("method_7, l2norm of serial version: %f\n", l2norm(result_v));
#endif

    // All 16 omp versions are the same. 
    // But off by factor of approximately 6 from exact version. 


    _mm_free(result_vt);
    _mm_free(vec_vt);
    _mm_free(data_t);
    _mm_free(col_id_t);

}
//----------------------------------------------------------------------
template <typename T>
void ELL_OPENMP<T>::method_8(int nbit)
{
    // Align the vectors so I can use vector instrincis and other technique.s 
    // Replace std::vector by pointers to floats and ints. 
    // Process 4 matrices and 4 vectors. Make 4 matrices identical. 
	printf("============== METHOD 8 ===================\n");
    printf("Implement streaming\n");
    // vectors are aligned. Start using vector _mm_ constructs. 

    fill_random(mat, vec_v);

    int nb_mat = 4; // must be 4
    int nb_vec = 4; // must be 4
    int nz = mat.ell_num;
    std::vector<int>& col_id = mat.ell_col_id;
    std::vector<float>& data = mat.ell_data;
    int nb_rows = vec_v.size();

    //std::vector<float> vec_vt(nb_vec*vec_v.size());
    //std::vector<float> result_vt(nb_vec*nb_mat*vec_v.size());
    //std::vector<int> col_id_t(col_id.size()*4);
    //std::vector<float> data_t(data.size()*4);

    float* vec_vt    = (float*) _mm_malloc(sizeof(float) * nb_vec * vec_v.size(), 64);
    float* result_vt = (float*) _mm_malloc(sizeof(float) * nb_vec * nb_mat * vec_v.size(), 64);
    int*   col_id_t  = (int*)   _mm_malloc(sizeof(int)   * col_id.size(), 16);
    float* data_t    = (float*) _mm_malloc(sizeof(float) * nb_mat * col_id.size(), 64);

    if (vec_vt == 0 || result_vt == 0 || col_id_t == 0 || data_t == 0) {
        printf("1. memory allocation failed\n");
        exit(0);
    }

    // transpose col_id array 
    // Current version, from ell_matrix:  [nz][nrow]
    // Transform to [nrow][nz]
    #define COL(c,r)   col_id_t[(c) + nz*(r)]
    #define DATA(m,c,r)    data_t[(m) + nb_mat*((c) + nz*(r))]
    #define VEC(m,r)       vec_vt[(m) + nb_mat*(r)]
    #define RES(m,v,r)  result_vt[(m) + nb_mat*((v) + nb_vec*(r))]

    // Create 4 identical vectors, 4 identical matrices (easier to check results at first)
    for (int m=0; m < nb_mat; m++) {
        for (int r=0; r < nb_rows; r++) {
            for (int n=0; n < nz; n++) {
                DATA(m,n,r) = data[r+n*nb_rows];
                //printf("data= %f\n", DATA(m,n,r));
            }
            VEC(m,r) = vec_v[r] * (1*m+1);
            //printf("vec= %f\n", VEC(m,r));
        }
    }

    for (int r=0; r < nb_rows; r++) {
        for (int n=0; n < nz; n++) {
            COL(n,r) = col_id[r + nb_rows*n];
        }
    }

    //for (int i=0; i < 128; i++) {
        //printf("col_id_t[%d] = %d\n", i, col_id_t[i]);
    //}
 
#if 0
    printf("nz in row: %d\n", nz);
    printf("nb rows: %d\n", vec_v.size());
    printf("aligned_length= %d\n", aligned_length);
    printf("vector length= %d\n", vec_v.size());
    printf("result_length= %d\n", result_v.size());
    printf("nz*aligned_length= %d\n", nz*aligned_length);
    //printf("maximum nb threads: %d\n", omp_get_max_threads());
    printf("size of col_id: %d\n", col_id.size());
    printf("size of data: %d\n", data.size());
#endif

    float gflops;
    float elapsed; 

    //----------------------------
    // Restructure data array
    // order:[r=0,m=0,n=(0,1,2,3)],[r=0,m=1,n=(0,1,2,3)],...,[r=0,m=3,n=(0,1,2,3)]
    //       [r=1,m=0,n=(0,1,2,3)],....
    int nz4 = nz / 4;
    int n_skip = 4;
    for (int r=0; r < nb_rows; r++) {
    for (int n=0; n < nz; n+=n_skip) {
        int n4 = n >> 2; // if n_skip == 4
        for (int m=0; m < nb_mat; m++) {
            for (int in=0; in < n_skip; in++) {
                //data(in,m,c,r)
                //data_t[in+nb_mat*(m+nz*r)] = data[r+(n+in)*nb_rows];
                
                data_t[in+n_skip*(m+nb_mat*(n4+nz4*r))] = data[r+(n+in)*nb_rows] * (1*m+1);
                ;
            }
        }
    }}

    print_f(vec_vt, "vec_vt, first 16");
    print_f(vec_vt+16, "vec_vt, second 16");
    print_i(col_id_t, "col_id_t, first 16");
    print_i(col_id_t+16, "col_id_t, second 16");
    print_f(&data[0], "data, first 16");
    print_f(data_t, "data_t, first 16");
    print_f(data_t+16, "data_t, second 16");

#if 1
//.......................................................
    // Must now work on alignmentf vectors. 
    // Produces the correct serial result
    for (int it=0; it < 10; it++) {
        tm["spmv"]->start();
#pragma omp parallel firstprivate(nb_rows, nz)
{
    //const int skip = 1;
    const int int_mask_lo = (1 << 0) + (1 << 4) + (1 << 8) + (1 << 12);
    const int nb_mat = 4;
    const int nb_vec = 4;
    const int scale = 4;

    __m512 v1_old = _mm512_setzero_ps();
    __m512 v2_old = _mm512_setzero_ps();
    __m512 v3_old = _mm512_setzero_ps();
    __m512i i3_old = _mm512_setzero_ps();
   // 0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3 (OK)
   const __m512i offsets = _mm512_set4_epi32(3,2,1,0); 
   const __m512i four = _mm512_set4_epi32(4,4,4,4); 
   const __m512i offsets1 = _mm512_set_epi32(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0);
   // (c4,c4,c4,c4,c3,c3,c3,c3,c2,c2,c2,c2,c1,c1,c1,c1)
   // value of vperm has no effect
   const __m512i vperm = _mm512_set_epi32(15,11,7,3,14,10,6,2,13,9,5,1,12,8,4,0);
   //const __m512i vperm = _mm512_set_epi32(15,11,7,3,14,10,6,2,13,9,5,1,12,8,4,0);
   //const __m512i vperm = _mm512_set_epi32(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15);
   //const __m512i vperm = _mm512_set_epi32(0,4,8,,4,5,6,7,8,9,10,11,12,13,14,15);
   //const __m512i vperm = _mm512_set_epi32(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1);
   //print_epi32(offsets, "offsets");

#pragma omp for 
        for (int r=0; r < nb_rows; r++) {
            //printf("****************** row %d\n", r);
            //if (r == 1) exit(0);
//#pragma simd
            __m512 accu = _mm512_setzero_ps(); // 16 floats for 16 matrices

#pragma simd
            for (int n=0; n < nz; n+=4) {  // nz is multiple of 32 (for now)
                //if (n == 4) exit(0);
                int    icol;
                float* addr_vector;
                //printf("*** n= %d, %d\n", n, nz);

                v1_old = _mm512_load_ps(data_t + nb_mat*(n + r*nz)); // load 16 at a time
                //print_ps(v1_old, "--- v1_old, data_t");

                //v3_old = _mm512_extload_ps(addr_vector, _MM_UPCONV_PS_NONE, _MM_BROADCAST_4X16, _MM_HINT_NT);
         
        // if col_id is (2,35,52,30)=(c1,c2,c3,c4), the offsets into the vector are
        //   (4*2+0,4*2+1,4*2+2,4*2+3,  4*35+0,4*35+1,4*35+2,4*35+3,   4*52+0,4*52+1,4*52+2,4*52+3,  4*30+0,4*30+1, ... )
        //   4*(c1,c1,c1,c1,  c2,c2,c2,c2,  c3,c3,c3,c3,   c4,c4,c4,c4)
        // +   ( 0, 1, 2, 3,   0, 1, 2, 3,   0, 1, 2, 3,    0, 1, 2, 3
       //__m512i vecidv = _mm512_load_epi32(col_id_ta+matoffset);  // only need 1/2 registers if dealing with doubles

       //Using broadcast, I can read  c1,c2,c3,c4   c1,c2,c3,c4,  c1,c2,c3,c4,  c1,c2,c3,c4)


       // Change hint to _MM_HINT_NT?
       // (c4,c3,c2,c1,  c4,c3,c2,c1,   )
        // read 4 values, broadcast to 16
       // m0c0,m0c1,m0c2,m0c3,  m1c0,m1c1,m1c2,m2c3,  ...., m3c0,m3c1,m3c2,m3c3
       //__m512i v3_oldi = _mm512_extload_epi32(col_id_t+n, _MM_UPCONV_EPI32_NONE, _MM_BROADCAST_4X16, _MM_HINT_NONE);
       __m512i v3_oldi = read_aaaa(&col_id_t[0]+n);
       //printf("col_id: %d, %d, %d, %d\n", col_id[0], col_id[1], col_id[2], col_id[3]);
       //print_epi32(v3_oldi, "v3_oldi (col_id, load, 4x16)");
       //v3_oldi = _mm512_permutevar_epi32(vperm, v3_oldi);
       //print_epi32(v3_oldi, "a. v3_oldi");
       //print_epi32(vperm, "vperm");
       //print_epi32(v3_oldi, "permuted v3_oldi");
       //print_epi32(offsets1, "offsets1");
       //v3_oldi = _mm512_mul_epi32(v3_oldi, four);
       //v3_oldi = _mm512_add_epi32(v3_oldi, offsets);
       v3_oldi = _mm512_fmadd_epi32(v3_oldi, four, offsets);
       //print_epi32(v3_oldi, "permuted v3_oldi + offsets");
       //print_epi32(v3_oldi, "b. v3_oldi");
       __m512  v     = _mm512_i32gather_ps(v3_oldi, vec_vt, scale); // scale = 4 bytes (floats)
       //print_ps(v, "v after gather");
       //print_ps(v, "v");

                v3_old = permute(v, _MM_PERM_AAAA);
       //print_ps(v3_old, "v3_old, vec_vt");
       //print_ps(v1_old, "v1_old, data_t, before swizzle (multiply with v3_old)"); // WRONG: all values the same. 
                v2_old = _mm512_swizzle_ps(v1_old, _MM_SWIZ_REG_AAAA);
       //print_ps(v2_old, "v2_old, data_t, swizzle (multiply with v3_old)"); // WRONG: all values the same. 
                accu = _mm512_fmadd_ps(v3_old, v2_old, accu);

                v3_old = permute(v, _MM_PERM_BBBB);
                v2_old = _mm512_swizzle_ps(v1_old, _MM_SWIZ_REG_BBBB);
       //print_ps(v3_old, "v3_old");
       //print_ps(v2_old, "v2_old, data_t, swizzle (multiply with v3_old)"); // WRONG: all values the same. 
                accu = _mm512_fmadd_ps(v3_old, v2_old, accu);

                v3_old = permute(v, _MM_PERM_CCCC);
                v2_old = _mm512_swizzle_ps(v1_old, _MM_SWIZ_REG_CCCC);
       //print_ps(v3_old, "v3_old");
       //print_ps(v2_old, "v2_old, data_t, swizzle (multiply with v3_old)"); // WRONG: all values the same. 
                accu = _mm512_fmadd_ps(v3_old, v2_old, accu);

                v3_old = permute(v, _MM_PERM_DDDD);
                v2_old = _mm512_swizzle_ps(v1_old, _MM_SWIZ_REG_DDDD);
       //print_ps(v3_old, "v3_old");
       //print_ps(v2_old, "v2_old, data_t, swizzle (multiply with v3_old)"); // WRONG: all values the same. 
                accu = _mm512_fmadd_ps(v3_old, v2_old, accu);
                //accu = _mm512_setzero_ps();  generates zero norm as expected

      //printf("---- end of n=%d loop\n", n);
            }
            _mm512_storenrngo_ps(result_vt+nb_mat*nb_vec*r, accu);
        } 
}
    tm["spmv"]->end();  // time for each matrix/vector multiply
    elapsed = tm["spmv"]->getTime();
    gflops = nb_mat * nb_vec * 2.*nz*vec_v.size()*1e-9 / (1e-3*elapsed); // assumes count of 1
    printf("Gflops: %f, time: %f (ms)\n", gflops, elapsed);
   }
#endif


#if 1
    std::vector<float> one_res(nb_rows);  // single result
    for (int w=0; w < 16; w++) {
        for (int i=0; i < nb_rows; i++) {
            one_res[i] = result_vt[16*i+w];
        }
        printf("method_8, l2norm[%d]=of omp version: %f\n", w, l2norm(one_res));
    }

    // compute serial result for all 16 matrices
    std::vector<float> one_mat(nb_rows*nz);
    std::vector<float> one_vec(nb_rows);
    for (int m=0; m < 4; m++) {
        //printf("m= %d\n", m);
        int nz4 = nz / 4;
        int n_skip = 4;
        //printf("after n_skip\n");
        for (int in=0; in < n_skip; in++) {
            //printf("in= %d\n", in);
        for (int n=0; n < nz; n += n_skip) {
            int n4 = (n << 2);
        for (int r=0; r < nb_rows; r++) {
            one_mat[r+nb_rows*(n+in)] = data_t[in+n_skip*(m+nb_mat*(n4+nz4*r))];
        }}}

        int nb_vec =4;
        for (int v=0; v < nb_vec; v++) {
            for (int r=0; r < nb_rows; r++) {
                one_vec[r] = vec_vt[v+nb_vec*r];
            }
            spmv_serial_row(mat, one_mat, one_vec, result_v);
            printf("method_8, l2norm of serial version: %f\n", l2norm(result_v));
        }
        //printf("before serial row\n");
    }

    //spmv_serial(mat, vec_v, result_v);
    //printf("method_8, l2norm of serial version: %f\n", l2norm(result_v));
#endif


    _mm_free(result_vt);
    _mm_free(vec_vt);
    _mm_free(data_t);
    _mm_free(col_id_t);

}
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
template <typename T>
void ELL_OPENMP<T>::spmv_serial(ell_matrix<int, T>& mat, std::vector<T>& v, std::vector<T>& result)
{
    T accumulant;
    int vecid;
    int aligned = mat.ell_height_aligned;
    int nz = mat.ell_num;
    std::vector<int>& col_id = mat.ell_col_id;
    std::vector<T>& data = mat.ell_data;
    //printf(" mat.ell_num = %d\n", mat.ell_num);
    //printf("tot nb nonzeros: %d\n", mat.matinfo.nnz);
    //printf("nb rows: %d\n", v.size());
    //printf("aligned= %d\n", aligned);
    std::fill(result.begin(), result.end(), (T) 0.);

    for (int row=0; row < v.size(); row++) {
        //if (row >= 1) break;
        int matoffset = row;
        float accumulant = 0.;

        for (int i = 0; i < nz; i++) {
            vecid = col_id[matoffset];
            float d= data[matoffset];
            //printf("vecid= %d, d = %f\n", vecid, d);
            accumulant += data[matoffset] * v[vecid];
            //printf("col, i=%d, accu= %f,  ", i, accumulant);
            //printf("vecid= %d, d = %f\n, v= %f\n", vecid, d, v[vecid]);
            matoffset += aligned;
        }
        //printf("accumulant= %f\n", accumulant);
        result[row] = accumulant;
    }
    
    //for (int i=0; i < 10; i++) {
        //printf("serial result: %f\n", result[i]);
    //}
}
//----------------------------------------------------------------------
template <typename T>
void ELL_OPENMP<T>::spmv_serial_row(ell_matrix<int, T>& mat, std::vector<T>& v, std::vector<T>& result)
// transpose col_id from mat, and apply alternate algorithm to compute col_id (more efficient)
// results should be same as spmv_serial (which is really spmv_serial_col)
{
    T accumulant;
    int vecid;
    int aligned = mat.ell_height_aligned;
    int nz = mat.ell_num;
    std::vector<int>& col_id = mat.ell_col_id;
    std::vector<T>& data = mat.ell_data;
    int nb_rows = v.size();
#if 0
    printf(" mat.ell_num = %d\n", mat.ell_num);
    printf("tot nb nonzeros: %d\n", mat.matinfo.nnz);
    printf("nb rows: %d\n", v.size());
    printf("aligned= %d\n", aligned);
    printf("nz*nb_rows= %d\n", nz*nb_rows);
    printf("data size= %d\n", data.size());
    printf("col_id size= %d\n", col_id.size());
    mat.print();
#endif

    std::fill(result.begin(), result.end(), (T) 0.);

    std::vector<int> col_id_t(col_id.size());
    std::vector<T> data_t(data.size());

    // Transpose rows and columns
    for (int row=0; row < nb_rows; row++) {
        for (int n=0; n < nz; n++) {
            //col_id_t[row+n*nb_rows] = col_id[n+nz*row];
            //data_t[row+n*nb_rows] = data[n+nz*row];
            //ct[nrows][nz], c[nz][nrows]
            // ct[nz][n] = c[n][nz]
            col_id_t[n+nz*row] = col_id[row+n*nb_rows];
              data_t[n+nz*row]   = data[row+n*nb_rows];
        }
    }
    
    for (int row=0; row < nb_rows; row++) {
        //if (row >= 1) break;
        float accumulant = 0.;
        for (int i=0; i < nz; i++) {
            int matoffset = row*nz+i;
            int vecid = col_id_t[matoffset];
            float   d = data_t[matoffset];
            accumulant += d * v[vecid];  // most efficient, 12 Gflops
            //printf("row, i=%d, accu= %f,  ", i, accumulant);
            //printf("vecid= %d, d = %f\n, v= %f\n", vecid, d, v[vecid]);
        }
        //printf("accumulant= %f\n", accumulant);
        result[row] = accumulant;
    }
}
//----------------------------------------------------------------------
template <typename T>
void ELL_OPENMP<T>::spmv_serial_row(ell_matrix<int, T>& mat, std::vector<T>& data, std::vector<T>& v, std::vector<T>& result)
// transpose col_id from mat, and apply alternate algorithm to compute col_id (more efficient)
// results should be same as spmv_serial (which is really spmv_serial_col)
{
    T accumulant;
    int vecid;
    int aligned = mat.ell_height_aligned;
    int nz = mat.ell_num;
    std::vector<int>& col_id = mat.ell_col_id;
    //std::vector<T>& data = mat.ell_data;
    int nb_rows = v.size();
#if 0
    printf(" mat.ell_num = %d\n", mat.ell_num);
    printf("tot nb nonzeros: %d\n", mat.matinfo.nnz);
    printf("nb rows: %d\n", v.size());
    printf("aligned= %d\n", aligned);
    printf("nz*nb_rows= %d\n", nz*nb_rows);
    printf("data size= %d\n", data.size());
    printf("col_id size= %d\n", col_id.size());
    mat.print();
#endif

    std::fill(result.begin(), result.end(), (T) 0.);

    std::vector<int> col_id_t(col_id.size());
    std::vector<T> data_t(data.size());

    // Transpose rows and columns
    for (int row=0; row < nb_rows; row++) {
        for (int n=0; n < nz; n++) {
            //col_id_t[row+n*nb_rows] = col_id[n+nz*row];
            //data_t[row+n*nb_rows] = data[n+nz*row];
            //ct[nrows][nz], c[nz][nrows]
            // ct[nz][n] = c[n][nz]
            col_id_t[n+nz*row] = col_id[row+n*nb_rows];
              data_t[n+nz*row]   = data[row+n*nb_rows];
        }
    }
    
    for (int row=0; row < nb_rows; row++) {
        //if (row >= 1) break;
        float accumulant = 0.;
        for (int i=0; i < nz; i++) {
            int matoffset = row*nz+i;
            int vecid = col_id_t[matoffset];
            float   d = data_t[matoffset];
            accumulant += d * v[vecid];  // most efficient, 12 Gflops
            //printf("row, i=%d, accu= %f,  ", i, accumulant);
            //printf("vecid= %d, d = %f\n, v= %f\n", vecid, d, v[vecid]);
        }
        //printf("accumulant= %f\n", accumulant);
        result[row] = accumulant;
    }
}
//----------------------------------------------------------------------
//----------------------------------------------------------------------
template <typename T>
T ELL_OPENMP<T>::l2norm(std::vector<T>& v)
{
    T norm = (T) 0.;
    for (int i=0; i < v.size(); i++) {
            norm += v[i]*v[i];
    }
    return (T) sqrt(norm);
}
//----------------------------------------------------------------------
template <typename T>
T ELL_OPENMP<T>::l2norm(T* v, int n)
{
    T norm = (T) 0.;
    for (int i=0; i < n; i++) {
            norm += v[i]*v[i];
    }
    return (T) sqrt(norm);
}
//----------------------------------------------------------------------
template <typename T>
void ELL_OPENMP<T>::fill_random(ell_matrix<int, T>& mat, std::vector<T>& v)
{
    for (int i=0; i < v.size(); i++) {
            v[i] = (T) getRandf();  
            v[i] = 1.0;   // Did not work
    }

    for (int i=0; i < mat.ell_col_id.size(); i++) {
            mat.ell_data[i] = getRandf(); // problem is with random matrices
            mat.ell_data[i] = 1.0;  // worked
    }

#if 0
    for (int i=0; i < 32; i++) {
    for (int j=0; j < 32; j++) {
        mat.ell_data[i+32*j] = (float) j; // worked
        mat.ell_data[i+32*j] = (float) i; // get a zero product. NOT POSSIBLE.
    }}
#endif
    /**
     *   (0 0 0 0) 1
     *   (1 1 1 1) 1
     *   (2 2 2 2) 1  = 32*(0,1,2,3,...,31)
     *   (3 3 3 3) 1
     *
     *   (0 1 2 3) 1
     *   (0 1 2 3) 1
     *   (0 1 2 3) 1  = (0+1+2+3+...+31)*(1,1,...,1) = 
     *   (0 1 2 3) 1
    **/
}
//----------------------------------------------------------------------
template <typename T>
__m512 ELL_OPENMP<T>::tensor_product(float* a, float* b)
{
        __m512 va = read_aaaa(a);
        __m512 vb = read_abcd(b);
        return _mm512_mul_ps(va, vb);
}
//----------------------------------------------------------------------
template <typename T>
__m512 ELL_OPENMP<T>::permute(__m512 v1, _MM_PERM_ENUM perm)
//  Read 4 floats and copy them to four other lanes
//  trick: __m512i _mm512_castps_si512(__m512 IN)
//  Use _mm512_shuffle_epi32(__m512i v2, _MM_PERM_ENUM permute)
//  permute =  255  (all lanes the smae
{
    __m512i vi = _mm512_castps_si512(v1);
    //vi = _mm512_shuffle_epi32(vi, _MM_PERM_AAAA);
    vi = _mm512_permute4f128_epi32(vi, perm);
    // shuffle is like a swizzle
    v1 = _mm512_castsi512_ps(vi);
    return v1;
}
//----------------------------------------------------------------------
template <typename T>
__m512i ELL_OPENMP<T>::read_aaaa(int* a)
{
    // only works with ints
    // read in 4 ints (a,b,c,d) and create the 
    // 16-float vector dddd,cccc,bbbb,aaaa  (a is least significant)

    int int_mask_lo = (1 << 0) + (1 << 4) + (1 << 8) + (1 << 12);
    //int int_mask_lo = (1 << 0) + (1 << 1) + (1 << 2) + (1 << 3);
    __mmask16 mask_lo = _mm512_int2mask(int_mask_lo);
    __m512i v1_old;
    v1_old = _mm512_setzero_epi32();
    v1_old = _mm512_mask_loadunpacklo_epi32(v1_old, mask_lo, a);
    v1_old = _mm512_mask_loadunpackhi_epi32(v1_old, mask_lo, a);
    v1_old = _mm512_swizzle_epi32(v1_old, _MM_SWIZ_REG_AAAA);
    return v1_old;
}
//----------------------------------------------------------------------
template <typename T>
__m512 ELL_OPENMP<T>::read_aaaa(float* a)
{
    // only works with floats
    // read in 4 floats (a,b,c,d) and create the 
    // 16-float vector dddd,cccc,bbbb,aaaa

    int int_mask_lo = (1 << 0) + (1 << 4) + (1 << 8) + (1 << 12);
    __mmask16 mask_lo = _mm512_int2mask(int_mask_lo);
    __m512 v1_old;
    v1_old = _mm512_setzero_ps();
    v1_old = _mm512_mask_loadunpacklo_ps(v1_old, mask_lo, a);
    v1_old = _mm512_mask_loadunpackhi_ps(v1_old, mask_lo, a);
    v1_old = _mm512_swizzle_ps(v1_old, _MM_SWIZ_REG_AAAA);
    return v1_old;
}
//----------------------------------------------------------------------
template <typename T>
void ELL_OPENMP<T>::print_f(float* res, const std::string msg)
{
    printf("--- %s ---\n", msg.c_str());
    for (int i=0; i < 8; i++) { printf("(%d,%f), ", i, res[i]); }
    printf("\n");
    for (int i=8; i < 16; i++) { printf("(%d,%f), ", i, res[i]); }
    printf("\n\n");
}
//----------------------------------------------------------------------
template <typename T>
void ELL_OPENMP<T>::print_i(int* res, const std::string msg)
{
    printf("--- %s ---\n", msg.c_str());
    for (int i=0; i < 8; i++) { printf("(%d,%d), ", i, res[i]); }
    printf("\n");
    for (int i=8; i < 16; i++) { printf("(%d,%d), ", i, res[i]); }
    printf("\n\n");
}
//----------------------------------------------------------------------
template <typename T>
void ELL_OPENMP<T>::print_ps(const __m512 v1, const std::string msg)
{
    return;
    float* res = (float*) _mm_malloc(32*sizeof(float), 64);
    _mm512_store_ps(res, v1);
    printf("--- %s ---\n", msg.c_str());
    for (int i=0; i < 8; i++) { printf("(%d,%f), ", i, res[i]); }
    printf("\n");
    for (int i=8; i < 16; i++) { printf("(%d,%f), ", i, res[i]); }
    printf("\n\n");
    _mm_free(res);
}
//----------------------------------------------------------------------
template <typename T>
void ELL_OPENMP<T>::print_epi32(const __m512i v1, const std::string msg)
{
    return;
    int* res = (int*) _mm_malloc(32*sizeof(int), 64);
    _mm512_store_epi32(res, v1);
    printf("--- %s ---\n", msg.c_str());
    for (int i=0; i < 8; i++) { printf("(%d,%d), ", i, res[i]); }
    printf("\n");
    for (int i=8; i < 16; i++) { printf("(%d,%d), ", i, res[i]); }
    printf("\n\n");
    _mm_free(res);
}
//----------------------------------------------------------------------
template <typename T>
__m512 ELL_OPENMP<T>::read_abcd(float* a)
{
    // read in 4 floats (a,b,c,d) and create the 
    // 16-float vector dcba,dcba,dcba,dcba

    __m512 v1_old = _mm512_extload_ps(a, _MM_UPCONV_PS_NONE, _MM_BROADCAST_4X16, _MM_HINT_NONE);
    return v1_old;
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
//----------------------------------------------------------------------
