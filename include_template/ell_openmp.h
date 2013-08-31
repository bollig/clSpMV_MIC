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
#include "projectsettings.h"

#include <omp.h>

//#include "oclcommon.h"
#include "openmp_base.h"

#include "timer_eb.h"
#include "timestamp.hpp"
#include "rbffd_io.h"
#include "runs.h"
//#include "rcm.h"
#include "burkardt_rcm.hpp"

#include "reorder.h"
#include "vcl_bandwidth_reduction.h"

void genrcmi(const int n, const int flags,
            const int *xadj, const int *adj,
            int *perm, signed char *mask, int *deg);


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

    // used for multidomain case
    std::vector<int> Qbeg;
    std::vector<int> Qend;
    std::vector<int> beg_row; // in col_id (for each subdomain)

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

    RunData rd;

    double overallopttime;
	std::vector<T> paddedvec_v;
	std::vector<T> vec_v;
	std::vector<T> result_v;
    std::vector<int> col_id;

    // read fro data file test.conf
    std::string sparsity;
    int diag_sep;
    int inner_bandwidth;
    enum stencilType {COMPACT=0, RANDOM, RANDOMWITHDIAG, RANDOMDIAGS, SUPERCOMPACT, NONE};
    enum nonzeroStats {UNIFORM=0, NORMAL};
    stencilType stencil_type;

    // Set to 1 in order to sort the column indices (smallest to largest) when using 
    // random elements (read from .conf file)
    int sort_col_indices;

    USE(dim2); // relates to workgroups

    //USE(vec);
    //USE(result);
    USE(coores);

	USE(getKernelName);
	//USECL(loadKernel);
	//USECL(enqueueKernel);

    ell_matrix<int, T> mat;

    // variables for spmv and checking results. Will contain composite arrays formed for multiple
    // matrices and multiple vectors
    float* vec_vt;
    float* result_vt;
    int* col_id_t;
    float* data_t;

    class Subdomain {
    public:
        float* vec_vt;
        float* result_vt;
        int* col_id_t;
        float* data_t;
        std::vector<int> Qbeg;
        std::vector<int> Qend;
    };
    std::vector<Subdomain> subdomains;
    int nb_subdomains; 
    std::vector<int> nb_rows_multi;
    std::vector<int> nb_vec_elem_multi;
    std::vector<int> offsets_multi;

public:
	ELL_OPENMP(std::string filename, int ntimes);
    // multi=1: multidomain input file (third argument is not used)
	ELL_OPENMP(std::string filename, int ntimes, int multi);
	ELL_OPENMP(coo_matrix<int, T>* mat, int dim2Size, int ntimes);
	ELL_OPENMP(std::vector<int>& col_id, int nb_rows, int nb_nonzeros_per_row);
	~ELL_OPENMP<T>() {
	}

    void spmv_serial(ell_matrix<int, T>& mat, std::vector<T>& v, std::vector<T>& result);
    void spmv_serial_row(ell_matrix<int, T>& mat, std::vector<T>& v, std::vector<T>& result);
    void spmv_serial_row(ell_matrix<int, T>& mat, std::vector<T>& data, std::vector<T>& v, std::vector<T>& result);
    void spmv_serial_row(int* col_id, T* data, T* v, T* res, int nbz, int nb_rows) ;
    void fill_random(ell_matrix<int, T>& mat, std::vector<T>& v);
    T l2norm(std::vector<T>& v);
    T l2norm(T*, int n);

	virtual void run();
	void method_3(int nb=0);
	void method_4(int nb=0);
	void method_5(int nb=0);
	void method_6(int nb=0);
	void method_7(int nb=0); // 4 matrices, 4 vectors (correct results)
    void method_rd_wr(int nbit);
    // read/write benchmarks. Memory is local to each thread. 
    void method_rd_wr_local_thread(int nbit);
	void method_7a(int nb=0); // 4 matrices, 4 vectors, using new matrix generators
	//void method_8(int nb=0); // 4 matrices, 4 vectors
	void method_8a(int nb=0); // 4 matrices, 4 vectors
    void method_8a_multi(int nb=0);

    inline __m512 permute(__m512 v1, _MM_PERM_ENUM perm);
    inline __m512 read_aaaa(float* a);
    inline __m512i read_aaaa(int* a);
    inline __m512 read_abcd(float* a);
    inline __m512 tensor_product(float* a, float* b);
    void print_f(float* res, const std::string msg="");
    void print_i(int* res, const std::string msg="");
    void print_ps(const __m512 v1, const std::string msg="");
    void print_epi32(const __m512i v1, const std::string msg="");
    void generate_ell_matrix_by_row(std::vector<int>& col_id, std::vector<T>& data, int nb_elem);
    void generate_ell_matrix_data(T* data, int nbz, int nb_rows, int nb_mat);
    //void generate_ell_matrix_data(std::vector<T> data, int nbz, int nb_rows, int nb_mat);
    void generate_col_id(int* col_id, int nbz, int nb_rows);
    void generate_col_id_multi(int* col_id, int nbz, int nb_rows);
    void generate_vector(T* vec, int nb_rows, int nb_vec);
    //void generate_vector(std::vector<T>& vec, int nb_rows, int nb_vec);
    //void retrieve_vector(std::vector<T>& vec, std::vector<T>& retrieved, int vec_id, int nb_vec);
    void retrieve_vector(T* vec, T* retrieved, int vec_id, int nb_vec, int nb_rows);
    //void retrieve_data(std::vector<T>& data_in, std::vector<T>& data_out, int mat_id, int nbz, int nb_rows, int nb_mat);
    void retrieve_data(T* data_in, T* data_out, int mat_id, int nbz, int nb_rows, int nb_mat);
    void checkAllSerialSolutions(T* data_t, int* col_id_t, T* vec_vt, T* one_res, int nz, int nb_mat, int nb_vec, int nb_rows);
    void generateInputMatricesAndVectors();
    void generateInputMatricesAndVectorsMulti();
    void freeInputMatricesAndVectors();
    void freeInputMatricesAndVectorsMulti();
    void checkSolutions();
    void bandwidth(int* col_id, int nb_rows, int stencil_size, int vec_size);
    void bandwidthQ(Subdomain& s, int nb_rows, int stencil_size, int vec_size);
    void permInverse3(int n, int* perm, int* perm_inv);
    int adj_bandwidth(int node_num, int adj_num, int* adj_row, int* adj);
    int adj_perm_bandwidth(int node_num, int adj_num, int* adj_row, int* adj, int* perm, int* perm_inv);
    int i4_max (int i1, int i2);
    //void freeInputMatricesAndVectors(float* vec_vt, float* result_vt, int* col_id_t, float* data_t);
    //void generateInputMatricesAndVectors(float* vec_vt, float* result_vt, int* col_id_t, float* data_t);
    void cuthillMcKee(std::vector<int>& col_id, int nb_rows, int stencil_size);

protected:
	virtual void method_0(int nb=0);
	virtual void method_1(int nb=0);
	virtual void method_2(int nb=0);
};

//----------------------------------------------------------------------
template <typename T>
void ELL_OPENMP<T>::run()
{
    int num_threads[] = {1,2,4,8,16,32,64,96,128,160,192,224,244};
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

#if 1
    rd.nb_rows = rd.n3d*rd.n3d*rd.n3d;
    printf("rd.nb_rows: %d\n", rd.nb_rows);
    int save_nb_threads = omp_get_num_threads();
    //for (int i=0; i < 13; i++) {
    for (int i=12; i < 13; i++) {
        //num_threads[i] = 1;// TEMPORARY
        omp_set_num_threads(num_threads[i]);
        printf("rd/wr, nb threads: %d\n", num_threads[i]);
	    method_rd_wr(4); // correct results
    }
    //for (int i=0; i < 13; i++) {
    for (int i=12; i < 13; i++) {
        omp_set_num_threads(num_threads[i]);
        method_rd_wr_local_thread(4);
    }
    exit(0);
#endif

	//method_7(4); // correct results
    //exit(0);
	//method_7a(4); // Something wrong with random matrices

    int count = 0;
    int max_nb_runs = 1;

#if 0
    while (1) {
        rd.nb_rows = rd.n3d*rd.n3d*rd.n3d;

	    method_8a(4);
        count++;

        printf("update inner bandwidth: %d\n", rd.inner_bandwidth);
        if (rd.inner_bandwidth >= 233000) break;
        if (rd.diag_sep >= 200000) break;
        if (rd.n3d > 80) break; // maximum size grid to experiment with. 
        if (rd.sort_col_indices > 1) break;
        if (count >= max_nb_runs) break;
    }
#endif

    if (rd.use_subdomains == 0 || nb_subdomains == 1) {
        //method_8a(4);
        method_8a_multi(4); // temporary? 
    } else {
        method_8a_multi(4);
    }
    exit(0);

    if (nb_subdomains > 0) {
        method_8a_multi(4);
    } else {
        method_8a(4);
    }

    exit(0);
}
//----------------------------------------------------------------------
template <typename T>
ELL_OPENMP<T>::ELL_OPENMP(std::vector<int>& col_id, int nb_rows, int nb_nonzeros_per_row) :
            OPENMP_BASE<T>(col_id, nb_rows, nb_nonzeros_per_row)
            // MUST DEBUG THE BASE CODE
{
}
//----------------------------------------------------------------------
template <typename T>
ELL_OPENMP<T>::ELL_OPENMP(coo_matrix<int, T>* coo_mat, int dim2Size, int ntimes) : 
   OPENMP_BASE<T>(coo_mat, dim2Size, ntimes)
{
    ;
}
//----------------------------------------------------------------------
template <typename T>
ELL_OPENMP<T>::ELL_OPENMP(std::string filename, int ntimes) : 
   OPENMP_BASE<T>(ntimes)
{
    int offset = 3;
    tm["spmv"] = new EB::Timer("[method_0] Matrix-Vector multiply", offset);

    // Read relevant parameters
	ProjectSettings pj("test.conf");
	std::string asci_binary = REQUIRED<std::string>("asci_binary");
    std::string sparsity = REQUIRED<std::string>("sparsity");
    diag_sep = REQUIRED<int>("diag_sep");
    inner_bandwidth = REQUIRED<int>("inner_bandwidth");
    sort_col_indices = REQUIRED<int>("sort_col_indices");
    rd.n3d = REQUIRED<int>("n3d");
    rd.random_seed = REQUIRED<int>("random_seed");
    if (rd.random_seed == 1) setSeed();
    rd.use_subdomains = REQUIRED<int>("use_subdomains");

    printf("sparsity = %s\n", sparsity.c_str());
    if (sparsity == "NONE")  stencil_type = NONE;
    else if (sparsity == "COMPACT")  stencil_type = COMPACT;
    else if (sparsity == "SUPERCOMPACT") stencil_type = SUPERCOMPACT;
    else if (sparsity == "RANDOM") stencil_type = RANDOM;
    else if (sparsity == "RANDOM_WITH_DIAG") stencil_type = RANDOMWITHDIAG;
    else if (sparsity == "RANDOM_DIAGS") stencil_type = RANDOMDIAGS;
    else { printf("Unknown sparsity\n"); exit(1); }

    std::string in_format = REQUIRED<std::string>("in_format");

    coo_matrix<int, float> cmat;
    coo_matrix<int, double> cmat_d;
	RBFFD_IO<float> io;
	std::vector<int> rows, cols;
	std::vector<float> values;
	int width, height;
    std::vector<int> col_id;
    int nb_rows;
    int stencil_size;
    nb_subdomains = -1; // for all formats execept ELL_MULTI

    if (in_format == "MM") {
        rd.use_subdomains = 0;
        nb_subdomains = 1;
        init_coo_matrix(cmat);
        init_coo_matrix(cmat_d);
        if (asci_binary == "asci" || asci_binary == "ascii") {
            printf("*** load ASCI FILE %s ***\n", filename.c_str());
            io.loadFromAsciMMFile(rows, cols, values, width, height, filename);
        } else if (asci_binary == "binary") {
            printf("*** load BINARY FILE %s ***\n", filename.c_str());
            io.loadFromBinaryMMFile(rows, cols, values, width, height, filename);
        } else {
            printf("*** unknown read format type (asci/binary)\n");
        }

        cmat.matinfo.height = height;
        cmat.matinfo.width = width;
        cmat.matinfo.nnz = rows.size();
        cmat_d.matinfo.height = height;
        cmat_d.matinfo.width = width;
        cmat_d.matinfo.nnz = rows.size();
        cmat.coo_row_id = rows; // copy is expensive
        cmat.coo_col_id = cols;
        cmat.coo_data = values;
        cmat_d.coo_row_id = rows; // copy is expensive
        cmat_d.coo_col_id = cols;
        cmat_d.coo_data.resize(values.size());
        for (int i=0; i < values.size(); i++)  {
            cmat_d.coo_data[i] = (double) values[i];
        }
	    printf("rows.size: %d\n", rows.size());
	    printf("READ INPUT FILE: \n");
	    cmat.print();
        printMatInfo_T(&cmat);
        //Initialize values
        coo2ell<int, T>(&cmat, &mat, GPU_ALIGNMENT, 0);
        aligned_length = mat.ell_height_aligned;
        nnz = mat.matinfo.nnz;
        rownum = mat.matinfo.height;
        vecsize = mat.matinfo.width;
        ellnum = mat.ell_num;
        stencil_size = ellnum;
        vec_v.resize(cmat.matinfo.width);
        result_v.resize(cmat.matinfo.height);
    } else if (in_format == "ELL") {
        nb_rows_multi.resize(1);
        nb_vec_elem_multi.resize(1);
        rd.use_subdomains = 0;
        subdomains.resize(1);
        nb_subdomains = 1;
        //mat.col_id.resize(nb_rows*stencil_size); // perhaps allocate inside load routine?
        io.loadFromBinaryEllpackFile(mat.ell_col_id, nb_rows, stencil_size, filename);
        vec_v.resize(nb_rows);
        result_v.resize(nb_rows); // might have to change if height aligned is wrong. 
        mat.matinfo.height = nb_rows;
        mat.matinfo.width = nb_rows;
        mat.matinfo.nnz = nb_rows * stencil_size;
        // if divisible by 32, else next largest multiple of 32. MUST IMPLEMENT GE. 
        mat.ell_height_aligned = nb_rows;
        mat.ell_num = stencil_size;
	    //spmv::spmv_ell_openmp(col_id, nb_rows, stencil_size); // choice == 1
        printf("\nRead ELLPACK file\n");
        //printf("col_id size: %d\n", mat.ell_col_id.size());
        nnz = stencil_size * nb_rows; // nnz not used except in print
        nb_rows_multi[0] = nb_rows;
        nb_vec_elem_multi[0] = nb_rows;
        offsets_multi.resize(2);
        offsets_multi[0] = 0;
        offsets_multi[1] = nb_rows_multi[0]*stencil_size;
        //printf("cuthill from ELL input format\n");
        //cuthillMcKer(mat.ell_col_id, nb_rows,stencil_size);
    } else if (in_format == "ELL_MULTI") {
        // need variables mat_multi, vec_v_multi, result_v_multi
        // stencil size is identical for all subdomains
        // all arguments by reference 
        printf("before loadFromBinaryEllpackFileMulti\n");
        io.loadFromBinaryEllpackFileMulti(nb_subdomains, mat.ell_col_id, nb_rows_multi, 
            nb_vec_elem_multi, offsets_multi, Qbeg, Qend, stencil_size, filename);
        printf("*** after load multi, print col_id\n");
        printf("sizeof Qbeg, Qend: %d, %d\n", Qbeg.size(), Qend.size());
        //for (int i=0; i < 20; i++) {
            //printf("col_id[%d]= %d\n", i, mat.ell_col_id[i]);
            //printf("Qbeg/Qend= %d, %d\n", Qbeg[i], Qend[i]);
            // Qbeg[i] and Qend[i] are equal!!! NOT POSSIBLE (refernece error somewere?)
        //}
        //exit(0);
        rd.use_subdomains = 1; // IGNORE INPUT FROM TEST.CONF
        subdomains.resize(nb_subdomains);
        nb_rows = mat.ell_col_id.size() / stencil_size;
        printf("top, nb_rows= %d\n", nb_rows);
        beg_row.resize(nb_subdomains+1);
        beg_row[0] = 0;
        for (int i=0; i < nb_subdomains; i++) {
            beg_row[i+1] += nb_rows_multi[i];
        }
        for (int n=0; n < nb_subdomains; n++) {
            Subdomain& s = subdomains[n];
            s.Qbeg.resize(nb_rows_multi[n]);
            s.Qend.resize(nb_rows_multi[n]);
            for (int i=0; i < nb_rows_multi[n]; i++) {
                s.Qbeg[i] = Qbeg[beg_row[n]+i];
                s.Qend[i] = Qend[beg_row[n]+i];
            }
        }
        for (int i=0; i < nb_subdomains; i++) {
                printf("nb_rows_multi[%d] = %d\n", i, nb_rows_multi[i]);
                printf("nb_vec_elem_multi[%d] = %d\n", i, nb_vec_elem_multi[i]);
        }



#if 0
void GENRCM(const INT n, const int flags,
            const INT *xadj, const INT *adj,
            INT *perm, signed char *mask, INT *deg)
#endif

#if 0
    class Subdomain {
    public:
        float* vec_vt;
        float* result_vt;
        int* col_id_t;
        float* data_t;
    };
#endif
    } else {
        printf("input format not supported. Must be MM or ELL\n");
        exit(1);
    }

	std::fill(vec_v.begin(), vec_v.end(), 1.);
	std::fill(result_v.begin(), result_v.end(), 0.);

    rd.nb_rows = nb_rows;
    rd.nb_mats = 4;
    rd.nb_vecs = 4;
    rd.stencil_size = stencil_size;
    rd.diag_sep = diag_sep;
    rd.inner_bandwidth = inner_bandwidth;
    rd.sort_col_indices = sort_col_indices;
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
    //for (int it=0; it < 10; it++) { // }
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
void ELL_OPENMP<T>::method_rd_wr_local_thread(int nbit)
{
	printf("\n============== METHOD RD/WR LOCAL THREAD ===================\n");
    double tot_bw = 0.;
    int nb_threads = omp_get_max_threads();
    std::vector<double> tim(nb_threads);
    std::vector<double> bw(244); // 244 = max nb threads
    //const int size = 8*128*128; // seg fault if size is 4*128*128 (WHY WOULD THAT BE?)
    // bandwidth goes up for 4*128*128 size bytes. WHY? This equals 262,144 bytes which is 50% 
    // of cache size. THIS MUST BE RELEVANT. BUT DO NOT KNOW WHY. 
    //const int size = 128*128; // I get up to 619 Gbytes/sec aggregate. IMPOSSIBLE
    const int size = 16*128*128;
    util::timestamp* timeof = new util::timestamp[nb_threads];

#pragma omp parallel firstprivate(size)
  {
      // assign memory on each thread
      //EB::Timer tm("tid");
      int* buf_orig = (int*) _mm_malloc(sizeof(int)*size, 64);
      int* buf_dest = (int*) _mm_malloc(sizeof(int)*size, 64);
      int* reg_fill = (int*) _mm_malloc(sizeof(int)*1024*128, 64); // size of 512 cache
      int tid = omp_get_thread_num();

      // initialize these arrays before benchmarks to improve locality (relates to ccNuma). I do 
      // not fully understand. 
      // Source: http://www.multicore-challenge.org/resources/FMCCII_Tutorial_Treibig.pdf
      //  page 87. 
      
#if 1
// Writing to them first ade the benchmarks realistic (I do not understand the reasons)
      for (int i=0; i < size; i++) {
            buf_orig[i] = 0.0;
            buf_dest[i] = 0.0;
      }
#endif

#pragma omp barrier
#if 0
    // ensure that registers are full so there will be cache misses
    for (int r=0; r < 1024*128; r += 16) {
        const __m512 v1_old = _mm512_load_ps(reg_fill + r);
        _mm512_storenrngo_ps(buf_dest+r, v1_old);
    }
    // The above loop segfaults if size if 128*128, but NOT if size = 32*128*128. 
    // NOT LOGICAL. DO NOT UNDERSTAND
      printf("1 nbthread= %d\n", nb_threads);
#endif

#pragma omp barrier
#if 0
      //printf("size= %d\n", size);
      for (int d=0; d < 10; d++) {
        //tm.start();
	    util::timestamp beg;
        for (int r=0; r  < size; r += 16) {
            //printf("r= %d\n", r);
       	   //_mm_prefetch ((const char*) vec_vt+r+2*4096, _MM_HINT_T1);
            __m512 v1_old = _mm512_load_ps(buf_orig + r);
            //v1_old = _mm512_mul_ps(v1_old, v1_old);
            //_mm512_store_ps(buf_dest + r, v1_old);
            _mm512_storenrngo_ps(buf_dest+r, v1_old);
        }
        //tm.end();
	    util::timestamp end;
        if (d == 6) {
            //tim[tid] = tm.getTime();
	        tim[tid] = (end-beg) * 1000; // in ms
        }
      }
#endif


      for (int d=0; d < 5; d++) {
        for (int r=0; r  < size; r += 16) {
            __m512 v1_old = _mm512_load_ps(buf_orig + r);
            _mm512_storenrngo_ps(buf_dest+r, v1_old);
        }
      }

    int repeat = 1000;
#pragma omp barrier
	  util::timestamp beg;
      for (int d=0; d < repeat; d++) {
        for (int r=0; r  < size; r += 16) {
            __m512 v1_old = _mm512_load_ps(buf_orig + r);
            _mm512_storenrngo_ps(buf_dest+r, v1_old);
        }
       }
#pragma omp barrier
	   util::timestamp end;
	   tim[tid] = (1./repeat) * (end-beg) * 1000; // in ms  (time per sizeo(int)*size bytes per thread)

      _mm_free(buf_orig);
      _mm_free(buf_dest);
  }
  float mean = 0.;
  float mx = 0.;
  float mn = 1.e6;
  printf("size= %f Mbytes\n", sizeof(int) * size/1000000.);
  for (int i=0; i < nb_threads; i++) {
    bw[i] = 2*sizeof(int)*size *1.e-9/ (tim[i]*1.e-3); // rd + wr (Gbytes/sec) 
    //bw[i] = sizeof(int)*size *1.e-9/ (tim[i]*1.e-3); // rd + wr (Gbytes/sec)  // rd
    tot_bw += bw[i];
    mean += tim[i];
    mx = tim[i] > mx ? tim[i] : mx;
    mn = tim[i] < mn ? tim[i] : mn;
  }
  mean /= nb_threads;
  printf("nb threads: %d, tot_bw= %f Gbytes/sec, avg/max/min time per thread: %5.2f/%5.2f/%5.2f (ms)\n", nb_threads, tot_bw, mean, mx, mn);
}
//----------------------------------------------------------------------
template <typename T>
void ELL_OPENMP<T>::method_rd_wr(int nbit)
{
	printf("============== METHOD RD/WR ===================\n");
    printf("Implement streaming\n");

    float gflops;
    float max_gflops = 0.;
    float max_bandwidth = 0.;
    float elapsed = 0.; 
    float min_elapsed = 0.; 
    int nb_rows = rd.nb_rows;
    int nz = rd.stencil_size;

    // make array large enough
    //nb_rows = 128*128*128*64; // too large? 134 Mrows
    nb_rows = 128*128*128*16; // too large? 134 Mrows
    //nb_rows = 128;
    //nb_rows = 128*128*128; // too large? 134 Mrows
    printf("nb_rows= %d\n", nb_rows);
    vec_vt = (float*) _mm_malloc(sizeof(float)*nb_rows, 64);
    result_vt = (float*) _mm_malloc(sizeof(float)*nb_rows, 64);
    printf("*vec_vt= %ld\n", (long) vec_vt);
    col_id_t = (int*) _mm_malloc(sizeof(int)*nb_rows, 64);
    printf("col_id_t= %d\n", col_id_t);
    printf("after col_id_t allocated\n");

//#define COLCOMPACT
//#define COLREVERSE
#define COLRANDOM

#ifdef COLCOMPACT
    for (int i=0; i < nb_rows; i++) {
        col_id_t[i] = i;
    }
    printf("after col_id definition\n");
#endif
#ifdef COLREVERSE
    for (int i=0; i < nb_rows; i++) {
        col_id_t[i] = nb_rows-i-1;
    }
#endif
#ifdef COLRANDOM
    // Cache line is 16 floats, so this is a worst case. 
    for (int i=0; i < nb_rows; i+=16) {
        col_id_t[i]   = (i+16*0);
        col_id_t[i+1] = (i+16*1) % nb_rows;
        col_id_t[i+2] = (i+16*2) % nb_rows;
        col_id_t[i+3] = (i+16*3) % nb_rows;
        col_id_t[i+4] = (i+16*4) % nb_rows;
        col_id_t[i+5] = (i+16*5) % nb_rows;
        col_id_t[i+6] = (i+16*6) % nb_rows;
        col_id_t[i+7] = (i+16*7) % nb_rows;
        col_id_t[i+8] = (i+16*8) % nb_rows;
        col_id_t[i+9] = (i+16*9) % nb_rows;
        col_id_t[i+10] = (i+16*10) % nb_rows;
        col_id_t[i+11] = (i+16*11) % nb_rows;
        col_id_t[i+12] = (i+16*12) % nb_rows;
        col_id_t[i+13] = (i+16*13) % nb_rows;
        col_id_t[i+14] = (i+16*14) % nb_rows;
        col_id_t[i+15] = (i+16*15) % nb_rows;
    }
#endif
 
//#define BANDRDWR
#define BANDRD
#define BANDWR
//#define BANDCPP
//#define BANDCPPGATHER
//#define BANDUNPACK
//#define BANDGATHER

    // Time pure loads
    for (int it=0; it < 10; it++) {
        tm["spmv"]->start();
#pragma omp parallel
{


//-------------------------------------------------
#ifdef BANDRD
__m512 sum = _mm512_setzero_ps();

#pragma omp for
       for (int i=0; i  < nb_rows; i += 16) {
            sum = _mm512_add_ps(_mm512_load_ps(vec_vt+i), sum);
            //_mm512_storenrngo_ps(result_vt+i, sum);
        }
        _mm512_storenrngo_ps(result_vt, sum);
#endif  // BANDRD
//-------------------------------------------------
#ifdef BANDWR
__m512 sumwr = _mm512_setzero_ps();

#pragma omp for
       for (int i=0; i  < nb_rows; i += 16) {
            //sum = _mm512_add_ps(_mm512_load_ps(vec_vt+i), sum);
            _mm512_storenrngo_ps(result_vt+i, sumwr);
        }
#endif  // BANDWR
//-------------------------------------------------
#ifdef BANDRDWR
#pragma omp for
//#pragma noprefetch
        // only nb_rows*sizeof(float) bytes transferred
       for (int i=0; i  < nb_rows; i += 16) {
       	    //_mm_prefetch ((const char*) vec_vt+r+2048, _MM_HINT_T1); // slows down
            const __m512 v1_old = _mm512_load_ps(vec_vt + i);
            //_mm512_store_ps(result_vt + r, v1_old);
            _mm512_storenrngo_ps(result_vt+i, v1_old);
            //_mm512_store_ps(result_vt + r, _mm512_load_ps(vec_vt+r));
        }
#endif  // BANDRDWR
//-------------------------------------------------
#ifdef BANDUNPACK
       __m512 v3_old;
#pragma omp for
        for (int r=0; r < nb_rows; r+=16) {
            // retrieve 4 elements per read_aaaa, expand to 16 elements via masking
            v3_old = read_aaaa(vec_vt+r);
            // use addition to ensure that dead code is not eliminated
            v3_old = _mm512_add_ps(read_aaaa(vec_vt+r+4), v3_old);
            v3_old = _mm512_add_ps(read_aaaa(vec_vt+r+8), v3_old);
            v3_old = _mm512_add_ps(read_aaaa(vec_vt+r+12), v3_old);
            _mm512_storenrngo_ps(result_vt+r, v3_old);
       }
#endif  // BANDUNPACK
//-------------------------------------------------
#ifdef BANDCPP
#pragma omp for
       for (int r=0; r  < nb_rows; r++) {
#pragma SIMD
            result_vt[r] = vec_vt[r];
        }
#endif // BANDCPP
//----------------------------------------------------------------------
#ifdef BANDCPPGATHER
#pragma omp for
       for (int r=0; r  < nb_rows; r++) {
#pragma SIMD
            result_vt[r] = vec_vt[col_id_t[r]];
        }
#endif // BANDCPPGATHER
//----------------------------------------------------------------------
#ifdef BANDGATHER
       {
const __m512i offsets = _mm512_set4_epi32(3,2,1,0);  // original
//const __m512i four = _mm512_set4_epi32(4,4,4,4); 
const __m512i four = _mm512_set4_epi32(1,1,1,1); // only one vector 
__m512i v3_oldi;
__m512  v = _mm512_setzero_ps();
const int scale = 4;

// There will be differences depending on the matrix type. Create specialized col_id_t matrix for this
// experiment. 
#pragma omp for
        for (int i=0; i < nb_rows; i+=16) {
             v3_oldi = read_aaaa(&col_id_t[0] + i);    // ERROR: dom not known
             //if (i < 100) print_epi32(v3_oldi, "erad_aaaa v3_oldi");
             v3_oldi = _mm512_fmadd_epi32(v3_oldi, four, offsets); // offsets?
             //v = _mm512_castsi512_ps(v3_oldi); // temporary

             //  Error in next line ERROR ERROR ERROR
             //if (i < 100) print_epi32(v3_oldi, "v3_oldi");
             v     = _mm512_i32gather_ps(v3_oldi, vec_vt, scale); // scale = 4 bytes (floats)

             //printf("3\n");
             v3_oldi = read_aaaa(&col_id_t[0] + i + 4);
             v3_oldi = _mm512_fmadd_epi32(v3_oldi, four, offsets); // offsets?
             v     = _mm512_add_ps(v, _mm512_i32gather_ps(v3_oldi, vec_vt, scale) ); // scale = 4 bytes (floats)

#if 1
             v3_oldi = read_aaaa(&col_id_t[0] + i + 8);
             v3_oldi = _mm512_fmadd_epi32(v3_oldi, four, offsets); // offsets?
             v     = _mm512_add_ps(v, _mm512_i32gather_ps(v3_oldi, vec_vt, scale) ); // scale = 4 bytes (floats)

             v3_oldi = read_aaaa(&col_id_t[0] + i + 12);
             v3_oldi = _mm512_fmadd_epi32(v3_oldi, four, offsets); // offsets?
             v     = _mm512_add_ps(v, _mm512_i32gather_ps(v3_oldi, vec_vt, scale) ); // scale = 4 bytes (floats)
#endif
            _mm512_storenrngo_ps(result_vt+i, v); 
            // 138Gflops without the gathers
        }
       }
       //printf("nb_rows= %d\n", nb_rows); // temporary
#endif // BANDGATHER
//----------------------------------------------------------------------

} // OMP parallel
       tm["spmv"]->end();
       elapsed = tm["spmv"]->getTime();
       float gbytes = nb_rows*sizeof(float) * 2 * 1.e-9;
       float bandwidth = gbytes / (elapsed*1.e-3);
        if (bandwidth > max_bandwidth) {
            max_bandwidth = bandwidth;
            min_elapsed = elapsed;
        }
    }
    _mm_free(vec_vt);
    printf("read and write of %f Mbytes, using _mm512_store_ps and _m512_load_ps,\n",
            nb_rows*sizeof(float)*1.e-6);
    printf("16 at a time\n");
    printf("r/w, max bandwidth= %f (gbytes/sec), time: %f (ms)\n", max_bandwidth, min_elapsed);
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

    float gflops;
    float max_gflops = 0.;
    float max_bandwidth = 0.;
    float elapsed = 0.; 
    float min_elapsed = 0.; 
    int nb_rows = rd.nb_rows;
    int nz = rd.stencil_size;

    generateInputMatricesAndVectors();

// -----------------------------------------------
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
    __m512i i3_old  = _mm512_setzero_epi32();

#pragma omp for 
        for (int r=0; r < nb_rows; r++) {
//#pragma simd
            //printf("***** row %d\n", r);
            //if (r > 1) exit(0);
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
                //print_ps(v3_old, "1st v3old, extload_ps");
                //print_ps(v2_old, "1st swizzle");
                accu = _mm512_fmadd_ps(v3_old, v2_old, accu);

                //-----
                icol         = col_id_t[n+nz*r+1];
                addr_vector  = vec_vt + nb_mat*icol;
                // read 4 vectors (m0v0,m0v1,m0v2,m0v3) and create vector (m0v0,m0v1,m0v2,m0v3,m0v0,m0v1,m0v2,m0v3,...)
                v3_old = _mm512_extload_ps(addr_vector, _MM_UPCONV_PS_NONE, _MM_BROADCAST_4X16, _MM_HINT_NT);
                // m0c0,m0c0,m0c0,m0c0, m1c0,m1c0,m1c0,m1c0, m2c0,m2c0,m2c0,m2c0,   m3c0,m3c0,m3c0,m3c0
                v2_old = _mm512_swizzle_ps(v1_old, _MM_SWIZ_REG_BBBB);
                //printf("2nd icol= %d\n", icol);
                //print_ps(v3_old, "2nd v3old, extload_ps");
                //print_ps(v2_old, "2nd swizzle");
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
    gflops = rd.nb_mats * rd.nb_vecs * 2.*rd.stencil_size*rd.nb_rows*1e-9 / (1e-3*elapsed); 
    if (gflops > max_gflops) {
        max_gflops = gflops;
        min_elapsed = elapsed;
    }
}
    printf("Max gflops: %f, time: %f (ms)\n", max_gflops, elapsed);

    //checkSolutions();
    freeInputMatricesAndVectors();
}
//----------------------------------------------------------------------
template <typename T>
void ELL_OPENMP<T>::method_7a(int nbit)
{
    // Align the vectors so I can use vector instrincis and other technique.s 
    // Replace std::vector by pointers to floats and ints. 
    // Process 4 matrices and 4 vectors. Make 4 matrices identical. 
	printf("============== METHOD 7a ===================\n");
    printf("Method 7a with new matrix generators\n");

    float gflops;
    float max_gflops = 0.;
    float elapsed = 0.; 
    float min_elapsed = 0.; 
    int nb_rows = rd.nb_rows;
    int nz = rd.stencil_size;


    generateInputMatricesAndVectors();

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
            __m512 accu = _mm512_setzero_ps(); // 16 floats for 16 matrices
            float* addr_vector;
            int    icol;

#pragma simd
            for (int n=0; n < nz; n+=4) {  // nz is multiple of 32 (for now)
                // (m=[0,..,3],c=0),(m=[0,..,3],c=1),..,(m=[0,..,3],c=3) == 16 elements
                // Left is least significant (in Intel vectorization charts, right is least significant
                // Left here, is right in the Intel documents.
                // m0c0,m0c1,m0c2,m0c3,  m1c0,m1c1,m1c2,m2c3,  ...., m3c0,m3c1,m3c2,m3c3
                v1_old = _mm512_load_ps(data_t + nb_mat*(n + r*nz)); // load 16 at a time

                // icol is the same for all matrices
                icol         = col_id_t[n+nz*r+0];   // single element (but next 4 in cache)
                addr_vector  = vec_vt + nb_mat*icol; 
                // f0v0 means 0th element of function 0
                // read 4 vectors (f0v0,f1v0,f2v0,f3v0) and create vector (f0v0,f1v0,f2v0,f3v0, f0v0,f1v0,f2v0,f3v0,...)
                v3_old = _mm512_extload_ps(addr_vector, _MM_UPCONV_PS_NONE, _MM_BROADCAST_4X16, _MM_HINT_NT);
                // m0c0,m0c0,m0c0,m0c0,   m1c0,m1c0,m1c0,m1c0,   m2c0,m2c0,m2c0,m2c0,   m3c0,m3c0,m3c0,m3c0
                v2_old = _mm512_swizzle_ps(v1_old, _MM_SWIZ_REG_AAAA);
                accu = _mm512_fmadd_ps(v3_old, v2_old, accu);

                //-----
                icol         = col_id_t[n+nz*r+1];
                addr_vector  = vec_vt + nb_mat*icol;
                // read 4 vectors (m0v0,m0v1,m0v2,m0v3) and create vector (m0v0,m0v1,m0v2,m0v3,m0v0,m0v1,m0v2,m0v3,...)
                v3_old = _mm512_extload_ps(addr_vector, _MM_UPCONV_PS_NONE, _MM_BROADCAST_4X16, _MM_HINT_NT);
                // m0c0,m0c0,m0c0,m0c0, m1c0,m1c0,m1c0,m1c0, m2c0,m2c0,m2c0,m2c0,   m3c0,m3c0,m3c0,m3c0
                v2_old = _mm512_swizzle_ps(v1_old, _MM_SWIZ_REG_BBBB);
                accu = _mm512_fmadd_ps(v3_old, v2_old, accu);

                //-----
                icol         = col_id_t[n+nz*r+2];
                addr_vector  = vec_vt + nb_mat*icol;
                v3_old = _mm512_extload_ps(addr_vector, _MM_UPCONV_PS_NONE, _MM_BROADCAST_4X16, _MM_HINT_NT);
                // m0c2,m0c2,m0c2,m0c2, m1c2,m1c2,m1c2,m1c2, m2c2,m2c2,m2c2,m2c2,   m3c2,m3c2,m3c2,m3c2
                v2_old = _mm512_swizzle_ps(v1_old, _MM_SWIZ_REG_CCCC);
                accu = _mm512_fmadd_ps(v3_old, v2_old, accu);

                //-----
                icol         = col_id_t[n+nz*r+3];
                addr_vector  = vec_vt + nb_mat*icol;
                v3_old = _mm512_extload_ps(addr_vector, _MM_UPCONV_PS_NONE, _MM_BROADCAST_4X16, _MM_HINT_NT);
                // m0c3,m0c3,m0c3,m0c3, m1c3,m1c3,m1c3,m1c3, m2c3,m2c3,m2c3,m2c3,   m3c3,m3c3,m3c3,m3c3
                v2_old = _mm512_swizzle_ps(v1_old, _MM_SWIZ_REG_DDDD);
                accu = _mm512_fmadd_ps(v3_old, v2_old, accu);
            }
            _mm512_store_ps(result_vt+nb_mat*nb_vec*r, accu);
            // nr: no register
            //_mm512_storenr_ps(result_vt+nb_mat*nb_vec*r, accu);
            // no ordering and no read of cache lines from memory
            //_mm512_storenrngo_ps(result_vt+nb_mat*nb_vec*r, accu);
        } 
}
        tm["spmv"]->end();  // time for each matrix/vector multiply
        float gflops;
        float elapsed; 
        elapsed = tm["spmv"]->getTime();
        gflops = rd.nb_mats * rd.nb_vecs * 2.*rd.stencil_size*rd.nb_rows*1e-9 / (1e-3*elapsed); // assumes count of 1
        if (gflops > max_gflops) {
            max_gflops = gflops;
            min_elapsed = elapsed;
        }
    }
    printf("Max Gflops: %f, min time: %f (ms)\n", max_gflops, min_elapsed);

    //checkSolutions();
    freeInputMatricesAndVectors();
}
//----------------------------------------------------------------------
//----------------------------------------------------------------------
template <typename T>
void ELL_OPENMP<T>::method_8a(int nbit)
{
	printf("============== METHOD 8a ===================\n");

    //generateInputMatricesAndVectors();
    generateInputMatricesAndVectorsMulti();
    // I created a dummy subdomain that is the entire domain
    //printf("subdomains[0].col_id_t[262000] = %d\n", subdomains[0].col_id_t[262000]); exit(0);
    bandwidth(subdomains[0].col_id_t, rd.nb_rows, rd.stencil_size, rd.nb_rows);

    float gflops;
    float max_gflops = 0.;
    float elapsed = 0.; 
    float min_elapsed = 0.; 
    int nb_rows = rd.nb_rows;
    int nz = rd.stencil_size;

    // Must now work on alignmentf vectors. 
    // Produces the correct serial result
    for (int it=0; it < 10; it++) {
        tm["spmv"]->start();
#pragma omp parallel firstprivate(nb_rows, nz)
{
    const int int_mask_lo = (1 << 0) + (1 << 4) + (1 << 8) + (1 << 12);
    const int nb_mat = 4;
    const int nb_vec = 4;
    const int scale = 4;
    const int nz = rd.stencil_size;

    __m512 v1_old = _mm512_setzero_ps();
    __m512 v2_old = _mm512_setzero_ps();
    __m512 v3_old = _mm512_setzero_ps();
    __m512i i3_old = _mm512_setzero_epi32();
   const __m512i offsets = _mm512_set4_epi32(3,2,1,0);  // original
   const __m512i four = _mm512_set4_epi32(4,4,4,4); 


#pragma omp for 
        for (int r=0; r < nb_rows; r++) {
            __m512 accu = _mm512_setzero_ps(); // 16 floats for 16 matrices

#pragma simd
            for (int n=0; n < nz; n+=4) {  // nz is multiple of 32 (for now)
                v1_old = _mm512_load_ps(data_t + nb_mat*(n + r*nz)); // load 16 at a time

                __m512i v3_oldi = read_aaaa(&col_id_t[0]+n+nz*r);
                v3_oldi = _mm512_fmadd_epi32(v3_oldi, four, offsets);
                __m512  v     = _mm512_i32gather_ps(v3_oldi, vec_vt, scale); // scale = 4 bytes (floats)

                v3_old = permute(v, _MM_PERM_AAAA);
                v2_old = _mm512_swizzle_ps(v1_old, _MM_SWIZ_REG_AAAA);
                accu = _mm512_fmadd_ps(v3_old, v2_old, accu);

                v3_old = permute(v, _MM_PERM_BBBB);
                v2_old = _mm512_swizzle_ps(v1_old, _MM_SWIZ_REG_BBBB);
                accu = _mm512_fmadd_ps(v3_old, v2_old, accu);

                v3_old = permute(v, _MM_PERM_CCCC);
                v2_old = _mm512_swizzle_ps(v1_old, _MM_SWIZ_REG_CCCC);
                accu = _mm512_fmadd_ps(v3_old, v2_old, accu);

                v3_old = permute(v, _MM_PERM_DDDD);
                v2_old = _mm512_swizzle_ps(v1_old, _MM_SWIZ_REG_DDDD);
                accu = _mm512_fmadd_ps(v3_old, v2_old, accu);
            }
            _mm512_storenrngo_ps(result_vt+nb_mat*nb_vec*r, accu);
        } 
}
        tm["spmv"]->end();  // time for each matrix/vector multiply
        elapsed = tm["spmv"]->getTime();
        gflops = rd.nb_mats * rd.nb_vecs * 2.*rd.stencil_size*rd.nb_rows*1e-9 / (1e-3*elapsed); // assumes count of 1
        printf("gflops= %f\n", gflops);
        if (gflops > max_gflops) {
            max_gflops = gflops;
            min_elapsed = elapsed;
        }
   }

   printf("Max Gflops: %f, min time: %f (ms)\n", max_gflops, min_elapsed);
   //checkSolutions();
   freeInputMatricesAndVectors();
}
//----------------------------------------------------------------------
template <typename T>
void ELL_OPENMP<T>::method_8a_multi(int nbit)
{
	printf("============== METHOD 8a Multi, %d domain  ===================\n", nb_subdomains);
    printf("nb subdomains= %d\n", nb_subdomains);
    printf("nb_rows_multi = %d\n", nb_rows_multi[0]);

    generateInputMatricesAndVectorsMulti();
    bandwidth(subdomains[0].col_id_t, nb_rows_multi[0], rd.stencil_size, nb_vec_elem_multi[0]);
    bandwidthQ(subdomains[0], nb_rows_multi[0], rd.stencil_size, nb_vec_elem_multi[0]);
 
    float gflops;
    float max_gflops = 0.;
    float elapsed = 0.; 
    float min_elapsed = 0.; 
    //int nb_rows = rd.nb_rows;
    int nz = rd.stencil_size;
    printf("*** nb_subdomains: %d\n", nb_subdomains);

    // Must now work on alignmentf vectors. 
    // Produces the correct serial result
    for (int it=0; it < 10; it++) {
      tm["spmv"]->start();
      for (int s=0; s < nb_subdomains; s++) {
        //printf("iter %d, subdom %d\n", it, s);
// Should all 4 subdomains be contained in a single omp parallel pragma?
#pragma omp parallel firstprivate(nz)
{
    const int nb_rows = nb_rows_multi[s];
    const int int_mask_lo = (1 << 0) + (1 << 4) + (1 << 8) + (1 << 12);
    const int nb_mat = 4;
    const int nb_vec = 4;
    const int scale = 4;
    const int nz = rd.stencil_size;
    const Subdomain& dom = subdomains[s];

    __m512 v1_old = _mm512_setzero_ps();
    __m512 v2_old = _mm512_setzero_ps();
    __m512 v3_old = _mm512_setzero_ps();
    __m512i i3_old = _mm512_setzero_epi32();
    __m512  v = _mm512_setzero_ps(); // TEMPORARY
   const __m512i offsets = _mm512_set4_epi32(3,2,1,0);  // original
   const __m512i four = _mm512_set4_epi32(4,4,4,4); 
    //printf("after subdomain\n");

//#define GATHER
//#undef GATHER
//#define PERMUTE

#if 1
#pragma omp for 
        for (int r=0; r < nb_rows; r++) {
            __m512 accu = _mm512_setzero_ps(); // 16 floats for 16 matrices

#pragma simd
            for (int n=0; n < nz; n+=4) {  // nz is multiple of 32 (for now)
                v1_old = _mm512_load_ps(dom.data_t + nb_mat*(n + r*nz)); // load 16 at a time

                __m512i v3_oldi = read_aaaa(&dom.col_id_t[0]+n+nz*r);
                v3_oldi = _mm512_fmadd_epi32(v3_oldi, four, offsets);
                //printf("after v3_oldi\n");
#ifdef GATHER
                v     = _mm512_i32gather_ps(v3_oldi, dom.vec_vt, scale); // scale = 4 bytes (floats)
#else
                v = _mm512_load_ps(dom.vec_vt);
#endif

#ifndef PERMUTE 
                v3_old = v;
#endif
//#endif
                //printf("after v\n");

#ifdef PERMUTE
                v3_old = permute(v, _MM_PERM_AAAA);
#endif
                v2_old = _mm512_swizzle_ps(v1_old, _MM_SWIZ_REG_AAAA);
                accu = _mm512_fmadd_ps(v3_old, v2_old, accu);

#ifdef PERMUTE
                v3_old = permute(v, _MM_PERM_BBBB);
#endif
                v2_old = _mm512_swizzle_ps(v1_old, _MM_SWIZ_REG_BBBB);
                accu = _mm512_fmadd_ps(v3_old, v2_old, accu);

#ifdef PERMUTE
                v3_old = permute(v, _MM_PERM_CCCC);
#endif
                v2_old = _mm512_swizzle_ps(v1_old, _MM_SWIZ_REG_CCCC);
                accu = _mm512_fmadd_ps(v3_old, v2_old, accu);

#ifdef PERMUTE
                v3_old = permute(v, _MM_PERM_DDDD);
#endif
                v2_old = _mm512_swizzle_ps(v1_old, _MM_SWIZ_REG_DDDD);
                accu = _mm512_fmadd_ps(v3_old, v2_old, accu);
            }
            _mm512_storenrngo_ps(dom.result_vt+nb_mat*nb_vec*r, accu);  
        } 
#endif
}
     }
        tm["spmv"]->end();  // time for each matrix/vector multiply
        elapsed = tm["spmv"]->getTime();
        // nb_rows is wrong. 
        //printf("%d, %d, %d, %d\n", rd.nb_mats, rd.nb_vecs, rd.stencil_size, rd.nb_rows);
        gflops = rd.nb_mats * rd.nb_vecs * 2.*rd.stencil_size*rd.nb_rows*1e-9 / (1e-3*elapsed); // assumes count of 1
        printf("%f gflops, %f (ms)\n", gflops, elapsed);
        if (gflops > max_gflops) {
            max_gflops = gflops;
            min_elapsed = elapsed;
        }
   }

   printf("Max Gflops: %f, min time: %f (ms)\n", max_gflops, min_elapsed);
   //checkSolutions();
   freeInputMatricesAndVectorsMulti();
}
//----------------------------------------------------------------------
template <typename T>
void ELL_OPENMP<T>::checkAllSerialSolutions(T* data_t, int* col_id_t, T* vec_vt, T* one_res, int nz, int nb_mat, int nb_vec, int nb_rows)
{
    T* v = (T*) _mm_malloc(sizeof(T)*nb_rows, 16);
    T* d = (T*) _mm_malloc(sizeof(T)*nb_rows*nz, 16);
    
    for (int mat_id=0; mat_id < nb_mat; mat_id++) {
        retrieve_data(&data_t[0], d, mat_id, nz, nb_rows, nb_mat);
    for (int vec_id=0; vec_id < nb_vec; vec_id++) {
        retrieve_vector(vec_vt, v, vec_id, nb_vec, nb_rows); 
        spmv_serial_row(&col_id_t[0], d, v, &one_res[0], nz, nb_rows);
        printf("method_8a, l2norm of serial vec/mat= %d/%d, %f\n", vec_id, mat_id, l2norm(one_res, nb_rows)); // need matrix and vector index
    }}

    _mm_free(v);
    _mm_free(d);
}
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
    std::fill(result.begin(), result.end(), (T) 0.);

    for (int row=0; row < v.size(); row++) {
        int matoffset = row;
        float accumulant = 0.;

        for (int i = 0; i < nz; i++) {
            vecid = col_id[matoffset];
            float d= data[matoffset];
            accumulant += data[matoffset] * v[vecid];
            matoffset += aligned;
        }
        result[row] = accumulant;
    }
    
#if 0
    for (int i=0; i < 10; i++) {
        printf("serial result: %f\n", result[i]);
    }
#endif
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
template <typename T>
void ELL_OPENMP<T>::spmv_serial_row(int* col_id, T* data, T* v, T* res, int nbz, int nb_rows) 
// transpose col_id from mat, and apply alternate algorithm to compute col_id (more efficient)
// results should be same as spmv_serial (which is really spmv_serial_col)
{
    T accumulant;
   int matoffset;

    for (int i=0; i < nb_rows; i++) res[i] = 0.;

    for (int row=0; row < nb_rows; row++) {
        float accumulant = 0.;
        for (int i=0; i < nbz; i++) {
            matoffset = row*nbz+i;
            int vecid = col_id[matoffset];
            float   d = data[matoffset];
            accumulant += d * v[vecid];  // most efficient, 12 Gflops
        }
        res[row] = accumulant;
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

#if 1
    int int_mask_lo = (1 << 0) + (1 << 4) + (1 << 8) + (1 << 12);
    //int int_mask_lo = (1 << 0) + (1 << 1) + (1 << 2) + (1 << 3);
    __mmask16 mask_lo = _mm512_int2mask(int_mask_lo);
    __m512i v1_old;
    v1_old = _mm512_setzero_epi32();
    v1_old = _mm512_mask_loadunpacklo_epi32(v1_old, mask_lo, a);
    v1_old = _mm512_mask_loadunpackhi_epi32(v1_old, mask_lo, a);
    v1_old = _mm512_swizzle_epi32(v1_old, _MM_SWIZ_REG_AAAA);
#endif
    return v1_old;
}
//----------------------------------------------------------------------
template <typename T>
__m512 ELL_OPENMP<T>::read_aaaa(float* a)
{
    // only works with floats
    // read in 4 floats (a,b,c,d) and create the 
    // 16-float vector dddd,cccc,bbbb,aaaa
#if 1
    const int int_mask_lo = (1 << 0) + (1 << 4) + (1 << 8) + (1 << 12);
    __mmask16 mask_lo = _mm512_int2mask(int_mask_lo);
    __m512 v1_old;
    v1_old = _mm512_setzero_ps();
    v1_old = _mm512_mask_loadunpacklo_ps(v1_old, mask_lo, a);
    v1_old = _mm512_mask_loadunpackhi_ps(v1_old, mask_lo, a);
    v1_old = _mm512_swizzle_ps(v1_old, _MM_SWIZ_REG_AAAA);
#endif
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
    //return;
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
    //return;
    int* res = (int*) _mm_malloc(32*sizeof(int), 64);
    _mm512_store_epi32(res, v1);
    printf("\n--- %s ---\n", msg.c_str());
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
#if 0
    __m512 v1_old = _mm512_extload_ps(a, _MM_UPCONV_PS_NONE, _MM_BROADCAST_4X16, _MM_HINT_NONE);
    return v1_old;
#endif
}
//----------------------------------------------------------------------
template <typename T>
void ELL_OPENMP<T>::generate_ell_matrix_by_row(std::vector<int>& col_id, std::vector<T>& data, int nb_elem)
{
    // successive nonzeros are contiguous in memory. This is different than the standard
    // ordering for col_id
}
//----------------------------------------------------------------------
template <typename T>
void ELL_OPENMP<T>::generate_ell_matrix_data(T* data, int nbz, int nb_rows, int nb_mat)
{
    int n_skip = nb_mat;
    int nz4 = nbz / n_skip;
    for (int m=0; m < nb_mat; m++) {
        for (int r=0; r < nb_rows; r++) {
            for (int n=0; n < nbz; n += n_skip) {
                int n4 = n / n_skip;
                for (int in=0; in < n_skip; in++) {
                    data[in+n_skip*(m+nb_mat*(n4+nz4*r))] = getRandf(); // correct norms (if vector = 1)
                }
            }
        }
    }
}
//----------------------------------------------------------------------
template <typename T>
void ELL_OPENMP<T>::generate_vector(T* vec, int nb_rows, int nb_vec)
{
    //assert(vec.size() == nb_rows*nb_vec);
    int sz = nb_rows*nb_vec;

    for (int i=0; i < sz; i++) {
        //if (!(i % 10000))  printf("i= %d\n", i);
        // works with matrix random, vector non-random
        vec[i] = (T) getRandf();// incorrect norms 
        //vec[i] = 1.0; // wrong norms when matrices random, although only 4 different norms
    }
}
//----------------------------------------------------------------------
template <typename T>
void ELL_OPENMP<T>::generate_col_id_multi(int* col_id, int nbz, int nb_rows)
{
}
//----------------------------------------------------------------------
template <typename T>
void ELL_OPENMP<T>::generate_col_id(int* col_id, int nbz, int nb_rows)
{
    printf("======= Reconstruct col_id matrix for special effects =====\n");
    //assert(col_id.size() == nbz*nb_rows);
    int left;
    int right;
    int width;
    int half_width;
    int sz2 = nbz >> 1;

    switch (stencil_type) {
    case NONE:
        break;

    case SUPERCOMPACT:
        printf("SUPERCOMPACT\n");
        // the first 32 columns of col_id are 1 to nbz-1 (for each row)
        // retrieving "x" is much more efficient
        for (int i=0; i < nb_rows; i++) {
            for (int j=0; j < nbz; j++) {
                col_id[j+i*nbz] = j;
            }
        }
        break;

    case COMPACT:
        printf("COMPACT\n");

        for (int i=0; i < nb_rows; i++) {
            if (i < nbz) {
                left = 0;
                right = left + nbz;
            }
            else if (i > nb_rows-nbz) { 
                right = nb_rows;
                left  = nb_rows-nbz;
            }
            else {
                left  = i-sz2;
                right = left + nbz;
            }
            for (int j=left; j < right; j++) {
                //col_id[i][j-left] = j;
                col_id[nbz*i+j-left] = j;
            }
            //col_id[i][0] = i;
            col_id[nbz*i] = i;
		    //col_id[i][i-left] = left;
		    col_id[nbz*i+i-left] = left;
	    }
        break;

    case RANDOM:
        // limit the randomness to a specified band about the center node. 
        // random with repeat. Would probably create a problem  with smaller arrays? Do not know.
        printf("RANDOM CASE\n");
        left = 0;
        right = 0;
        width = 6000;
        half_width = width / 2;
        for (int i=0; i < nb_rows; i++) {
                left = i - half_width;
                right = i + half_width;
                if (left < 0) {
                    right = right - left;
                    left = 0;
                }
                if (right >= nb_rows) {
                    left = left - (right-nb_rows);
                    right = nb_rows - 1;
                }
            for (int j=0; j < nbz; j++) {
                int r = getRand(width);
                col_id[i*nbz+j] = left + r;
                if ((left+r) >= nb_rows) {
                    printf("(left+r) >= nb_rows: should not happened. Fix code\n");
                }
            }
        }
        break;

    case RANDOMWITHDIAG:
        printf("RANDOM CASE WITH DIAGONAL\n");
        printf("inner_bandwidth= %d\n", rd.inner_bandwidth);
        printf("diagonal separation= %d\n", rd.diag_sep);
        // same as RANDOM, but add two additional diagonals a fixed distance away from the main diagaonal
        // Periodicity imposed so that the two diagonals extend across all the rows
        left = 0;
        right = 0;
        half_width = rd.inner_bandwidth / 2;
        if (rd.diag_sep <= half_width) {
            printf("diag_sep (%d) from main diagonal must be greater than half_width (%d)\n", 
                rd.diag_sep, half_width);
            exit(0);
        }
        for (int i=0; i < nb_rows; i++) {
                left = i - half_width;
                right = i + half_width;
                if (left < 0) {
                    right = right - left;
                    left = 0;
                }
                if (right >= nb_rows) {
                    left = left - (right-nb_rows);
                    right = nb_rows - 1;
                }
            for (int j=1; j < (nbz-1); j++) {
                int r = getRand(rd.inner_bandwidth);
                col_id[i*nbz+j] = left + r;
                if ((left+r) >= nb_rows) {
                    printf("(left+r) >= nb_rows: should not happened. Fix code\n");
                }
            }

            // These two lines create a segmentation fault. WHY? 
            // The two diagonals will cross through the middle band
            col_id[0    +nbz*i] = (i-rd.diag_sep+nb_rows) % nb_rows;
            col_id[nbz-1+nbz*i] = (i+rd.diag_sep+nb_rows) % nb_rows;
        }
#if 0
        for (int i=0; i < nb_rows*nbz; i++) {
            if (col_id[i] >= nb_rows) {
                printf("col_id[%] = %d, too large\n");
                exit(0);
            }
        }
#endif
        break;

    case RANDOMDIAGS:
        printf("RANDOM CASE WITH DIAGONAL\n");
        // nbz diagonals. Periodicity.
        left = nbz + 1;
        right = nb_rows - nbz - 1;
        int inner_bandwidth = right - left;
        // Generate stencil_size diagonals between left and right
        // Top row
        for (int j=0; j < nbz; j++) {
            col_id[j] = getRand(inner_bandwidth);
        }
        std::sort(col_id, col_id+nbz);
        //for (int i=0; i < nbz; i++) printf("col_id[%d]= %d\n", i, col_id[i]);
 
        // Create diagonals
        for (int i=1; i < nb_rows; i++) {
            for (int j=0; j < nbz; j++) {
                col_id[j+nbz*i] = (col_id[j+nbz*(i-1)] + 1) % nb_rows;
#if 0
                if (col_id[j+nbz*i] >= nb_rows) {
                    printf("illegal matrix entry\n"); 
                    printf("i,j= %d, %d, col_id= %d\n", i, j, col_id[j+nbz*i]);
                    exit(1);
                }
#endif
            }
        }
        break;

    default:
        printf("generate_col_id, case not treated\n");
        break;
    }

    switch(stencil_type) {
    case(RANDOMWITHDIAG):
    case(RANDOM):
#if 0
        for (int i=0; i < 20; i++) {
            printf("bef, row %d: ");
            for (int j=0; j < nbz; j++) {
                printf("%d,", col_id[i*nbz+j]);
            }
            printf("\n");
        }
#endif
        if (rd.sort_col_indices == 0) break;
        for (int i=0; i < nb_rows; i++) {
            std::sort(col_id+i*nbz, col_id+i*nbz+nbz);
        }
#if 0
        for (int i=0; i < 20; i++) {
            printf("aft, row %d: ");
            for (int j=0; j < nbz; j++) {
                printf("%d,", col_id[i*nbz+j]);
            }
            printf("\n");
        }
#endif
        break;
    }
}
//----------------------------------------------------------------------
template <typename T>
void ELL_OPENMP<T>::retrieve_vector(T* vec, T* retrieved, int vec_id, int nb_vec, int nb_rows)
//void ELL_OPENMP<T>::retrieve_vector(std::vector<T>& vec, std::vector<T>& retrieved, int vec_id, int nb_vec)
{
    // vec is stored with vec_id as the fastest varying index

    for (int i=0; i < nb_rows; i++) {
        retrieved[i] = vec[vec_id + nb_vec*i];
    }
}
//----------------------------------------------------------------------
template <typename T>
void ELL_OPENMP<T>::retrieve_data(T* data_in, T* data_out, int mat_id, int nbz, int nb_rows, int nb_mat)
{
    assert(nb_mat == 4);
    //assert(data_out.size() == nbz*nb_rows);

    int n_skip = nb_mat;
    int nz4 = nbz / n_skip;
    for (int r=0; r < nb_rows; r++) {
        for (int n=0; n < nbz; n += n_skip) {
            int n4 = n / nb_mat;
            for (int in=0; in < nb_mat; in++) {
                //data_out[r+nb_rows*(n+in)] = data_in[in+n_skip*(mat_id+nb_mat*(n4+nz4*r))];
                data_out[n+in + nbz*r] = data_in[in+n_skip*(mat_id+nb_mat*(n4+nz4*r))];
            }
        }
    }
}
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
#if 1
template <typename T>
void spmv_ell_openmp(std::string filename)
{
    int ntimes = 10;
    //ell_subdomain_64x_64y_64z.bmtx
#if 0
    if (filename.find("subdomain") != std::string::npos) {
        printf("multidomain file\n");
        int multi = 1;
        ELL_OPENMP<T> ell_ocl(filename, ntimes, multi); 
        // multidomain read
        ;
    }
#endif

    // input file type is in the configuration file (test.conf)
    ELL_OPENMP<T> ell_ocl(filename, ntimes);
	ell_ocl.run();
}
//----------------------------------------------------------------------
template <typename T>
//void spmv_ell_openmp(coo_matrix<int, T>* coo_mat, int dim2Size, int ntimes)
void spmv_ell_openmp(std::vector<int>& col_id, int& nb_rows, int& stencil_size)
{
    printf("spmv_ell_openmp: handle ELLPACK MATRIX\n");
	ELL_OPENMP<T> ell_ocl(col_id, nb_rows, stencil_size);
	ell_ocl.run();
}

template <typename T>
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
template <typename T>
void ELL_OPENMP<T>::generateInputMatricesAndVectors()
{
    // vectors are aligned. Start using vector _mm_ constructs. 
    int nz = rd.stencil_size;
    int nb_mat = rd.nb_mats;
    int nb_vec = rd.nb_vecs;
    int nb_rows = rd.nb_rows;
    int tot_nz = nz*nb_rows;

    //std::vector<int>& col_id = mat.ell_col_id; // I SHOULD NOT NEED THIS
    printf("nz= %d, nb_rows= %d, nb_rows*nz = %d\n", nz, nb_rows, nb_rows*nz);
 
    // Create variables new_col_id_size, new_nb_vec (4 or 1), new_nb_mats (4 or 1)
    // new_nz, new_sparsity, new_diag_sep, new_inner_bandwidth, new_non_zero_stats (uniform, normal, etc. distribution 
    // of nonzeros in inner_bandwidth section of the matrix. 
   
   //newVars(int new_col_id_size, int new_nb_vec, int new_nb_mats, int new_nz, stencilType new_sparsity, int new_diag_step, int new_inner_bandwidth, int  new_nonzero_stats)
   // newVars(new_col_id_size, new_nb_vec, new_nb_mats, new_nz, new_sparsity, new_diag_step, new_inner_bandwidth, 
   //   new_nonzero_stats)

//----------------------------------------------------------------
    vec_vt    = (float*) _mm_malloc(sizeof(float) * nb_vec * nb_rows, 64);
    result_vt = (float*) _mm_malloc(sizeof(float) * nb_vec * nb_mat * nb_rows, 64);
    col_id_t  = (int*)   _mm_malloc(sizeof(int)   * tot_nz, 16);
    data_t    = (float*) _mm_malloc(sizeof(float) * nb_mat * tot_nz, 64);

#if 0
    printf("nb_vec= %d\n", nb_vec);
    printf("nb_mat= %d\n", nb_mat);
    printf("nb_rows= %d\n", nb_rows);
    printf("tot_nz= %d\n", tot_nz);
#endif

    if (vec_vt == 0 || result_vt == 0 || col_id_t == 0 || data_t == 0) {
        printf("1. memory allocation failed\n");
        exit(0);
    }
 
    generate_vector(vec_vt, nb_rows, nb_vec);
    generate_col_id(col_id_t, nz, nb_rows);
    generate_ell_matrix_data(data_t, nz, nb_rows, nb_mat);
}
//----------------------------------------------------------------
template <typename T>
void ELL_OPENMP<T>::generateInputMatricesAndVectorsMulti()
{
    // vectors are aligned. Start using vector _mm_ constructs. 
    int nb_mat = rd.nb_mats;
    int nb_vec = rd.nb_vecs;

    printf("inside generate Multi\n");

    if (rd.use_subdomains == 0) {
        printf("generate: nb_subdomains = %d\n", nb_subdomains);
        nb_subdomains = 1;
        nb_rows_multi[0] = rd.nb_rows;
        nb_vec_elem_multi[0] = rd.nb_rows;
    }

    for (int i=0; i < nb_subdomains; i++) {
        printf("subdomain %d\n", i);
        int nz = rd.stencil_size;
        int nb_rows = nb_rows_multi[i];
        int tot_nz = nz*nb_rows;
        int vec_size = nb_vec_elem_multi[i];
        printf("nb_rows= %d\n", nb_rows);
        printf("vec_size= %d\n", vec_size);
        printf("nb nonzeros per row= %d\n", nz);

        Subdomain& s = subdomains[i];

        s.vec_vt    = (float*) _mm_malloc(sizeof(float) * nb_vec * vec_size, 64);
        s.result_vt = (float*) _mm_malloc(sizeof(float) * nb_vec * nb_mat * nb_rows, 64);
        s.col_id_t  = (int*)   _mm_malloc(sizeof(int)   * tot_nz, 16);
        s.data_t    = (float*) _mm_malloc(sizeof(float) * nb_mat * tot_nz, 64);

        //printf("before generate vector\n");

        for (int j=offsets_multi[i], k=0; j < offsets_multi[i+1]; j++) {
            s.col_id_t[k++] = mat.ell_col_id[j];
        }

        // FIX col_ids'
        generate_vector(s.vec_vt, vec_size, nb_vec);
        generate_col_id(s.col_id_t, nz, nb_rows);
        //printf("before generate data\n");
        generate_ell_matrix_data(s.data_t, nz, nb_rows, nb_mat);

        //printf("before offsets\n");
        //printf("imin= %d, imax= %d\n", offsets_multi[i], offsets_multi[i+1]);
        //printf("after offsets\n");

        //for (int i=262000; i < 262100; i++) {
            //printf("s.col_id_t[%d] = %d\n", i, s.col_id_t[i]);
        //}
        //printf("inside generate...Multi\n"); exit(0);

        if (s.vec_vt == 0 || s.result_vt == 0 || s.col_id_t == 0 || s.data_t == 0) {
            printf("1. memory allocation failed\n");
            exit(0);
        }
    }
}
//----------------------------------------------------------------
template <typename T>
void ELL_OPENMP<T>::freeInputMatricesAndVectors()
{
    _mm_free(result_vt);
    _mm_free(vec_vt);
    _mm_free(data_t);
    _mm_free(col_id_t);
}
//----------------------------------------------------------------------
template <typename T>
void ELL_OPENMP<T>::freeInputMatricesAndVectorsMulti()
{
    for (int s=0; s < nb_subdomains; s++) {
        Subdomain& dom = subdomains[s];
        _mm_free(dom.result_vt);
        _mm_free(dom.vec_vt);
        _mm_free(dom.data_t);
        _mm_free(dom.col_id_t);
    }
}
//----------------------------------------------------------------------
template <typename T>
void ELL_OPENMP<T>::checkSolutions()
{
    std::vector<float> one_res(rd.nb_rows);  // single result
    for (int w=0; w < 16; w++) {
        for (int i=0; i < rd.nb_rows; i++) {
            one_res[i] = result_vt[16*i+w];
        }
        printf("method_8a, l2norm[%d]=of omp version: %f\n", w, l2norm(one_res));
    }

    checkAllSerialSolutions(data_t, col_id_t, vec_vt, &one_res[0], rd.stencil_size, rd.nb_mats, rd.nb_vecs, rd.nb_rows);
}
//----------------------------------------------------------------------
template <typename T>
void ELL_OPENMP<T>::bandwidthQ(Subdomain& s, int nb_rows, int stencil_size, int vec_size)
// compute bandwidth, average bandwidth, rms bandwidth
// Only consider Q matrix
// given row r, and stencil element s, col_id[s+stencil_size*r] is the index into the 
// vector vec of length vec_size
{
    // stencil ordering is assumed to be sorted by this time
    std::vector<int> bw(nb_rows);
    int* col_id = s.col_id_t;
    std::vector<int>& Qbeg = s.Qbeg;
    std::vector<int>& Qend = s.Qend;
    if (Qbeg.size() == 0 || Qend.size() == 0) return;
    int max_bandwidth = 0;


    printf("col_id, nb rows: %d\n", nb_rows);
    for (int r=0; r < nb_rows; r++) {
        //if (r < (nb_rows-1) && Qbeg[r+1] != Qend[r]) printf("bandwidthQ:: beg/end= %d, %d\n", Qbeg[r], Qend[r]);
        //std::sort(col_id+r*stencil_size, col_id+(r+1)*stencil_size);
        bw[r] = col_id[Qend[r]-1] - col_id[Qbeg[r]];
        max_bandwidth = bw[r] > max_bandwidth ? bw[r] : max_bandwidth;
    }
    printf("\n");
    printf("max bandwidthQ: %d\n", max_bandwidth);
}
//----------------------------------------------------------------------
//----------------------------------------------------------------------
template <typename T>
void ELL_OPENMP<T>::bandwidth(int* col_id, int nb_rows, int stencil_size, int vec_size)
// compute bandwidth, average bandwidth, rms bandwidth
// given row r, and stencil element s, col_id[s+stencil_size*r] is the index into the 
// vector vec of length vec_size
{
    // assume col_id are sorted
    //printf("nb_rows= %d\n", nb_rows);
    //printf("stencil_size= %d\n", stencil_size);
    std::vector<int> bw(nb_rows);
    int max_bandwidth = 0;

    printf("col_id, nb rows: %d\n", nb_rows);
    for (int r=0; r < nb_rows; r++) {
        // sort row that is unsorted in Evan code
         //printf("before sort, size: %d\n", stencil_size);
        std::sort(col_id+r*stencil_size, col_id+(r+1)*stencil_size);
        bw[r] = col_id[(r+1)*stencil_size-1] - col_id[r*stencil_size] + 1;
        //printf("bw[%d]= %d\n", r, bw[r]);
        max_bandwidth = bw[r] > max_bandwidth ? bw[r] : max_bandwidth;

        #if 0
        if (r > 262000) {
            printf("\n==== bandwidth, row %d: \n", r);
            for (int j=0; j < stencil_size; j++) {
                printf("%d,", col_id[j+stencil_size*r]);
            }
        } else { exit(0); }
        #endif
    }
    printf("\n");
    printf("max bandwidth: %d\n", max_bandwidth);
    //exit(0);
}
//----------------------------------------------------------------------
//****************************************************************************80
template <typename T>
void ELL_OPENMP<T>::permInverse3(int n, int* perm, int* perm_inv )

//****************************************************************************80
//
//  Purpose:
//
//    PERM_INVERSE3 produces the inverse of a given permutation.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    05 January 2007
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, int N, the number of items permuted.
//
//    Input, int PERM[N], a permutation.
//
//    Output, int PERM_INV[N], the inverse permutation.
//
{
  int i;

  for ( i = 0; i < n; i++ )
  {
    perm_inv[perm[i]-1] = i + 1;
  }

  return;
}
//----------------------------------------------------------------------
template <typename T>
int ELL_OPENMP<T>::adj_bandwidth(int node_num, int adj_num, int* adj_row, int* adj)

//****************************************************************************80
//
//  Purpose:
//
//    ADJ_BANDWIDTH computes the bandwidth of an adjacency matrix.
//
//  Discussion:
//
//    Thanks to Man Yuan, of Southeast University, China, for pointing out
//    an inconsistency in the indexing of ADJ_ROW, 06 June 2011.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    06 June 2011
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    Alan George, Joseph Liu,
//    Computer Solution of Large Sparse Positive Definite Systems,
//    Prentice Hall, 1981.
//
//  Parameters:
//
//    Input, int NODE_NUM, the number of nodes.
//
//    Input, int ADJ_NUM, the number of adjacency entries.
//
//    Input, int ADJ_ROW[NODE_NUM+1].  Information about row I is stored
//    in entries ADJ_ROW(I) through ADJ_ROW(I+1)-1 of ADJ.
//
//    Input, int ADJ[ADJ_NUM], the adjacency structure.
//    For each row, it contains the column indices of the nonzero entries.
//
//    Output, int ADJ_BANDWIDTH, the bandwidth of the adjacency
//    matrix.
//
{
  int band_hi;
  int band_lo;
  int col;
  int i;
  int j;
  int value;

  band_lo = 0;
  band_hi = 0;

  for ( i = 0; i < node_num; i++ )
  {
    for ( j = adj_row[i]; j <= adj_row[i+1]-1; j++ )
    {
      col = adj[j-1] - 1;
      band_lo = i4_max ( band_lo, i - col );
      band_hi = i4_max ( band_hi, col - i );
    }
  }

  value = band_lo + 1 + band_hi;

  return value;
}
//----------------------------------------------------------------------
template <typename T>
void ELL_OPENMP<T>::cuthillMcKee(std::vector<int>& col_id, int nb_rows, int stencil_size)
{
    std::vector<int> row_ptr;
    std::vector<int> new_row_ptr;
    std::vector<int> new_col_ind;

    row_ptr.push_back(0);
    for (int i=0; i < nb_rows; i++) {
        row_ptr.push_back(stencil_size);
    }

    printf("Bandwidth reduction using ViennaCL\n");
    new_col_ind.resize(0);
    new_row_ptr.resize(0);
    vcl::ConvertMatrix d(nb_rows, col_id, row_ptr, new_col_ind, new_row_ptr);
    int bw = d.calcOrigBandwidth();
    printf("initial bw (VCL): %d\n", bw);
    d.reduceBandwidthRCM();
    //d.reorderMatrix(); // done in reduceBandwidthwithRCM
    exit(0);
}
//----------------------------------------------------------------------
template <typename T>
int ELL_OPENMP<T>::adj_perm_bandwidth(int node_num, int adj_num, int* adj_row, int* adj,
  int* perm, int* perm_inv)

//****************************************************************************80
//
//  Purpose:
//
//    ADJ_PERM_BANDWIDTH computes the bandwidth of a permuted adjacency matrix.
//
//  Discussion:
//
//    The matrix is defined by the adjacency information and a permutation.
//
//    The routine also computes the bandwidth and the size of the envelope.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    05 January 2007
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    Alan George, Joseph Liu,
//    Computer Solution of Large Sparse Positive Definite Systems,
//    Prentice Hall, 1981.
//
//  Parameters:
//
//    Input, int NODE_NUM, the number of nodes.
//
//    Input, int ADJ_NUM, the number of adjacency entries.
//
//    Input, int ADJ_ROW[NODE_NUM+1].  Information about row I is stored
//    in entries ADJ_ROW(I) through ADJ_ROW(I+1)-1 of ADJ.
//
//    Input, int ADJ[ADJ_NUM], the adjacency structure.
//    For each row, it contains the column indices of the nonzero entries.
//
//    Input, int PERM[NODE_NUM], PERM_INV(NODE_NUM), the permutation
//    and inverse permutation.
//
//    Output, int ADJ_PERM_BANDWIDTH, the bandwidth of the permuted
//    adjacency matrix.
//
{
  int band_hi;
  int band_lo;
  int bandwidth;
  int col;
  int i;
  int j;

  band_lo = 0;
  band_hi = 0;

  for ( i = 0; i < node_num; i++ )
  {
    for ( j = adj_row[perm[i]-1]; j <= adj_row[perm[i]]-1; j++ )
    {
      col = perm_inv[adj[j-1]-1];
      band_lo = i4_max ( band_lo, i - col );
      band_hi = i4_max ( band_hi, col - i );
    }
  }

  bandwidth = band_lo + 1 + band_hi;

  return bandwidth;
}
//----------------------------------------------------------------------
template <typename T>
int ELL_OPENMP<T>::i4_max ( int i1, int i2 )

//****************************************************************************80
//
//  Purpose:
//
//    I4_MAX returns the maximum of two I4's.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    13 October 1998
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, int I1, I2, are two integers to be compared.
//
//    Output, int I4_MAX, the larger of I1 and I2.
//
{
  if ( i2 < i1 )
  {
    return i1;
  }
  else
  {
    return i2;
  }

}
//----------------------------------------------------------------------
//----------------------------------------------------------------------
}; // namespace

#endif
//----------------------------------------------------------------------
