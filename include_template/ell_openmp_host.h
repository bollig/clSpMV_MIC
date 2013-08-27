//TODO (disable from Makefile)
//FIX	two_vec_compare_T(coores, tmpresult, rownum);

#ifndef __CLASS_SPMV_ELL_OPENMP_HOST_H__
#define __CLASS_SPMV_ELL_OPENMP_HOST_H__

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


// for ic
//#include <immintrin.h>
#define _mm512_loadnr_pd(block) _mm512_extload_pd(block, _MM_UPCONV_PD_NONE, _MM_BROADCAST64_NONE, _MM_HINT_NT)
#define _mm512_loadnr_ps(block) _mm512_extload_ps(block, _MM_UPCONV_PD_NONE, _MM_BROADCAST64_NONE, _MM_HINT_NT)

namespace spmv {

#define USE(x) using OPENMP_BASE<T>::x
//#define USECL(x) using CLBaseClass::x

template <typename T>
class ELL_OPENMP_HOST : public OPENMP_BASE<T>
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
	ELL_OPENMP_HOST(std::string filename, int ntimes);
    // multi=1: multidomain input file (third argument is not used)
	ELL_OPENMP_HOST(std::string filename, int ntimes, int multi);
	ELL_OPENMP_HOST(coo_matrix<int, T>* mat, int dim2Size, int ntimes);
	ELL_OPENMP_HOST(std::vector<int>& col_id, int nb_rows, int nb_nonzeros_per_row);
	~ELL_OPENMP_HOST<T>() {
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
#if 0
    void method_rd_wr(int nbit);
#endif
    // read/write benchmarks. Memory is local to each thread. 
    void method_rd_wr_local_thread(int nbit);
	void method_7a(int nb=0); // 4 matrices, 4 vectors, using new matrix generators
	//void method_8(int nb=0); // 4 matrices, 4 vectors
	void method_8a(int nb=0); // 4 matrices, 4 vectors
//#endif
    void method_8a_multi(int nb=0);

    //inline __m512 permute(__m512 v1, _MM_PERM_ENUM perm);
    //inline __m512 read_aaaa(float* a);
    //inline __m512i read_aaaa(int* a);
    //inline __m512 read_abcd(float* a);
    //inline __m512 tensor_product(float* a, float* b);
    void print_f(float* res, const std::string msg="");
    void print_i(int* res, const std::string msg="");
    //void print_ps(const __m512 v1, const std::string msg="");
    //void print_epi32(const __m512i v1, const std::string msg="");
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
void ELL_OPENMP_HOST<T>::run()
{
    int num_threads[] = {1,2,4,8,16,32,64,96,128,160,192,224,244};

    int count = 0;
    int max_nb_runs = 1;

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
ELL_OPENMP_HOST<T>::ELL_OPENMP_HOST(std::vector<int>& col_id, int nb_rows, int nb_nonzeros_per_row) :
            OPENMP_BASE<T>(col_id, nb_rows, nb_nonzeros_per_row)
            // MUST DEBUG THE BASE CODE
{
}
//----------------------------------------------------------------------
template <typename T>
ELL_OPENMP_HOST<T>::ELL_OPENMP_HOST(coo_matrix<int, T>* coo_mat, int dim2Size, int ntimes) : 
   OPENMP_BASE<T>(coo_mat, dim2Size, ntimes)
{
    ;
}
//----------------------------------------------------------------------
template <typename T>
ELL_OPENMP_HOST<T>::ELL_OPENMP_HOST(std::string filename, int ntimes) : 
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
void ELL_OPENMP_HOST<T>::method_0(int nbit)
{
}
//----------------------------------------------------------------------
template <typename T>
void ELL_OPENMP_HOST<T>::method_1(int nb_vectors)
{
}
//----------------------------------------------------------------------
template <typename T>
void ELL_OPENMP_HOST<T>::method_2(int nb_vectors)
{
}
//----------------------------------------------------------------------
template <typename T>
void ELL_OPENMP_HOST<T>::method_3(int nbit)
{
}
//----------------------------------------------------------------------
template <typename T>
void ELL_OPENMP_HOST<T>::method_4(int nbit)
{
}
//----------------------------------------------------------------------
//----------------------------------------------------------------------
template <typename T>
void ELL_OPENMP_HOST<T>::method_5(int nbit)
{
}
//----------------------------------------------------------------------
template <typename T>
void ELL_OPENMP_HOST<T>::method_6(int nbit)
{
}
//----------------------------------------------------------------------
template <typename T>
void ELL_OPENMP_HOST<T>::method_rd_wr_local_thread(int nbit)
{
}
//----------------------------------------------------------------------
template <typename T>
void ELL_OPENMP_HOST<T>::method_7(int nbit)
{
}
//----------------------------------------------------------------------
template <typename T>
void ELL_OPENMP_HOST<T>::method_7a(int nbit)
{
}
//----------------------------------------------------------------------
template <typename T>
void ELL_OPENMP_HOST<T>::method_8a(int nbit)
{
#if 0
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
#endif
}
//----------------------------------------------------------------------
template <typename T>
void ELL_OPENMP_HOST<T>::method_8a_multi(int nbit)
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

#if 0
    __m512 v1_old = _mm512_setzero_ps();
    __m512 v2_old = _mm512_setzero_ps();
    __m512 v3_old = _mm512_setzero_ps();
    __m512i i3_old = _mm512_setzero_epi32();
    __m512  v = _mm512_setzero_ps(); // TEMPORARY
   const __m512i offsets = _mm512_set4_epi32(3,2,1,0);  // original
   const __m512i four = _mm512_set4_epi32(4,4,4,4); 
    //printf("after subdomain\n");


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
#endif
} // omp parallel
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
void ELL_OPENMP_HOST<T>::checkAllSerialSolutions(T* data_t, int* col_id_t, T* vec_vt, T* one_res, int nz, int nb_mat, int nb_vec, int nb_rows)
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
void ELL_OPENMP_HOST<T>::spmv_serial(ell_matrix<int, T>& mat, std::vector<T>& v, std::vector<T>& result)
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
void ELL_OPENMP_HOST<T>::spmv_serial_row(ell_matrix<int, T>& mat, std::vector<T>& v, std::vector<T>& result)
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

    std::fill(result.begin(), result.end(), (T) 0.);

    std::vector<int> col_id_t(col_id.size());
    std::vector<T> data_t(data.size());

    // Transpose rows and columns
    for (int row=0; row < nb_rows; row++) {
        for (int n=0; n < nz; n++) {
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
void ELL_OPENMP_HOST<T>::spmv_serial_row(ell_matrix<int, T>& mat, std::vector<T>& data, std::vector<T>& v, std::vector<T>& result)
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

    std::fill(result.begin(), result.end(), (T) 0.);

    std::vector<int> col_id_t(col_id.size());
    std::vector<T> data_t(data.size());

    // Transpose rows and columns
    for (int row=0; row < nb_rows; row++) {
        for (int n=0; n < nz; n++) {
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
void ELL_OPENMP_HOST<T>::spmv_serial_row(int* col_id, T* data, T* v, T* res, int nbz, int nb_rows) 
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
T ELL_OPENMP_HOST<T>::l2norm(std::vector<T>& v)
{
    T norm = (T) 0.;
    for (int i=0; i < v.size(); i++) {
            norm += v[i]*v[i];
    }
    return (T) sqrt(norm);
}
//----------------------------------------------------------------------
template <typename T>
T ELL_OPENMP_HOST<T>::l2norm(T* v, int n)
{
    T norm = (T) 0.;
    for (int i=0; i < n; i++) {
            norm += v[i]*v[i];
    }
    return (T) sqrt(norm);
}
//----------------------------------------------------------------------
template <typename T>
void ELL_OPENMP_HOST<T>::fill_random(ell_matrix<int, T>& mat, std::vector<T>& v)
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
#if 0
template <typename T>
__m512 ELL_OPENMP_HOST<T>::tensor_product(float* a, float* b)
{
        __m512 va = read_aaaa(a);
        __m512 vb = read_abcd(b);
        return _mm512_mul_ps(va, vb);
}
#endif
//----------------------------------------------------------------------
#if 0
template <typename T>
__m512 ELL_OPENMP_HOST<T>::permute(__m512 v1, _MM_PERM_ENUM perm)
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
#endif
//----------------------------------------------------------------------
#if 0
template <typename T>
__m512i ELL_OPENMP_HOST<T>::read_aaaa(int* a)
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
#endif
//----------------------------------------------------------------------
#if 0
template <typename T>
__m512 ELL_OPENMP_HOST<T>::read_aaaa(float* a)
{
    // only works with floats
    // read in 4 floats (a,b,c,d) and create the 
    // 16-float vector dddd,cccc,bbbb,aaaa

    const int int_mask_lo = (1 << 0) + (1 << 4) + (1 << 8) + (1 << 12);
    __mmask16 mask_lo = _mm512_int2mask(int_mask_lo);
    __m512 v1_old;
    v1_old = _mm512_setzero_ps();
    v1_old = _mm512_mask_loadunpacklo_ps(v1_old, mask_lo, a);
    v1_old = _mm512_mask_loadunpackhi_ps(v1_old, mask_lo, a);
    v1_old = _mm512_swizzle_ps(v1_old, _MM_SWIZ_REG_AAAA);
    return v1_old;
}
#endif
//----------------------------------------------------------------------
template <typename T>
void ELL_OPENMP_HOST<T>::print_f(float* res, const std::string msg)
{
    printf("--- %s ---\n", msg.c_str());
    for (int i=0; i < 8; i++) { printf("(%d,%f), ", i, res[i]); }
    printf("\n");
    for (int i=8; i < 16; i++) { printf("(%d,%f), ", i, res[i]); }
    printf("\n\n");
}
//----------------------------------------------------------------------
template <typename T>
void ELL_OPENMP_HOST<T>::print_i(int* res, const std::string msg)
{
    printf("--- %s ---\n", msg.c_str());
    for (int i=0; i < 8; i++) { printf("(%d,%d), ", i, res[i]); }
    printf("\n");
    for (int i=8; i < 16; i++) { printf("(%d,%d), ", i, res[i]); }
    printf("\n\n");
}
//----------------------------------------------------------------------
#if 0
template <typename T>
void ELL_OPENMP_HOST<T>::print_ps(const __m512 v1, const std::string msg)
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
#endif
//----------------------------------------------------------------------
#if 0
template <typename T>
void ELL_OPENMP_HOST<T>::print_epi32(const __m512i v1, const std::string msg)
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
#endif
//----------------------------------------------------------------------
#if 0
template <typename T>
__m512 ELL_OPENMP_HOST<T>::read_abcd(float* a)
{
    // read in 4 floats (a,b,c,d) and create the 
    // 16-float vector dcba,dcba,dcba,dcba

    __m512 v1_old = _mm512_extload_ps(a, _MM_UPCONV_PS_NONE, _MM_BROADCAST_4X16, _MM_HINT_NONE);
    return v1_old;
}
#endif
//----------------------------------------------------------------------
template <typename T>
void ELL_OPENMP_HOST<T>::generate_ell_matrix_by_row(std::vector<int>& col_id, std::vector<T>& data, int nb_elem)
{
    // successive nonzeros are contiguous in memory. This is different than the standard
    // ordering for col_id
}
//----------------------------------------------------------------------
template <typename T>
void ELL_OPENMP_HOST<T>::generate_ell_matrix_data(T* data, int nbz, int nb_rows, int nb_mat)
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
void ELL_OPENMP_HOST<T>::generate_vector(T* vec, int nb_rows, int nb_vec)
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
void ELL_OPENMP_HOST<T>::generate_col_id_multi(int* col_id, int nbz, int nb_rows)
{
}
//----------------------------------------------------------------------
template <typename T>
void ELL_OPENMP_HOST<T>::generate_col_id(int* col_id, int nbz, int nb_rows)
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
void ELL_OPENMP_HOST<T>::retrieve_vector(T* vec, T* retrieved, int vec_id, int nb_vec, int nb_rows)
//void ELL_OPENMP_HOST<T>::retrieve_vector(std::vector<T>& vec, std::vector<T>& retrieved, int vec_id, int nb_vec)
{
    // vec is stored with vec_id as the fastest varying index

    for (int i=0; i < nb_rows; i++) {
        retrieved[i] = vec[vec_id + nb_vec*i];
    }
}
//----------------------------------------------------------------------
template <typename T>
void ELL_OPENMP_HOST<T>::retrieve_data(T* data_in, T* data_out, int mat_id, int nbz, int nb_rows, int nb_mat)
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
void spmv_ell_openmp_host(std::string filename)
{
    int ntimes = 10;
    printf("host\n");

#if 0
    if (filename.find("subdomain") != std::string::npos) {
        printf("multidomain file\n");
        int multi = 1;
        ELL_OPENMP_HOST<T> ell_ocl(filename, ntimes, multi); 
        // multidomain read
        ;
    }
#endif

    // input file type is in the configuration file (test.conf)
    ELL_OPENMP_HOST<T> ell_ocl(filename, ntimes);
	ell_ocl.run();
}
#endif
//----------------------------------------------------------------------
template <typename T>
//void spmv_ell_openmp(coo_matrix<int, T>* coo_mat, int dim2Size, int ntimes)
void spmv_ell_openmp_host(std::vector<int>& col_id, int& nb_rows, int& stencil_size)
{
    printf("spmv_ell_openmp: handle ELLPACK MATRIX\n");
	ELL_OPENMP_HOST<T> ell_ocl(col_id, nb_rows, stencil_size);
	ell_ocl.run();
}

template <typename T>
void spmv_ell_openmp_host(coo_matrix<int, T>* coo_mat, int dim2Size, int ntimes)
{

	printf("** GORDON, spmv_ell\n");
	ELL_OPENMP_HOST<T> ell_ocl(coo_mat, dim2Size, ntimes);
	ell_ocl.run();

	printf("GORDON: after ell_ocl.run\n");

	double opttime = ell_ocl.getOptTime();
	int optmethod = ell_ocl.getOptMethod();

	printf("\n------------------------------------------------------------------------\n");
	printf("ELL_OPENMP_HOST best time %f ms best method %d", opttime*1000.0, optmethod);
	printf("\n------------------------------------------------------------------------\n");
}
//----------------------------------------------------------------------
template <typename T>
void ELL_OPENMP_HOST<T>::generateInputMatricesAndVectors()
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
void ELL_OPENMP_HOST<T>::generateInputMatricesAndVectorsMulti()
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
void ELL_OPENMP_HOST<T>::freeInputMatricesAndVectors()
{
    _mm_free(result_vt);
    _mm_free(vec_vt);
    _mm_free(data_t);
    _mm_free(col_id_t);
}
//----------------------------------------------------------------------
template <typename T>
void ELL_OPENMP_HOST<T>::freeInputMatricesAndVectorsMulti()
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
void ELL_OPENMP_HOST<T>::checkSolutions()
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
void ELL_OPENMP_HOST<T>::bandwidthQ(Subdomain& s, int nb_rows, int stencil_size, int vec_size)
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
void ELL_OPENMP_HOST<T>::bandwidth(int* col_id, int nb_rows, int stencil_size, int vec_size)
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
//----------------------------------------------------------------------
}; // namespace spmv

#endif
//----------------------------------------------------------------------
