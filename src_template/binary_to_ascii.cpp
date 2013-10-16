//TODO (disable from Makefile)
//FIX	two_vec_compare_T(coores, tmpresult, rownum);

//#ifndef __CLASS_SPMV_ELL_OPENMP_H__
//#define __CLASS_SPMV_ELL_OPENMP_H__

#if 0
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
#endif

#include "rbffd_io.h"

#if 0
#include "runs.h"
//#include "rcm.h"
#include "burkardt_rcm.hpp"

//#include "reorder.h"
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
    int num_threads; // controlled from test.conf
    std::string method_name;

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
    virtual void runSingleAlgo();
    void gather_test(int do_transpose);
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
    int setNbThreads(int nb_threads); // set the variable num_threads
	void method_8a(int nb=0); // 4 matrices, 4 vectors
    void method_8a_multi(int nb=0);
    void method_8a_base(int nbit);
    void method_8a_base_bflops(int nbit);
    void method_8a_multi_novec(int nb=0);
    void transpose1(float* mat, int nr_slow, int nc_fast);
    void transpose1(int* mat, int nr_slow, int nc_fast);
    void transpose1(float* mat, float* mat_t, int nr_slow, int nc_fast);
    void transpose1(int* mat, int* mat_t, int nr_slow, int nc_fast);

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
    void retrieve_data_base(T* data_in, T* data_out, int mat_id, int nbz, int nb_rows);
    void checkAllSerialSolutions(T* data_t, int* col_id_t, T* vec_vt, T* one_res, int nz, int nb_mat, int nb_vec, int nb_rows);
    void checkAllSerialSolutionsBase(T* data_t, int* col_id_t, T* vec_vt, T* one_res, int nz, int nb_mat, int nb_vec, int nb_rows);
    void generateInputMatricesAndVectors();
    void generateInputMatricesAndVectorsBase(int align=16); // for col_id
    void generateInputMatricesAndVectorsMulti();
    void freeInputMatricesAndVectors();
    void freeInputMatricesAndVectorsMulti();
    void checkSolutions();
    void checkSolutionsBase();
    void checkSolutionsNoVec();
    void bandwidth(int* col_id, int nb_rows, int stencil_size, int vec_size);
    void bandwidthQ(Subdomain& s, int nb_rows, int stencil_size, int vec_size);
    void permInverse3(int n, int* perm, int* perm_inv);
    int adj_bandwidth(int node_num, int adj_num, int* adj_row, int* adj);
    int adj_perm_bandwidth(int node_num, int adj_num, int* adj_row, int* adj, int* perm, int* perm_inv);
    int i4_max (int i1, int i2);
    //void freeInputMatricesAndVectors(float* vec_vt, float* result_vt, int* col_id_t, float* data_t);
    //void generateInputMatricesAndVectors(float* vec_vt, float* result_vt, int* col_id_t, float* data_t);
    void cuthillMcKee(std::vector<int>& col_id, int nb_rows, int stencil_size);
    void cuthillMcKee_new(std::vector<int>& col_id, int nb_rows, int stencil_size);

    // functions meant to treat the case of 4mat/4vec running in straighforward mode, where
    // data (weights) are ordered 4 matrix el 1, 4 matrix el2, etc.
    //  In the functions above, the order is more complex for supposedly faster benchmarks. 
    void generateInputMatricesAndVectorsMultiNoVec();
    void generate_ell_matrix_data_novec(T* data, int nbz, int nb_rows, int nb_mat);
    void retrieve_data_novec(T* data_in, T* data_out, int mat_id, int nbz, int nb_rows, int nb_mat);
    void checkAllSerialSolutionsNoVec(T* data_t, int* col_id_t, T* vec_vt, T* one_res, int nz, int nb_mat, int nb_vec, int nb_rows);

protected:
	virtual void method_0(int nb=0);
	virtual void method_1(int nb=0);
	virtual void method_2(int nb=0);
};

//----------------------------------------------------------------------
template <typename T>
void ELL_OPENMP<T>::runSingleAlgo()
{
    // 244 on S2, 240 on S3
    int num_threads_v[] = {1,2,4,8,16,32,64,96,128,160,192,224,240};
    int max_threads = 240;
    int nb_slots = 13;
    //nb_slots = 1; num_s_v[0] = 244; // for testing

    int count = 0;
    int max_nb_runs = 1;

    //for (int i=0; i < nb_slots; i++) {
    for (int i=0; i < 1; i++) {
        //setNbThreads(num_threads_v[i]);
        setNbThreads(max_threads);
        method_8a_multi(4);
    }
    return;

    exit(0);
}
//----------------------------------------------------------------------
template <typename T>
void ELL_OPENMP<T>::run()
{
	runSingleAlgo();  // single problem size, single algorithm
    exit(0);


    // 244 on S2, 240 on S3
    int num_threads_v[] = {1,2,4,8,16,32,64,96,128,160,192,224,240};
    int nb_slots = 13;
    //nb_slots = 1; num_threads_v[0] = 244; // for testing

    int count = 0;
    int max_nb_runs = 1;

#if 0
    for (int i=0; i < nb_slots; i++) {
        printf("i= %d\n", i);
        setNbThreads(num_threads_v[i]);
        //setNbThreads(240);
        method_8a_base_bflops(4);
        //exit(0);
    }
    exit(0);
#endif

#if 0
    setNbThreads(1);
    setNbThreads(244);
    method_8a_base(4);
    exit(0);
#endif

    for (int i=0; i < nb_slots; i++) {
        setNbThreads(num_threads_v[i]);
        method_8a_multi_novec(4);
    }
    for (int i=0; i < nb_slots; i++) {
        setNbThreads(num_threads_v[i]);
        method_8a_multi(4);
    }
    for (int i=0; i < nb_slots; i++) {
        setNbThreads(num_threads_v[i]);
        method_8a_base(4);
    }
    return;


    if (rd.use_subdomains == 0 || nb_subdomains == 1) {
        //method_8a(4);
        method_8a_multi(4); // temporary? 
    } else {
        method_8a_multi(4);
    }
    method_8a_multi_novec(4);   // exact solution not correct. 
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
#endif

int main()
{
    std::vector<int> ell_col_id;
    //std::string filename = "matrix/ell_kd-tree_rcm_sym_1_x_weights_direct__no_hv_stsize_32_3d_8x_8y_8z.bmtx";
    //std::string filename = "matrix/ell_kd-tree_x_weights_direct__no_hv_stsize_32_3d_8x_8y_8z.bmtx";
    std::string filename = "ell_kd-tree_rcm_sym_0_x_weights_direct__no_hv_stsize_32_2d_23x_23y_1z.bmtx";
    //std::string filename = "ell_kd-tree_x_weights_direct__no_hv_stsize_32_2d_23x_23y_1z.bmtx";

	RBFFD_IO<float> io;
    int stencil_size;
    int nb_rows;
    io.loadFromBinaryEllpackFile(ell_col_id, nb_rows, stencil_size, filename);
    filename = filename + ".asc";
    FILE* fd = fopen(filename.c_str(), "w");
    fprintf(fd, "%d %d\n", nb_rows, stencil_size);
    for (int i=0; i < nb_rows; i++) {
        for (int j=0; j < stencil_size; j++) {
            fprintf(fd, "%d ", ell_col_id[i*stencil_size+j]);
        }
        fprintf(fd, "\n");
    }
    return(0);
}
