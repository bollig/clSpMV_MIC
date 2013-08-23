#ifndef _BANDWIDTH_TESTS_H_
#define _BANDWIDTH_TESTS_H_

#include <omp.h>
#include <string>
#include "timer_eb.h"
#include "projectsettings.h"
#include <immintrin.h>


namespace spmv {

class MemoryBandwidth
{
private:
    ProjectSettings* pj;

public:
    float gflops;
    float max_gflops = 0.;
    float max_bandwidth = 0.;
    float elapsed = 0.; 
    float min_elapsed = 0.; 
    int nb_rows;
    //int nz = rd.stencil_size;
    std::string col_id_type_s;
    std::string experiment_s;
    int nb_iter;
    EB::TimerList tm; // timers


    float* vec_vt;
    float* result_vt;
    int* col_id_t;

public:
    MemoryBandwidth(int nb_rows);
    //~MemoryBandwidth(int nb_rows);
    void initialize();
    void free();
    void run();
    void benchRead();
    void benchWrite();
    void benchReadWrite();
    void benchGather();
    void benchGatherCpp();
    void benchUnpack();
    void benchReadWriteCpp();

private:
    __m512  permute(__m512 v1, _MM_PERM_ENUM perm);
    __m512i read_aaaa(int* a);
    __m512  read_aaaa(float* a);

}; // class
}; // namespace

#endif 
//_BANDWIDTH_TESTS_H_
