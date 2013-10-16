#ifndef _BANDWIDTH_TESTS_HOST_H_
#define _BANDWIDTH_TESTS_HOST_H_

#include <omp.h>
#include <string>
#include <vector>
#include "timer_eb.h"
#include "projectsettings.h"
#include <immintrin.h>
#include <malloc.h>


namespace spmv {

class MemoryBandwidthHost
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
    int nb_bytes_per_row;


    float* vec_vt;
    float* result_vt;
    int* col_id_t;

public:
    MemoryBandwidthHost();
    //~MemoryBandwidthHost(int nb_rows);
    void setNbRows(int nb_rows_) { this->nb_rows = nb_rows_; }
    void initialize();
    void free();
    void run();
    void benchRead();
    void benchReadCpp();
    void benchWrite();
    void benchWriteCpp();
    void benchReadWrite();
    //void benchReadWriteCppAlone();
    //void benchGather();
    //void benchGatherCpp();
    //void benchUnpack();
    void benchReadWriteCpp();

private:
    __m512  permute(__m512 v1, _MM_PERM_ENUM perm);
    __m512i read_aaaa(int* a);
    __m512  read_aaaa(float* a);

}; // class
}; // namespace

#endif 
//_BANDWIDTH_TESTS_HOST_H_
