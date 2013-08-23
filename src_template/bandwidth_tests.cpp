#include "bandwidth_tests.h"
#include "timer_eb.h"

namespace spmv {

//----------------------------------------------------------------------
MemoryBandwidth::MemoryBandwidth(int nb_rows)
{
    max_gflops = 0.;
    max_bandwidth = 0.;
    elapsed = 0.; 
    min_elapsed = 0.; 
    this->nb_rows = nb_rows;

    ProjectSettings pj("test.conf");
    experiment_s  = REQUIRED<std::string>("experiment");
    col_id_type_s = REQUIRED<std::string>("col_id_type");
}
//----------------------------------------------------------------------
void MemoryBandwidth::initialize()
{
    vec_vt = (float*) _mm_malloc(sizeof(float)*nb_rows, 64);
    result_vt = (float*) _mm_malloc(sizeof(float)*nb_rows, 64);
    col_id_t = (int*) _mm_malloc(sizeof(int)*nb_rows, 64);
    nb_iter = 10;

    if (co_id_type_s == "compact") {
        for (int i=0; i < nb_rows; i++) {
            col_id_t[i] = i;
        }
    }

    else if (col_id_type_s == "reverse") {
        for (int i=0; i < nb_rows; i++) {
            col_id_t[i] = nb_rows-i-1;
        }
    }

    else if (col_id_type_s == "random") {
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
    }
}
//----------------------------------------------------------------------
void MemoryBandwidth::run()
{
    for (int i=0; i < nb_iter; i++) {
        tm["spmv"]->start();
#pragma omp parallel
  {
#pragma omp master
        if (experiment_s == "write") benchWrite();
        else if (experiment_s == "read") benchRead();
        else if (experiment_s == "read_write") benchReadWrite();
        else if (experiment_s == "gather") benchGather();
        else if (experiment_s == "unpack") benchUnpack();
        else if (experiment_s == "read_cpp") benchReadCpp();
        else if (experiment_s == "write_cpp") benchWriteCpp();
        else if (experiment_s == "read_write_cpp") benchReadWriteCpp();
        else if (experiment_s == "gather_cpp") benchGatherCpp();
  }
        tm["spmv"]->end();
    }

    float elapsed = 0.; 
    float min_elapsed = 0.; 
    float max_bandwidth = 0.; 
    float max_flops = 0.; 
    float gflops;
    elapsed = tm["spmv"]->getTime();
    float gbytes = nb_rows*sizeof(float) * 2 * 1.e-9;
    float bandwidth = gbytes / (elapsed*1.e-3);

    if (bandwidth > max_bandwidth) {
        max_bandwidth = bandwidth;
        min_elapsed = elapsed;
    }

    printf("r/w, max bandwidth= %f (gbytes/sec), time: %f (ms)\n", max_bandwidth, min_elapsed);
}
//----------------------------------------------------------------------
void MemoryBandwidth::benchRead()
{
    __m512 sum = _mm512_setzero_ps();

#pragma omp for
   for (int i=0; i  < nb_rows; i += 16) {
        sum = _mm512_add_ps(_mm512_load_ps(vec_vt+i), sum);
        //_mm512_storenrngo_ps(result_vt+i, sum);
    }
    _mm512_storenrngo_ps(result_vt, sum);
}
//----------------------------------------------------------------------
void MemoryBandwidth::benchWrite()
{
    __m512 sumwr = _mm512_setzero_ps();

#pragma omp for
   for (int i=0; i  < nb_rows; i += 16) {
        _mm512_storenrngo_ps(result_vt+i, sumwr);
    }
}
//----------------------------------------------------------------------
void MemoryBandwidth::benchReadWrite()
{
#pragma omp for
   for (int i=0; i  < nb_rows; i += 16) {
        const __m512 v1_old = _mm512_load_ps(vec_vt + i);
        _mm512_storenrngo_ps(result_vt+i, v1_old);
    }
}
//----------------------------------------------------------------------
void MemoryBandwidth::benchCpp()
{
#pragma omp for
       for (int r=0; r  < nb_rows; r++) {
#pragma SIMD
            result_vt[r] = vec_vt[r];
        }
}
//----------------------------------------------------------------------
void MemoryBandwidth::benchCppGather()
{
#pragma omp for
       for (int r=0; r  < nb_rows; r++) {
#pragma SIMD
            result_vt[r] = vec_vt[col_id_t[r]];
        }
}
//----------------------------------------------------------------------
void MemoryBandwidth::benchCppUnpack()
{
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
}
//----------------------------------------------------------------------
void MemoryBandwidth::benchGather()
{
    const __m512i offsets = _mm512_set4_epi32(3,2,1,0);  // original
    //const __m512i four = _mm512_set4_epi32(4,4,4,4); 
    const __m512i four = _mm512_set4_epi32(1,1,1,1); // only one vector 
    __m512i v3_oldi;
    __m512  v = _mm512_setzero_ps();
    const int scale = 4;

// There will be differences depending on the matrix type. Create specialized col_id_t matrix for this experiment. 

#pragma omp for
    for (int i=0; i < nb_rows; i+=16) {
         v3_oldi = read_aaaa(&col_id_t[0] + i);    // ERROR: dom not known
         v3_oldi = _mm512_fmadd_epi32(v3_oldi, four, offsets); // offsets?

         v     = _mm512_i32gather_ps(v3_oldi, vec_vt, scale); // scale = 4 bytes (floats)

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
//----------------------------------------------------------------------




__m512 MemoryBandwidth::tensor_product(float* a, float* b)
{
        __m512 va = read_aaaa(a);
        __m512 vb = read_abcd(b);
        return _mm512_mul_ps(va, vb);
}
//----------------------------------------------------------------------
__m512 MemoryBandwidth::permute(__m512 v1, _MM_PERM_ENUM perm)
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
__m512i MemoryBandwidth::read_aaaa(int* a)
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
__m512 MemoryBandwidth::read_aaaa(float* a)
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
//----------------------------------------------------------------------

int main()
{
    int nb_rows = 128*128*128*16;
    MemoryBandwidth mem(nb_rows);
    mem.initialize();
    mem.run();
    mem.free();
}
