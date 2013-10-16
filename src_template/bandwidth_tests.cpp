#include "bandwidth_tests.h"
#include "timer_eb.h"
#include <vector>
#include <algorithm>

namespace spmv {

//----------------------------------------------------------------------
MemoryBandwidth::MemoryBandwidth()
{
    max_gflops = 0.;
    max_bandwidth = 0.;
    elapsed = 0.; 
    min_elapsed = 0.; 

    int offset = 3; // ignore first offset measurements
    tm["spmv"] = new EB::Timer("memory benchmarks", offset);
    tm["spmv1"] = new EB::Timer("readWriteAlone", offset);

    pj = new ProjectSettings("./test.conf");
 
	// Parse file created by python script
	// Uncomment if not using a script
	bool use_script = REQUIRED<bool>("USE_PYTHON_SCRIPT");
    printf("use_script: %d\n", (int) use_script);

	if (use_script) {
    	pj->ParseFile("bench.conf");
	}

    //pj = new ProjectSettings("/mnt/global/LCSE/gerlebacher/mic/test.conf");
    experiment_s  = REQUIRED<std::string>("bandwidth_experiment");
    col_id_type_s = REQUIRED<std::string>("col_id_type");
    std::string rr = REQUIRED<std::string>("nb_rows");
    sscanf(rr.c_str(), "%d", &this->nb_rows);
    printf("constructor: nb_rows= %d\n", this->nb_rows);
    delete pj;
}
//----------------------------------------------------------------------
void MemoryBandwidth::free()
{
    _mm_free(vec_vt);
    _mm_free(result_vt);
    _mm_free(col_id_t);
}
//----------------------------------------------------------------------
void MemoryBandwidth::initialize()
{
    printf("initialize, entered\n");
    printf("initialize: nb rows: %d\n", nb_rows);
    vec_vt = (float*) _mm_malloc(sizeof(float)*nb_rows, 64);
    result_vt = (float*) _mm_malloc(sizeof(float)*nb_rows, 64);
    col_id_t = (int*) _mm_malloc(sizeof(int)*nb_rows, 64);
    nb_iter = 10;

    if (col_id_type_s == "compact") {
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
    printf("end initialize: nb rows: %d\n", nb_rows);
}
//----------------------------------------------------------------------
void MemoryBandwidth::run()
{
    printf("*** run: nb_rows= %d\n", nb_rows);

    //omp_set_num_threads(num_threads[i]);
    //omp_set_num_threads(244);
    max_bandwidth = 0.; 
    min_elapsed = 0.; 
    nb_bytes_per_row = 0;

    std::vector<float> bw;
    bw.resize(0);

    for (int i=0; i < nb_iter; i++) {
        //tm["spmv"]->start();
        //benchReadWriteCpp();
#if 1
        if (experiment_s == "write") benchWrite();
        else if (experiment_s == "write_cpp") benchWriteCpp();
        else if (experiment_s == "read_write_cpp") benchReadWriteCpp();
        else if (experiment_s == "read_write_cpp_alone") { benchReadWriteCppAlone();} // exit(0); }
        else if (experiment_s == "read") benchRead();
        else if (experiment_s == "read_cpp") benchReadCpp();
        else if (experiment_s == "read_write") benchReadWrite();
        else if (experiment_s == "gather") benchGather();
        else if (experiment_s == "unpack") benchUnpack();
        else if (experiment_s == "gather_cpp") benchGatherCpp();
        else { printf("experiment not implemented\n"); exit(0); }
#endif
        //tm["spmv"]->end();
        if (i < 3) continue;

        float elapsed = 0.; 
        float max_flops = 0.; 
        float gflops;
        elapsed = tm["spmv"]->getTime();
        float gbytes = nb_rows * nb_bytes_per_row * 1.e-9;
        float bandwidth = gbytes / (elapsed*1.e-3);
        bw.push_back(bandwidth);

    }  // for loop

    std::sort(bw.begin(), bw.end());
    int sz = bw.size();
    //for (int i=0; i < 7; i++) { printf("bw[%d] = %f\n", i, bw[i]); }
    // 2nd highest bandwidth (in case highest one is an outlier)
    //printf("r/w, max bandwidth= %f (gbytes/sec)\n", bw[sz-1]);
    printf("rows: %d (x 128^3), max bandwidth= %f (gbytes/sec)\n", nb_rows/(128*128*128), bw[sz-2]);
    printf("nb_rows: %d,mx_bw: %f\n", nb_rows/(128*128*128), bw[sz-2]);
}
//----------------------------------------------------------------------
void MemoryBandwidth::benchWriteCpp()
{
    //printf("== benchReadCpp ==\n");
    nb_bytes_per_row = 4;
    __assume_aligned(result_vt, 64);
    __assume_aligned(vec_vt, 64);

    tm["spmv"]->start();
#pragma omp parallel
{
    float sum = 5.;
    const int nr = nb_rows;
#pragma ivdep
#pragma omp for
   for (int i=0; i < nr; i++) {
        result_vt[i] = sum;
    }
} // omp parallel
     tm["spmv"]->end();
}
//----------------------------------------------------------------------
//----------------------------------------------------------------------
void MemoryBandwidth::benchReadCpp()
{
    //printf("== benchReadCpp ==\n");
    nb_bytes_per_row = 4;
    __assume_aligned(result_vt, 64);
    __assume_aligned(vec_vt, 64);

    tm["spmv"]->start();
#pragma omp parallel
{
    float sum;
    const int nr = nb_rows;
#pragma ivdep
#pragma omp for
   for (int i=0; i < nr; i++) {
        sum += vec_vt[i];
        //_mm512_storenrngo_ps(result_vt+i, sum);
    }
   result_vt[0] = sum;
} // omp parallel
     tm["spmv"]->end();
}
//----------------------------------------------------------------------
void MemoryBandwidth::benchRead()
{
    //printf("== benchRead ==\n");
    nb_bytes_per_row = 4;

    tm["spmv"]->start();
#pragma omp parallel
{
    int nr = nb_rows;
    __m512 sum = _mm512_setzero_ps();

// Unroll loop to speed up.  TRY IT OUT. 
#pragma omp for
   for (int i=0; i < nr; i += 16) {
        sum = _mm512_add_ps(_mm512_load_ps(vec_vt+i), sum);
        //_mm512_storenrngo_ps(result_vt+i, sum);
    }
    _mm512_storenrngo_ps(result_vt, sum);
} // omp parallel
     tm["spmv"]->end();
}
//----------------------------------------------------------------------
void MemoryBandwidth::benchWrite()
{
    //printf("== benchWrite ==\n");
    nb_bytes_per_row = 4;

//firstprivate does not work with members of this class
    tm["spmv"]->start();
#pragma omp parallel 
{
    int nr = nb_rows;
    __m512 sumwr = _mm512_setzero_ps();

#pragma omp for
   for (int i=0; i < nr; i += 16) {
        _mm512_storenrngo_ps(result_vt+i, sumwr);
        //_mm512_store_ps(result_vt+i, sumwr);
    }
} // omp parallel
    tm["spmv"]->end();
}
//----------------------------------------------------------------------
void MemoryBandwidth::benchReadWrite()
{
    //printf("== benchReadWrite ==\n");
    nb_bytes_per_row = 8;

    tm["spmv"]->start();
#pragma omp parallel
{
    int nr = nb_rows;

#pragma omp for
   for (int i=0; i < nr; i += 16) {
        const __m512 v1_old = _mm512_load_ps(vec_vt + i);
        _mm512_storenrngo_ps(result_vt+i, v1_old);
        //_mm512_store_ps(result_vt+i, v1_old);
    }
}  // omp parallel
    tm["spmv"]->end();
}
//----------------------------------------------------------------------
void MemoryBandwidth::benchReadWriteCppAlone()
{
    //bw.resize(0);
    nb_bytes_per_row = 8;
    int nr = nb_rows;
    __assume_aligned(result_vt, 64);
    __assume_aligned(vec_vt, 64);
    //printf("-- benchReadWriteCppAlone() --\n");

    //for (int i=0; i < 10; i++) {
    
    tm["spmv"]->start();
#pragma omp parallel firstprivate(nr)
{
#pragma ivdep
#pragma omp for 
   for (int r=0; r < nr; r++) {
        result_vt[r] = vec_vt[r];
   }
} // omp parallel
    tm["spmv"]->end();

#if 0
        tm["spmv1"]->end();
        if (i < 3) continue;
        float elapsed = tm["spmv1"]->getTime();
        float gbytes = nr * nb_bytes_per_row * 1.e-9;
        float bandwidth = gbytes / (elapsed*1.e-3);
        bw.push_back(bandwidth);
    }
    int sz = bw.size();
    std::sort(bw.begin(), bw.end());
    // print 2nd maximum bandwidth to avoid outliers
    printf("r/w alone, max bandwidth= %f (gbytes/sec)\n", bw[sz-2]);
#endif
}
//----------------------------------------------------------------------
void MemoryBandwidth::benchReadWriteCpp()
{
    //printf("== benchReadWriteCpp ==\n");
    nb_bytes_per_row = 8;
    const int nr = nb_rows;
    __assume_aligned(result_vt, 64);
    __assume_aligned(vec_vt, 64);

    tm["spmv"]->start();
#pragma omp parallel firstprivate(nr)
{
#pragma ivdep
#pragma omp for
   for (int r=0; r < nr; r++) {
        //float a = vec_vt[r];
        //result_vt[r] = a; // did not help
        result_vt[r] = vec_vt[r];
   }
}  // omp parallel

     tm["spmv"]->end();
#if 0
        //float elapsed = tm["spmv"]->getTime();
        //float gbytes = nr * nb_bytes_per_row * 1.e-9;
        //float bandwidth = gbytes / (elapsed*1.e-3);
    // print 2nd maximum bandwidth to avoid outliers
    //printf("xxx alone, max bandwidth= %f (gbytes/sec)\n", bandwidth);
#endif
}
//----------------------------------------------------------------------
void MemoryBandwidth::benchGatherCpp()
{
    //printf("== benchGatherCpp ==\n");
    nb_bytes_per_row = 12;
    __assume_aligned(result_vt, 64);
    __assume_aligned(vec_vt, 64);

    tm["spmv"]->start();
#pragma omp parallel
{
    int nr = nb_rows;
#pragma ivdep
#pragma omp for
       for (int r=0; r < nr; r++) {
            result_vt[r] = vec_vt[col_id_t[r]];
        }
}  // omp parallel
     tm["spmv"]->end();
}
//----------------------------------------------------------------------
void MemoryBandwidth::benchUnpack()
{
    //printf("== benchUnpack ==\n");
    nb_bytes_per_row = 8;

    tm["spmv"]->start();
#pragma omp parallel
{
    int nr = nb_rows;
   __m512 v3_old;
#pragma omp for
    for (int r=0; r < nr; r+=16) {
        // retrieve 4 elements per read_aaaa, expand to 16 elements via masking
        v3_old = read_aaaa(vec_vt+r);
        // use addition to ensure that dead code is not eliminated
        v3_old = _mm512_add_ps(read_aaaa(vec_vt+r+4), v3_old);
        v3_old = _mm512_add_ps(read_aaaa(vec_vt+r+8), v3_old);
        v3_old = _mm512_add_ps(read_aaaa(vec_vt+r+12), v3_old);
        _mm512_storenrngo_ps(result_vt+r, v3_old);
    }
}  // omp parallel
     tm["spmv"]->end();
}
//----------------------------------------------------------------------
void MemoryBandwidth::benchGather()
{
    //printf("== benchGather ==\n");
    nb_bytes_per_row = 12;

    tm["spmv"]->start();
#pragma omp parallel
{
    int nr = nb_rows;
    const __m512i offsets = _mm512_set4_epi32(3,2,1,0);  // original
    //const __m512i four = _mm512_set4_epi32(4,4,4,4); 
    const __m512i four = _mm512_set4_epi32(1,1,1,1); // only one vector 
    __m512i v3_oldi;
    __m512  v = _mm512_setzero_ps();
    const int scale = 4;

// There will be differences depending on the matrix type. Create specialized col_id_t matrix for this experiment. 

#pragma omp for
    for (int i=0; i < nr; i+=16) {
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
        //_mm512_store_ps(result_vt+i, v); 
        // 138Gflops without the gathers
    }
} // omp parallel
     tm["spmv"]->end();
} // benchGather
//----------------------------------------------------------------------




//----------------------------------------------------------------------
#if 0
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
#endif
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

}; // end namespace

int varyRows()
{
    //int nbr[] = {1,2,4,8,16,32,64};
    //int nbr[] = {1,2,4,8,16,32,64};
    std::vector<int> nb_rows(7); 
    nb_rows.resize(0);
    int count=0;
    for (int i=2; i < 66; i+=2) {
        nb_rows.push_back(i);  
        count++;
    }
    //std::copy(nbr, nbr+count, nb_rows.begin());
    spmv::MemoryBandwidth mem;
    int mult = 128*128*128;

    for (int i=0; i < nb_rows.size(); i++) {
        printf("--------------------------------\n");
        printf("i= %d, rows: %d\n" , i), nb_rows[i];
        mem.setNbRows(mult*nb_rows[i]);
        mem.initialize();
        mem.run();
        mem.free();
    }
    return(0);
}

//----------------------------------------------------------------------
int main()
{
    // different tests
    varyRows();
    return(0);
}
