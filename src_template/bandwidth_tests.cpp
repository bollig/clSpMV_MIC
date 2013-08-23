#include "projectsettings.h"

namespace spmv {

class MemoryBandwidth
{
public:
    float gflops;
    float max_gflops = 0.;
    float max_bandwidth = 0.;
    float elapsed = 0.; 
    float min_elapsed = 0.; 
    int nb_rows = rd.nb_rows;
    //int nz = rd.stencil_size;
    int nb_rows;
    int col_id_type;
    int experiment;

public:
    void MemoryBandwidth(int nb_rows)
    {
	    max_gflops = 0.;
	    max_bandwidth = 0.;
	    elapsed = 0.; 
	    min_elapsed = 0.; 
	    this->nb_rows = nb_rows;

	    ProjectSettings pj("test.conf");
	    std::string experiment_s = REQUIRED<std::string>("experiment");
	    std::string col_id_type_s = REQUIRED<std::string>("col_id_type");

        if (experiment_s == ....) {
        } else if (experiment_s == ...) {
        } else if (experiment_s == ...) {
        } else {
            printf("experiment not implemented\n");
            exit(0);
        }

        if (col_id_type_s_s == ....) {
        } else if (col_id_type_s_s == ...) {
        } else if (col_id_type_s_s == ...) {
        } else {
            printf("col_id_type_s not implemented\n");
            exit(0);
        }
    }
    //----------------------------------------------------------------------
    void MemoryBandwidth::initialize()
    {
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
    }
    //----------------------------------------------------------------------
    void MemoryBandwidth::free()
    {
        _mm_free(vec_vt);
        _mm_free(result_vt);
        _mm_free(col_id_t);
    }
    //----------------------------------------------------------------------
	void MemoryBandwidth::method_rd_wr(int nbit)
	{
		printf("============== METHOD RD/WR ===================\n");
	    printf("Implement streaming\n");
	
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
}; // class
}; // namespace spmv
//----------------------------------------------------------------------
