
template <typename T>
void ELL_OPENMP<T>::method_8a_base_bflops(int nbit)
{
// Single matrix, single vector, base case. 
// Measure raw speed by increasing number of calculations to overwhelm communcation costs. 
// Make sure all data can reside in cache. Use supercompact, which will help in this regard. 

	printf("============== METHOD 8a Base Gflops, %d domain  ===================\n", nb_subdomains);
    method_name = "method_8a_base_gflops";
    printf("nb subdomains= %d\n", nb_subdomains);
    printf("nb_rows_multi = %d\n", nb_rows_multi[0]);


    generateInputMatricesAndVectorsBase(64);
    // vectorize across rows requires a transpose
    int nz = rd.stencil_size;
    transpose1(subdomains[0].data_t, rd.nb_rows, nz); // nz varies fastest
    transpose1(subdomains[0].col_id_t, rd.nb_rows, nz); // nz varies fastest

    float gflops;
    float max_gflops = 0.;
    float elapsed = 0.; 
    float min_elapsed = 0.; 
    int nb_repeats = 2*32;
    //int nb_rows = rd.nb_rows;
    printf("*** nb_subdomains: %d\n", nb_subdomains);

    // Must now work on alignmentf vectors. 
    // Produces the correct serial result
    for (int it=0; it < 10; it++) {
      tm["spmv"]->start();
      for (int s=0; s < nb_subdomains; s++) {

#pragma omp parallel firstprivate(nz, nb_repeats)
{
    const int nb_rows = nb_rows_multi[s];
    //const int scale = 4; // seg fault if 1 (WHY?)
    const int nz = rd.stencil_size;
    const Subdomain& dom = subdomains[s];

    //__m512 v3_oldi;
    //__m512  v = _mm512_setzero_ps(); // TEMPORARY
    __m512 v1_old = _mm512_setzero_ps();
    __m512  accu;

    //printf("nb_repeats= %d\n", nb_repeats);
    //printf("nb_rows= %d\n", nb_rows);


#pragma omp for 
        for (int r=0; r < nb_rows; r+=16) {
            accu = _mm512_setzero_ps(); // 16 floats for 16 rows

#pragma simd
                for (int n=0; n < nz; n++) {  // nz is multiple of 32 (for now)
                    v1_old = _mm512_load_ps(dom.data_t       + (r+n*nb_rows)); // load 16 rows at column n
            for (int iter=0; iter < nb_repeats; iter++) {
                    //v3_oldi = _mm512_load_epi32(dom.col_id_t + (r+n*nb_rows)); // ERROR SOMEHOW!
                    //v  = _mm512_i32gather_ps(v3_oldi, dom.vec_vt, scale); // scale = 4 bytes (floats) HOW DOES SCALE WORK?
                    accu = _mm512_fmadd_ps(v1_old, v1_old, accu);
                    accu = _mm512_fmadd_ps(v1_old, v1_old, accu);
                    accu = _mm512_fmadd_ps(v1_old, v1_old, accu);
                    accu = _mm512_fmadd_ps(v1_old, v1_old, accu);
#if 0
                    accu = _mm512_fmadd_ps(v1_old, v1_old, accu);
                    accu = _mm512_fmadd_ps(v1_old, v1_old, accu);
                    accu = _mm512_fmadd_ps(v1_old, v1_old, accu);
                    accu = _mm512_fmadd_ps(v1_old, v1_old, accu);
#endif
                }
            }
            _mm512_storenrngo_ps(dom.result_vt+r, accu);  
        } 
} // omp parallel

      }
        tm["spmv"]->end();  // time for each matrix/vector multiply
        if (it < 3) continue;
        elapsed = tm["spmv"]->getTime();
        // nb_rows is wrong. 
        //printf("%d, %d, %d, %d\n", rd.nb_mats, rd.nb_vecs, rd.stencil_size, rd.nb_rows);
        gflops = 4*nb_repeats * 2.*rd.stencil_size*rd.nb_rows*1e-9 / (1e-3*elapsed); // assumes count of 1
        printf("%f gflops, %f (ms)\n", gflops, elapsed);
        if (gflops > max_gflops) {
            max_gflops = gflops;
            min_elapsed = elapsed;
        }
   }

    printf("%s, threads: %d, Max Gflops: %f, min time: %f (ms)\n", method_name.c_str(), num_threads, max_gflops, min_elapsed);
    freeInputMatricesAndVectorsMulti();
}
//----------------------------------------------------------------------
