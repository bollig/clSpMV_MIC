#include <stdio.h>
#include <omp.h>
#include <malloc.h>
#include <immintrin.h> 

//----------------------------------------------------------------------
//  Read 4 floats and copy them to four other lanes
//  trick: __m512i _mm512_castps_si512(__m512 IN)
//  Use _mm512_shuffle_epi32(__m512i v2, _MM_PERM_ENUM permute)
//  permute =  255  (all lanes the smae
__m512 permute(float* a)
{
    __m512 v1 = _mm512_load_ps(a);
    __m512i vi = _mm512_castps_si512(v1);
    //vi = _mm512_shuffle_epi32(vi, _MM_PERM_AAAA);
    vi = _mm512_permute4f128_epi32(vi, _MM_PERM_AAAA);
    // shuffle is like a swizzle
    v1 = _mm512_castsi512_ps(vi);
    return v1;
}
//----------------------------------------------------------------------
__m512 read_aaaa(float* a)
{
    // read in 4 floats (a,b,c,d) and create the 
    // 16-float vector dddd,cccc,bbbb,aaaa

    int int_mask_lo = (1 << 0) + (1 << 4) + (1 << 8) + (1 << 12);
    __mmask16 mask_lo = _mm512_int2mask(int_mask_lo);
    __m512 v1_old;
    v1_old = _mm512_setzero_ps();
    v1_old = _mm512_mask_loadunpacklo_ps(v1_old, mask_lo, a);
    v1_old = _mm512_swizzle_ps(v1_old, _MM_SWIZ_REG_AAAA);
    return v1_old;
}
//----------------------------------------------------------------------
__m512 read_abcd(float* a)
{
    // read in 4 floats (a,b,c,d) and create the 
    // 16-float vector dcba,dcba,dcba,dcba

    __m512 v1_old = _mm512_extload_ps(a, _MM_UPCONV_PS_NONE, _MM_BROADCAST_4X16, _MM_HINT_NONE);
    return v1_old;
}
//----------------------------------------------------------------------
//void init_array()
int main()
{
    float* a = (float*) _mm_malloc(sizeof(float)*100, 64);
    float* b = (float*) _mm_malloc(sizeof(float)*100, 64);
    float* c = (float*) _mm_malloc(sizeof(float)*100, 64);

    for (int i=0; i < 32; i++) {
        a[i] = 1.+i;
        b[i] = 11.+i;
    }

    __m512 p = permute(a+4);
    _mm512_store_ps(c, p);
    for (int i=0; i < 16; i++) {
        printf("c[%d]= %f\n", i, c[i]);
    }
    return(0);

    __m512 v = read_aaaa(a);
    __m512 w = read_abcd(b);
    __m512 prod = _mm512_mul_ps(v,w);
    _mm512_store_ps(c, prod);

    for (int i=0; i < 16*2; i++) {
        printf("c[%d]= %f\n", i, c[i]);
    }

    _mm_free(a);
    _mm_free(b);
    _mm_free(c);

    return(0);
}
//----------------------------------------------------------------------

