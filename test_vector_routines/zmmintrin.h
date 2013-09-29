/**
*** Copyright (C) 2007-2012 Intel Corporation.  All rights reserved.
***
*** The information and source code contained herein is the exclusive
*** property of Intel Corporation and may not be disclosed, examined
*** or reproduced in whole or in part without explicit written authorization
*** from the company.
***
**/

#ifndef _ZMMINTRIN_H_INCLUDED
#define _ZMMINTRIN_H_INCLUDED

#ifndef _INCLUDED_IMM
#error "Header should only be included from <immintrin.h>."
#endif

/*
 * Definitions and declarations for use with 512-bit compiler intrinsics.
 */

/*
 *
 * A word about intrinsic naming conventions.  Most 512-bit vector
 * instructions have names such as v<operation><type>.  For example
 * "vaddps" is an addition operation (add) on packed single precision (ps)
 * values.  The corresponding intrinsic is usually (not always) named
 * "_mm512_<operation>_<type>", for example _mm512_add_ps.  The corresponding
 * write-masked flavor has "_mask" in the name, _mm512_mask_add_ps.
 *
 * The types are:
 *
 *    ps    -- packed single precision
 *    pd    -- packed double precision
 *    epi32 -- packed 32-bit integers
 *    epu32 -- packed 32-bit unsigned integers
 *    epi64 -- packed 64-bit integers
 */

typedef unsigned char   __mmask8;
typedef unsigned short  __mmask16;
/*
 * __mmask is deprecated, use __mmask16 instead.
 */
typedef __mmask16 __mmask;

#if !defined(__INTEL_COMPILER) && defined(_MSC_VER)
# define _MM512INTRIN_TYPE(X) __declspec(intrin_type)
#else
# define _MM512INTRIN_TYPE(X) _MMINTRIN_TYPE(X)
#endif

typedef union _MM512INTRIN_TYPE(64) __m512 {
    float       __m512_f32[16];
} __m512;

typedef union _MM512INTRIN_TYPE(64) __m512d {
    double      __m512d_f64[8];
} __m512d;

typedef union _MM512INTRIN_TYPE(64) __m512i {
    int         __m512i_i32[16];
} __m512i;


#ifdef __cplusplus
extern "C" {
/* Intrinsics use C name-mangling. */
#endif /* __cplusplus */

/* Conversion from one type to another, no change in value. */

extern __m512  __ICL_INTRINCC _mm512_castpd_ps(__m512d);
extern __m512i __ICL_INTRINCC _mm512_castpd_si512(__m512d);
extern __m512d __ICL_INTRINCC _mm512_castps_pd(__m512);
extern __m512i __ICL_INTRINCC _mm512_castps_si512(__m512);
extern __m512  __ICL_INTRINCC _mm512_castsi512_ps(__m512i);
extern __m512d __ICL_INTRINCC _mm512_castsi512_pd(__m512i);


/* Constant for special read-only mask register 'k0'. */
#define _MM_K0_REG (0xffff)


/* Constants for register swizzle primitives. */
typedef enum {
    _MM_SWIZ_REG_NONE,      /* hgfe dcba - Nop */
#define _MM_SWIZ_REG_DCBA _MM_SWIZ_REG_NONE
    _MM_SWIZ_REG_CDAB,      /* ghef cdab - Swap pairs */
    _MM_SWIZ_REG_BADC,      /* fehg badc - Swap with two-away */
    _MM_SWIZ_REG_AAAA,      /* eeee aaaa - broadcast a element */
    _MM_SWIZ_REG_BBBB,      /* ffff bbbb - broadcast b element */
    _MM_SWIZ_REG_CCCC,      /* gggg cccc - broadcast c element */
    _MM_SWIZ_REG_DDDD,      /* hhhh dddd - broadcast d element */
    _MM_SWIZ_REG_DACB       /* hegf dacb - cross-product */
} _MM_SWIZZLE_ENUM;

/* Constants for broadcasts to vectors with 32-bit elements. */
typedef enum {
    _MM_BROADCAST32_NONE,   /* identity swizzle/convert */
#define _MM_BROADCAST_16X16 _MM_BROADCAST32_NONE
    _MM_BROADCAST_1X16,     /* broadcast x 16 ( aaaa aaaa aaaa aaaa ) */
    _MM_BROADCAST_4X16      /* broadcast x 4  ( dcba dcba dcba dcba ) */
} _MM_BROADCAST32_ENUM;

/* Constants for broadcasts to vectors with 64-bit elements. */
typedef enum {
    _MM_BROADCAST64_NONE,   /* identity swizzle/convert */
#define _MM_BROADCAST_8X8 _MM_BROADCAST64_NONE
    _MM_BROADCAST_1X8,      /* broadcast x 8 ( aaaa aaaa ) */
    _MM_BROADCAST_4X8       /* broadcast x 2 ( dcba dcba ) */
} _MM_BROADCAST64_ENUM;

/*
 * Constants for rounding mode.
 * These names beginnig with "_MM_ROUND" are deprecated.
 * Use the names beginning with "_MM_FROUND" going forward.
 */
typedef enum {
    _MM_ROUND_MODE_NEAREST,             /* round to nearest (even) */
    _MM_ROUND_MODE_DOWN,                /* round toward negative infinity */
    _MM_ROUND_MODE_UP,                  /* round toward positive infinity */
    _MM_ROUND_MODE_TOWARD_ZERO,         /* round toward zero */
    _MM_ROUND_MODE_DEFAULT              /* round mode from MXCSR */
} _MM_ROUND_MODE_ENUM;

/* Constants for exponent adjustment. */
typedef enum {
    _MM_EXPADJ_NONE,               /* 2**0  (32.0 - no exp adjustment) */
    _MM_EXPADJ_4,                  /* 2**4  (28.4)  */
    _MM_EXPADJ_5,                  /* 2**5  (27.5)  */
    _MM_EXPADJ_8,                  /* 2**8  (24.8)  */
    _MM_EXPADJ_16,                 /* 2**16 (16.16) */
    _MM_EXPADJ_24,                 /* 2**24 (8.24)  */
    _MM_EXPADJ_31,                 /* 2**31 (1.31)  */
    _MM_EXPADJ_32                  /* 2**32 (0.32)  */
} _MM_EXP_ADJ_ENUM;

/* Constants for index scale (vgather/vscatter). */
typedef enum {
    _MM_SCALE_1 = 1,
    _MM_SCALE_2 = 2,
    _MM_SCALE_4 = 4,
    _MM_SCALE_8 = 8
} _MM_INDEX_SCALE_ENUM;

/*
 * Constants for load/store temporal hints.
 */
#define _MM_HINT_NONE           0x0
#define _MM_HINT_NT             0x1     /* Load or store is non-temporal. */

typedef enum {
    _MM_PERM_AAAA = 0x00, _MM_PERM_AAAB = 0x01, _MM_PERM_AAAC = 0x02,
    _MM_PERM_AAAD = 0x03, _MM_PERM_AABA = 0x04, _MM_PERM_AABB = 0x05,
    _MM_PERM_AABC = 0x06, _MM_PERM_AABD = 0x07, _MM_PERM_AACA = 0x08,
    _MM_PERM_AACB = 0x09, _MM_PERM_AACC = 0x0A, _MM_PERM_AACD = 0x0B,
    _MM_PERM_AADA = 0x0C, _MM_PERM_AADB = 0x0D, _MM_PERM_AADC = 0x0E,
    _MM_PERM_AADD = 0x0F, _MM_PERM_ABAA = 0x10, _MM_PERM_ABAB = 0x11,
    _MM_PERM_ABAC = 0x12, _MM_PERM_ABAD = 0x13, _MM_PERM_ABBA = 0x14,
    _MM_PERM_ABBB = 0x15, _MM_PERM_ABBC = 0x16, _MM_PERM_ABBD = 0x17,
    _MM_PERM_ABCA = 0x18, _MM_PERM_ABCB = 0x19, _MM_PERM_ABCC = 0x1A,
    _MM_PERM_ABCD = 0x1B, _MM_PERM_ABDA = 0x1C, _MM_PERM_ABDB = 0x1D,
    _MM_PERM_ABDC = 0x1E, _MM_PERM_ABDD = 0x1F, _MM_PERM_ACAA = 0x20,
    _MM_PERM_ACAB = 0x21, _MM_PERM_ACAC = 0x22, _MM_PERM_ACAD = 0x23,
    _MM_PERM_ACBA = 0x24, _MM_PERM_ACBB = 0x25, _MM_PERM_ACBC = 0x26,
    _MM_PERM_ACBD = 0x27, _MM_PERM_ACCA = 0x28, _MM_PERM_ACCB = 0x29,
    _MM_PERM_ACCC = 0x2A, _MM_PERM_ACCD = 0x2B, _MM_PERM_ACDA = 0x2C,
    _MM_PERM_ACDB = 0x2D, _MM_PERM_ACDC = 0x2E, _MM_PERM_ACDD = 0x2F,
    _MM_PERM_ADAA = 0x30, _MM_PERM_ADAB = 0x31, _MM_PERM_ADAC = 0x32,
    _MM_PERM_ADAD = 0x33, _MM_PERM_ADBA = 0x34, _MM_PERM_ADBB = 0x35,
    _MM_PERM_ADBC = 0x36, _MM_PERM_ADBD = 0x37, _MM_PERM_ADCA = 0x38,
    _MM_PERM_ADCB = 0x39, _MM_PERM_ADCC = 0x3A, _MM_PERM_ADCD = 0x3B,
    _MM_PERM_ADDA = 0x3C, _MM_PERM_ADDB = 0x3D, _MM_PERM_ADDC = 0x3E,
    _MM_PERM_ADDD = 0x3F, _MM_PERM_BAAA = 0x40, _MM_PERM_BAAB = 0x41,
    _MM_PERM_BAAC = 0x42, _MM_PERM_BAAD = 0x43, _MM_PERM_BABA = 0x44,
    _MM_PERM_BABB = 0x45, _MM_PERM_BABC = 0x46, _MM_PERM_BABD = 0x47,
    _MM_PERM_BACA = 0x48, _MM_PERM_BACB = 0x49, _MM_PERM_BACC = 0x4A,
    _MM_PERM_BACD = 0x4B, _MM_PERM_BADA = 0x4C, _MM_PERM_BADB = 0x4D,
    _MM_PERM_BADC = 0x4E, _MM_PERM_BADD = 0x4F, _MM_PERM_BBAA = 0x50,
    _MM_PERM_BBAB = 0x51, _MM_PERM_BBAC = 0x52, _MM_PERM_BBAD = 0x53,
    _MM_PERM_BBBA = 0x54, _MM_PERM_BBBB = 0x55, _MM_PERM_BBBC = 0x56,
    _MM_PERM_BBBD = 0x57, _MM_PERM_BBCA = 0x58, _MM_PERM_BBCB = 0x59,
    _MM_PERM_BBCC = 0x5A, _MM_PERM_BBCD = 0x5B, _MM_PERM_BBDA = 0x5C,
    _MM_PERM_BBDB = 0x5D, _MM_PERM_BBDC = 0x5E, _MM_PERM_BBDD = 0x5F,
    _MM_PERM_BCAA = 0x60, _MM_PERM_BCAB = 0x61, _MM_PERM_BCAC = 0x62,
    _MM_PERM_BCAD = 0x63, _MM_PERM_BCBA = 0x64, _MM_PERM_BCBB = 0x65,
    _MM_PERM_BCBC = 0x66, _MM_PERM_BCBD = 0x67, _MM_PERM_BCCA = 0x68,
    _MM_PERM_BCCB = 0x69, _MM_PERM_BCCC = 0x6A, _MM_PERM_BCCD = 0x6B,
    _MM_PERM_BCDA = 0x6C, _MM_PERM_BCDB = 0x6D, _MM_PERM_BCDC = 0x6E,
    _MM_PERM_BCDD = 0x6F, _MM_PERM_BDAA = 0x70, _MM_PERM_BDAB = 0x71,
    _MM_PERM_BDAC = 0x72, _MM_PERM_BDAD = 0x73, _MM_PERM_BDBA = 0x74,
    _MM_PERM_BDBB = 0x75, _MM_PERM_BDBC = 0x76, _MM_PERM_BDBD = 0x77,
    _MM_PERM_BDCA = 0x78, _MM_PERM_BDCB = 0x79, _MM_PERM_BDCC = 0x7A,
    _MM_PERM_BDCD = 0x7B, _MM_PERM_BDDA = 0x7C, _MM_PERM_BDDB = 0x7D,
    _MM_PERM_BDDC = 0x7E, _MM_PERM_BDDD = 0x7F, _MM_PERM_CAAA = 0x80,
    _MM_PERM_CAAB = 0x81, _MM_PERM_CAAC = 0x82, _MM_PERM_CAAD = 0x83,
    _MM_PERM_CABA = 0x84, _MM_PERM_CABB = 0x85, _MM_PERM_CABC = 0x86,
    _MM_PERM_CABD = 0x87, _MM_PERM_CACA = 0x88, _MM_PERM_CACB = 0x89,
    _MM_PERM_CACC = 0x8A, _MM_PERM_CACD = 0x8B, _MM_PERM_CADA = 0x8C,
    _MM_PERM_CADB = 0x8D, _MM_PERM_CADC = 0x8E, _MM_PERM_CADD = 0x8F,
    _MM_PERM_CBAA = 0x90, _MM_PERM_CBAB = 0x91, _MM_PERM_CBAC = 0x92,
    _MM_PERM_CBAD = 0x93, _MM_PERM_CBBA = 0x94, _MM_PERM_CBBB = 0x95,
    _MM_PERM_CBBC = 0x96, _MM_PERM_CBBD = 0x97, _MM_PERM_CBCA = 0x98,
    _MM_PERM_CBCB = 0x99, _MM_PERM_CBCC = 0x9A, _MM_PERM_CBCD = 0x9B,
    _MM_PERM_CBDA = 0x9C, _MM_PERM_CBDB = 0x9D, _MM_PERM_CBDC = 0x9E,
    _MM_PERM_CBDD = 0x9F, _MM_PERM_CCAA = 0xA0, _MM_PERM_CCAB = 0xA1,
    _MM_PERM_CCAC = 0xA2, _MM_PERM_CCAD = 0xA3, _MM_PERM_CCBA = 0xA4,
    _MM_PERM_CCBB = 0xA5, _MM_PERM_CCBC = 0xA6, _MM_PERM_CCBD = 0xA7,
    _MM_PERM_CCCA = 0xA8, _MM_PERM_CCCB = 0xA9, _MM_PERM_CCCC = 0xAA,
    _MM_PERM_CCCD = 0xAB, _MM_PERM_CCDA = 0xAC, _MM_PERM_CCDB = 0xAD,
    _MM_PERM_CCDC = 0xAE, _MM_PERM_CCDD = 0xAF, _MM_PERM_CDAA = 0xB0,
    _MM_PERM_CDAB = 0xB1, _MM_PERM_CDAC = 0xB2, _MM_PERM_CDAD = 0xB3,
    _MM_PERM_CDBA = 0xB4, _MM_PERM_CDBB = 0xB5, _MM_PERM_CDBC = 0xB6,
    _MM_PERM_CDBD = 0xB7, _MM_PERM_CDCA = 0xB8, _MM_PERM_CDCB = 0xB9,
    _MM_PERM_CDCC = 0xBA, _MM_PERM_CDCD = 0xBB, _MM_PERM_CDDA = 0xBC,
    _MM_PERM_CDDB = 0xBD, _MM_PERM_CDDC = 0xBE, _MM_PERM_CDDD = 0xBF,
    _MM_PERM_DAAA = 0xC0, _MM_PERM_DAAB = 0xC1, _MM_PERM_DAAC = 0xC2,
    _MM_PERM_DAAD = 0xC3, _MM_PERM_DABA = 0xC4, _MM_PERM_DABB = 0xC5,
    _MM_PERM_DABC = 0xC6, _MM_PERM_DABD = 0xC7, _MM_PERM_DACA = 0xC8,
    _MM_PERM_DACB = 0xC9, _MM_PERM_DACC = 0xCA, _MM_PERM_DACD = 0xCB,
    _MM_PERM_DADA = 0xCC, _MM_PERM_DADB = 0xCD, _MM_PERM_DADC = 0xCE,
    _MM_PERM_DADD = 0xCF, _MM_PERM_DBAA = 0xD0, _MM_PERM_DBAB = 0xD1,
    _MM_PERM_DBAC = 0xD2, _MM_PERM_DBAD = 0xD3, _MM_PERM_DBBA = 0xD4,
    _MM_PERM_DBBB = 0xD5, _MM_PERM_DBBC = 0xD6, _MM_PERM_DBBD = 0xD7,
    _MM_PERM_DBCA = 0xD8, _MM_PERM_DBCB = 0xD9, _MM_PERM_DBCC = 0xDA,
    _MM_PERM_DBCD = 0xDB, _MM_PERM_DBDA = 0xDC, _MM_PERM_DBDB = 0xDD,
    _MM_PERM_DBDC = 0xDE, _MM_PERM_DBDD = 0xDF, _MM_PERM_DCAA = 0xE0,
    _MM_PERM_DCAB = 0xE1, _MM_PERM_DCAC = 0xE2, _MM_PERM_DCAD = 0xE3,
    _MM_PERM_DCBA = 0xE4, _MM_PERM_DCBB = 0xE5, _MM_PERM_DCBC = 0xE6,
    _MM_PERM_DCBD = 0xE7, _MM_PERM_DCCA = 0xE8, _MM_PERM_DCCB = 0xE9,
    _MM_PERM_DCCC = 0xEA, _MM_PERM_DCCD = 0xEB, _MM_PERM_DCDA = 0xEC,
    _MM_PERM_DCDB = 0xED, _MM_PERM_DCDC = 0xEE, _MM_PERM_DCDD = 0xEF,
    _MM_PERM_DDAA = 0xF0, _MM_PERM_DDAB = 0xF1, _MM_PERM_DDAC = 0xF2,
    _MM_PERM_DDAD = 0xF3, _MM_PERM_DDBA = 0xF4, _MM_PERM_DDBB = 0xF5,
    _MM_PERM_DDBC = 0xF6, _MM_PERM_DDBD = 0xF7, _MM_PERM_DDCA = 0xF8,
    _MM_PERM_DDCB = 0xF9, _MM_PERM_DDCC = 0xFA, _MM_PERM_DDCD = 0xFB,
    _MM_PERM_DDDA = 0xFC, _MM_PERM_DDDB = 0xFD, _MM_PERM_DDDC = 0xFE,
    _MM_PERM_DDDD = 0xFF
} _MM_PERM_ENUM;

/*
 * Helper type and macro for computing the values of the immediate
 * used in mm512_fixup_ps.
 */
typedef enum {
    _MM_FIXUP_NO_CHANGE,
    _MM_FIXUP_NEG_INF,
    _MM_FIXUP_NEG_ZERO,
    _MM_FIXUP_POS_ZERO,
    _MM_FIXUP_POS_INF,
    _MM_FIXUP_NAN,
    _MM_FIXUP_MAX_FLOAT,
    _MM_FIXUP_MIN_FLOAT
} _MM_FIXUPRESULT_ENUM;

#define _MM_FIXUP(_NegInf, \
                  _Neg, \
                  _NegZero, \
                  _PosZero, \
                  _Pos, \
                  _PosInf, \
                  _Nan) \
   ((int) (_NegInf) | \
   ((int) (_Neg) << 3) | \
   ((int) (_NegZero) << 6) | \
   ((int) (_PosZero) << 9) | \
   ((int) (_Pos) << 12) | \
   ((int) (_PosInf) << 15) | \
   ((int) (_Nan) << 18))


/*
 * Write-masked vector copy.
 */
extern __m512  __ICL_INTRINCC _mm512_mask_mov_ps(__m512, __mmask16, __m512);
extern __m512d __ICL_INTRINCC _mm512_mask_mov_pd(__m512d, __mmask8, __m512d);

#define _mm512_mask_mov_epi32(v_old, k1, src) \
    _mm512_mask_swizzle_epi32((v_old), (k1), (src), _MM_SWIZ_REG_NONE)

#define _mm512_mask_mov_epi64(v_old, k1, src) \
    _mm512_mask_swizzle_epi64((v_old), (k1), (src), _MM_SWIZ_REG_NONE)


/* Constants for upconversion to packed single precision. */

typedef enum {

    _MM_UPCONV_PS_NONE,         /* no conversion      */
    _MM_UPCONV_PS_FLOAT16,      /* float16 => float32 */
    _MM_UPCONV_PS_UINT8,        /* uint8   => float32 */
    _MM_UPCONV_PS_SINT8,        /* sint8   => float32 */
    _MM_UPCONV_PS_UINT16,       /* uint16  => float32 */
    _MM_UPCONV_PS_SINT16        /* sint16  => float32 */


} _MM_UPCONV_PS_ENUM;

extern __m512 __ICL_INTRINCC _mm512_extload_ps(void const*,
                                               _MM_UPCONV_PS_ENUM,
                                               _MM_BROADCAST32_ENUM,
                                               int /* mem hint */);
extern __m512 __ICL_INTRINCC _mm512_mask_extload_ps(__m512, __mmask16,
                                                    void const*,
                                                    _MM_UPCONV_PS_ENUM,
                                                    _MM_BROADCAST32_ENUM,
                                                    int /* mem hint */);

extern __m512 __ICL_INTRINCC _mm512_load_ps(void const*);
extern __m512 __ICL_INTRINCC _mm512_mask_load_ps(__m512, __mmask16,
                                                 void const*);


/* Constants for upconversion to packed 32-bit integers. */

typedef enum {

    _MM_UPCONV_EPI32_NONE,      /* no conversion      */
    _MM_UPCONV_EPI32_UINT8,     /* uint8   => uint32  */
    _MM_UPCONV_EPI32_SINT8,     /* sint8   => sint32  */
    _MM_UPCONV_EPI32_UINT16,    /* uint16  => uint32  */
    _MM_UPCONV_EPI32_SINT16     /* sint16  => sint32  */

} _MM_UPCONV_EPI32_ENUM;

extern __m512i __ICL_INTRINCC _mm512_extload_epi32(void const*,
                                                   _MM_UPCONV_EPI32_ENUM,
                                                   _MM_BROADCAST32_ENUM,
                                                   int /* mem hint */);
extern __m512i __ICL_INTRINCC _mm512_mask_extload_epi32(__m512i, __mmask16,
                                                        void const*,
                                                        _MM_UPCONV_EPI32_ENUM,
                                                        _MM_BROADCAST32_ENUM,
                                                        int /* mem hint */);

extern __m512i __ICL_INTRINCC _mm512_load_epi32(void const*);
extern __m512i __ICL_INTRINCC _mm512_mask_load_epi32(__m512i, __mmask16,
                                                     void const*);

/* Constants for upconversion to packed double precision. */

typedef enum {
    _MM_UPCONV_PD_NONE          /* no conversion */
} _MM_UPCONV_PD_ENUM;

extern __m512d __ICL_INTRINCC _mm512_extload_pd(void const*,
                                                _MM_UPCONV_PD_ENUM,
                                                _MM_BROADCAST64_ENUM,
                                                int /* mem hint */);
extern __m512d __ICL_INTRINCC _mm512_mask_extload_pd(__m512d, __mmask8,
                                                     void const*,
                                                     _MM_UPCONV_PD_ENUM,
                                                     _MM_BROADCAST64_ENUM,
                                                     int /* mem hint */);

extern __m512d __ICL_INTRINCC _mm512_load_pd(void const*);
extern __m512d __ICL_INTRINCC _mm512_mask_load_pd(__m512d, __mmask8,
                                                  void const*);


/* Constants for upconversion to packed 64-bit integers. */

typedef enum {
    _MM_UPCONV_EPI64_NONE       /* no conversion */
} _MM_UPCONV_EPI64_ENUM;

extern __m512i __ICL_INTRINCC _mm512_extload_epi64(void const*,
                                                   _MM_UPCONV_EPI64_ENUM,
                                                   _MM_BROADCAST64_ENUM,
                                                   int /* mem hint */);

extern __m512i __ICL_INTRINCC _mm512_mask_extload_epi64(__m512i, __mmask8,
                                                        void const*,
                                                        _MM_UPCONV_EPI64_ENUM,
                                                        _MM_BROADCAST64_ENUM,
                                                        int /* mem hint */);

extern __m512i __ICL_INTRINCC _mm512_load_epi64(void const*);
extern __m512i __ICL_INTRINCC _mm512_mask_load_epi64(__m512i, __mmask8,
                                                     void const*);

/*
 * Swizzle/broadcast/upconversion operations.
 */
extern __m512  __ICL_INTRINCC _mm512_swizzle_ps(__m512, _MM_SWIZZLE_ENUM);
extern __m512d __ICL_INTRINCC _mm512_swizzle_pd(__m512d, _MM_SWIZZLE_ENUM);
extern __m512i __ICL_INTRINCC _mm512_swizzle_epi32(__m512i, _MM_SWIZZLE_ENUM);
extern __m512i __ICL_INTRINCC _mm512_swizzle_epi64(__m512i, _MM_SWIZZLE_ENUM);

extern __m512  __ICL_INTRINCC _mm512_mask_swizzle_ps(__m512, __mmask16,
                                                     __m512,
                                                     _MM_SWIZZLE_ENUM);
extern __m512d __ICL_INTRINCC _mm512_mask_swizzle_pd(__m512d, __mmask8,
                                                     __m512d,
                                                     _MM_SWIZZLE_ENUM);
extern __m512i __ICL_INTRINCC _mm512_mask_swizzle_epi32(__m512i, __mmask16,
                                                        __m512i,
                                                        _MM_SWIZZLE_ENUM);
extern __m512i __ICL_INTRINCC _mm512_mask_swizzle_epi64(__m512i, __mmask8,
                                                        __m512i,
                                                        _MM_SWIZZLE_ENUM);

/* Constants for downconversion from packed single precision. */

typedef enum {

    _MM_DOWNCONV_PS_NONE,         /* no conversion      */
    _MM_DOWNCONV_PS_FLOAT16,      /* float32 => float16 */
    _MM_DOWNCONV_PS_UINT8,        /* float32 => uint8   */
    _MM_DOWNCONV_PS_SINT8,        /* float32 => sint8   */
    _MM_DOWNCONV_PS_UINT16,       /* float32 => uint16  */
    _MM_DOWNCONV_PS_SINT16        /* float32 => sint16  */


} _MM_DOWNCONV_PS_ENUM;

/* Constants for downconversion from packed 32-bit integers. */

typedef enum {
    _MM_DOWNCONV_EPI32_NONE,      /* no conversion      */
    _MM_DOWNCONV_EPI32_UINT8,     /* uint32 => uint8    */
    _MM_DOWNCONV_EPI32_SINT8,     /* sint32 => sint8    */
    _MM_DOWNCONV_EPI32_UINT16,    /* uint32 => uint16   */
    _MM_DOWNCONV_EPI32_SINT16     /* sint32 => sint16   */
} _MM_DOWNCONV_EPI32_ENUM;

/* Constants for downconversion from packed double precision. */

typedef enum {
    _MM_DOWNCONV_PD_NONE          /* no conversion      */
} _MM_DOWNCONV_PD_ENUM;

/* Constants for downconversion from packed 64-bit integers. */

typedef enum {
    _MM_DOWNCONV_EPI64_NONE       /* no conversion      */
} _MM_DOWNCONV_EPI64_ENUM;

extern void __ICL_INTRINCC _mm512_extstore_ps(void*, __m512,
                                              _MM_DOWNCONV_PS_ENUM,
                                              int /* mem hint */);
extern void __ICL_INTRINCC _mm512_extstore_epi32(void*, __m512i,
                                                 _MM_DOWNCONV_EPI32_ENUM,
                                                 int /* mem hint */);
extern void __ICL_INTRINCC _mm512_extstore_pd(void*, __m512d,
                                              _MM_DOWNCONV_PD_ENUM,
                                              int /* mem hint */);
extern void __ICL_INTRINCC _mm512_extstore_epi64(void*, __m512i,
                                                 _MM_DOWNCONV_EPI64_ENUM,
                                                 int /* mem hint */);
extern void __ICL_INTRINCC _mm512_mask_extstore_ps(void*, __mmask16, __m512,
                                                   _MM_DOWNCONV_PS_ENUM,
                                                   int /* mem hint */);
extern void __ICL_INTRINCC _mm512_mask_extstore_pd(void*, __mmask8, __m512d,
                                                   _MM_DOWNCONV_PD_ENUM,
                                                   int /* mem hint */);
extern void __ICL_INTRINCC _mm512_mask_extstore_epi32(void*, __mmask16,
                                                      __m512i,
                                                      _MM_DOWNCONV_EPI32_ENUM,
                                                      int /* mem hint */);
extern void __ICL_INTRINCC _mm512_mask_extstore_epi64(void*, __mmask8, __m512i,
                                                      _MM_DOWNCONV_EPI64_ENUM,
                                                      int /* mem hint */);

extern void __ICL_INTRINCC _mm512_store_ps(void*, __m512);
extern void __ICL_INTRINCC _mm512_store_epi32(void*, __m512i);
extern void __ICL_INTRINCC _mm512_store_pd(void*, __m512d);
extern void __ICL_INTRINCC _mm512_store_epi64(void*, __m512i);
extern void __ICL_INTRINCC _mm512_mask_store_ps(void*, __mmask16, __m512);
extern void __ICL_INTRINCC _mm512_mask_store_pd(void*, __mmask8, __m512d);
extern void __ICL_INTRINCC _mm512_mask_store_epi32(void*, __mmask16, __m512i);
extern void __ICL_INTRINCC _mm512_mask_store_epi64(void*, __mmask8, __m512i);


/*
 * Store aligned float32/float64 vector with No-Read hint.
 */

extern void __ICL_INTRINCC _mm512_storenr_ps(void*, __m512);
extern void __ICL_INTRINCC _mm512_storenr_pd(void*, __m512d);

/*
 * Non-globally ordered store aligned float32/float64 vector with No-Read hint.
 */

extern void __ICL_INTRINCC _mm512_storenrngo_ps(void*, __m512);
extern void __ICL_INTRINCC _mm512_storenrngo_pd(void*, __m512d);

/*
 * Compute absolute value of the difference between float32 or int32 vectors.
 */

extern __m512 __ICL_INTRINCC _mm512_absdiff_round_ps(__m512, __m512,
                                                     int /* rounding */);
extern __m512 __ICL_INTRINCC _mm512_mask_absdiff_round_ps(__m512, __mmask16,
                                                          __m512, __m512,
                                                          int /* rounding */);
#define _mm512_absdiff_ps(v2, v3) \
    _mm512_absdiff_round_ps((v2), (v3), _MM_FROUND_CUR_DIRECTION)

#define _mm512_mask_absdiff_ps(v1_old, k1, v2, v3) \
    _mm512_mask_absdiff_round_ps((v1_old), (k1), (v2), (v3), \
                                 _MM_FROUND_CUR_DIRECTION)

extern __m512i __ICL_INTRINCC _mm512_absdiff_epi32(__m512i, __m512i);
extern __m512i __ICL_INTRINCC _mm512_mask_absdiff_epi32(__m512i, __mmask16,
                                                        __m512i, __m512i);

/*
 * Add int32 vectors with carry.
 * The carry of the sum is returned via the __mmask16 pointer.
 */
extern __m512i __ICL_INTRINCC _mm512_adc_epi32(__m512i, __mmask16, __m512i,
                                               __mmask16*);
extern __m512i __ICL_INTRINCC _mm512_mask_adc_epi32(__m512i, __mmask16,
                                                    __mmask16,
                                                    __m512i, __mmask16*);
/*
 * Add float32 or float64 vectors and negate the sum.
 */
extern __m512d __ICL_INTRINCC _mm512_addn_pd(__m512d, __m512d);
extern __m512d __ICL_INTRINCC _mm512_mask_addn_pd(__m512d, __mmask8,
                                                  __m512d, __m512d);
extern __m512  __ICL_INTRINCC _mm512_addn_ps(__m512, __m512);
extern __m512  __ICL_INTRINCC _mm512_mask_addn_ps(__m512, __mmask16,
                                                  __m512, __m512);

extern __m512d __ICL_INTRINCC _mm512_addn_round_pd(__m512d, __m512d,
                                                   int /* rounding */);
extern __m512d __ICL_INTRINCC _mm512_mask_addn_round_pd(__m512d, __mmask8,
                                                        __m512d, __m512d,
                                                        int /* rounding */);

extern __m512 __ICL_INTRINCC _mm512_addn_round_ps(__m512, __m512,
                                                  int /* rounding */);
extern __m512 __ICL_INTRINCC _mm512_mask_addn_round_ps(__m512, __mmask16,
                                                       __m512, __m512,
                                                       int /* rounding */);

/*
 * Add, subtract or multiply float64, float32, int64 or int32 vectors.
 */
extern __m512d __ICL_INTRINCC _mm512_add_pd(__m512d, __m512d);
extern __m512d __ICL_INTRINCC _mm512_mask_add_pd(__m512d, __mmask8,
                                                 __m512d, __m512d);
extern __m512  __ICL_INTRINCC _mm512_add_ps(__m512, __m512);
extern __m512  __ICL_INTRINCC _mm512_mask_add_ps(__m512, __mmask16,
                                                 __m512, __m512);

extern __m512d __ICL_INTRINCC _mm512_mul_pd(__m512d, __m512d);
extern __m512d __ICL_INTRINCC _mm512_mask_mul_pd(__m512d, __mmask8,
                                                 __m512d, __m512d);

extern __m512 __ICL_INTRINCC _mm512_mul_ps(__m512, __m512);
extern __m512 __ICL_INTRINCC _mm512_mask_mul_ps(__m512, __mmask16,
                                                __m512, __m512);

extern __m512d __ICL_INTRINCC _mm512_sub_pd(__m512d, __m512d);
extern __m512d __ICL_INTRINCC _mm512_mask_sub_pd(__m512d, __mmask8,
                                                 __m512d, __m512d);

extern __m512 __ICL_INTRINCC _mm512_sub_ps(__m512, __m512);
extern __m512 __ICL_INTRINCC _mm512_mask_sub_ps(__m512, __mmask16,
                                                __m512, __m512);

extern __m512d __ICL_INTRINCC _mm512_subr_pd(__m512d, __m512d);
extern __m512d __ICL_INTRINCC _mm512_mask_subr_pd(__m512d, __mmask8,
                                                  __m512d, __m512d);

extern __m512 __ICL_INTRINCC _mm512_subr_ps(__m512,__m512);
extern __m512 __ICL_INTRINCC _mm512_mask_subr_ps(__m512, __mmask16,
                                                 __m512, __m512);

extern __m512d __ICL_INTRINCC _mm512_add_round_pd(__m512d, __m512d,
                                                  int /* rounding */);
extern __m512d __ICL_INTRINCC _mm512_mask_add_round_pd(__m512d, __mmask8,
                                                       __m512d, __m512d,
                                                       int /* rounding */);

extern __m512 __ICL_INTRINCC _mm512_add_round_ps(__m512, __m512,
                                                 int /* rounding */);
extern __m512 __ICL_INTRINCC _mm512_mask_add_round_ps(__m512, __mmask16,
                                                      __m512, __m512,
                                                      int /* rounding */);

extern __m512i __ICL_INTRINCC _mm512_add_epi32(__m512i, __m512i);
extern __m512i __ICL_INTRINCC _mm512_mask_add_epi32(__m512i, __mmask16,
                                                    __m512i, __m512i);

extern __m512i __ICL_INTRINCC _mm512_add_epi64(__m512i, __m512i);
extern __m512i __ICL_INTRINCC _mm512_mask_add_epi64(__m512i, __mmask8,
                                                    __m512i, __m512i);

extern __m512d __ICL_INTRINCC _mm512_mul_round_pd(__m512d, __m512d,
                                                  int /* rounding */);
extern __m512d __ICL_INTRINCC _mm512_mask_mul_round_pd(__m512d, __mmask8,
                                                       __m512d, __m512d,
                                                       int /* rounding */);

extern __m512 __ICL_INTRINCC _mm512_mul_round_ps(__m512, __m512,
                                                 int /* rounding */);
extern __m512 __ICL_INTRINCC _mm512_mask_mul_round_ps(__m512, __mmask16,
                                                      __m512, __m512,
                                                      int /* rounding */);

extern __m512d __ICL_INTRINCC _mm512_sub_round_pd(__m512d, __m512d,
                                                  int /* rounding */);
extern __m512d __ICL_INTRINCC _mm512_mask_sub_round_pd(__m512d, __mmask8,
                                                       __m512d, __m512d,
                                                       int /* rounding */);

extern __m512 __ICL_INTRINCC _mm512_sub_round_ps(__m512, __m512,
                                                 int /* rounding */);
extern __m512 __ICL_INTRINCC _mm512_mask_sub_round_ps(__m512, __mmask16,
                                                      __m512, __m512,
                                                      int /* rounding */);

extern __m512i __ICL_INTRINCC _mm512_sub_epi32(__m512i, __m512i);
extern __m512i __ICL_INTRINCC _mm512_mask_sub_epi32(__m512i, __mmask16,
                                                    __m512i, __m512i);

extern __m512d __ICL_INTRINCC _mm512_subr_round_pd(__m512d, __m512d,
                                                   int /* rounding */);
extern __m512d __ICL_INTRINCC _mm512_mask_subr_round_pd(__m512d, __mmask8,
                                                        __m512d, __m512d,
                                                        int /* rounding */);

extern __m512 __ICL_INTRINCC _mm512_subr_round_ps(__m512, __m512,
                                                  int /* rounding */);
extern __m512 __ICL_INTRINCC _mm512_mask_subr_round_ps(__m512, __mmask16,
                                                       __m512, __m512,
                                                       int /* rounding */);

extern __m512i __ICL_INTRINCC _mm512_subr_epi32(__m512i, __m512i);
extern __m512i __ICL_INTRINCC _mm512_mask_subr_epi32(__m512i, __mmask16,
                                                     __m512i, __m512i);

/*
 * Add int32 vectors and set carry.
 * The carry from the sum is returned via the __mmask16 pointer.
 */
extern __m512i __ICL_INTRINCC _mm512_addsetc_epi32(__m512i, __m512i,
                                                   __mmask16*);
extern __m512i __ICL_INTRINCC _mm512_mask_addsetc_epi32(__m512i, __mmask16,
                                                        __mmask16, __m512i,
                                                        __mmask16*);

/*
 * Add int32 or float32 Vectors and Set Mask to Sign.  The sign of the result
 * for the n-th element is returned via the __mmask16 pointer.
 */
extern __m512i __ICL_INTRINCC _mm512_addsets_epi32(__m512i, __m512i,
                                                   __mmask16*);
extern __m512i __ICL_INTRINCC _mm512_mask_addsets_epi32(__m512i, __mmask16,
                                                        __m512i, __m512i,
                                                        __mmask16*);

extern __m512 __ICL_INTRINCC _mm512_addsets_ps(__m512, __m512, __mmask16*);
extern __m512 __ICL_INTRINCC _mm512_mask_addsets_ps(__m512, __mmask16,
                                                    __m512, __m512,
                                                    __mmask16*);

extern __m512 __ICL_INTRINCC _mm512_addsets_round_ps(__m512, __m512,
                                                     __mmask16*,
                                                     int /* rounding */);
extern __m512 __ICL_INTRINCC _mm512_mask_addsets_round_ps(__m512, __mmask16,
                                                          __m512, __m512,
                                                          __mmask16*,
                                                          int /* rounding */);

/*
 * Concatenate vectors, shift right by 'count' int32 elements,
 * and return the low 16 elements.
 */
extern __m512i __ICL_INTRINCC _mm512_alignr_epi32(__m512i, __m512i,
                                                  const int /* count */);
extern __m512i __ICL_INTRINCC _mm512_mask_alignr_epi32(__m512i, __mmask16,
                                                       __m512i, __m512i,
                                                       const int /* count */);
/*
 * Blending between two vectors.
 */
extern __m512i __ICL_INTRINCC _mm512_mask_blend_epi32(__mmask16, __m512i,
                                                      __m512i);
extern __m512i __ICL_INTRINCC _mm512_mask_blend_epi64(__mmask8, __m512i,
                                                      __m512i);
extern __m512  __ICL_INTRINCC _mm512_mask_blend_ps(__mmask16, __m512,
                                                   __m512);
extern __m512d __ICL_INTRINCC _mm512_mask_blend_pd(__mmask8, __m512d,
                                                   __m512d);

/*
 * Subtract int32 vectors and set borrow.
 * The borrow from the subtraction for the n-th element
 * is written into the n-th bit of vector mask, via the __mmask16 pointer.
 */
extern __m512i __ICL_INTRINCC _mm512_subsetb_epi32(__m512i, __m512i,
                                                   __mmask16*);
extern __m512i __ICL_INTRINCC _mm512_mask_subsetb_epi32(__m512i, __mmask16,
                                                        __mmask16, __m512i,
                                                        __mmask16*);

/*
 * Reverse subtract int32 vectors and set borrow.
 * The borrow from the subtraction for the n-th element
 * is written into the n-th bit of vector mask, via the __mmask16 pointer.
 */
extern __m512i __ICL_INTRINCC _mm512_subrsetb_epi32(__m512i, __m512i,
                                                    __mmask16*);
extern __m512i __ICL_INTRINCC _mm512_mask_subrsetb_epi32(__m512i, __mmask16,
                                                         __mmask16, __m512i,
                                                         __mmask16*);
/*
 * Subtract int32 vectors with borrow.
 *    Performs an element-by-element three-input subtraction of second int32
 *    vector as well as the corresponding bit of the first mask, from the
 *    first int32 vector.
 *
 *    In addition, the borrow from the subtraction difference for the n-th
 *    element is written into the n-th mask bit via the __mmask16 pointer.
 */
extern __m512i __ICL_INTRINCC _mm512_sbb_epi32(__m512i, __mmask16,
                                               __m512i, __mmask16*);
extern __m512i __ICL_INTRINCC _mm512_mask_sbb_epi32(__m512i, __mmask16,
                                                    __mmask16, __m512i,
                                                    __mmask16*);

/*
 * Reverse subtract int32 vectors with borrow.
 * In addition, the borrow from the subtraction difference for the n-th
 * element is written via the n-th bit of __mmask16 pointer.
 */
extern __m512i __ICL_INTRINCC _mm512_sbbr_epi32(__m512i, __mmask16,
                                                __m512i, __mmask16*);
extern __m512i __ICL_INTRINCC _mm512_mask_sbbr_epi32(__m512i, __mmask16,
                                                     __mmask16, __m512i,
                                                     __mmask16*);

/*
 * Bitwise and, and not, or, and xor of int32 or int64 vectors.
 * "and not" ands the ones complement of the first vector operand
 * with the second.
 */

extern __m512i __ICL_INTRINCC _mm512_and_epi32(__m512i, __m512i);
extern __m512i __ICL_INTRINCC _mm512_mask_and_epi32(__m512i, __mmask16,
                                                    __m512i, __m512i);

extern __m512i __ICL_INTRINCC _mm512_and_epi64(__m512i, __m512i);
extern __m512i __ICL_INTRINCC _mm512_mask_and_epi64(__m512i, __mmask8,
                                                    __m512i, __m512i);

extern __m512i __ICL_INTRINCC _mm512_andnot_epi32(__m512i, __m512i);
extern __m512i __ICL_INTRINCC _mm512_mask_andnot_epi32(__m512i, __mmask16,
                                                       __m512i, __m512i);

extern __m512i __ICL_INTRINCC _mm512_andnot_epi64(__m512i, __m512i);
extern __m512i __ICL_INTRINCC _mm512_mask_andnot_epi64(__m512i, __mmask8,
                                                       __m512i, __m512i);

extern __m512i __ICL_INTRINCC _mm512_or_epi32(__m512i, __m512i);
extern __m512i __ICL_INTRINCC _mm512_mask_or_epi32(__m512i, __mmask16,
                                                   __m512i, __m512i);

extern __m512i __ICL_INTRINCC _mm512_or_epi64(__m512i, __m512i);
extern __m512i __ICL_INTRINCC _mm512_mask_or_epi64(__m512i, __mmask8,
                                                   __m512i, __m512i);

extern __m512i __ICL_INTRINCC _mm512_xor_epi32(__m512i, __m512i);
extern __m512i __ICL_INTRINCC _mm512_mask_xor_epi32(__m512i, __mmask16,
                                                    __m512i, __m512i);

extern __m512i __ICL_INTRINCC _mm512_xor_epi64(__m512i, __m512i);
extern __m512i __ICL_INTRINCC _mm512_mask_xor_epi64(__m512i, __mmask8,
                                                    __m512i, __m512i);

/*
 * Compare float32, float64 or int32 vectors and set mask.
 */

/* Constants for integer comparison predicates */
typedef enum {
    _MM_CMPINT_EQ,      /* Equal */
    _MM_CMPINT_LT,      /* Less than */
    _MM_CMPINT_LE,      /* Less than or Equal */
    _MM_CMPINT_UNUSED,
    _MM_CMPINT_NE,      /* Not Equal */
    _MM_CMPINT_NLT,     /* Not Less than */
#define _MM_CMPINT_GE   _MM_CMPINT_NLT  /* Greater than or Equal */
    _MM_CMPINT_NLE      /* Not Less than or Equal */
#define _MM_CMPINT_GT   _MM_CMPINT_NLE  /* Greater than */
} _MM_CMPINT_ENUM;

extern __mmask16 __ICL_INTRINCC _mm512_cmp_epi32_mask(__m512i, __m512i,
                                                      const _MM_CMPINT_ENUM);
extern __mmask16 __ICL_INTRINCC _mm512_mask_cmp_epi32_mask(__mmask16, __m512i,
                                                       __m512i,
                                                       const _MM_CMPINT_ENUM);

#define _mm512_cmpeq_epi32_mask(v1, v2) \
    _mm512_cmp_epi32_mask((v1), (v2), _MM_CMPINT_EQ)
#define _mm512_mask_cmpeq_epi32_mask(k1, v1, v2) \
    _mm512_mask_cmp_epi32_mask((k1), (v1), (v2), _MM_CMPINT_EQ)
#define _mm512_cmplt_epi32_mask(v1, v2) \
    _mm512_cmp_epi32_mask((v1), (v2), _MM_CMPINT_LT)
#define _mm512_mask_cmplt_epi32_mask(k1, v1, v2) \
    _mm512_mask_cmp_epi32_mask((k1), (v1), (v2), _MM_CMPINT_LT)
#define _mm512_cmple_epi32_mask(v1, v2) \
    _mm512_cmp_epi32_mask((v1), (v2), _MM_CMPINT_LE)
#define _mm512_mask_cmple_epi32_mask(k1, v1, v2) \
    _mm512_mask_cmp_epi32_mask((k1), (v1), (v2), _MM_CMPINT_LE)
#define _mm512_cmpneq_epi32_mask(v1, v2) \
    _mm512_cmp_epi32_mask((v1), (v2), _MM_CMPINT_NE)
#define _mm512_mask_cmpneq_epi32_mask(k1, v1, v2) \
    _mm512_mask_cmp_epi32_mask((k1), (v1), (v2), _MM_CMPINT_NE)
#define _mm512_cmpge_epi32_mask(v1, v2) \
    _mm512_cmp_epi32_mask((v1), (v2), _MM_CMPINT_GE)
#define _mm512_mask_cmpge_epi32_mask(k1, v1, v2) \
    _mm512_mask_cmp_epi32_mask((k1), (v1), (v2), _MM_CMPINT_GE)
#define _mm512_cmpgt_epi32_mask(v1, v2) \
    _mm512_cmp_epi32_mask((v1), (v2), _MM_CMPINT_GT)
#define _mm512_mask_cmpgt_epi32_mask(k1, v1, v2) \
    _mm512_mask_cmp_epi32_mask((k1), (v1), (v2), _MM_CMPINT_GT)

extern __mmask16 __ICL_INTRINCC _mm512_cmp_epu32_mask(__m512i, __m512i,
                                                      const _MM_CMPINT_ENUM);
extern __mmask16 __ICL_INTRINCC _mm512_mask_cmp_epu32_mask(__mmask16, __m512i,
                                                      __m512i,
                                                      const _MM_CMPINT_ENUM);

#define _mm512_cmpeq_epu32_mask(v1, v2) \
    _mm512_cmp_epu32_mask((v1), (v2), _MM_CMPINT_EQ)
#define _mm512_mask_cmpeq_epu32_mask(k1, v1, v2) \
    _mm512_mask_cmp_epu32_mask((k1), (v1), (v2), _MM_CMPINT_EQ)
#define _mm512_cmplt_epu32_mask(v1, v2) \
    _mm512_cmp_epu32_mask((v1), (v2), _MM_CMPINT_LT)
#define _mm512_mask_cmplt_epu32_mask(k1, v1, v2) \
    _mm512_mask_cmp_epu32_mask((k1), (v1), (v2), _MM_CMPINT_LT)
#define _mm512_cmple_epu32_mask(v1, v2) \
    _mm512_cmp_epu32_mask((v1), (v2), _MM_CMPINT_LE)
#define _mm512_mask_cmple_epu32_mask(k1, v1, v2) \
    _mm512_mask_cmp_epu32_mask((k1), (v1), (v2), _MM_CMPINT_LE)
#define _mm512_cmpneq_epu32_mask(v1, v2) \
    _mm512_cmp_epu32_mask((v1), (v2), _MM_CMPINT_NE)
#define _mm512_mask_cmpneq_epu32_mask(k1, v1, v2) \
    _mm512_mask_cmp_epu32_mask((k1), (v1), (v2), _MM_CMPINT_NE)
#define _mm512_cmpge_epu32_mask(v1, v2) \
    _mm512_cmp_epu32_mask((v1), (v2), _MM_CMPINT_GE)
#define _mm512_mask_cmpge_epu32_mask(k1, v1, v2) \
    _mm512_mask_cmp_epu32_mask((k1), (v1), (v2), _MM_CMPINT_GE)
#define _mm512_cmpgt_epu32_mask(v1, v2) \
    _mm512_cmp_epu32_mask((v1), (v2), _MM_CMPINT_GT)
#define _mm512_mask_cmpgt_epu32_mask(k1, v1, v2) \
    _mm512_mask_cmp_epu32_mask((k1), (v1), (v2), _MM_CMPINT_GT)

extern __mmask8 __ICL_INTRINCC _mm512_cmp_pd_mask(__m512d, __m512d, const int);
extern __mmask8 __ICL_INTRINCC _mm512_mask_cmp_pd_mask(__mmask8, __m512d,
                                                       __m512d,
                                                       const int);

#define _mm512_cmpeq_pd_mask(v1, v2) _mm512_cmp_pd_mask((v1), (v2), _CMP_EQ_OQ)
#define _mm512_mask_cmpeq_pd_mask(k1, v1, v2) \
    _mm512_mask_cmp_pd_mask((k1), (v1), (v2), _CMP_EQ_OQ)
#define _mm512_cmplt_pd_mask(v1, v2) _mm512_cmp_pd_mask((v1), (v2), _CMP_LT_OS)
#define _mm512_mask_cmplt_pd_mask(k1, v1, v2) \
    _mm512_mask_cmp_pd_mask((k1), (v1), (v2), _CMP_LT_OS)
#define _mm512_cmple_pd_mask(v1, v2) _mm512_cmp_pd_mask((v1), (v2), _CMP_LE_OS)
#define _mm512_mask_cmple_pd_mask(k1, v1, v2) \
    _mm512_mask_cmp_pd_mask((k1), (v1), (v2), _CMP_LE_OS)
#define _mm512_cmpunord_pd_mask(v1, v2) \
    _mm512_cmp_pd_mask((v1), (v2), _CMP_UNORD_Q)
#define _mm512_mask_cmpunord_pd_mask(k1, v1, v2) \
    _mm512_mask_cmp_pd_mask((k1), (v1), (v2), _CMP_UNORD_Q)
#define _mm512_cmpneq_pd_mask(v1, v2) \
    _mm512_cmp_pd_mask((v1), (v2), _CMP_NEQ_UQ)
#define _mm512_mask_cmpneq_pd_mask(k1, v1, v2) \
    _mm512_mask_cmp_pd_mask((k1), (v1), (v2), _CMP_NEQ_UQ)
#define _mm512_cmpnlt_pd_mask(v1, v2) \
    _mm512_cmp_pd_mask((v1), (v2), _CMP_NLT_US)
#define _mm512_mask_cmpnlt_pd_mask(k1, v1, v2) \
    _mm512_mask_cmp_pd_mask((k1), (v1), (v2), _CMP_NLT_US)
#define _mm512_cmpnle_pd_mask(v1, v2) \
    _mm512_cmp_pd_mask((v1), (v2), _CMP_NLE_US)
#define _mm512_mask_cmpnle_pd_mask(k1, v1, v2) \
    _mm512_mask_cmp_pd_mask((k1), (v1), (v2), _CMP_NLE_US)
#define _mm512_cmpord_pd_mask(v1, v2) \
    _mm512_cmp_pd_mask((v1), (v2), _CMP_ORD_Q)
#define _mm512_mask_cmpord_pd_mask(k1, v1, v2) \
    _mm512_mask_cmp_pd_mask((k1), (v1), (v2), _CMP_ORD_Q)

extern __mmask16 __ICL_INTRINCC _mm512_cmp_ps_mask(__m512, __m512, const int);
extern __mmask16 __ICL_INTRINCC _mm512_mask_cmp_ps_mask(__mmask16, __m512,
                                                        __m512, const int);

#define _mm512_cmpeq_ps_mask(v1, v2) _mm512_cmp_ps_mask((v1), (v2), _CMP_EQ_OQ)
#define _mm512_mask_cmpeq_ps_mask(k1, v1, v2) \
    _mm512_mask_cmp_ps_mask((k1), (v1), (v2), _CMP_EQ_OQ)
#define _mm512_cmplt_ps_mask(v1, v2) _mm512_cmp_ps_mask((v1), (v2), _CMP_LT_OS)
#define _mm512_mask_cmplt_ps_mask(k1, v1, v2) \
    _mm512_mask_cmp_ps_mask((k1), (v1), (v2), _CMP_LT_OS)
#define _mm512_cmple_ps_mask(v1, v2) _mm512_cmp_ps_mask((v1), (v2), _CMP_LE_OS)
#define _mm512_mask_cmple_ps_mask(k1, v1, v2) \
    _mm512_mask_cmp_ps_mask((k1), (v1), (v2), _CMP_LE_OS)
#define _mm512_cmpunord_ps_mask(v1, v2) \
    _mm512_cmp_ps_mask((v1), (v2), _CMP_UNORD_Q)
#define _mm512_mask_cmpunord_ps_mask(k1, v1, v2) \
    _mm512_mask_cmp_ps_mask((k1), (v1), (v2), _CMP_UNORD_Q)
#define _mm512_cmpneq_ps_mask(v1, v2) \
    _mm512_cmp_ps_mask((v1), (v2), _CMP_NEQ_UQ)
#define _mm512_mask_cmpneq_ps_mask(k1, v1, v2) \
    _mm512_mask_cmp_ps_mask((k1), (v1), (v2), _CMP_NEQ_UQ)
#define _mm512_cmpnlt_ps_mask(v1, v2) \
    _mm512_cmp_ps_mask((v1), (v2), _CMP_NLT_US)
#define _mm512_mask_cmpnlt_ps_mask(k1, v1, v2) \
    _mm512_mask_cmp_ps_mask((k1), (v1), (v2), _CMP_NLT_US)
#define _mm512_cmpnle_ps_mask(v1, v2) \
    _mm512_cmp_ps_mask((v1), (v2), _CMP_NLE_US)
#define _mm512_mask_cmpnle_ps_mask(k1, v1, v2) \
    _mm512_mask_cmp_ps_mask((k1), (v1), (v2), _CMP_NLE_US)
#define _mm512_cmpord_ps_mask(v1, v2) \
    _mm512_cmp_ps_mask((v1), (v2), _CMP_ORD_Q)
#define _mm512_mask_cmpord_ps_mask(k1, v1, v2) \
    _mm512_mask_cmp_ps_mask((k1), (v1), (v2), _CMP_ORD_Q)


extern __m512 __ICL_INTRINCC _mm512_cvt_roundpd_pslo(__m512d, int);
extern __m512 __ICL_INTRINCC _mm512_mask_cvt_roundpd_pslo(__m512, __mmask8,
                                                          __m512d, int);
#define _mm512_cvtpd_pslo(v2) \
    _mm512_cvt_roundpd_pslo((v2), _MM_FROUND_CUR_DIRECTION)
#define _mm512_mask_cvtpd_pslo(v1_old, k1, v2) \
    _mm512_mask_cvt_roundpd_pslo((v1_old), (k1), (v2), \
                                 _MM_FROUND_CUR_DIRECTION)

extern __m512i __ICL_INTRINCC _mm512_cvtfxpnt_roundpd_epi32lo(__m512d, int);
extern __m512i __ICL_INTRINCC _mm512_mask_cvtfxpnt_roundpd_epi32lo(__m512i,
                                                                   __mmask8,
                                                                   __m512d,
                                                                   int);
extern __m512i __ICL_INTRINCC _mm512_cvtfxpnt_roundpd_epu32lo(__m512d, int);
extern __m512i __ICL_INTRINCC _mm512_mask_cvtfxpnt_roundpd_epu32lo(__m512i,
                                                                   __mmask8,
                                                                   __m512d,
                                                                   int);

extern __m512d  __ICL_INTRINCC _mm512_cvtpslo_pd(__m512);
extern __m512d  __ICL_INTRINCC _mm512_mask_cvtpslo_pd(__m512d, __mmask8,
                                                      __m512);

extern __m512i __ICL_INTRINCC _mm512_cvtfxpnt_round_adjustps_epi32(__m512,
                                                    int /* rounding */,
                                                    _MM_EXP_ADJ_ENUM);
extern __m512i __ICL_INTRINCC _mm512_mask_cvtfxpnt_round_adjustps_epi32(
                                                    __m512i,
                                                    __mmask16, __m512,
                                                    int /* rounding */,
                                                    _MM_EXP_ADJ_ENUM);

extern __m512i __ICL_INTRINCC _mm512_cvtfxpnt_round_adjustps_epu32(__m512,
                                                    int /* rounding */,
                                                    _MM_EXP_ADJ_ENUM);
extern __m512i __ICL_INTRINCC _mm512_mask_cvtfxpnt_round_adjustps_epu32(
                                                    __m512i,
                                                    __mmask16, __m512,
                                                    int /* rounding */,
                                                    _MM_EXP_ADJ_ENUM);

/*
 * Convert int32 or unsigned int32 vector to float32 or float64 vector.
 */

extern __m512d __ICL_INTRINCC _mm512_cvtepi32lo_pd(__m512i);
extern __m512d __ICL_INTRINCC _mm512_mask_cvtepi32lo_pd(__m512d, __mmask8,
                                                        __m512i);
extern __m512d __ICL_INTRINCC _mm512_cvtepu32lo_pd(__m512i);
extern __m512d __ICL_INTRINCC _mm512_mask_cvtepu32lo_pd(__m512d, __mmask8,
                                                        __m512i);

extern __m512 __ICL_INTRINCC _mm512_cvtfxpnt_round_adjustepi32_ps(__m512i,
                                                       int /* rounding */,
                                                       _MM_EXP_ADJ_ENUM);

extern __m512 __ICL_INTRINCC _mm512_mask_cvtfxpnt_round_adjustepi32_ps(
                                                       __m512,
                                                       __mmask16,
                                                       __m512i,
                                                       int /* rounding */,
                                                       _MM_EXP_ADJ_ENUM);

extern __m512 __ICL_INTRINCC _mm512_cvtfxpnt_round_adjustepu32_ps(__m512i,
                                                       int /* rounding */,
                                                       _MM_EXP_ADJ_ENUM);

extern __m512 __ICL_INTRINCC _mm512_mask_cvtfxpnt_round_adjustepu32_ps(__m512,
                                                       __mmask16, __m512i,
                                                       int /* rounding */,
                                                       _MM_EXP_ADJ_ENUM);

/*
 * Approximate the base-2 exponential of an int32 vector representing
 * fixed point values with 8 bits for sign and integer part, and 24 bits
 * for the fraction.
 */
extern __m512 __ICL_INTRINCC _mm512_exp223_ps(__m512i);
extern __m512 __ICL_INTRINCC _mm512_mask_exp223_ps(__m512, __mmask16, __m512i);

extern __m512d __ICL_INTRINCC _mm512_fixupnan_pd(__m512d, __m512d, __m512i);
extern __m512d __ICL_INTRINCC _mm512_mask_fixupnan_pd(__m512d, __mmask8,
                                                      __m512d, __m512i);

extern __m512  __ICL_INTRINCC _mm512_fixupnan_ps(__m512, __m512, __m512i);
extern __m512  __ICL_INTRINCC _mm512_mask_fixupnan_ps(__m512, __mmask16,
                                                      __m512, __m512i);

extern __m512i __ICL_INTRINCC _mm512_i32extgather_epi32(__m512i, void const*,
                                                        _MM_UPCONV_EPI32_ENUM,
                                                        int,
                                                        int /* mem hint */);

extern __m512i __ICL_INTRINCC _mm512_mask_i32extgather_epi32(__m512i,
                                                      __mmask16,
                                                      __m512i /* index */,
                                                      void const*,
                                                      _MM_UPCONV_EPI32_ENUM,
                                                      int, int /* mem hint */);

extern __m512i __ICL_INTRINCC _mm512_i32loextgather_epi64(__m512i, void const*,
                                                      _MM_UPCONV_EPI64_ENUM,
                                                      int,
                                                      int /* mem hint */);

extern __m512i __ICL_INTRINCC _mm512_mask_i32loextgather_epi64(__m512i,
                                                        __mmask8,
                                                        __m512i,
                                                        void const*,
                                                        _MM_UPCONV_EPI64_ENUM,
                                                        int,
                                                        int /* mem hint */);

extern __m512  __ICL_INTRINCC _mm512_i32extgather_ps(__m512i, void const*,
                                                     _MM_UPCONV_PS_ENUM, int,
                                                     int /* mem hint */);

extern __m512  __ICL_INTRINCC _mm512_mask_i32extgather_ps(__m512, __mmask16,
                                                          __m512i, void const*,
                                                          _MM_UPCONV_PS_ENUM,
                                                          int,
                                                          int /* mem hint */);

extern __m512d __ICL_INTRINCC _mm512_i32loextgather_pd(__m512i, void const*,
                                                       _MM_UPCONV_PD_ENUM, int,
                                                       int /* mem hint */);

extern __m512d __ICL_INTRINCC _mm512_mask_i32loextgather_pd(__m512d, __mmask8,
                                                         __m512i,
                                                         void const*,
                                                         _MM_UPCONV_PD_ENUM,
                                                         int,
                                                         int /* mem hint */);

#define _mm512_i32gather_epi32(index, addr, scale) \
    _mm512_i32extgather_epi32((index), (addr), _MM_UPCONV_EPI32_NONE, \
                              (scale), _MM_HINT_NONE)

#define _mm512_mask_i32gather_epi32(v1_old, k1, index, addr, scale) \
    _mm512_mask_i32extgather_epi32((v1_old), (k1), (index), (addr), \
                                   _MM_UPCONV_EPI32_NONE, (scale), \
                                   _MM_HINT_NONE)

#define _mm512_i32logather_epi64(index, addr, scale) \
    _mm512_i32loextgather_epi64((index), (addr), _MM_UPCONV_EPI64_NONE, \
                                (scale), _MM_HINT_NONE)

#define _mm512_mask_i32logather_epi64(v1_old, k1, index, addr, scale) \
    _mm512_mask_i32loextgather_epi64((v1_old), (k1), (index), (addr), \
                                     _MM_UPCONV_EPI64_NONE, (scale), \
                                     _MM_HINT_NONE)

#define _mm512_i32gather_ps(index, addr, scale) \
    _mm512_i32extgather_ps((index), (addr), _MM_UPCONV_PS_NONE, \
                           (scale), _MM_HINT_NONE)

#define _mm512_mask_i32gather_ps(v1_old, k1, index, addr, scale) \
    _mm512_mask_i32extgather_ps((v1_old), (k1), (index), (addr), \
                                _MM_UPCONV_PS_NONE, (scale), _MM_HINT_NONE)

#define _mm512_i32logather_pd(index, addr, scale) \
    _mm512_i32loextgather_pd((index), (addr), _MM_UPCONV_PD_NONE, \
                             (scale), _MM_HINT_NONE)

#define _mm512_mask_i32logather_pd(v1_old, k1, index, addr, scale) \
    _mm512_mask_i32loextgather_pd((v1_old), (k1), (index), (addr), \
                                  _MM_UPCONV_PD_NONE, (scale), _MM_HINT_NONE)

extern void __ICL_INTRINCC _mm512_prefetch_i32extgather_ps(__m512i,
                                                           void const*,
                                                           _MM_UPCONV_PS_ENUM,
                                                           int /* scale */,
                                                           int /* pf hint */);

extern void __ICL_INTRINCC _mm512_mask_prefetch_i32extgather_ps(
                                                         __m512i /* index */,
                                                         __mmask16,
                                                         void const*,
                                                         _MM_UPCONV_PS_ENUM,
                                                         int /* scale */,
                                                         int /* pf hint */);

#define _mm512_prefetch_i32gather_ps(index, addr, scale, pf_hint) \
    _mm512_prefetch_i32extgather_ps((index), (addr), _MM_UPCONV_PS_NONE, \
                                    (scale), (pf_hint))

#define _mm512_mask_prefetch_i32gather_ps(index, k1, addr, scale, pf_hint) \
    _mm512_mask_prefetch_i32extgather_ps((index), (k1), (addr), \
                                         _MM_UPCONV_PS_NONE, (scale), \
                                         (pf_hint))

extern void __ICL_INTRINCC _mm512_i32extscatter_ps(void*, __m512i, __m512,
                                                   _MM_DOWNCONV_PS_ENUM,
                                                   int /* scale */,
                                                   int /* mem hint */);

extern void __ICL_INTRINCC _mm512_mask_i32extscatter_ps(void*, __mmask16,
                                                        __m512i, __m512,
                                                        _MM_DOWNCONV_PS_ENUM,
                                                        int /* scale */,
                                                        int /* mem hint */);

extern void __ICL_INTRINCC _mm512_i32loextscatter_pd(void*, __m512i, __m512d,
                                                     _MM_DOWNCONV_PD_ENUM,
                                                     int /* scale */,
                                                     int /* mem hint */);

extern void __ICL_INTRINCC _mm512_mask_i32loextscatter_pd(void*, __mmask8,
                                                          __m512i, __m512d,
                                                          _MM_DOWNCONV_PD_ENUM,
                                                          int /* scale */,
                                                          int /* mem hint */);

extern void __ICL_INTRINCC _mm512_i32extscatter_epi32(void*, __m512i, __m512i,
                                                      _MM_DOWNCONV_EPI32_ENUM,
                                                      int /* scale */,
                                                      int /* mem hint */);

extern void __ICL_INTRINCC _mm512_mask_i32extscatter_epi32(void*, __mmask16,
                                                    __m512i, __m512i,
                                                    _MM_DOWNCONV_EPI32_ENUM,
                                                    int /* scale */,
                                                    int /* mem hint */);

extern void __ICL_INTRINCC _mm512_i32loextscatter_epi64(void*, __m512i,
                                                 __m512i,
                                                 _MM_DOWNCONV_EPI64_ENUM,
                                                 int /* scale */,
                                                 int /* mem hint */);

extern void __ICL_INTRINCC _mm512_mask_i32loextscatter_epi64(void*, __mmask8,
                                                      __m512i, __m512i,
                                                      _MM_DOWNCONV_EPI64_ENUM,
                                                      int /* scale */,
                                                      int /* mem hint */);

#define _mm512_i32scatter_ps(addr, index, v1, scale) \
    _mm512_i32extscatter_ps((addr), (index), (v1), _MM_DOWNCONV_PS_NONE, \
                            (scale), _MM_HINT_NONE)

#define _mm512_mask_i32scatter_ps(addr, k1, index, v1, scale) \
    _mm512_mask_i32extscatter_ps((addr), (k1), (index), (v1), \
                                 _MM_DOWNCONV_PS_NONE, (scale), _MM_HINT_NONE)

#define _mm512_i32loscatter_pd(addr, index, v1, scale) \
    _mm512_i32loextscatter_pd((addr), (index), (v1), _MM_DOWNCONV_PD_NONE, \
                              (scale), _MM_HINT_NONE)

#define _mm512_mask_i32loscatter_pd(addr, k1, index, v1, scale) \
    _mm512_mask_i32loextscatter_pd((addr), (k1), (index), (v1), \
                                   _MM_DOWNCONV_PD_NONE, (scale), \
                                   _MM_HINT_NONE)

#define _mm512_i32scatter_epi32(addr, index, v1, scale) \
    _mm512_i32extscatter_epi32((addr), (index), (v1), \
                               _MM_DOWNCONV_EPI32_NONE, (scale), _MM_HINT_NONE)

#define _mm512_mask_i32scatter_epi32(addr, k1, index, v1, scale) \
    _mm512_mask_i32extscatter_epi32((addr), (k1), (index), (v1), \
                                    _MM_DOWNCONV_EPI32_NONE, (scale), \
                                    _MM_HINT_NONE)

#define _mm512_i32loscatter_epi64(addr, index, v1, scale) \
    _mm512_i32loextscatter_epi64((addr), (index), (v1), \
                                 _MM_DOWNCONV_EPI64_NONE, (scale), \
                                 _MM_HINT_NONE)

#define _mm512_mask_i32loscatter_epi64(addr, k1, index, v1, scale) \
    _mm512_mask_i32loextscatter_epi64((addr), (k1), (index), (v1), \
                                      _MM_DOWNCONV_EPI64_NONE, (scale), \
                                      _MM_HINT_NONE)

/*
 * Scatter prefetch element vector.
 */

extern void __ICL_INTRINCC _mm512_prefetch_i32extscatter_ps(void*, __m512i,
                                                            _MM_UPCONV_PS_ENUM,
                                                            int /* scale */,
                                                            int /* pf hint */);

extern void __ICL_INTRINCC _mm512_mask_prefetch_i32extscatter_ps(void*,
                                                          __mmask16, __m512i,
                                                          _MM_UPCONV_PS_ENUM,
                                                          int /* scale */,
                                                          int /* pf hint */);

#define _mm512_prefetch_i32scatter_ps(addr, index, scale, pf_hint) \
    _mm512_prefetch_i32extscatter_ps((addr), (index), _MM_UPCONV_PS_NONE, \
                                     (scale), (pf_hint))

#define _mm512_mask_prefetch_i32scatter_ps(addr, k1, index, scale, pf_hint) \
    _mm512_mask_prefetch_i32extscatter_ps((addr), (k1), (index), \
                                          _MM_UPCONV_PS_NONE, (scale), \
                                          (pf_hint))


/*
 * Extract float32 vector of exponents.
 */
extern __m512 __ICL_INTRINCC _mm512_getexp_ps(__m512);
extern __m512 __ICL_INTRINCC _mm512_mask_getexp_ps(__m512, __mmask16, __m512);

extern __m512d __ICL_INTRINCC _mm512_getexp_pd(__m512d);
extern __m512d __ICL_INTRINCC _mm512_mask_getexp_pd(__m512d, __mmask8,
                                                    __m512d);

/*
 * Extract float32 or float64 normalized mantissas.
 */

/* Constants for mantissa extraction */
typedef enum {
    _MM_MANT_NORM_1_2,      /* interval [1, 2)      */
    _MM_MANT_NORM_p5_2,     /* interval [1.5, 2)    */
    _MM_MANT_NORM_p5_1,     /* interval [1.5, 1)    */
    _MM_MANT_NORM_p75_1p5   /* interval [0.75, 1.5) */
} _MM_MANTISSA_NORM_ENUM;

typedef enum {
    _MM_MANT_SIGN_src,      /* sign = sign(SRC)     */
    _MM_MANT_SIGN_zero,     /* sign = 0             */
    _MM_MANT_SIGN_nan       /* DEST = NaN if sign(SRC) = 1 */
} _MM_MANTISSA_SIGN_ENUM;

extern __m512d __ICL_INTRINCC _mm512_getmant_pd(__m512d,
                                                _MM_MANTISSA_NORM_ENUM,
                                                _MM_MANTISSA_SIGN_ENUM);
extern __m512d __ICL_INTRINCC _mm512_mask_getmant_pd(__m512d, __mmask8,
                                                     __m512d,
                                                     _MM_MANTISSA_NORM_ENUM,
                                                     _MM_MANTISSA_SIGN_ENUM);

extern __m512  __ICL_INTRINCC _mm512_getmant_ps(__m512,
                                                _MM_MANTISSA_NORM_ENUM,
                                                _MM_MANTISSA_SIGN_ENUM);
extern __m512  __ICL_INTRINCC _mm512_mask_getmant_ps(__m512, __mmask16, __m512,
                                                     _MM_MANTISSA_NORM_ENUM,
                                                     _MM_MANTISSA_SIGN_ENUM);

/*
 * Load unaligned high and unpack to doubleword vector.
 *    The high-64-byte portion of the byte/word/doubleword stream starting
 *    at the element-aligned address is loaded, converted and expanded
 *    into the writemask-enabled elements of doubleword vector.
 *    Doubleword vector is returned.
 *
 *    The number of set bits in the writemask determines the length of the
 *    converted doubleword stream, as each converted doubleword is mapped
 *    to exactly one of the doubleword elements in returned vector, skipping
 *    over writemasked elements.
 */
extern __m512i __ICL_INTRINCC _mm512_extloadunpackhi_epi32(__m512i,
                                                    void const*,
                                                    _MM_UPCONV_EPI32_ENUM,
                                                    int /* mem hint */);

extern __m512i __ICL_INTRINCC _mm512_mask_extloadunpackhi_epi32(__m512i,
                                                         __mmask16,
                                                         void const*,
                                                         _MM_UPCONV_EPI32_ENUM,
                                                         int /* mem hint */);

extern __m512i __ICL_INTRINCC _mm512_extloadunpacklo_epi32(__m512i,
                                                    void const*,
                                                    _MM_UPCONV_EPI32_ENUM,
                                                    int /* mem hint */);

extern __m512i __ICL_INTRINCC _mm512_mask_extloadunpacklo_epi32(__m512i,
                                                         __mmask16,
                                                         void const*,
                                                         _MM_UPCONV_EPI32_ENUM,
                                                         int /* mem hint */);

extern __m512i __ICL_INTRINCC _mm512_extloadunpackhi_epi64(__m512i,
                                                    void const*,
                                                    _MM_UPCONV_EPI64_ENUM,
                                                    int /* mem hint */);

extern __m512i __ICL_INTRINCC _mm512_mask_extloadunpackhi_epi64(__m512i,
                                                         __mmask8,
                                                         void const*,
                                                         _MM_UPCONV_EPI64_ENUM,
                                                         int /* mem hint */);

extern __m512i __ICL_INTRINCC _mm512_extloadunpacklo_epi64(__m512i,
                                                    void const*,
                                                    _MM_UPCONV_EPI64_ENUM,
                                                    int /* mem hint */);

extern __m512i __ICL_INTRINCC _mm512_mask_extloadunpacklo_epi64(__m512i,
                                                         __mmask8,
                                                         void const*,
                                                         _MM_UPCONV_EPI64_ENUM,
                                                         int /* mem hint */);

extern __m512  __ICL_INTRINCC _mm512_extloadunpackhi_ps(__m512, void const*,
                                                        _MM_UPCONV_PS_ENUM,
                                                        int /* mem hint */);

extern __m512  __ICL_INTRINCC _mm512_mask_extloadunpackhi_ps(__m512, __mmask16,
                                                      void const*,
                                                      _MM_UPCONV_PS_ENUM,
                                                      int /* mem hint */);

extern __m512  __ICL_INTRINCC _mm512_extloadunpacklo_ps(__m512, void const*,
                                                        _MM_UPCONV_PS_ENUM,
                                                        int /* mem hint */);

extern __m512  __ICL_INTRINCC _mm512_mask_extloadunpacklo_ps(__m512,
                                                      __mmask16,
                                                      void const*,
                                                      _MM_UPCONV_PS_ENUM,
                                                      int /* mem hint */);

extern __m512d __ICL_INTRINCC _mm512_extloadunpackhi_pd(__m512d,
                                                 void const*,
                                                 _MM_UPCONV_PD_ENUM,
                                                 int /* mem hint */);

extern __m512d __ICL_INTRINCC _mm512_mask_extloadunpackhi_pd(__m512d, __mmask8,
                                                      void const*,
                                                      _MM_UPCONV_PD_ENUM,
                                                      int /* mem hint */);

extern __m512d __ICL_INTRINCC _mm512_extloadunpacklo_pd(__m512d, void const*,
                                                        _MM_UPCONV_PD_ENUM,
                                                        int /* mem hint */);

extern __m512d __ICL_INTRINCC _mm512_mask_extloadunpacklo_pd(__m512d, __mmask8,
                                                      void const*,
                                                      _MM_UPCONV_PD_ENUM,
                                                      int /* mem hint */);

#define _mm512_loadunpackhi_epi32(v1_old, addr) \
    _mm512_extloadunpackhi_epi32((v1_old), (addr), \
                                 _MM_UPCONV_EPI32_NONE, _MM_HINT_NONE)
#define _mm512_mask_loadunpackhi_epi32(v1_old, k1, addr) \
    _mm512_mask_extloadunpackhi_epi32((v1_old), (k1), (addr), \
                                      _MM_UPCONV_EPI32_NONE, _MM_HINT_NONE)

#define _mm512_loadunpacklo_epi32(v1_old, addr) \
    _mm512_extloadunpacklo_epi32((v1_old), (addr), \
                                 _MM_UPCONV_EPI32_NONE, _MM_HINT_NONE)
#define _mm512_mask_loadunpacklo_epi32(v1_old, k1, addr) \
    _mm512_mask_extloadunpacklo_epi32((v1_old), (k1), (addr), \
                                      _MM_UPCONV_EPI32_NONE, _MM_HINT_NONE)

#define _mm512_loadunpackhi_epi64(v1_old, addr) \
    _mm512_extloadunpackhi_epi64((v1_old), (addr), \
                                 _MM_UPCONV_EPI64_NONE, _MM_HINT_NONE)
#define _mm512_mask_loadunpackhi_epi64(v1_old, k1, addr) \
    _mm512_mask_extloadunpackhi_epi64((v1_old), (k1), (addr), \
                                      _MM_UPCONV_EPI64_NONE, _MM_HINT_NONE)

#define _mm512_loadunpacklo_epi64(v1_old, addr) \
    _mm512_extloadunpacklo_epi64((v1_old), (addr), \
                                 _MM_UPCONV_EPI64_NONE, _MM_HINT_NONE)
#define _mm512_mask_loadunpacklo_epi64(v1_old, k1, addr) \
    _mm512_mask_extloadunpacklo_epi64((v1_old), (k1), (addr), \
                                      _MM_UPCONV_EPI64_NONE, _MM_HINT_NONE)

#define _mm512_loadunpackhi_ps(v1_old, addr) \
    _mm512_extloadunpackhi_ps((v1_old), (addr), \
                              _MM_UPCONV_PS_NONE, _MM_HINT_NONE)
#define _mm512_mask_loadunpackhi_ps(v1_old, k1, addr) \
    _mm512_mask_extloadunpackhi_ps((v1_old), (k1), (addr), \
                                   _MM_UPCONV_PS_NONE, _MM_HINT_NONE)

#define _mm512_loadunpacklo_ps(v1_old, addr) \
    _mm512_extloadunpacklo_ps((v1_old), (addr), \
                              _MM_UPCONV_PS_NONE, _MM_HINT_NONE)
#define _mm512_mask_loadunpacklo_ps(v1_old, k1, addr) \
    _mm512_mask_extloadunpacklo_ps((v1_old), (k1), (addr), \
                                   _MM_UPCONV_PS_NONE, _MM_HINT_NONE)

#define _mm512_loadunpackhi_pd(v1_old, addr) \
    _mm512_extloadunpackhi_pd((v1_old), (addr), \
                              _MM_UPCONV_PD_NONE, _MM_HINT_NONE)
#define _mm512_mask_loadunpackhi_pd(v1_old, k1, addr) \
    _mm512_mask_extloadunpackhi_pd((v1_old), (k1), (addr), \
                                   _MM_UPCONV_PD_NONE, _MM_HINT_NONE)

#define _mm512_loadunpacklo_pd(v1_old, addr) \
    _mm512_extloadunpacklo_pd((v1_old), (addr), \
                              _MM_UPCONV_PD_NONE, _MM_HINT_NONE)
#define _mm512_mask_loadunpacklo_pd(v1_old, k1, addr) \
    _mm512_mask_extloadunpacklo_pd((v1_old), (k1), (addr), \
                                   _MM_UPCONV_PD_NONE, _MM_HINT_NONE)


extern void __ICL_INTRINCC _mm512_extpackstorehi_epi32(void*, __m512i,
                                                       _MM_DOWNCONV_EPI32_ENUM,
                                                       int /* mem hint */);
extern void __ICL_INTRINCC _mm512_mask_extpackstorehi_epi32(void*, __mmask16,
                                                     __m512i,
                                                     _MM_DOWNCONV_EPI32_ENUM,
                                                     int /* mem hint */);

extern void __ICL_INTRINCC _mm512_extpackstorelo_epi32(void*, __m512i,
                                                       _MM_DOWNCONV_EPI32_ENUM,
                                                       int /* mem hint */);
extern void __ICL_INTRINCC _mm512_mask_extpackstorelo_epi32(void*, __mmask16,
                                                     __m512i,
                                                     _MM_DOWNCONV_EPI32_ENUM,
                                                     int /* mem hint */);

extern void __ICL_INTRINCC _mm512_extpackstorehi_epi64(void*, __m512i,
                                                       _MM_DOWNCONV_EPI64_ENUM,
                                                       int /* mem hint */);
extern void __ICL_INTRINCC _mm512_mask_extpackstorehi_epi64(void*, __mmask8,
                                                     __m512i,
                                                     _MM_DOWNCONV_EPI64_ENUM,
                                                     int /* mem hint */);

extern void __ICL_INTRINCC _mm512_extpackstorelo_epi64(void*, __m512i,
                                                       _MM_DOWNCONV_EPI64_ENUM,
                                                       int /* mem hint */);
extern void __ICL_INTRINCC _mm512_mask_extpackstorelo_epi64(void*, __mmask8,
                                                     __m512i,
                                                     _MM_DOWNCONV_EPI64_ENUM,
                                                     int /* mem hint */);

extern void __ICL_INTRINCC _mm512_extpackstorehi_ps(void*, __m512,
                                                    _MM_DOWNCONV_PS_ENUM,
                                                    int /* mem hint */);
extern void __ICL_INTRINCC _mm512_mask_extpackstorehi_ps(void*, __mmask16,
                                                  __m512,
                                                  _MM_DOWNCONV_PS_ENUM,
                                                  int /* mem hint */);

extern void __ICL_INTRINCC _mm512_extpackstorelo_ps(void*, __m512,
                                                    _MM_DOWNCONV_PS_ENUM,
                                                    int /* mem hint */);
extern void __ICL_INTRINCC _mm512_mask_extpackstorelo_ps(void*, __mmask16,
                                                         __m512,
                                                         _MM_DOWNCONV_PS_ENUM,
                                                         int /* mem hint */);

extern void __ICL_INTRINCC _mm512_extpackstorehi_pd(void*, __m512d,
                                                    _MM_DOWNCONV_PD_ENUM,
                                                    int /* mem hint */);
extern void __ICL_INTRINCC _mm512_mask_extpackstorehi_pd(void*, __mmask8,
                                                         __m512d,
                                                         _MM_DOWNCONV_PD_ENUM,
                                                         int /* mem hint */);

extern void __ICL_INTRINCC _mm512_extpackstorelo_pd(void*, __m512d,
                                                    _MM_DOWNCONV_PD_ENUM,
                                                    int /* mem hint */);
extern void __ICL_INTRINCC _mm512_mask_extpackstorelo_pd(void*, __mmask8,
                                                  __m512d,
                                                  _MM_DOWNCONV_PD_ENUM,
                                                  int /* mem hint */);

#define _mm512_packstorehi_epi32(addr, v1) \
    _mm512_extpackstorehi_epi32((addr), (v1), \
                                _MM_DOWNCONV_EPI32_NONE, 0)
#define _mm512_mask_packstorehi_epi32(addr, k1, v1) \
    _mm512_mask_extpackstorehi_epi32((addr), (k1), (v1), \
                                _MM_DOWNCONV_EPI32_NONE, 0)

#define _mm512_packstorelo_epi32(addr, v1) \
    _mm512_extpackstorelo_epi32((addr), (v1), \
                                _MM_DOWNCONV_EPI32_NONE, 0)
#define _mm512_mask_packstorelo_epi32(addr, k1, v1) \
    _mm512_mask_extpackstorelo_epi32((addr), (k1), (v1), \
                                _MM_DOWNCONV_EPI32_NONE, 0)

#define _mm512_packstorehi_epi64(addr, v1) \
    _mm512_extpackstorehi_epi64((addr), (v1), _MM_DOWNCONV_EPI64_NONE, 0)
#define _mm512_mask_packstorehi_epi64(addr, k1, v1) \
    _mm512_mask_extpackstorehi_epi64((addr), (k1), (v1), \
                                     _MM_DOWNCONV_EPI64_NONE, 0)

#define _mm512_packstorelo_epi64(addr, v1) \
    _mm512_extpackstorelo_epi64((addr), (v1), _MM_DOWNCONV_EPI64_NONE, 0)
#define _mm512_mask_packstorelo_epi64(addr, k1, v1) \
    _mm512_mask_extpackstorelo_epi64((addr), (k1), (v1), \
                                     _MM_DOWNCONV_EPI64_NONE, 0)

#define _mm512_packstorehi_ps(addr, v1) \
    _mm512_extpackstorehi_ps((addr), (v1), _MM_DOWNCONV_PS_NONE, 0)
#define _mm512_mask_packstorehi_ps(addr, k1, v1) \
    _mm512_mask_extpackstorehi_ps((addr), (k1), (v1), _MM_DOWNCONV_PS_NONE, 0)

#define _mm512_packstorelo_ps(addr, v1) \
    _mm512_extpackstorelo_ps((addr), (v1), _MM_DOWNCONV_PS_NONE, 0)
#define _mm512_mask_packstorelo_ps(addr, k1, v1) \
    _mm512_mask_extpackstorelo_ps((addr), (k1), (v1), _MM_DOWNCONV_PS_NONE, 0)

#define _mm512_packstorehi_pd(addr, v1) \
    _mm512_extpackstorehi_pd((addr), (v1), _MM_DOWNCONV_PD_NONE, 0)
#define _mm512_mask_packstorehi_pd(addr, k1, v1) \
    _mm512_mask_extpackstorehi_pd((addr), (k1), (v1) ,_MM_DOWNCONV_PD_NONE, 0)

#define _mm512_packstorelo_pd(addr, v1) \
    _mm512_extpackstorelo_pd((addr), (v1), _MM_DOWNCONV_PD_NONE, 0)
#define _mm512_mask_packstorelo_pd(addr, k1, v1) \
    _mm512_mask_extpackstorelo_pd((addr), (k1), (v1), _MM_DOWNCONV_PD_NONE, 0)


/*
 * Logarithm base-2 of float32 vector, with absolute error
 * bounded by 2^(-23).
 */

extern __m512 __ICL_INTRINCC _mm512_log2ae23_ps(__m512);
extern __m512 __ICL_INTRINCC _mm512_mask_log2ae23_ps(__m512, __mmask16,
                                                     __m512);

/*
 * Fused multiply and add of float32, float64 or int32 vectors.
 *
 * This group of FMA instructions computes the following
 *
 *  fmadd       (v1 * v2) + v3
 *  fmsub       (v1 * v2) - v3
 *  fnmadd     -(v1 * v2) + v3
 *  fnmsub     -(v1 * v2) - v3
 *  fnmadd1    -(v1 * v2) + 1.0
 *
 * When a write-mask is used, the pass-through values come from the
 * vector parameter immediately preceding the mask parameter.  For example,
 * for _mm512_mask_fmadd_ps(__m512 v1, __mmask16 k1, __m512 v2, __m512 v3) the
 * pass through values come from v1, while for
 * _mm512_mask3_fmadd_ps(__m512 v1, __m512 v2, __m512 v3, __mmask16 k3)
 * the pass through values come from v3.  To get pass through values
 * from v2, just reverse the order of v1 and v2 in the "_mask_" form.
 */

extern __m512  __ICL_INTRINCC _mm512_fmadd_round_ps(__m512, __m512, __m512,
                                                    int /* rounding */);
extern __m512  __ICL_INTRINCC _mm512_mask_fmadd_round_ps(__m512, __mmask16,
                                                         __m512, __m512,
                                                         int /* rounding */);
extern __m512  __ICL_INTRINCC _mm512_mask3_fmadd_round_ps(__m512, __m512,
                                                          __m512, __mmask16,
                                                          int /* rounding */);
#define _mm512_fmadd_ps(v1, v2, v3) \
    _mm512_fmadd_round_ps((v1), (v2), (v3), _MM_FROUND_CUR_DIRECTION)

#define _mm512_mask_fmadd_ps(v1, k1, v2, v3) \
    _mm512_mask_fmadd_round_ps((v1), (k1), (v2), (v3), \
                               _MM_FROUND_CUR_DIRECTION)

#define _mm512_mask3_fmadd_ps(v1, v2, v3, k3) \
    _mm512_mask3_fmadd_round_ps((v1), (v2), (v3), (k3), \
                                _MM_FROUND_CUR_DIRECTION)

extern __m512d __ICL_INTRINCC _mm512_fmadd_round_pd(__m512d, __m512d,
                                                    __m512d,
                                                    int /* rounding */);
extern __m512d __ICL_INTRINCC _mm512_mask_fmadd_round_pd(__m512d, __mmask8,
                                                         __m512d, __m512d,
                                                         int /* rounding */);
extern __m512d __ICL_INTRINCC _mm512_mask3_fmadd_round_pd(__m512d, __m512d,
                                                          __m512d, __mmask8,
                                                          int /* rounding */);
#define _mm512_fmadd_pd(v1, v2, v3) \
    _mm512_fmadd_round_pd((v1), (v2), (v3), _MM_FROUND_CUR_DIRECTION)

#define _mm512_mask_fmadd_pd(v1, k1, v2, v3) \
    _mm512_mask_fmadd_round_pd((v1), (k1), (v2), (v3), \
                               _MM_FROUND_CUR_DIRECTION)

#define _mm512_mask3_fmadd_pd(v1, v2, v3, k3) \
    _mm512_mask3_fmadd_round_pd((v1), (v2), (v3), (k3), \
                                _MM_FROUND_CUR_DIRECTION)


extern __m512i __ICL_INTRINCC _mm512_fmadd_epi32(__m512i, __m512i, __m512i);
extern __m512i __ICL_INTRINCC _mm512_mask_fmadd_epi32(__m512i, __mmask16,
                                                      __m512i, __m512i);
extern __m512i __ICL_INTRINCC _mm512_mask3_fmadd_epi32(__m512i, __m512i,
                                                       __m512i, __mmask16);

extern __m512  __ICL_INTRINCC _mm512_fmsub_round_ps(__m512, __m512, __m512,
                                                    int /* rounding */);
extern __m512  __ICL_INTRINCC _mm512_mask_fmsub_round_ps(__m512, __mmask16,
                                                         __m512, __m512,
                                                         int /* rounding */);
extern __m512  __ICL_INTRINCC _mm512_mask3_fmsub_round_ps(__m512, __m512,
                                                          __m512, __mmask16,
                                                          int /* rounding */);
#define _mm512_fmsub_ps(v1, v2, v3) \
    _mm512_fmsub_round_ps((v1), (v2), (v3), _MM_FROUND_CUR_DIRECTION)

#define _mm512_mask_fmsub_ps(v1, k1, v2, v3) \
    _mm512_mask_fmsub_round_ps((v1), (k1), (v2), (v3), \
                               _MM_FROUND_CUR_DIRECTION)

#define _mm512_mask3_fmsub_ps(v1, v2, v3, k3) \
    _mm512_mask3_fmsub_round_ps((v1), (v2), (v3), (k3), \
                                _MM_FROUND_CUR_DIRECTION)

extern __m512d __ICL_INTRINCC _mm512_fmsub_round_pd(__m512d, __m512d, __m512d,
                                                    int /* rounding */);
extern __m512d __ICL_INTRINCC _mm512_mask_fmsub_round_pd(__m512d, __mmask8,
                                                         __m512d, __m512d,
                                                         int /* rounding */);
extern __m512d __ICL_INTRINCC _mm512_mask3_fmsub_round_pd(__m512d, __m512d,
                                                          __m512d, __mmask8,
                                                          int /* rounding */);
#define _mm512_fmsub_pd(v1, v2, v3) \
    _mm512_fmsub_round_pd((v1), (v2), (v3), _MM_FROUND_CUR_DIRECTION)

#define _mm512_mask_fmsub_pd(v1, k1, v2, v3) \
    _mm512_mask_fmsub_round_pd((v1), (k1), (v2), (v3), \
                               _MM_FROUND_CUR_DIRECTION)

#define _mm512_mask3_fmsub_pd(v1, v2, v3, k3) \
    _mm512_mask3_fmsub_round_pd((v1), (v2), (v3), (k3), \
                                _MM_FROUND_CUR_DIRECTION)

extern __m512  __ICL_INTRINCC _mm512_fnmadd_round_ps(__m512, __m512, __m512,
                                                     int /* rounding */);
extern __m512  __ICL_INTRINCC _mm512_mask_fnmadd_round_ps(__m512, __mmask16,
                                                          __m512, __m512,
                                                          int /* rounding */);
extern __m512  __ICL_INTRINCC _mm512_mask3_fnmadd_round_ps(__m512, __m512,
                                                           __m512, __mmask16,
                                                           int /* rounding */);
#define _mm512_fnmadd_ps(v1, v2, v3) \
    _mm512_fnmadd_round_ps((v1), (v2), (v3), _MM_FROUND_CUR_DIRECTION)

#define _mm512_mask_fnmadd_ps(v1, k1, v2, v3) \
    _mm512_mask_fnmadd_round_ps((v1), (k1), (v2), (v3), \
                                _MM_FROUND_CUR_DIRECTION)

#define _mm512_mask3_fnmadd_ps(v1, v2, v3, k3) \
    _mm512_mask3_fnmadd_round_ps((v1), (v2), (v3), (k3), \
                                 _MM_FROUND_CUR_DIRECTION)

extern __m512d __ICL_INTRINCC _mm512_fnmadd_round_pd(__m512d, __m512d, __m512d,
                                                     int /* rounding */);
extern __m512d __ICL_INTRINCC _mm512_mask_fnmadd_round_pd(__m512d, __mmask8,
                                                          __m512d, __m512d,
                                                          int /* rounding */);
extern __m512d __ICL_INTRINCC _mm512_mask3_fnmadd_round_pd(__m512d, __m512d,
                                                           __m512d, __mmask8,
                                                           int /* rounding */);
#define _mm512_fnmadd_pd(v1, v2, v3) \
    _mm512_fnmadd_round_pd((v1), (v2), (v3), _MM_FROUND_CUR_DIRECTION)

#define _mm512_mask_fnmadd_pd(v1, k1, v2, v3) \
    _mm512_mask_fnmadd_round_pd((v1), (k1), (v2), (v3), \
                                _MM_FROUND_CUR_DIRECTION)

#define _mm512_mask3_fnmadd_pd(v1, v2, v3, k3) \
    _mm512_mask3_fnmadd_round_pd((v1), (v2), (v3), (k3), \
                                 _MM_FROUND_CUR_DIRECTION)

extern __m512  __ICL_INTRINCC _mm512_fnmsub_round_ps(__m512, __m512, __m512,
                                                     int /* rounding */);
extern __m512  __ICL_INTRINCC _mm512_mask_fnmsub_round_ps(__m512, __mmask16,
                                                          __m512, __m512,
                                                          int /* rounding */);
extern __m512  __ICL_INTRINCC _mm512_mask3_fnmsub_round_ps(__m512, __m512,
                                                           __m512, __mmask16,
                                                           int /* rounding */);
#define _mm512_fnmsub_ps(v1, v2, v3) \
    _mm512_fnmsub_round_ps((v1), (v2), (v3), _MM_FROUND_CUR_DIRECTION)

#define _mm512_mask_fnmsub_ps(v1, k1, v2, v3) \
    _mm512_mask_fnmsub_round_ps((v1), (k1), (v2), (v3), \
                                _MM_FROUND_CUR_DIRECTION)

#define _mm512_mask3_fnmsub_ps(v1, v2, v3, k3) \
    _mm512_mask3_fnmsub_round_ps((v1), (v2), (v3), (k3), \
                                 _MM_FROUND_CUR_DIRECTION)

extern __m512d __ICL_INTRINCC _mm512_fnmsub_round_pd(__m512d, __m512d, __m512d,
                                                     int /* rounding */);
extern __m512d __ICL_INTRINCC _mm512_mask_fnmsub_round_pd(__m512d, __mmask8,
                                                          __m512d, __m512d,
                                                          int /* rounding */);
extern __m512d __ICL_INTRINCC _mm512_mask3_fnmsub_round_pd(__m512d, __m512d,
                                                           __m512d, __mmask8,
                                                           int /* rounding */);
#define _mm512_fnmsub_pd(v1, v2, v3) \
    _mm512_fnmsub_round_pd((v1), (v2), (v3), _MM_FROUND_CUR_DIRECTION)

#define _mm512_mask_fnmsub_pd(v1, k1, v2, v3) \
    _mm512_mask_fnmsub_round_pd((v1), (k1), (v2), (v3), \
                                _MM_FROUND_CUR_DIRECTION)

#define _mm512_mask3_fnmsub_pd(v1, v2, v3, k3) \
    _mm512_mask3_fnmsub_round_pd((v1), (v2), (v3), (k3), \
                                 _MM_FROUND_CUR_DIRECTION)

/*
 * Multiply and add int32 or float32 vectors with alternating elements.
 *
 *    Multiply vector v2 by certain elements of vector v3, and add that
 *    result to certain other elements of v3.
 *
 *    This intrinsic is built around the concept of 4-element sets, of which
 *    there are four elements 0-3, 4-7, 8-11, and 12-15.
 *    Each element 0-3 of vector v2 is multiplied by element 1 of v3,
 *    the result is added to element 0 of v3, and the final sum is written
 *    into the corresponding element 0-3 of the result vector.
 *    Similarly each element 4-7 of v2 is multiplied by element 5 of v3,
 *    and added to element 4 of v3.
 *    Each element 8-11 of v2 is multiplied by element 9 of v3,
 *    and added to element 8 of v3.
 *    Each element 12-15 of vector v2 is multiplied by element 13 of v3,
 *    and added to element 12 of v3.
 */
extern __m512i __ICL_INTRINCC _mm512_fmadd233_epi32(__m512i, __m512i);
extern __m512i __ICL_INTRINCC _mm512_mask_fmadd233_epi32(__m512i, __mmask16,
                                                         __m512i, __m512i);

extern __m512  __ICL_INTRINCC _mm512_fmadd233_round_ps(__m512, __m512,
                                                       int /* rounding */);
extern __m512  __ICL_INTRINCC _mm512_mask_fmadd233_round_ps(__m512, __mmask16,
                                                          __m512, __m512,
                                                          int /* rounding */);
#define _mm512_fmadd233_ps(v2, v3) \
    _mm512_fmadd233_round_ps((v2), (v3), _MM_FROUND_CUR_DIRECTION)

#define _mm512_mask_fmadd233_ps(v1_old, k1, v2, v3) \
    _mm512_mask_fmadd233_round_ps((v1_old), (k1), (v2), (v3), \
                                  _MM_FROUND_CUR_DIRECTION)

/*
 * Minimum or maximum of float32, float64, int32 or unsigned int32 vectors.
 *
 * gmaxabs returns maximum of absolute values of source operands.
 * gmax, gmaxabs and gmin have DX10 and IEEE 754R semantics:
 *
 * gmin     dest = src0 < src1 ? src0 : src1
 * gmax:    dest = src0 >= src1 ? src0 : src1
 *          >= is used instead of > so that
 *          if gmin(x,y) = x then gmax(x,y) = y.
 *
 *    NaN has special handling: If one source operand is NaN, then the other
 *    source operand is returned (choice made per-component).  If both are NaN,
 *    then the quietized NaN from the first source is returned.
 */

extern __m512 __ICL_INTRINCC _mm512_max_ps(__m512, __m512);
extern __m512 __ICL_INTRINCC _mm512_mask_max_ps(__m512, __mmask16,
                                                __m512, __m512);

extern __m512 __ICL_INTRINCC _mm512_maxabs_ps(__m512, __m512);
extern __m512 __ICL_INTRINCC _mm512_mask_maxabs_ps(__m512, __mmask16,
                                                   __m512, __m512);

extern __m512d __ICL_INTRINCC _mm512_max_pd(__m512d, __m512d);
extern __m512d __ICL_INTRINCC _mm512_mask_max_pd(__m512d, __mmask8,
                                                 __m512d, __m512d);

extern __m512i __ICL_INTRINCC _mm512_max_epi32(__m512i, __m512i);
extern __m512i __ICL_INTRINCC _mm512_mask_max_epi32(__m512i, __mmask16,
                                                    __m512i, __m512i);

extern __m512i __ICL_INTRINCC _mm512_max_epu32(__m512i,__m512i);
extern __m512i __ICL_INTRINCC _mm512_mask_max_epu32(__m512i, __mmask16,
                                                    __m512i,__m512i);

extern __m512 __ICL_INTRINCC _mm512_min_ps(__m512, __m512);
extern __m512 __ICL_INTRINCC _mm512_mask_min_ps(__m512, __mmask16,
                                                __m512, __m512);

extern __m512d __ICL_INTRINCC _mm512_min_pd(__m512d,__m512d);
extern __m512d __ICL_INTRINCC _mm512_mask_min_pd(__m512d, __mmask8,
                                                 __m512d,__m512d);

extern __m512i __ICL_INTRINCC _mm512_min_epi32(__m512i, __m512i);
extern __m512i __ICL_INTRINCC _mm512_mask_min_epi32(__m512i, __mmask16,
                                                    __m512i, __m512i);

extern __m512i __ICL_INTRINCC _mm512_min_epu32(__m512i, __m512i);
extern __m512i __ICL_INTRINCC _mm512_mask_min_epu32(__m512i, __mmask16,
                                                    __m512i, __m512i);

extern __m512  __ICL_INTRINCC _mm512_gmax_ps(__m512, __m512);
extern __m512  __ICL_INTRINCC _mm512_mask_gmax_ps(__m512, __mmask16,
                                                  __m512, __m512);

extern __m512  __ICL_INTRINCC _mm512_gmaxabs_ps(__m512, __m512);
extern __m512  __ICL_INTRINCC _mm512_mask_gmaxabs_ps(__m512, __mmask16,
                                                     __m512, __m512);

extern __m512d __ICL_INTRINCC _mm512_gmax_pd(__m512d, __m512d);
extern __m512d __ICL_INTRINCC _mm512_mask_gmax_pd(__m512d, __mmask8,
                                                  __m512d, __m512d);

extern __m512  __ICL_INTRINCC _mm512_gmin_ps(__m512, __m512);
extern __m512  __ICL_INTRINCC _mm512_mask_gmin_ps(__m512, __mmask16,
                                                  __m512, __m512);

extern __m512d __ICL_INTRINCC _mm512_gmin_pd(__m512d, __m512d);
extern __m512d __ICL_INTRINCC _mm512_mask_gmin_pd(__m512d, __mmask8,
                                                  __m512d, __m512d);

/*
 * Multiply int32 or unsigned int32 vectors, and select the high or low
 * half of the 64-bit result.
 */
extern __m512i __ICL_INTRINCC _mm512_mulhi_epi32(__m512i, __m512i);
extern __m512i __ICL_INTRINCC _mm512_mask_mulhi_epi32(__m512i, __mmask16,
                                                      __m512i, __m512i);

extern __m512i __ICL_INTRINCC _mm512_mulhi_epu32(__m512i, __m512i);
extern __m512i __ICL_INTRINCC _mm512_mask_mulhi_epu32(__m512i, __mmask16,
                                                      __m512i, __m512i);

extern __m512i __ICL_INTRINCC _mm512_mullo_epi32(__m512i, __m512i);
extern __m512i __ICL_INTRINCC _mm512_mask_mullo_epi32(__m512i, __mmask16,
                                                      __m512i, __m512i);

/*
 * Permute 32-bit elements of last vector according to indexes in next
 * to last vector.
 * The i'th element of the result is the j'th element of last vector,
 * where j is the i'th element of next to last vector.
 */
extern __m512i __ICL_INTRINCC _mm512_permutevar_epi32(__m512i, __m512i);
extern __m512i __ICL_INTRINCC _mm512_mask_permutevar_epi32(__m512i, __mmask16,
                                                           __m512i, __m512i);

/*
 * These "permutev" names are deprecated and will be removed.
 * Use the "permutevar" names going forward.
 */
#define _mm512_permutev_epi32 _mm512_permutevar_epi32
#define _mm512_mask_permutev_epi32 _mm512_mask_permutevar_epi32

/*
 * Permute the four 128-bit elements of v2 according to indexes in 'perm'.
 */
extern __m512i __ICL_INTRINCC _mm512_permute4f128_epi32(__m512i,
                                                        _MM_PERM_ENUM);
extern __m512i __ICL_INTRINCC _mm512_mask_permute4f128_epi32(__m512i,
                                                             __mmask16,
                                                             __m512i,
                                                             _MM_PERM_ENUM);

extern __m512 __ICL_INTRINCC _mm512_permute4f128_ps(__m512, _MM_PERM_ENUM);
extern __m512 __ICL_INTRINCC _mm512_mask_permute4f128_ps(__m512, __mmask16,
                                                         __m512,
                                                         _MM_PERM_ENUM);

/*
 * Approximate the reciprocals of the float32 elements in v2 with
 * 23 bits of accuracy.
 */
extern __m512 __ICL_INTRINCC _mm512_rcp23_ps(__m512);
extern __m512 __ICL_INTRINCC _mm512_mask_rcp23_ps(__m512, __mmask16, __m512);

/*
 * Round float32 or float64 vector.
 */
extern __m512 __ICL_INTRINCC _mm512_round_ps(__m512, int /* rounding */,
                                             _MM_EXP_ADJ_ENUM);

extern __m512 __ICL_INTRINCC _mm512_mask_round_ps(__m512, __mmask16,
                                                  __m512, int /* rounding */,
                                                  _MM_EXP_ADJ_ENUM);

extern __m512 __ICL_INTRINCC _mm512_roundfxpnt_adjust_ps(__m512,
                                                         int /* rounding */,
                                                         _MM_EXP_ADJ_ENUM);

extern __m512 __ICL_INTRINCC _mm512_mask_roundfxpnt_adjust_ps(__m512,
                                                       __mmask16, __m512,
                                                       int /* rounding */,
                                                       _MM_EXP_ADJ_ENUM);

extern __m512d __ICL_INTRINCC _mm512_roundfxpnt_adjust_pd(__m512d,
                                                   int /* rounding */,
                                                   _MM_EXP_ADJ_ENUM);

extern __m512d __ICL_INTRINCC _mm512_mask_roundfxpnt_adjust_pd(__m512d,
                                                        __mmask8, __m512d,
                                                        int /* rounding */,
                                                        _MM_EXP_ADJ_ENUM);

/*
 * Reciprocal square root of float32 vector to 0.775ULP accuracy.
 */
extern __m512 __ICL_INTRINCC _mm512_rsqrt23_ps(__m512);
extern __m512 __ICL_INTRINCC _mm512_mask_rsqrt23_ps(__m512, __mmask16, __m512);

/*
 * Scale float32 vectors.
 */
extern __m512  __ICL_INTRINCC _mm512_scale_ps(__m512, __m512i);
extern __m512  __ICL_INTRINCC _mm512_mask_scale_ps(__m512, __mmask16,
                                                   __m512, __m512i);

extern __m512  __ICL_INTRINCC _mm512_scale_round_ps(__m512, __m512i,
                                                    int /* rounding */);
extern __m512  __ICL_INTRINCC _mm512_mask_scale_round_ps(__m512, __mmask16,
                                                         __m512, __m512i,
                                                         int /* rounding */);

extern __m512i __ICL_INTRINCC _mm512_shuffle_epi32(__m512i, _MM_PERM_ENUM);
extern __m512i __ICL_INTRINCC _mm512_mask_shuffle_epi32(__m512i, __mmask16,
                                                        __m512i,
                                                        _MM_PERM_ENUM);

/*
 * Shift int32 vector by full variable count.
 *
 *    Performs an element-by-element shift of int32 vector, shifting by the
 *    number of bits given by the corresponding int32 element of last vector.
 *    If the shift count is greater than 31 then for logical shifts the result
 *    is zero, and for arithmetic right shifts the result is all ones or all
 *    zeroes depending on the original sign bit.
 *
 *    sllv   logical shift left
 *    srlv   logical shift right
 *    srav   arithmetic shift right
 */

extern __m512i __ICL_INTRINCC _mm512_sllv_epi32(__m512i, __m512i);
extern __m512i __ICL_INTRINCC _mm512_mask_sllv_epi32(__m512i, __mmask16,
                                                     __m512i,__m512i);

extern __m512i __ICL_INTRINCC _mm512_srav_epi32(__m512i, __m512i);
extern __m512i __ICL_INTRINCC _mm512_mask_srav_epi32(__m512i, __mmask16,
                                                     __m512i, __m512i);

extern __m512i __ICL_INTRINCC _mm512_srlv_epi32(__m512i, __m512i);
extern __m512i __ICL_INTRINCC _mm512_mask_srlv_epi32(__m512i, __mmask16,
                                                     __m512i, __m512i);

/*
 * Shift int32 vector by full immediate count.
 *
 *    Performs an element-by-element shift of int32 vector , shifting
 *    by the number of bits given by count.  If the count is greater than
 *    31 then for logical shifts the result is zero, and for arithmetic
 *    right shifts the result is all ones or all zeroes depending on the
 *    original sign bit.
 *
 *    slli   logical shift left
 *    srli   logical shift right
 *    srai   arithmetic shift right
 */

extern __m512i __ICL_INTRINCC _mm512_slli_epi32(__m512i,
                                                unsigned int /* count */);
extern __m512i __ICL_INTRINCC _mm512_mask_slli_epi32(__m512i, __mmask16,
                                                     __m512i, unsigned int);

extern __m512i __ICL_INTRINCC _mm512_srai_epi32(__m512i, unsigned int);
extern __m512i __ICL_INTRINCC _mm512_mask_srai_epi32(__m512i, __mmask16,
                                                     __m512i, unsigned int);

extern __m512i __ICL_INTRINCC _mm512_srli_epi32(__m512i, unsigned int);
extern __m512i __ICL_INTRINCC _mm512_mask_srli_epi32(__m512i, __mmask16,
                                                     __m512i, unsigned int);

/*
 * Logical AND and set vector mask.
 *
 *    Performs an element-by-element bitwise AND between int32 vectors
 *    and uses the result to construct a 16-bit
 *    vector mask, with a 0-bit for each element for which the result of
 *    the AND was 0, and a 1-bit where the result of the AND was not zero.
 *    Vector mask is returned.
 *
 *    The writemask does not perform the normal writemasking function
 *    for this instruction.  While it does enable/disable comparisons,
 *    it does not block updating of the result; instead, if a writemask
 *    bit is 0, the corresponding destination bit is set to 0.
 */

extern __mmask16 __ICL_INTRINCC _mm512_test_epi32_mask(__m512i, __m512i);
extern __mmask16 __ICL_INTRINCC _mm512_mask_test_epi32_mask(__mmask16, __m512i,
                                                            __m512i);

/*
 * Return 512 vector with undefined elements.  It is recommended to use the
 * result of this intrinsic as the old value for masked versions of intrinsics
 * when the old values will never be meaningfully used.
 */
extern __m512 __ICL_INTRINCC _mm512_undefined(void);
#define _mm512_undefined_pd() _mm512_castps_pd(_mm512_undefined())
#define _mm512_undefined_ps() _mm512_undefined()
#define _mm512_undefined_epi32() _mm512_castps_si512(_mm512_undefined())

/*
 * Return 512 vector with all elements 0.
 */
extern __m512 __ICL_INTRINCC _mm512_setzero(void);

#define _mm512_setzero_pd() _mm512_castps_pd(_mm512_setzero())
#define _mm512_setzero_ps() _mm512_setzero()
#define _mm512_setzero_epi32() _mm512_castps_si512(_mm512_setzero())

/*
 * Return float64 vector with all 8 elements equal to given scalar.
 */
extern __m512d __ICL_INTRINCC _mm512_set1_pd(double);
#define _mm512_set_1to8_pd(x) _mm512_set1_pd((x))

/*
 * Return int64 vector with all 8 elements equal to given scalar.
 */
extern __m512i __ICL_INTRINCC _mm512_set1_epi64(__int64);
#define _mm512_set_1to8_pq(x) _mm512_set1_epi64((x))
#define _mm512_set_1to8_epi64(x) _mm512_set1_epi64((x))

/*
 * Return float32 vector with all 16 elements equal to given scalar.
 */
extern __m512  __ICL_INTRINCC _mm512_set1_ps(float);
#define _mm512_set_1to16_ps(x) _mm512_set1_ps((x))

/*
 * Return int32 vector with all 16 elements equal to given scalar.
 */
extern __m512i __ICL_INTRINCC _mm512_set1_epi32(int);
#define _mm512_set_1to16_pi(x) _mm512_set1_epi32((x))
#define _mm512_set_1to16_epi32(x) _mm512_set1_epi32((x))

/*
 * Return float64 vector dcbadcba.
 * (v4, v0 = a; v5, v1 = b; v6, v2 = c; v7, v3 = d).
 */
extern __m512d __ICL_INTRINCC _mm512_set4_pd(double /* d */, double /* c */,
                                             double /* b */, double /* a */);
#define _mm512_setr4_pd(a,b,c,d) \
    _mm512_set4_pd((d),(c),(b),(a))
#define _mm512_set_4to8_pd(a,b,c,d) \
    _mm512_set4_pd((d),(c),(b),(a))

/*
 * Return int64 vector dcbadcba.
 * (v4, v0 = a; v5, v1 = b; v6, v2 = c; v7, v3 = d).
 */
extern __m512i __ICL_INTRINCC _mm512_set4_epi64(__int64 /* d */,
                                                __int64 /* c */,
                                                __int64 /* b */,
                                                __int64 /* a */);
#define _mm512_setr4_epi64(a,b,c,d) \
    _mm512_set4_epi64((d),(c),(b),(a))
#define _mm512_set_4to8_pq(a,b,c,d) \
    _mm512_set4_epi64((d),(c),(b),(a))
#define _mm512_set_4to8_epi64(a,b,c,d) \
    _mm512_set4_epi64((d),(c),(b),(a))

/*
 * Return float32 vector dcbadcbadcbadcba.
 * (v12, v8, v4, v0 = a; v13, v9, v5, v1 = b; v14, v10, v6, v2 = c;
 *  v15, v11, v7, v3 = d).
 */
extern __m512  __ICL_INTRINCC _mm512_set4_ps(float /* d */, float /* c */,
                                             float /* b */, float /* a */);
#define _mm512_setr4_ps(a,b,c,d) \
    _mm512_set4_ps((d),(c),(b),(a))
#define _mm512_set_4to16_ps(a,b,c,d) \
    _mm512_set4_ps((d),(c),(b),(a))

/*
 * Return int32 vector dcbadcbadcbadcba.
 * (v12, v8, v4, v0 = a; v13, v9, v5, v1 = b; v14, v10, v6, v2 = c;
 *  v15, v11, v7, v3 = d).
 */
extern __m512i __ICL_INTRINCC _mm512_set4_epi32(int /* d */, int /* c */,
                                                int /* b */, int /* a */);

#define _mm512_setr4_epi32(a,b,c,d) \
    _mm512_set4_epi32((d),(c),(b),(a))
#define _mm512_set_4to16_pi(a,b,c,d) \
    _mm512_set4_epi32((d),(c),(b),(a))
#define _mm512_set_4to16_epi32(a,b,c,d) \
    _mm512_set4_epi32((d),(c),(b),(a))

/*
 * Return float32 vector e15 e14 e13 ... e1 e0 (v15=e15, v14=e14, ..., v0=e0).
 */
extern __m512 __ICL_INTRINCC _mm512_set_ps(float /* e15 */, float, float,
                                           float, float, float,
                                           float, float, float,
                                           float, float, float,
                                           float, float, float,
                                           float /* e0 */);
#define _mm512_setr_ps(e0,e1,e2,e3,e4,e5,e6,e7,e8, \
                       e9,e10,e11,e12,e13,e14,e15) \
    _mm512_set_ps((e15),(e14),(e13),(e12),(e11),(e10), \
                  (e9),(e8),(e7),(e6),(e5),(e4),(e3),(e2),(e1),(e0))

#define _mm512_set_16to16_ps(e0,e1,e2,e3,e4,e5,e6,e7,e8, \
                             e9,e10,e11,e12,e13,e14,e15) \
    _mm512_set_ps((e0),(e1),(e2),(e3),(e4),(e5),(e6),(e7), \
                  (e8),(e9),(e10),(e11),(e12),(e13),(e14),(e15))

/*
 * Return int32 vector e15 e14 e13 ... e1 e0 (v15=e15, v14=e14, ..., v0=e0).
 */
extern __m512i __ICL_INTRINCC _mm512_set_epi32(int /* e15 */, int, int, int,
                                               int, int, int, int,
                                               int, int, int, int,
                                               int, int, int, int /* e0 */);

#define _mm512_setr_epi32(e0,e1,e2,e3,e4,e5,e6,e7,e8, \
                          e9,e10,e11,e12,e13,e14,e15) \
    _mm512_set_epi32((e15),(e14),(e13),(e12),(e11),(e10), \
                     (e9),(e8),(e7),(e6),(e5),(e4),(e3),(e2),(e1),(e0))

#define _mm512_set_16to16_pi(e0,e1,e2,e3,e4,e5,e6,e7,e8, \
                             e9,e10,e11,e12,e13,e14,e15) \
    _mm512_set_epi32((e0),(e1),(e2),(e3),(e4),(e5),(e6),(e7), \
                     (e8),(e9),(e10),(e11),(e12),(e13),(e14),(e15))

#define _mm512_set_16to16_epi32(e0,e1,e2,e3,e4,e5,e6,e7,e8, \
                                e9,e10,e11,e12,e13,e14,e15) \
    _mm512_set_epi32((e0),(e1),(e2),(e3),(e4),(e5),(e6),(e7), \
                     (e8),(e9),(e10),(e11),(e12),(e13),(e14),(e15))


/*
 * Return float64 vector e7 e6 e5 ... e1 e0 (v7=e7, v6=e6, ..., v0=e0).
 */
extern __m512d __ICL_INTRINCC _mm512_set_pd(double /* e7 */, double, double,
                                            double, double, double,
                                            double, double /* e0 */);

#define _mm512_setr_pd(e0,e1,e2,e3,e4,e5,e6,e7) \
    _mm512_set_pd((e7),(e6),(e5),(e4),(e3),(e2),(e1),(e0))
#define _mm512_set_8to8_pd(e0,e1,e2,e3,e4,e5,e6,e7) \
    _mm512_set_pd((e0),(e1),(e2),(e3),(e4),(e5),(e6),(e7))

/*
 * Return int64 vector e7 e6 e5 ... e1 e0 (v7=e7, v6=e6, ..., v0=e0).
 */
extern __m512i __ICL_INTRINCC _mm512_set_epi64(__int64 /* e7 */, __int64,
                                               __int64, __int64,
                                               __int64, __int64,
                                               __int64, __int64 /* e0 */);

#define _mm512_setr_epi64(e0,e1,e2,e3,e4,e5,e6,e7) \
    _mm512_set_epi64((e7),(e6),(e5),(e4),(e3),(e2),(e1),(e0))

#define _mm512_set_8to8_pq(e0,e1,e2,e3,e4,e5,e6,e7) \
    _mm512_set_epi64((e0),(e1),(e2),(e3),(e4),(e5),(e6),(e7))

#define _mm512_set_8to8_epi64(e0,e1,e2,e3,e4,e5,e6,e7) \
    _mm512_set_epi64((e0),(e1),(e2),(e3),(e4),(e5),(e6),(e7))


/*
 * Math intrinsics.
 */

extern __m512d __ICL_INTRINCC _mm512_acos_pd(__m512d);
extern __m512d __ICL_INTRINCC _mm512_mask_acos_pd(__m512d, __mmask8, __m512d);

extern __m512  __ICL_INTRINCC _mm512_acos_ps(__m512);
extern __m512  __ICL_INTRINCC _mm512_mask_acos_ps(__m512, __mmask16, __m512);

extern __m512d __ICL_INTRINCC _mm512_acosh_pd(__m512d);
extern __m512d __ICL_INTRINCC _mm512_mask_acosh_pd(__m512d, __mmask8, __m512d);

extern __m512  __ICL_INTRINCC _mm512_acosh_ps(__m512);
extern __m512  __ICL_INTRINCC _mm512_mask_acosh_ps(__m512, __mmask16, __m512);

extern __m512d __ICL_INTRINCC _mm512_asin_pd(__m512d);
extern __m512d __ICL_INTRINCC _mm512_mask_asin_pd(__m512d, __mmask8, __m512d);

extern __m512  __ICL_INTRINCC _mm512_asin_ps(__m512);
extern __m512  __ICL_INTRINCC _mm512_mask_asin_ps(__m512, __mmask16, __m512);

extern __m512d __ICL_INTRINCC _mm512_asinh_pd(__m512d);
extern __m512d __ICL_INTRINCC _mm512_mask_asinh_pd(__m512d, __mmask8, __m512d);

extern __m512  __ICL_INTRINCC _mm512_asinh_ps(__m512);
extern __m512  __ICL_INTRINCC _mm512_mask_asinh_ps(__m512, __mmask16, __m512);

extern __m512d __ICL_INTRINCC _mm512_atan2_pd(__m512d, __m512d);
extern __m512d __ICL_INTRINCC _mm512_mask_atan2_pd(__m512d, __mmask8, __m512d,
                                                   __m512d);

extern __m512  __ICL_INTRINCC _mm512_atan2_ps(__m512, __m512);
extern __m512  __ICL_INTRINCC _mm512_mask_atan2_ps(__m512, __mmask16, __m512,
                                                   __m512);

extern __m512d __ICL_INTRINCC _mm512_atan_pd(__m512d);
extern __m512d __ICL_INTRINCC _mm512_mask_atan_pd(__m512d, __mmask8, __m512d);

extern __m512  __ICL_INTRINCC _mm512_atan_ps(__m512);
extern __m512  __ICL_INTRINCC _mm512_mask_atan_ps(__m512, __mmask16, __m512);

extern __m512d __ICL_INTRINCC _mm512_atanh_pd(__m512d);
extern __m512d __ICL_INTRINCC _mm512_mask_atanh_pd(__m512d, __mmask8, __m512d);

extern __m512  __ICL_INTRINCC _mm512_atanh_ps(__m512);
extern __m512  __ICL_INTRINCC _mm512_mask_atanh_ps(__m512, __mmask16, __m512);

extern __m512d __ICL_INTRINCC _mm512_cbrt_pd(__m512d);
extern __m512d __ICL_INTRINCC _mm512_mask_cbrt_pd(__m512d, __mmask8, __m512d);

extern __m512  __ICL_INTRINCC _mm512_cbrt_ps(__m512);
extern __m512  __ICL_INTRINCC _mm512_mask_cbrt_ps(__m512, __mmask16, __m512);

extern __m512d __ICL_INTRINCC _mm512_cdfnorminv_pd(__m512d);
extern __m512d __ICL_INTRINCC _mm512_mask_cdfnorminv_pd(__m512d, __mmask8,
                                                        __m512d);

extern __m512  __ICL_INTRINCC _mm512_cdfnorminv_ps(__m512);
extern __m512  __ICL_INTRINCC _mm512_mask_cdfnorminv_ps(__m512, __mmask16,
                                                        __m512);

extern __m512d __ICL_INTRINCC _mm512_ceil_pd(__m512d);
extern __m512d __ICL_INTRINCC _mm512_mask_ceil_pd(__m512d, __mmask8, __m512d);

extern __m512  __ICL_INTRINCC _mm512_ceil_ps(__m512);
extern __m512  __ICL_INTRINCC _mm512_mask_ceil_ps(__m512, __mmask16, __m512);

extern __m512d __ICL_INTRINCC _mm512_cos_pd(__m512d);
extern __m512d __ICL_INTRINCC _mm512_mask_cos_pd(__m512d, __mmask8, __m512d);

extern __m512  __ICL_INTRINCC _mm512_cos_ps(__m512);
extern __m512  __ICL_INTRINCC _mm512_mask_cos_ps(__m512, __mmask16, __m512);

extern __m512d __ICL_INTRINCC _mm512_cosd_pd(__m512d);
extern __m512d __ICL_INTRINCC _mm512_mask_cosd_pd(__m512d, __mmask8, __m512d);

extern __m512  __ICL_INTRINCC _mm512_cosd_ps(__m512);
extern __m512  __ICL_INTRINCC _mm512_mask_cosd_ps(__m512, __mmask16, __m512);

extern __m512d __ICL_INTRINCC _mm512_cosh_pd(__m512d);
extern __m512d __ICL_INTRINCC _mm512_mask_cosh_pd(__m512d, __mmask8, __m512d);

extern __m512  __ICL_INTRINCC _mm512_cosh_ps(__m512);
extern __m512  __ICL_INTRINCC _mm512_mask_cosh_ps(__m512, __mmask16, __m512);

extern __m512d __ICL_INTRINCC _mm512_erf_pd(__m512d);
extern __m512d __ICL_INTRINCC _mm512_mask_erf_pd(__m512d, __mmask8, __m512d);

extern __m512d __ICL_INTRINCC _mm512_erfc_pd(__m512d);
extern __m512d __ICL_INTRINCC _mm512_mask_erfc_pd(__m512d, __mmask8, __m512d);

extern __m512  __ICL_INTRINCC _mm512_erf_ps(__m512);
extern __m512  __ICL_INTRINCC _mm512_mask_erf_ps(__m512, __mmask16, __m512);

extern __m512  __ICL_INTRINCC _mm512_erfc_ps(__m512);
extern __m512  __ICL_INTRINCC _mm512_mask_erfc_ps(__m512, __mmask16, __m512);

extern __m512d __ICL_INTRINCC _mm512_erfinv_pd(__m512d);
extern __m512d __ICL_INTRINCC _mm512_mask_erfinv_pd(__m512d, __mmask8,
                                                    __m512d);

extern __m512  __ICL_INTRINCC _mm512_erfinv_ps(__m512);
extern __m512  __ICL_INTRINCC _mm512_mask_erfinv_ps(__m512, __mmask16, __m512);

extern __m512d __ICL_INTRINCC _mm512_exp10_pd(__m512d);
extern __m512d __ICL_INTRINCC _mm512_mask_exp10_pd(__m512d, __mmask8, __m512d);

extern __m512  __ICL_INTRINCC _mm512_exp10_ps(__m512);
extern __m512  __ICL_INTRINCC _mm512_mask_exp10_ps(__m512, __mmask16, __m512);

extern __m512d __ICL_INTRINCC _mm512_exp2_pd(__m512d);
extern __m512d __ICL_INTRINCC _mm512_mask_exp2_pd(__m512d, __mmask8, __m512d);

extern __m512  __ICL_INTRINCC _mm512_exp2_ps(__m512);
extern __m512  __ICL_INTRINCC _mm512_mask_exp2_ps(__m512, __mmask16, __m512);

extern __m512d __ICL_INTRINCC _mm512_exp_pd(__m512d);
extern __m512d __ICL_INTRINCC _mm512_mask_exp_pd(__m512d, __mmask8, __m512d);

extern __m512  __ICL_INTRINCC _mm512_exp_ps(__m512);
extern __m512  __ICL_INTRINCC _mm512_mask_exp_ps(__m512, __mmask16, __m512);

extern __m512d __ICL_INTRINCC _mm512_expm1_pd(__m512d);
extern __m512d __ICL_INTRINCC _mm512_mask_expm1_pd(__m512d, __mmask8, __m512d);

extern __m512  __ICL_INTRINCC _mm512_expm1_ps(__m512);
extern __m512  __ICL_INTRINCC _mm512_mask_expm1_ps(__m512, __mmask16, __m512);

extern __m512d __ICL_INTRINCC _mm512_floor_pd(__m512d);
extern __m512d __ICL_INTRINCC _mm512_mask_floor_pd(__m512d, __mmask8, __m512d);

extern __m512  __ICL_INTRINCC _mm512_floor_ps(__m512);
extern __m512  __ICL_INTRINCC _mm512_mask_floor_ps(__m512, __mmask16, __m512);

extern __m512d __ICL_INTRINCC _mm512_hypot_pd(__m512d, __m512d);
extern __m512d __ICL_INTRINCC _mm512_mask_hypot_pd(__m512d, __mmask8, __m512d,
                                                   __m512d);

extern __m512  __ICL_INTRINCC _mm512_hypot_ps(__m512, __m512);
extern __m512  __ICL_INTRINCC _mm512_mask_hypot_ps(__m512, __mmask16, __m512,
                                                   __m512);

extern __m512i __ICL_INTRINCC _mm512_div_epi32(__m512i, __m512i);
extern __m512i __ICL_INTRINCC _mm512_mask_div_epi32(__m512i, __mmask16,
                                                    __m512i, __m512i);

extern __m512i __ICL_INTRINCC _mm512_div_epi8(__m512i, __m512i);
extern __m512i __ICL_INTRINCC _mm512_div_epi16(__m512i, __m512i);
extern __m512i __ICL_INTRINCC _mm512_div_epi64(__m512i, __m512i);

extern __m512  __ICL_INTRINCC _mm512_div_ps(__m512, __m512);
extern __m512  __ICL_INTRINCC _mm512_mask_div_ps(__m512, __mmask16,
                                                 __m512, __m512);

extern __m512d __ICL_INTRINCC _mm512_div_pd(__m512d, __m512d);
extern __m512d __ICL_INTRINCC _mm512_mask_div_pd(__m512d, __mmask8,
                                                 __m512d, __m512d);

extern __m512d __ICL_INTRINCC _mm512_invsqrt_pd(__m512d);
extern __m512d __ICL_INTRINCC _mm512_mask_invsqrt_pd(__m512d, __mmask8,
                                                     __m512d);

extern __m512  __ICL_INTRINCC _mm512_invsqrt_ps(__m512);
extern __m512  __ICL_INTRINCC _mm512_mask_invsqrt_ps(__m512, __mmask16,
                                                     __m512);

extern __m512i __ICL_INTRINCC _mm512_rem_epi32(__m512i, __m512i);
extern __m512i __ICL_INTRINCC _mm512_mask_rem_epi32(__m512i, __mmask16,
                                                    __m512i, __m512i);

extern __m512i __ICL_INTRINCC _mm512_rem_epi8(__m512i, __m512i);
extern __m512i __ICL_INTRINCC _mm512_rem_epi16(__m512i, __m512i);
extern __m512i __ICL_INTRINCC _mm512_rem_epi64(__m512i, __m512i);

extern __m512d __ICL_INTRINCC _mm512_log10_pd(__m512d);
extern __m512d __ICL_INTRINCC _mm512_mask_log10_pd(__m512d, __mmask8, __m512d);

extern __m512  __ICL_INTRINCC _mm512_log10_ps(__m512);
extern __m512  __ICL_INTRINCC _mm512_mask_log10_ps(__m512, __mmask16, __m512);

extern __m512d __ICL_INTRINCC _mm512_log1p_pd(__m512d);
extern __m512d __ICL_INTRINCC _mm512_mask_log1p_pd(__m512d, __mmask8, __m512d);

extern __m512  __ICL_INTRINCC _mm512_log1p_ps(__m512);
extern __m512  __ICL_INTRINCC _mm512_mask_log1p_ps(__m512, __mmask16, __m512);

extern __m512d __ICL_INTRINCC _mm512_log2_pd(__m512d);
extern __m512d __ICL_INTRINCC _mm512_mask_log2_pd(__m512d, __mmask8, __m512d);

extern __m512  __ICL_INTRINCC _mm512_log2_ps(__m512);
extern __m512  __ICL_INTRINCC _mm512_mask_log2_ps(__m512, __mmask16, __m512);

extern __m512d __ICL_INTRINCC _mm512_log_pd(__m512d);
extern __m512d __ICL_INTRINCC _mm512_mask_log_pd(__m512d, __mmask8, __m512d);

extern __m512  __ICL_INTRINCC _mm512_log_ps(__m512);
extern __m512  __ICL_INTRINCC _mm512_mask_log_ps(__m512, __mmask16, __m512);

extern __m512d __ICL_INTRINCC _mm512_logb_pd(__m512d);
extern __m512d __ICL_INTRINCC _mm512_mask_logb_pd(__m512d, __mmask8, __m512d);

extern __m512  __ICL_INTRINCC _mm512_logb_ps(__m512);
extern __m512  __ICL_INTRINCC _mm512_mask_logb_ps(__m512, __mmask16, __m512);

extern __m512d __ICL_INTRINCC _mm512_nearbyint_pd(__m512d);
extern __m512d __ICL_INTRINCC _mm512_mask_nearbyint_pd(__m512d, __mmask8,
                                                       __m512d);

extern __m512  __ICL_INTRINCC _mm512_nearbyint_ps(__m512);
extern __m512  __ICL_INTRINCC _mm512_mask_nearbyint_ps(__m512, __mmask16,
                                                       __m512);

extern __m512d __ICL_INTRINCC _mm512_pow_pd(__m512d, __m512d);
extern __m512d __ICL_INTRINCC _mm512_mask_pow_pd(__m512d, __mmask8,
                                                 __m512d, __m512d);

extern __m512  __ICL_INTRINCC _mm512_pow_ps(__m512, __m512);
extern __m512  __ICL_INTRINCC _mm512_mask_pow_ps(__m512, __mmask16,
                                                 __m512, __m512);

extern __m512d __ICL_INTRINCC _mm512_recip_pd(__m512d);
extern __m512d __ICL_INTRINCC _mm512_mask_recip_pd(__m512d, __mmask8, __m512d);

extern __m512  __ICL_INTRINCC _mm512_recip_ps(__m512);
extern __m512  __ICL_INTRINCC _mm512_mask_recip_ps(__m512, __mmask16, __m512);

extern __m512d __ICL_INTRINCC _mm512_rint_pd(__m512d);
extern __m512d __ICL_INTRINCC _mm512_mask_rint_pd(__m512d, __mmask8, __m512d);

extern __m512  __ICL_INTRINCC _mm512_rint_ps(__m512);
extern __m512  __ICL_INTRINCC _mm512_mask_rint_ps(__m512, __mmask16, __m512);

extern __m512d __ICL_INTRINCC _mm512_svml_round_pd(__m512d);
extern __m512d __ICL_INTRINCC _mm512_mask_svml_round_pd(__m512d, __mmask8,
                                                        __m512d);

extern __m512d __ICL_INTRINCC _mm512_sin_pd(__m512d);
extern __m512d __ICL_INTRINCC _mm512_mask_sin_pd(__m512d, __mmask8, __m512d);

extern __m512  __ICL_INTRINCC _mm512_sin_ps(__m512);
extern __m512  __ICL_INTRINCC _mm512_mask_sin_ps(__m512, __mmask16, __m512);

extern __m512d __ICL_INTRINCC _mm512_sinh_pd(__m512d);
extern __m512d __ICL_INTRINCC _mm512_mask_sinh_pd(__m512d, __mmask8, __m512d);

extern __m512  __ICL_INTRINCC _mm512_sinh_ps(__m512);
extern __m512  __ICL_INTRINCC _mm512_mask_sinh_ps(__m512, __mmask16, __m512);

extern __m512d __ICL_INTRINCC _mm512_sind_pd(__m512d);
extern __m512d __ICL_INTRINCC _mm512_mask_sind_pd(__m512d, __mmask8, __m512d);

extern __m512  __ICL_INTRINCC _mm512_sind_ps(__m512);
extern __m512  __ICL_INTRINCC _mm512_mask_sind_ps(__m512, __mmask16, __m512);

extern __m512d __ICL_INTRINCC _mm512_sqrt_pd(__m512d);
extern __m512d __ICL_INTRINCC _mm512_mask_sqrt_pd(__m512d, __mmask8, __m512d);

extern __m512  __ICL_INTRINCC _mm512_sqrt_ps(__m512);
extern __m512  __ICL_INTRINCC _mm512_mask_sqrt_ps(__m512, __mmask16, __m512);

extern __m512d __ICL_INTRINCC _mm512_tan_pd(__m512d);
extern __m512d __ICL_INTRINCC _mm512_mask_tan_pd(__m512d, __mmask8, __m512d);

extern __m512  __ICL_INTRINCC _mm512_tan_ps(__m512);
extern __m512  __ICL_INTRINCC _mm512_mask_tan_ps(__m512, __mmask16, __m512);

extern __m512d __ICL_INTRINCC _mm512_tand_pd(__m512d);
extern __m512d __ICL_INTRINCC _mm512_mask_tand_pd(__m512d, __mmask8, __m512d);

extern __m512  __ICL_INTRINCC _mm512_tand_ps(__m512);
extern __m512  __ICL_INTRINCC _mm512_mask_tand_ps(__m512, __mmask16, __m512);

extern __m512d __ICL_INTRINCC _mm512_tanh_pd(__m512d);
extern __m512d __ICL_INTRINCC _mm512_mask_tanh_pd(__m512d, __mmask8, __m512d);

extern __m512  __ICL_INTRINCC _mm512_tanh_ps(__m512);
extern __m512  __ICL_INTRINCC _mm512_mask_tanh_ps(__m512, __mmask16, __m512);

extern __m512d __ICL_INTRINCC _mm512_trunc_pd(__m512d);
extern __m512d __ICL_INTRINCC _mm512_mask_trunc_pd(__m512d, __mmask8, __m512d);

extern __m512  __ICL_INTRINCC _mm512_trunc_ps(__m512);
extern __m512  __ICL_INTRINCC _mm512_mask_trunc_ps(__m512, __mmask16, __m512);

extern __m512i __ICL_INTRINCC _mm512_div_epu32(__m512i, __m512i);
extern __m512i __ICL_INTRINCC _mm512_mask_div_epu32(__m512i, __mmask16,
                                                    __m512i, __m512i);

extern __m512i __ICL_INTRINCC _mm512_div_epu8(__m512i, __m512i);
extern __m512i __ICL_INTRINCC _mm512_div_epu16(__m512i, __m512i);
extern __m512i __ICL_INTRINCC _mm512_div_epu64(__m512i, __m512i);

extern __m512i __ICL_INTRINCC _mm512_rem_epu32(__m512i, __m512i);
extern __m512i __ICL_INTRINCC _mm512_mask_rem_epu32(__m512i, __mmask16,
                                                    __m512i, __m512i);

extern __m512i __ICL_INTRINCC _mm512_rem_epu8(__m512i, __m512i);
extern __m512i __ICL_INTRINCC _mm512_rem_epu16(__m512i, __m512i);
extern __m512i __ICL_INTRINCC _mm512_rem_epu64(__m512i, __m512i);

/*
 * Reduction intrinsics - perform corresponding operation on all elements
 * of source vector and return scalar value.
 * For example, _mm512_reduce_add_ps returns float32 value
 * calculated as v1[0] + v1[1] + ... + v1[15].
 */
extern float   __ICL_INTRINCC _mm512_reduce_add_ps(__m512);
extern float   __ICL_INTRINCC _mm512_mask_reduce_add_ps(__mmask16, __m512);

extern double  __ICL_INTRINCC _mm512_reduce_add_pd(__m512d);
extern double  __ICL_INTRINCC _mm512_mask_reduce_add_pd(__mmask8, __m512d);

extern int     __ICL_INTRINCC _mm512_reduce_add_epi32(__m512i);
extern int     __ICL_INTRINCC _mm512_mask_reduce_add_epi32(__mmask16, __m512i);

extern float   __ICL_INTRINCC _mm512_reduce_mul_ps(__m512);
extern float   __ICL_INTRINCC _mm512_mask_reduce_mul_ps(__mmask16, __m512);

extern double  __ICL_INTRINCC _mm512_reduce_mul_pd(__m512d);
extern double  __ICL_INTRINCC _mm512_mask_reduce_mul_pd(__mmask8, __m512d);

extern int     __ICL_INTRINCC _mm512_reduce_mul_epi32(__m512i);
extern int     __ICL_INTRINCC _mm512_mask_reduce_mul_epi32(__mmask16, __m512i);

extern float   __ICL_INTRINCC _mm512_reduce_min_ps(__m512);
extern float   __ICL_INTRINCC _mm512_mask_reduce_min_ps(__mmask16, __m512);

extern double  __ICL_INTRINCC _mm512_reduce_min_pd(__m512d);
extern double  __ICL_INTRINCC _mm512_mask_reduce_min_pd(__mmask8, __m512d);

extern int     __ICL_INTRINCC _mm512_reduce_min_epi32(__m512i);
extern int     __ICL_INTRINCC _mm512_mask_reduce_min_epi32(__mmask16, __m512i);

extern unsigned int __ICL_INTRINCC _mm512_reduce_min_epu32(__m512i);
extern unsigned int __ICL_INTRINCC _mm512_mask_reduce_min_epu32(__mmask16,
                                                                __m512i);

extern float   __ICL_INTRINCC _mm512_reduce_max_ps(__m512);
extern float   __ICL_INTRINCC _mm512_mask_reduce_max_ps(__mmask16, __m512);

extern double  __ICL_INTRINCC _mm512_reduce_max_pd(__m512d);
extern double  __ICL_INTRINCC _mm512_mask_reduce_max_pd(__mmask8, __m512d);

extern int     __ICL_INTRINCC _mm512_reduce_max_epi32(__m512i);
extern int     __ICL_INTRINCC _mm512_mask_reduce_max_epi32(__mmask16, __m512i);

extern unsigned int __ICL_INTRINCC _mm512_reduce_max_epu32(__m512i);
extern unsigned int __ICL_INTRINCC _mm512_mask_reduce_max_epu32(__mmask16,
                                                                __m512i);

extern int     __ICL_INTRINCC _mm512_reduce_or_epi32(__m512i);
extern int     __ICL_INTRINCC _mm512_mask_reduce_or_epi32(__mmask16, __m512i);

extern int     __ICL_INTRINCC _mm512_reduce_and_epi32(__m512i);
extern int     __ICL_INTRINCC _mm512_mask_reduce_and_epi32(__mmask16, __m512i);

extern float   __ICL_INTRINCC _mm512_reduce_gmin_ps(__m512);
extern float   __ICL_INTRINCC _mm512_mask_reduce_gmin_ps(__mmask16, __m512);

extern double  __ICL_INTRINCC _mm512_reduce_gmin_pd(__m512d);
extern double  __ICL_INTRINCC _mm512_mask_reduce_gmin_pd(__mmask8, __m512d);

extern float   __ICL_INTRINCC _mm512_reduce_gmax_ps(__m512);
extern float   __ICL_INTRINCC _mm512_mask_reduce_gmax_ps(__mmask16, __m512);

extern double  __ICL_INTRINCC _mm512_reduce_gmax_pd(__m512d);
extern double  __ICL_INTRINCC _mm512_mask_reduce_gmax_pd(__mmask8, __m512d);

/*
 * Scalar intrinsics.
 */

/* Trailing zero bit count */
extern int            __ICL_INTRINCC _mm_tzcnt_32(unsigned int);
extern __int64        __ICL_INTRINCC _mm_tzcnt_64(unsigned __int64);

/* Initialized trailing zero bit count */
extern int            __ICL_INTRINCC _mm_tzcnti_32(int, unsigned int);
extern __int64        __ICL_INTRINCC _mm_tzcnti_64(__int64, unsigned __int64);

/* Bit population count */
extern unsigned int      __ICL_INTRINCC _mm_countbits_32(unsigned int);
extern unsigned __int64  __ICL_INTRINCC _mm_countbits_64(unsigned __int64);

/* Stall thread.
 *
 *    Stall thread for specified clock without blocking other threads.
 *    Hints that the processor should not fetch/issue instructions for the
 *    current thread for the specified number of clock cycles.
 *    Any of the following events will cause the processor to start fetching
 *    instructions for the delayed thread again: the counter counting down
 *    to zero, an interrupt, an NMI or SMI, a debug exception, a machine check
 *    exception, the BINIT# signal, the INIT# signal, or the RESET# signal.
 *    Note that an interrupt will cause the processor to start fetching
 *    instructions for that thread only if the state was entered with
 *    interrupts enabled.
 */
extern void __ICL_INTRINCC _mm_delay_32(unsigned int);
extern void __ICL_INTRINCC _mm_delay_64(unsigned __int64);


/*
 * Set performance monitor filtering mask for current thread.
 */
extern void __ICL_INTRINCC _mm_spflt_32(unsigned int);
extern void __ICL_INTRINCC _mm_spflt_64(unsigned __int64);

/*
 * Evict cache line from specified cache level:
 * _MM_HINT_T0 -- first level
 * _MM_HINT_T1 -- second level
 */

extern void __ICL_INTRINCC _mm_clevict(const void*, int /* level */);

/*
 * Mask arithmetic operations.
 */
extern __mmask16 __ICL_INTRINCC _mm512_kand     (__mmask16, __mmask16);
extern __mmask16 __ICL_INTRINCC _mm512_kandn    (__mmask16, __mmask16);
extern __mmask16 __ICL_INTRINCC _mm512_kandnr   (__mmask16, __mmask16);
extern __mmask16 __ICL_INTRINCC _mm512_kmovlhb  (__mmask16, __mmask16);
extern __mmask16 __ICL_INTRINCC _mm512_knot     (__mmask16);
extern __mmask16 __ICL_INTRINCC _mm512_kor      (__mmask16, __mmask16);
extern __mmask16 __ICL_INTRINCC _mm512_kxnor    (__mmask16, __mmask16);
extern __mmask16 __ICL_INTRINCC _mm512_kxor     (__mmask16, __mmask16);
extern __mmask16 __ICL_INTRINCC _mm512_kswapb   (__mmask16, __mmask16);
extern int       __ICL_INTRINCC _mm512_kortestz (__mmask16, __mmask16);
extern int       __ICL_INTRINCC _mm512_kortestc (__mmask16, __mmask16);
extern __mmask16 __ICL_INTRINCC _mm512_kmov     (__mmask16);
extern int       __ICL_INTRINCC _mm512_mask2int (__mmask16);
extern __mmask16 __ICL_INTRINCC _mm512_int2mask (int);
extern __int64   __ICL_INTRINCC _mm512_kconcathi_64(__mmask16, __mmask16);
extern __int64   __ICL_INTRINCC _mm512_kconcatlo_64(__mmask16, __mmask16);
extern __mmask16 __ICL_INTRINCC _mm512_kextract_64(__int64,
                                                   const int /* select */);

#define _mm512_kmerge2l1h(k1, k2) _mm512_kswapb((k1), (k2))
#define _mm512_kmerge2l1l(k1, k2) _mm512_kmovlhb((k1), (k2))


#ifdef __cplusplus
};
#endif /* __cplusplus */

#endif /* _ZMMINTRIN_H_INCLUDED */
