#ifndef MEMUTILS_HEADER
#define MEMUTILS_HEADER

#include <xmmintrin.h>

void evict_array_from_cache (void* ptr, size_t size) {
  const size_t CACHELINE = 64;
  
  void* ptrend = ptr + size;
  for (; ptr< ptrend; ptr += CACHELINE) {
#ifdef MIC
    _mm_clevict(ptr,   _MM_HINT_T0);
    _mm_clevict(ptr,   _MM_HINT_T1);
#else
    _mm_clflush(ptr);
#endif
  }
}

#endif
