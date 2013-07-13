#ifndef __PADDED_2D_ARRAY__
#define __PADDED_2D_ARRAY__

#include <assert.h>
#include <stddef.h>

template<class T>
class Padded2DArray
{
  T* data;
  size_t paddedsize;
  int nbblock;
public:
  //most likely, there is a better way of doing this but that should be enough for now
  Padded2DArray(int nbblock, int nbelemperblock, int CACHELINE = 64)
    :nbblock(nbblock)
  {
//     int CACHELINE = cache_line_size();
    if (nbblock == 1)
      paddedsize = nbelemperblock;
    else
      {
	paddedsize = nbelemperblock;
	if (paddedsize*sizeof(T) % CACHELINE)
	  {
	    unsigned int d_orig = paddedsize*sizeof(T)/CACHELINE;
	    while (paddedsize*sizeof(T)/CACHELINE == d_orig)
	      paddedsize++;
	  }
      }
    data = new T[nbblock*paddedsize];
    paddedsize *= sizeof(T);
  }

  ~Padded2DArray()
  {
    close();
  }
  
  void close() {
      delete[] data;
  }

  inline T* operator[] (int i)
  {
    assert (i<nbblock);
    return (T*)((char*)data+i*paddedsize); //converting to char avoir a multiplication in assembly
  }

  inline const T* operator[] (int i) const
  {
    assert (i<nbblock);
    return (T*)((char*)data+i*paddedsize); //converting to char avoir a multiplication in assembly
  }
}; 


//the cacheline is not 64bytes on all machines. But it is on owens.
//one can read the cache line size in /sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size 
//it records it in byte. There is probably a programmatical way of getting it
template<class T>
class Padded1DArray
{
  T* data;
  size_t paddedsize;
  int nbblock;
public:
  //most likely, there is a better way of doing this but that should be enough for now
  Padded1DArray(int nbblock, int CACHELINE = 64)
    :nbblock(nbblock)
  {
//     int CACHELINE = cache_line_size();
    int nbelemperblock=1;
    if (nbblock == 1)
      paddedsize = nbelemperblock;
    else
      {
	paddedsize = nbelemperblock;
	if (paddedsize*sizeof(T) % CACHELINE)
	  {
	    unsigned int d_orig = paddedsize*sizeof(T)/CACHELINE;
	    while (paddedsize*sizeof(T)/CACHELINE == d_orig)
	      paddedsize++;
	  }
      }
    data = new T[nbblock*paddedsize];
    paddedsize *= sizeof(T);
  }

  ~Padded1DArray()
  {
    close();
  }

  void close() {
      delete[] data;
  }

  inline T& operator[] (int i)
  {
    assert (i<nbblock);
    return *(T*)((char*)data+i*paddedsize); //converting to char avoir a multiplication in assembly
  }

  inline const T& operator[] (int i) const
  {
    assert (i<nbblock);
    return *(T*)((char*)data+i*paddedsize); //converting to char avoir a multiplication in assembly
  }
};


#endif
