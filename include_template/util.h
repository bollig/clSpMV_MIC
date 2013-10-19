#ifndef UTIL__H__
#define UTIL__H__

#include <cmath>
#include <sys/time.h>
#include "matrix_storage.h"
#include <vector>
#include <algorithm>

double timestamp ();
int findPaddedSize(int realSize, int alignment);
void pad_csr(csr_matrix<int, float>* source, csr_matrix<int, float>* dest, int alignment);
double distance(float* vec1, float* vec2, int size);
void correctness_check(coo_matrix<int, float>* mat, float* vec, float* res);
void printMatInfo(coo_matrix<int, float>* mat);
void spmv_only(coo_matrix<int, float>* mat, float* vec, float* coores);

void two_vec_compare(float* coovec, float* newvec, int size);

//void rearrange_bell_4col(bell_matrix<int, float>* mat, int alignment);
unsigned int getRandMax();
unsigned int getRand(unsigned int upper);
double getRandf();
void setSeed();
void evict_array_from_cache(void* ptr, size_t size);

//----------------------------------------------------------------------
//----------------------------------------------------------------------
template <typename T>
double distance_T(std::vector<T>& vec1, std::vector<T>& vec2, int size)
{
	//printf("distance_T: size= %d, vec1.size= %d, vec2.size= %d\n", size, vec1.size(), vec2.size());
	assert(size <= vec1.size() && size <= vec2.size());
	double sum = 0.0f;

	for (int i = 0; i < size; i++)
	{
		double tmp = vec1[i] - vec2[i];
		sum += tmp * tmp;
	}
	return sqrt(sum);
}
//----------------------------------------------------------------------
template <typename T>
void two_vec_compare_T(std::vector<T>& coovec, std::vector<T>& newvec, int size) 
{
    double dist = distance_T(coovec, newvec, size);

    double maxdiff = 0.0f;
    int maxdiffid = 0;
    double maxratiodiff = 0.0f;
    int count = 0;

    for (int i = 0; i < size; i++)
    {
		T tmpa = coovec[i];
		if (tmpa < 0) tmpa *= -1;
		T tmpb = newvec[i];
		if (tmpb < 0) tmpb *= -1;
		double diff = tmpa - tmpb;
		if (diff < 0) diff *= -1;
		T maxab = (tmpa > tmpb)?tmpa:tmpb;
		double ratio = maxab > 0 ? diff/maxab : 0.0;
		if (diff > maxdiff)
		{
	    	maxdiff = diff;
	    	maxdiffid = i;
		}
		if (ratio > maxratiodiff) maxratiodiff = ratio;

		#if 0
		if (coovec[i] != newvec[i] && count < 10)
		{
	    	printf("Error i %d coo res %f res %f \n", i, coovec[i], newvec[i]);
	    	count++;
		}
		#endif
    }
    //printf("Max diff id %d coo res %f res %f \n", maxdiffid, coovec[maxdiffid], newvec[maxdiffid]);
    printf("\nCorrectness Check: Distance %e max diff %e max diff ratio %e vec size %d\n", dist, maxdiff, maxratiodiff, size);
}
//----------------------------------------------------------------------
template <typename T>
void spmv_only_T(coo_matrix<int, T>* mat, std::vector<T>& vec, std::vector<T>& coores)
{
    int ressize = mat->matinfo.height;
	assert(ressize == coores.size());
	// fake data for debugging purposes
	std::fill(coores.begin(), coores.end(), (T) 0.);  // T is float or double
    coo_spmv_T<int, T>(mat, vec, coores, mat->matinfo.width);
}
//----------------------------------------------------------------------
template <typename T>
void printMatInfo_T(coo_matrix<int, T>* mat)
{
    printf("\nMatInfo: Width %d Height %d NNZ %d\n", mat->matinfo.width, mat->matinfo.height, mat->matinfo.nnz);
    int minoffset = mat->matinfo.width;
    int maxoffset = -minoffset;
    int nnz = mat->matinfo.nnz;
    int lessn16 = 0;
    int inn16 = 0;
    int less16 = 0;
    int large16 = 0;
    for (int i = 0; i < nnz; i++)
    {
	int rowid = mat->coo_row_id[i];
	int colid = mat->coo_col_id[i];
	int diff = rowid - colid;
	if (diff < minoffset)
	    minoffset = diff;
	if (diff > maxoffset)
	    maxoffset = diff;
	if (diff < -15)
	    lessn16++;
	else if (diff < 0)
	    inn16++;
	else if (diff < 16)
	    less16++;
	else
	    large16++;
    }
    printf("Max Offset %d Min Offset %d\n", maxoffset, minoffset);
    printf("Histogram: <-15: %d -15~-1 %d < 0-15 %d > 16 %d\n", lessn16, inn16, less16, large16);

    if (!if_sorted_coo(mat))
    {
	assert(sort_coo(mat) == true);
    }

    int* cacheperrow = (int*)malloc(sizeof(int)*mat->matinfo.height);
    int* elemperrow = (int*)malloc(sizeof(int)*mat->matinfo.height);
    memset(cacheperrow, 0, sizeof(int)*mat->matinfo.height);
    memset(elemperrow, 0, sizeof(int)*mat->matinfo.height);
    int index = 0;
    for (int i = 0; i < mat->matinfo.height; i++)
    {
	if (i < mat->coo_row_id[index])
	    continue;
	int firstline = mat->coo_col_id[index]/16;
	cacheperrow[i] = 1;
	elemperrow[i] = 1;
	index++;
	while (mat->coo_row_id[index] == i)
	{
	    int nextline = mat->coo_col_id[index]/16;
	    if (nextline != firstline)
	    {
		firstline = nextline;
		cacheperrow[i]++;
	    }
	    elemperrow[i]++;
	    index++;
	}
    }
    int maxcacheline = 0;
    int mincacheline = 100000000;
    int sum = 0;
    for (int i = 0; i < mat->matinfo.height; i++)
    {
	if (cacheperrow[i] < mincacheline)
	    mincacheline = cacheperrow[i];
	if (cacheperrow[i] > maxcacheline)
	    maxcacheline = cacheperrow[i];
	sum += cacheperrow[i];
    }
    printf("Cacheline usage per row: max %d min %d avg %f\n", maxcacheline, mincacheline, (double)sum/(double)mat->matinfo.height);
}
//----------------------------------------------------------------------
//----------------------------------------------------------------------



#endif

