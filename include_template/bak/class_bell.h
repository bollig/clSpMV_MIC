#ifndef __CLASS_SPMV_BELL_H__
#define __CLASS_SPMV_BELL_H__

#include "util.h"
#include "CL/cl.h"
#include "matrix_storage.h"
#include <vector>
#include <string>

#include "oclcommon.h"
#include "class_base.h"

namespace spmv {

#define USE(x) using BASE<T>::x

template <typename T>
class BELL : public BASE<T>
{
public:
    USE(devices);
    USE(context);
	USE(cmdQueue);
    USE(program);
	USE(errorCode);

    //Create device memory objects
    USE(devColid);
    USE(devData);
    USE(devVec);
    USE(devRes);
    USE(devTexVec);

	USE(ntimes);

    USE(aligned_length);
    USE(nnz);
    USE(rownum);
	USE(vecsize);
    USE(ellnum);
	USE(coo_mat);

	USE(opttime);
	USE(optmethod);

    USE(dim2); // relates to workgroups

    USE(vec);
    USE(result);
    USE(coores);

	USE(getKernelName);

    int bestbw;
    int bestbh;

	int col_align;
   	int data_align;
    int blockrownum;
    int b4ellnum;
    int bwidth, bw;
    int bheight, bh;
    int width4num;
    int padveclen;
    T* paddedvec;

public:
	BELL(coo_matrix<int, T>* mat, int dim2Size, char* oclfilename, cl_device_type deviceType, int ntimes);
	~BELL<T>() { }
	virtual void run();
	virtual void method_0();
	virtual void method_1();
	virtual void method_2();
	virtual void method_3();
};

//----------------------------------------------------------------------
template <typename T>
void BELL<T>::run()
{
    double overallopttime = 10000.0f;
    int bestbw = 0;
    int bestbh = 0;

    for (int bwidth = 4; bwidth < 9; bwidth += 4)
    for (int bheight = 1; bheight < 9; bheight*=2)
    {
		bw = bwidth; // used in the method_i()
		bh = bheight;

		b4ell_matrix<int, T> mat;
		if (coo2b4ell<int, T>(coo_mat, &mat, bwidth, bheight, GPU_ALIGNMENT, 0) == false)
	  		continue;
		opttime = 10000.0f;
		optmethod = 0;

    	col_align = mat.b4ell_height_aligned;
    	data_align = mat.b4ell_float4_aligned;
    	nnz = mat.matinfo.nnz;
    	rownum = mat.matinfo.height;
    	blockrownum = mat.b4ell_row_num;
    	vecsize = mat.matinfo.width;
    	b4ellnum = mat.b4ell_block_num;
    	bwidth = mat.b4ell_bwidth;
    	bheight = mat.b4ell_bheight;
    	width4num = bwidth / 4;
    	padveclen = findPaddedSize(vecsize, 8);
    	paddedvec = (T*)malloc(sizeof(T)*padveclen);
    	memset(paddedvec, 0, sizeof(T)*padveclen);
    	memcpy(paddedvec, vec, sizeof(T)*vecsize);
    	ALLOCATE_GPU_READ(devColid, mat.b4ell_col_id, sizeof(int)*col_align*b4ellnum);
    	ALLOCATE_GPU_READ(devData, mat.b4ell_data, sizeof(T)*data_align*bheight*width4num*b4ellnum);
    	ALLOCATE_GPU_READ(devVec, paddedvec, sizeof(T)*padveclen);
    	int paddedres = findPaddedSize(rownum, 512);
    	devRes = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(T)*paddedres, NULL, &errorCode); CHECKERROR;
    	errorCode = clEnqueueWriteBuffer(cmdQueue, devRes, CL_TRUE, 0, sizeof(T)*rownum, result, 0, NULL, NULL); CHECKERROR;


		//spmv_b4ell_ocl(&b4ellmat, vec, result, dim2Size, opttime1, optmethod1, oclfilename, deviceType, coores, ntimes, bwidth, bheight);
		method_0();
		method_1();
		method_2();
		method_3();

		if (opttime < overallopttime) {
	    	overallopttime = opttime;
	    	bestbw = bwidth;
	    	bestbh = bheight;
		}
		double gflops = (double)nnz*2/opttime/(double)1e9;
		printf("BELL info: block row num %d ell num %d \n", mat.b4ell_row_num, mat.b4ell_block_num);
		printf("\n------------------------------------------------------------------------\n");
		printf("BELL best time %f ms best method %d GFLOPS %f", opttime*1000.0, optmethod, gflops);
		printf("\n------------------------------------------------------------------------\n");

		//free_b4ell_matrix(b4ellmat);
    }

	//double opttime = getOptTime();
}
//----------------------------------------------------------------------
template <typename T>
BELL<T>::BELL(coo_matrix<int, T>* coo_mat, int dim2Size, char* oclfilename, cl_device_type deviceType, int ntimes) : 
   BASE<T>(coo_mat, dim2Size, oclfilename, deviceType, ntimes)
{
#if 1
// From spmv_bell.cpp
    printMatInfo_T(coo_mat);

    vec = (T*)malloc(sizeof(T)*coo_mat->matinfo.width);
    result = (T*)malloc(sizeof(T)*coo_mat->matinfo.height);
    initVectorOne<int, T>(vec, coo_mat->matinfo.width);	
    initVectorZero<int, T>(result, coo_mat->matinfo.height);
    coores = (T*)malloc(sizeof(T)*coo_mat->matinfo.height);
    spmv_only_T<T>(coo_mat, vec, coores);
    double overallopttime = 10000.0f;
    int bestbw = 0;
    int bestbh = 0;
    nnz = coo_mat->matinfo.nnz;

    //for (int bwidth = 4; bwidth < 9; bwidth += 4)
    //for (int bheight = 1; bheight < 9; bheight*=2)
    //{}
	//int bwidth = 8;
	//int bheight = 4;
	//b4ell_matrix<int, float> mat;
	//if (coo2b4ell<int, float>(coo_mat, &mat, bwidth, bheight, GPU_ALIGNMENT, 0) == false)
	    //continue;
	//rearrange_b4ell_4col(&b4ellmat, GPU_ALIGNMENT);


#endif
	#if 0
    // Bell Initialize values
    int col_align = mat->b4ell_height_aligned;
    int data_align = mat->b4ell_float4_aligned;
    int nnz = mat->matinfo.nnz;
    int rownum = mat->matinfo.height;
    int blockrownum = mat->b4ell_row_num;
    int vecsize = mat->matinfo.width;
    int b4ellnum = mat->b4ell_block_num;
    int bwidth = mat->b4ell_bwidth;
    int bheight = mat->b4ell_bheight;
    int width4num = bwidth / 4;
    int padveclen = findPaddedSize(vecsize, 8);
    float* paddedvec = (float*)malloc(sizeof(float)*padveclen);
    memset(paddedvec, 0, sizeof(float)*padveclen);
    memcpy(paddedvec, vec, sizeof(float)*vecsize);
    ALLOCATE_GPU_READ(devColid, mat->b4ell_col_id, sizeof(int)*col_align*b4ellnum);
    ALLOCATE_GPU_READ(devData, mat->b4ell_data, sizeof(float)*data_align*bheight*width4num*b4ellnum);
    ALLOCATE_GPU_READ(devVec, paddedvec, sizeof(float)*padveclen);
    int paddedres = findPaddedSize(rownum, 512);
    devRes = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*paddedres, NULL, &errorCode); CHECKERROR;
    errorCode = clEnqueueWriteBuffer(cmdQueue, devRes, CL_TRUE, 0, sizeof(float)*rownum, result, 0, NULL, NULL); CHECKERROR;
	#endif

	#if 0
    int width = VEC2DWIDTH;
    int height = (vecsize + VEC2DWIDTH - 1)/VEC2DWIDTH;
    if (height % 4 != 0)
	height += (4 - (height % 4));
    float* image2dVec = (float*)malloc(sizeof(float)*width*height);
    memset(image2dVec, 0, sizeof(float)*width*height);
    for (int i = 0; i < vecsize; i++)
    {
	image2dVec[i] = vec[i];
    }
    size_t origin[] = {0, 0, 0};
    size_t vectorSize[] = {width, height/4, 1};
    devTexVec = clCreateImage2D(context, CL_MEM_READ_ONLY, &floatFormat, width, height/4, 0, NULL, &errorCode); CHECKERROR;
    errorCode = clEnqueueWriteImage(cmdQueue, devTexVec, CL_TRUE, origin, vectorSize, 0, 0, image2dVec, 0, NULL, NULL); CHECKERROR;
    clFinish(cmdQueue);
	#endif


#if 0
	// Create Ell matrices
    printMatInfo_T(coo_mat);
    ell_matrix<int, T> mat;
    coo2ell<int, T>(coo_mat, &mat, GPU_ALIGNMENT, 0);
    vec = (T*)malloc(sizeof(T)*coo_mat->matinfo.width);
    result = (T*)malloc(sizeof(T)*coo_mat->matinfo.height);
    initVectorOne<int, T>(vec, coo_mat->matinfo.width);	
    initVectorZero<int, T>(result, coo_mat->matinfo.height);
    coores = (T*)malloc(sizeof(T)*coo_mat->matinfo.height);
    spmv_only_T<T>(coo_mat, vec, coores);


    //Initialize values
    aligned_length = mat.ell_height_aligned;
    nnz = mat.matinfo.nnz;
    rownum = mat.matinfo.height;
    vecsize = mat.matinfo.width;
    ellnum = mat.ell_num;

	printf("nnz= %d\n", nnz);
	printf("rownum= %d\n", rownum);
	printf("vecsize= %d\n", vecsize);
	printf("ellnum= %d\n", ellnum);

    ALLOCATE_GPU_READ(devColid, mat.ell_col_id, sizeof(int)*aligned_length*ellnum);
    ALLOCATE_GPU_READ(devData, mat.ell_data, sizeof(T)*aligned_length*ellnum);
    ALLOCATE_GPU_READ(devVec, vec, sizeof(T)*vecsize);

    int paddedres = findPaddedSize(rownum, 512);
    devRes = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(T)*paddedres, NULL, &errorCode); CHECKERROR;
    errorCode = clEnqueueWriteBuffer(cmdQueue, devRes, CL_TRUE, 0, sizeof(T)*rownum, result, 0, NULL, NULL); CHECKERROR;
#endif
}
//----------------------------------------------------------------------
template <typename T>
void BELL<T>::method_0()
{
	int methodid = 0;
	cl_uint work_dim = 2;
	size_t blocksize[] = {BELL_GROUP_SIZE, 1};
	int gsize = ((blockrownum + BELL_GROUP_SIZE - 1)/BELL_GROUP_SIZE)*BELL_GROUP_SIZE;
	size_t globalsize[] = {gsize, dim2};
	int data_align4 = data_align / 4;
	char kernelname[100] = "gpu_bell00";
	kernelname[8] += bh;
	kernelname[9] += bw;

	cl_kernel csrKernel = NULL;
	csrKernel = clCreateKernel(program, kernelname, &errorCode); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 0, sizeof(cl_mem), &devColid); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 1, sizeof(cl_mem), &devData); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 2, sizeof(int),    &data_align4); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 3, sizeof(int),    &col_align); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 4, sizeof(int),    &b4ellnum); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 5, sizeof(cl_mem), &devVec); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 6, sizeof(cl_mem), &devRes); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 7, sizeof(int),    &blockrownum); CHECKERROR;

	errorCode = clEnqueueWriteBuffer(cmdQueue, devRes, CL_TRUE, 0, sizeof(float)*rownum, result, 0, NULL, NULL); CHECKERROR;
	clFinish(cmdQueue);
	errorCode = clEnqueueNDRangeKernel(cmdQueue, csrKernel, work_dim, NULL, globalsize, blocksize, 0, NULL, NULL); CHECKERROR;
	clFinish(cmdQueue);
	T* tmpresult = (T*)malloc(sizeof(T)*rownum);
	errorCode = clEnqueueReadBuffer(cmdQueue, devRes, CL_TRUE, 0, sizeof(float)*rownum, tmpresult, 0, NULL, NULL); CHECKERROR;
	clFinish(cmdQueue);
	two_vec_compare_T(coores, tmpresult, rownum);
	free(tmpresult);

	for (int k = 0; k < 3; k++)
	{
	    errorCode = clEnqueueNDRangeKernel(cmdQueue, csrKernel, work_dim, NULL, globalsize, blocksize, 0, NULL, NULL); CHECKERROR;
	}
	clFinish(cmdQueue);

	double teststart = timestamp();
	for (int i = 0; i < ntimes; i++)
	{
	    errorCode = clEnqueueNDRangeKernel(cmdQueue, csrKernel, work_dim, NULL, globalsize, blocksize, 0, NULL, NULL); CHECKERROR;
	}
	clFinish(cmdQueue);
	double testend = timestamp();
	double time_in_sec = (testend - teststart)/(double)dim2;
	printf("ntimes= %d, time_in_msec= %f, nnz= %d\n", ntimes, time_in_sec*1000., nnz);
	double gflops = (double)nnz*2/(time_in_sec/(double)ntimes)/(double)1e9;
	printf("\nBELL %dx%d block cpu time %lf ms GFLOPS %lf code %d \n\n", bh, bw,  time_in_sec / (double) ntimes * 1000, gflops, methodid);

	if (csrKernel)
	    clReleaseKernel(csrKernel);

	double onetime = time_in_sec / (double) ntimes;
	if (onetime < opttime)
	{
	    opttime = onetime;
	    optmethod = methodid;
	}
}
//------------------
template <typename T>
void BELL<T>::method_1()
{
	int methodid = 1;
	cl_uint work_dim = 2;
	size_t blocksize[] = {BELL_GROUP_SIZE, 1};
	int gsize = ((blockrownum + BELL_GROUP_SIZE - 1)/BELL_GROUP_SIZE)*BELL_GROUP_SIZE;
	size_t globalsize[] = {gsize, dim2};
	int data_align4 = data_align / 4;
	char kernelname[100] = "gpu_bell00_mad";
	kernelname[8] += bh;
	kernelname[9] += bw;

	cl_kernel csrKernel = NULL;
	csrKernel = clCreateKernel(program, kernelname, &errorCode); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 0, sizeof(cl_mem), &devColid); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 1, sizeof(cl_mem), &devData); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 2, sizeof(int),    &data_align4); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 3, sizeof(int),    &col_align); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 4, sizeof(int),    &b4ellnum); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 5, sizeof(cl_mem), &devVec); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 6, sizeof(cl_mem), &devRes); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 7, sizeof(int),    &blockrownum); CHECKERROR;

	errorCode = clEnqueueWriteBuffer(cmdQueue, devRes, CL_TRUE, 0, sizeof(float)*rownum, result, 0, NULL, NULL); CHECKERROR;
	clFinish(cmdQueue);
	errorCode = clEnqueueNDRangeKernel(cmdQueue, csrKernel, work_dim, NULL, globalsize, blocksize, 0, NULL, NULL); CHECKERROR;
	clFinish(cmdQueue);
	T* tmpresult = (T*)malloc(sizeof(T)*rownum);
	errorCode = clEnqueueReadBuffer(cmdQueue, devRes, CL_TRUE, 0, sizeof(float)*rownum, tmpresult, 0, NULL, NULL); CHECKERROR;
	clFinish(cmdQueue);
	two_vec_compare_T(coores, tmpresult, rownum);
	free(tmpresult);

	for (int k = 0; k < 3; k++)
	{
	    errorCode = clEnqueueNDRangeKernel(cmdQueue, csrKernel, work_dim, NULL, globalsize, blocksize, 0, NULL, NULL); CHECKERROR;
	}
	clFinish(cmdQueue);

	double teststart = timestamp();
	for (int i = 0; i < ntimes; i++)
	{
	    errorCode = clEnqueueNDRangeKernel(cmdQueue, csrKernel, work_dim, NULL, globalsize, blocksize, 0, NULL, NULL); CHECKERROR;
	}
	clFinish(cmdQueue);
	double testend = timestamp();
	double time_in_sec = (testend - teststart)/(double)dim2;
	double gflops = (double)nnz*2/(time_in_sec/(double)ntimes)/(double)1e9;
	printf("\nBELL %dx%d block mad cpu time %lf ms GFLOPS %lf code %d \n\n", bh, bw,  time_in_sec / (double) ntimes * 1000, gflops, methodid);

	if (csrKernel)
	    clReleaseKernel(csrKernel);

	double onetime = time_in_sec / (double) ntimes;
	if (onetime < opttime)
	{
	    opttime = onetime;
	    optmethod = methodid;
	}
}
//-----------------
template <typename T>
void BELL<T>::method_2()
{
	int methodid = 100;
	cl_uint work_dim = 2;
	size_t blocksize[] = {BELL_GROUP_SIZE, 1};
	int gsize = ((blockrownum + BELL_GROUP_SIZE - 1)/BELL_GROUP_SIZE)*BELL_GROUP_SIZE;
	size_t globalsize[] = {gsize, dim2};
	int data_align4 = data_align / 4;
	char kernelname[100] = "gpu_bell00_tx";
	kernelname[8] += bh;
	kernelname[9] += bw;

	cl_kernel csrKernel = NULL;
	csrKernel = clCreateKernel(program, kernelname, &errorCode); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 0, sizeof(cl_mem), &devColid); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 1, sizeof(cl_mem), &devData); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 2, sizeof(int),    &data_align4); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 3, sizeof(int),    &col_align); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 4, sizeof(int),    &b4ellnum); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 5, sizeof(cl_mem), &devTexVec); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 6, sizeof(cl_mem), &devRes); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 7, sizeof(int),    &blockrownum); CHECKERROR;

	errorCode = clEnqueueWriteBuffer(cmdQueue, devRes, CL_TRUE, 0, sizeof(T)*rownum, result, 0, NULL, NULL); CHECKERROR;
	clFinish(cmdQueue);
	errorCode = clEnqueueNDRangeKernel(cmdQueue, csrKernel, work_dim, NULL, globalsize, blocksize, 0, NULL, NULL); CHECKERROR;
	clFinish(cmdQueue);
	T* tmpresult = (T*)malloc(sizeof(T)*rownum);
	errorCode = clEnqueueReadBuffer(cmdQueue, devRes, CL_TRUE, 0, sizeof(T)*rownum, tmpresult, 0, NULL, NULL); CHECKERROR;
	clFinish(cmdQueue);
	two_vec_compare_T(coores, tmpresult, rownum);
	free(tmpresult);

	for (int k = 0; k < 3; k++)
	{
	    errorCode = clEnqueueNDRangeKernel(cmdQueue, csrKernel, work_dim, NULL, globalsize, blocksize, 0, NULL, NULL); CHECKERROR;
	}
	clFinish(cmdQueue);

	double teststart = timestamp();
	for (int i = 0; i < ntimes; i++)
	{
	    errorCode = clEnqueueNDRangeKernel(cmdQueue, csrKernel, work_dim, NULL, globalsize, blocksize, 0, NULL, NULL); CHECKERROR;
	}
	clFinish(cmdQueue);
	double testend = timestamp();
	double time_in_sec = (testend - teststart)/(double)dim2;
	double gflops = (double)nnz*2/(time_in_sec/(double)ntimes)/(double)1e9;
	printf("\nBELL %dx%d block tx cpu time %lf ms GFLOPS %lf code %d \n\n", bh, bw,  time_in_sec / (double) ntimes * 1000, gflops, methodid);

	if (csrKernel)
	    clReleaseKernel(csrKernel);

	double onetime = time_in_sec / (double) ntimes;
	if (onetime < opttime)
	{
	    opttime = onetime;
	    optmethod = methodid;
	}
}
//------------------
template <typename T>
void BELL<T>::method_3()
{
	int methodid = 101;
	cl_uint work_dim = 2;
	size_t blocksize[] = {BELL_GROUP_SIZE, 1};
	int gsize = ((blockrownum + BELL_GROUP_SIZE - 1)/BELL_GROUP_SIZE)*BELL_GROUP_SIZE;
	size_t globalsize[] = {gsize, dim2};
	int data_align4 = data_align / 4;
	char kernelname[100] = "gpu_bell00_mad_tx";
	kernelname[8] += bh;
	kernelname[9] += bw;

	cl_kernel csrKernel = NULL;
	csrKernel = clCreateKernel(program, kernelname, &errorCode); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 0, sizeof(cl_mem), &devColid); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 1, sizeof(cl_mem), &devData); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 2, sizeof(int),    &data_align4); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 3, sizeof(int),    &col_align); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 4, sizeof(int),    &b4ellnum); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 5, sizeof(cl_mem), &devTexVec); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 6, sizeof(cl_mem), &devRes); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 7, sizeof(int),    &blockrownum); CHECKERROR;

	errorCode = clEnqueueWriteBuffer(cmdQueue, devRes, CL_TRUE, 0, sizeof(T)*rownum, result, 0, NULL, NULL); CHECKERROR;
	clFinish(cmdQueue);
	errorCode = clEnqueueNDRangeKernel(cmdQueue, csrKernel, work_dim, NULL, globalsize, blocksize, 0, NULL, NULL); CHECKERROR;
	clFinish(cmdQueue);
	T* tmpresult = (T*)malloc(sizeof(T)*rownum);
	errorCode = clEnqueueReadBuffer(cmdQueue, devRes, CL_TRUE, 0, sizeof(T)*rownum, tmpresult, 0, NULL, NULL); CHECKERROR;
	clFinish(cmdQueue);
	two_vec_compare_T(coores, tmpresult, rownum);
	free(tmpresult);

	for (int k = 0; k < 3; k++)
	{
	    errorCode = clEnqueueNDRangeKernel(cmdQueue, csrKernel, work_dim, NULL, globalsize, blocksize, 0, NULL, NULL); CHECKERROR;
	}
	clFinish(cmdQueue);

	double teststart = timestamp();
	for (int i = 0; i < ntimes; i++)
	{
	    errorCode = clEnqueueNDRangeKernel(cmdQueue, csrKernel, work_dim, NULL, globalsize, blocksize, 0, NULL, NULL); CHECKERROR;
	}
	clFinish(cmdQueue);
	double testend = timestamp();
	double time_in_sec = (testend - teststart)/(double)dim2;
	double gflops = (double)nnz*2/(time_in_sec/(double)ntimes)/(double)1e9;
	printf("\nBELL %dx%d block mad tx cpu time %lf ms GFLOPS %lf code %d \n\n", bh, bw,  time_in_sec / (double) ntimes * 1000, gflops, methodid);

	if (csrKernel)
	    clReleaseKernel(csrKernel);

	double onetime = time_in_sec / (double) ntimes;
	if (onetime < opttime)
	{
	    opttime = onetime;
	    optmethod = methodid;
	}
}
//------------------------


//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------

	#if 0
    //Clean up
    if (image2dVec)
	free(image2dVec);

    if (devColid)
	clReleaseMemObject(devColid);
    if (devData)
	clReleaseMemObject(devData);
    if (devVec)
	clReleaseMemObject(devVec);
    if (devTexVec)
	clReleaseMemObject(devTexVec);
    if (devRes)
	clReleaseMemObject(devRes);


    freeObjects(devices, &context, &cmdQueue, &program);
	#endif
//----------------------------------------------------------------------

template <typename  T>
void spmv_bell(char* oclfilename, coo_matrix<int, T>* coo_mat, int dim2Size, int ntimes, cl_device_type deviceType)
{

	BELL<T> bell_ocl(coo_mat, dim2Size, oclfilename, deviceType, ntimes);
	bell_ocl.run();

	double opttime = bell_ocl.getOptTime();
	int optmethod = bell_ocl.getOptMethod();

	printf("\n------------------------------------------------------------------------\n");
	printf("BELL best time %f ms best method %d", opttime*1000.0, optmethod);
	printf("\n------------------------------------------------------------------------\n");
}
//----------------------------------------------------------------------

// namespace
}; 

#endif
