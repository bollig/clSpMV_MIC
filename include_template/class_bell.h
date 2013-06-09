#ifndef __CLASS_SPMV_BELL_H__
#define __CLASS_SPMV_BELL_H__

#include "util.h"
#include "CL/cl.h"
#include "matrix_storage.h"
#include <vector>
#include <string>

#include "oclcommon.h"
#include "cl_base_class.h"

using namespace std;

namespace spmv {

#define USE(x) using BASE<T>::x

template <typename T>
class BELL : public BASE<T>, public CLBaseClass
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
	vector<T> paddedvec_v;
	vector<T> vec_v;
	vector<T> result_v;
	vector<T> coores_v;

public:
	BELL(coo_matrix<int, T>* mat, int dim2Size, char* oclfilename, cl_device_type deviceType, int ntimes);
	~BELL<T>() { }
	virtual void run();
	virtual void method_0(int count);
	virtual void method_1();
	virtual void method_2();
	virtual void method_3();
};

//----------------------------------------------------------------------
template <typename T>
BELL<T>::BELL(coo_matrix<int, T>* coo_mat, int dim2Size, char* oclfilename, cl_device_type deviceType, int ntimes) : 
   BASE<T>(coo_mat, dim2Size, oclfilename, deviceType, ntimes)
{
// From spmv_bell.cpp
    printMatInfo_T(coo_mat);

    vec = new T [coo_mat->matinfo.width]; // (T*)malloc(sizeof(T)*coo_mat->matinfo.width);
    vec_v.resize(coo_mat->matinfo.width); // experimental use of vectors: xxx_v
    result = new T [coo_mat->matinfo.height]; // (T*)malloc(sizeof(T)*coo_mat->matinfo.height);
    result_v.resize(coo_mat->matinfo.height);
    initVectorOne<int, T>(vec, coo_mat->matinfo.width);	
    initVectorZero<int, T>(result, coo_mat->matinfo.height);
    coores = new T [coo_mat->matinfo.height]; //(T*)malloc(sizeof(T)*coo_mat->matinfo.height);
    coores_v.resize(coo_mat->matinfo.height);
    spmv_only_T<T>(coo_mat, vec, coores);   // matrix*vector, serial
    double overallopttime = 10000.0f;
    int bestbw = 0;
    int bestbh = 0;
    nnz = coo_mat->matinfo.nnz;
}
//----------------------------------------------------------------------
template <typename T>
void BELL<T>::run()
{
    	devices = NULL;
    	context = NULL;
		cmdQueue = NULL;
    	program = NULL;
		cl_device_type deviceType = CONTEXTTYPE;
		char* oclfilename = "./kernels/spmv_bell.cl";
    	assert(initialization(deviceType, devices, &context, &cmdQueue, &program, oclfilename) == 1);



    double overallopttime = 10000.0f;
    bestbw = 0;
    bestbh = 0;

	int count = -1;
    for (int bwidth = 4; bwidth < 9; bwidth += 4) {
    for (int bheight = 1; bheight < 9; bheight*=2) {
	// re-initialization Should not be required. Did not help solve the problem. 
	count++;




		printf("++++++ for loop: bwidth= %d, bheight= %d ++++\n", bwidth, bheight);

		bwidth = 4; // run same case twice
		bheight = 1;

		printf("\n\n** (before if) bwidth= %d, bheight= %d\n", bwidth, bheight);
		//if (!(bwidth == 4 && bheight == 4)) continue;
		if (count > 1) exit(0);
		//if (count > 1) continue;

//		printf("\n\n** (after if) bwidth= %d, bheight= %d\n", bwidth, bheight);
		bw = bwidth; // used in the method_i()
		bh = bheight;
//		printf("after if bw, bh= %d, %d\n", bw, bh);

		b4ell_matrix<int, T> mat;
		if (coo2b4ell<int, T>(coo_mat, &mat, bwidth, bheight, GPU_ALIGNMENT, 0) == false) {
			// array too large
			//printf("*** coo2b4ell is false ***\n"); exit(0); // original code had a continue
	  		continue;
		}
		//mat.print();
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
    	width4num = bwidth / 4;   // 8 in double precision?
    	padveclen = findPaddedSize(vecsize, 8);   // change for double precision?
		printf("*** vecsize= %d, padveclen= %d\n", vecsize, padveclen); // identical
		assert(padveclen == vecsize);
    	paddedvec = new T [padveclen]; // (T*)malloc(sizeof(T)*padveclen);
    	paddedvec_v.resize(padveclen);
    	//memset(paddedvec, 0, sizeof(T)*padveclen);
    	memcpy(paddedvec, vec, sizeof(T)*vecsize);
    	ALLOCATE_GPU_READ(devColid, mat.b4ell_col_id, sizeof(int)*col_align*b4ellnum); // ERROR
    	ALLOCATE_GPU_READ(devData, mat.b4ell_data, sizeof(T)*data_align*bheight*width4num*b4ellnum);
    	//ALLOCATE_GPU_READ(devData, mat.b4ell_data, sizeof(T)*data_align*bheight*width4num*b4ellnum);
    	ALLOCATE_GPU_READ(devVec, paddedvec, sizeof(T)*padveclen);
    	//ALLOCATE_GPU_READ(devVec, paddedvec, sizeof(T)*padveclen);
    	int paddedres = findPaddedSize(rownum, 512);
    	devRes = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(T)*paddedres, NULL, &errorCode); CHECKERROR;
    	//devRes = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(T)*paddedres, NULL, &errorCode); CHECKERROR;
		printf("rownum= %d, paddedres= %d\n", rownum, paddedres);
    	//errorCode = clEnqueueWriteBuffer(cmdQueue, devRes, CL_TRUE, 0, sizeof(T)*rownum, result, 0, NULL, NULL); CHECKERROR;
		// TEMP
    	errorCode = clEnqueueWriteBuffer(cmdQueue, devRes, CL_TRUE, 0, sizeof(T)*paddedres, result, 0, NULL, NULL); CHECKERROR;
    	//errorCode = clEnqueueWriteBuffer(cmdQueue, devRes, CL_TRUE, 0, sizeof(T)*rownum, result, 0, NULL, NULL); CHECKERROR;


		//spmv_b4ell_ocl(&b4ellmat, vec, result, dim2Size, opttime1, optmethod1, oclfilename, deviceType, coores, ntimes, bwidth, bheight);
		printf("\n\n** bw= %d, bh= %d\n", bw, bh);
		printf("col_align= %d\n", col_align);
		printf("data_align= %d\n", data_align);
		printf("blockrownum= %d\n", blockrownum);
		printf("vecsize= %d\n", vecsize);
		printf("b4ellnum= %d\n", b4ellnum);
		printf("width4num= %d\n", width4num);
		printf("padveclen= %d\n", padveclen);
		printf("paddedres= %d\n", paddedres);

		method_0(count);

		//method_0(count);
		//method_0(count);
		//method_0(count);
	//exit(0);
		//method_1();
		//method_2(); // images. Not support on mic
		//method_3(); // images

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

		delete [] paddedvec;
    	if (devColid) {clReleaseMemObject(devColid); printf("... release devColid\n");}
    	if (devData) {clReleaseMemObject(devData); printf("... release devData\n");}
    	if (devVec) {clReleaseMemObject(devVec); printf("... release devVec\n");}
    	//if (devTexVec) clReleaseMemObject(devTexVec);
	    if (devRes) {clReleaseMemObject(devRes); printf("... release devRes\n");}

    }}

    	freeObjects(devices, &context, &cmdQueue, &program); // should not be required

	//double opttime = getOptTime();
}
//----------------------------------------------------------------------
template <typename T>
void BELL<T>::method_0(int count)
{
	printf("======== METHOD 0 ======================================================\n");
	int methodid = 0;
	cl_uint work_dim = 2;
	size_t blocksize[] = {BELL_GROUP_SIZE, 1};
	int gsize = ((blockrownum + BELL_GROUP_SIZE - 1)/BELL_GROUP_SIZE)*BELL_GROUP_SIZE;
	size_t globalsize[] = {gsize, dim2};
	int data_align4 = data_align / 4;
	char kernelname[100] = "gpu_bell00";
	kernelname[8] += bh;
	kernelname[9] += bw;

	std::string kernel_name = getKernelName(kernelname);
	printf("****** kernel_name: %s ******\n", kernel_name.c_str());

	printf("Same ROUTINE CALLED TWICE ==> SEG ERROR\n");


	cl_kernel csrKernel = NULL;
	csrKernel = clCreateKernel(program, kernel_name.c_str(), &errorCode); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 0, sizeof(cl_mem), &devColid); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 1, sizeof(cl_mem), &devData); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 2, sizeof(int),    &data_align4); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 3, sizeof(int),    &col_align); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 4, sizeof(int),    &b4ellnum); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 5, sizeof(cl_mem), &devVec); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 6, sizeof(cl_mem), &devRes); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 7, sizeof(int),    &blockrownum); CHECKERROR;

//	Why would this be required if result = A*v? 
//	errorCode = clEnqueueWriteBuffer(cmdQueue, devRes, CL_TRUE, 0, sizeof(T)*rownum, result, 0, NULL, NULL); CHECKERROR;
//	clFinish(cmdQueue);
	// Somehow, a HW error on the next line. NO IDEA WHY, and only on the 2nd pass through the loop. 
	 printf("work_dim= %d\n", work_dim);
	 printf(" globalsize= %d,%d, blocksize= %d,%d\n", globalsize[0], globalsize[1], blocksize[0], blocksize[1]); // <<REASON FOR ERROR
	errorCode = clEnqueueNDRangeKernel(cmdQueue, csrKernel, work_dim, NULL, globalsize, blocksize, 0, NULL, NULL); CHECKERROR; // ERROR
	clFinish(cmdQueue);
		if (count == 2) exit(0);
	vector<T> tmpresult(rownum);
	
	//T* tmpresult = new T [rownum]; //(T*)malloc(sizeof(T)*rownum);
	errorCode = clEnqueueReadBuffer(cmdQueue, devRes, CL_TRUE, 0, sizeof(T)*rownum, &tmpresult[0], 0, NULL, NULL); CHECKERROR;
	clFinish(cmdQueue);
	for (int i=0; i < rownum; i++) {
		if (fabs(tmpresult[i]-coores[i]) > 1.e-4) 
		//printf("meth 0, (%d),  tmpresult= %f, coores= %f\n", i, tmpresult[i], coores[i]);
		;
	}
	two_vec_compare_T(coores, &tmpresult[0], rownum);
	//delete [] tmpresult;// free(tmpresult);

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
	#if 0
	printf("======== METHOD 1 ======================================================\n");
	int methodid = 1;
	cl_uint work_dim = 2;
	size_t blocksize[] = {BELL_GROUP_SIZE, 1};
	int gsize = ((blockrownum + BELL_GROUP_SIZE - 1)/BELL_GROUP_SIZE)*BELL_GROUP_SIZE;
	size_t globalsize[] = {gsize, dim2};
	int data_align4 = data_align / 4;      // should be /8 in double precision?
	char kernelname[100] = "gpu_bell00_mad";
	kernelname[8] += bh;
	kernelname[9] += bw;

	cl_kernel csrKernel = NULL;
	std::string kernel_name = getKernelName(kernelname);
	printf("****** kernel_name: %s ******\n", kernel_name.c_str());
	csrKernel = clCreateKernel(program, kernel_name.c_str(), &errorCode); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 0, sizeof(cl_mem), &devColid); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 1, sizeof(cl_mem), &devData); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 2, sizeof(int),    &data_align4); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 3, sizeof(int),    &col_align); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 4, sizeof(int),    &b4ellnum); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 5, sizeof(cl_mem), &devVec); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 6, sizeof(cl_mem), &devRes); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 7, sizeof(int),    &blockrownum); CHECKERROR;

	printf("*** method 1: rownum= %d\n", rownum);
	errorCode = clEnqueueWriteBuffer(cmdQueue, devRes, CL_TRUE, 0, sizeof(T)*rownum, result, 0, NULL, NULL); CHECKERROR;
	clFinish(cmdQueue);
	errorCode = clEnqueueNDRangeKernel(cmdQueue, csrKernel, work_dim, NULL, globalsize, blocksize, 0, NULL, NULL); CHECKERROR;
	clFinish(cmdQueue);
	T* tmpresult = (T*)malloc(sizeof(T)*rownum);
	errorCode = clEnqueueReadBuffer(cmdQueue, devRes, CL_TRUE, 0, sizeof(T)*rownum, tmpresult, 0, NULL, NULL); CHECKERROR;
	clFinish(cmdQueue);
	for (int i=0; i < rownum; i++) {
		if (fabs(tmpresult[i]-coores[i]) > 1.e-4) 
		printf("meth 1, (%d),  tmpresult= %f, coores= %f\n", i, tmpresult[i], coores[i]);
	}

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
	#endif
}
//-----------------
template <typename T>
void BELL<T>::method_2()
{
	#if 0
	printf("======== METHOD 2 ======================================================\n");
	int methodid = 100;
	cl_uint work_dim = 2;
	size_t blocksize[] = {BELL_GROUP_SIZE, 1};
	int gsize = ((blockrownum + BELL_GROUP_SIZE - 1)/BELL_GROUP_SIZE)*BELL_GROUP_SIZE;
	size_t globalsize[] = {gsize, dim2};
	int data_align4 = data_align / 4;
	char kernelname[100] = "gpu_bell00_tx";
	kernelname[8] += bh;
	kernelname[9] += bw;
	std::string kernel_name = getKernelName(kernelname);
	printf("****** kernel_name: %s ******\n", kernel_name.c_str());

	cl_kernel csrKernel = NULL;
	csrKernel = clCreateKernel(program, kernel_name.c_str(), &errorCode); CHECKERROR;
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
#endif
}
//------------------
template <typename T>
void BELL<T>::method_3()
{
#if 0
	int methodid = 101;
	cl_uint work_dim = 2;
	size_t blocksize[] = {BELL_GROUP_SIZE, 1};
	int gsize = ((blockrownum + BELL_GROUP_SIZE - 1)/BELL_GROUP_SIZE)*BELL_GROUP_SIZE;
	size_t globalsize[] = {gsize, dim2};
	int data_align4 = data_align / 4;
	char kernelname[100] = "gpu_bell00_mad_tx";
	kernelname[8] += bh;
	kernelname[9] += bw;
	std::string kernel_name = getKernelName(kernelname);
	printf("****** kernel_name: %s ******\n", kernel_name.c_str());

	cl_kernel csrKernel = NULL;
	csrKernel = clCreateKernel(program, kernel_name.c_str(), &errorCode); CHECKERROR;
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
#endif
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
