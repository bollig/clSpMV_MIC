//TODO (disable from Makefile)
//FIX	two_vec_compare_T(coores, tmpresult, rownum);

#ifndef __CLASS_SPMV_ELL_H__
#define __CLASS_SPMV_ELL_H__

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
class ELL : public BASE<T>
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

public:
	ELL(coo_matrix<int, T>* mat, int dim2Size, char* oclfilename, cl_device_type deviceType, int ntimes);
	~ELL<T>() {
    	//free_ell_matrix(mat);
    //	free(vec);
    	//free(result);
    	//free(coores);
	}

	virtual void run();

protected:
	virtual void method_0();
	virtual void method_1();
};

//----------------------------------------------------------------------
template <typename T>
void ELL<T>::run()
{
	method_0();
	method_1();
}
//----------------------------------------------------------------------
template <typename T>
ELL<T>::ELL(coo_matrix<int, T>* coo_mat, int dim2Size, char* oclfilename, cl_device_type deviceType, int ntimes) : 
   BASE<T>(coo_mat, dim2Size, oclfilename, deviceType, ntimes)
{
	// Create matrices
    printMatInfo_T(coo_mat);
    ell_matrix<int, T> mat;
    coo2ell<int, T>(coo_mat, &mat, GPU_ALIGNMENT, 0);
    vec = (T*)malloc(sizeof(T)*coo_mat->matinfo.width);
    result = (T*)malloc(sizeof(T)*coo_mat->matinfo.height);
    initVectorOne<int, T>(vec, coo_mat->matinfo.width);	
    initVectorZero<int, T>(result, coo_mat->matinfo.height);
    coores = (T*)malloc(sizeof(T)*coo_mat->matinfo.height);
	// CHECKING Supposedly on CPU, but execution is on GPU!!
    //spmv_only_T<T>(coo_mat, vec, coores);


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
}
//----------------------------------------------------------------------
template <typename T>
void ELL<T>::method_0()
{
	printf("inside method_0\n");
	int methodid = 0;
	cl_uint work_dim = 2;
	size_t blocksize[] = {WORK_GROUP_SIZE, 1};
	int gsize = ((rownum + WORK_GROUP_SIZE - 1)/WORK_GROUP_SIZE)*WORK_GROUP_SIZE;
	size_t globalsize[] = {gsize, dim2};


	cl_kernel csrKernel = NULL;
	std::string kernel_name = getKernelName("gpu_ell");
	printf("****** kernel_name: %s ******\n", kernel_name.c_str());
	csrKernel = clCreateKernel(program, kernel_name.c_str(), &errorCode); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 0, sizeof(cl_mem), &devColid); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 1, sizeof(cl_mem), &devData); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 2, sizeof(int),    &aligned_length); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 3, sizeof(int),    &ellnum); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 4, sizeof(cl_mem), &devVec); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 5, sizeof(cl_mem), &devRes); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 6, sizeof(int),    &rownum); CHECKERROR;

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
		printf("k= %d\n", k);
	    errorCode = clEnqueueNDRangeKernel(cmdQueue, csrKernel, work_dim, NULL, globalsize, blocksize, 0, NULL, NULL); CHECKERROR;
	}
	clFinish(cmdQueue);

	double teststart = timestamp();
	for (int i = 0; i < ntimes; i++)
	{
	    errorCode = clEnqueueNDRangeKernel(cmdQueue, csrKernel, work_dim, NULL, globalsize, blocksize, 0, NULL, NULL); CHECKERROR;
	}
	clFinish(cmdQueue);

	printf("finish queue\n");

	double testend = timestamp();
	double time_in_sec = (testend - teststart)/(double)dim2;
	printf("ntimes= %d, time_in_msec= %f, nnz= %d\n", ntimes, time_in_sec*1000., nnz);
	double gflops = (double)nnz*2/(time_in_sec/(double)ntimes)/(double)1e9;
	//printf("\nELL cpu time %lf ms GFLOPS %lf code %d \n\n",   time_in_sec / (double) ntimes * 1000, gflops, methodid);
	if (sizeof(T) == sizeof(float)) {
		printf("\nELL float cpu time %lf ms GFLOPS %lf code %d \n\n",   time_in_sec / (double) ntimes * 1000, gflops, methodid);
	} else {
		printf("\nELL double cpu time %lf ms GFLOPS %lf code %d \n\n",   time_in_sec / (double) ntimes * 1000, gflops, methodid);
	}

	if (csrKernel)
	    clReleaseKernel(csrKernel);

	double onetime = time_in_sec / (double) ntimes;
	if (onetime < opttime)
	{
	    opttime = onetime;
	    optmethod = methodid;
	}
}
//----------------------------------------------------------------------
template <typename T>
void ELL<T>::method_1()
{
	int methodid = 1;
	cl_uint work_dim = 2;
	size_t blocksize[] = {WORK_GROUP_SIZE, 1};
	int row4num = rownum / 4;
	if (rownum % 4 != 0)
	    row4num++;
	int aligned4 = aligned_length / 4;
	int row4 = rownum / 4;
	if (rownum % 4 != 0)
	    row4++;
	int gsize = ((row4num + WORK_GROUP_SIZE - 1)/WORK_GROUP_SIZE)*WORK_GROUP_SIZE;
	size_t globalsize[] = {gsize, dim2};

	cl_kernel csrKernel = NULL;
	std::string kernel_name = getKernelName("gpu_ell_v4");
	printf("****** kernel_name: %s ******\n", kernel_name.c_str());
	csrKernel = clCreateKernel(program, kernel_name.c_str(), &errorCode); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 0, sizeof(cl_mem), &devColid); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 1, sizeof(cl_mem), &devData); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 2, sizeof(int),    &aligned4); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 3, sizeof(int),    &ellnum); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 4, sizeof(cl_mem), &devVec); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 5, sizeof(cl_mem), &devRes); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 6, sizeof(int),    &row4); CHECKERROR;

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
	printf("ntimes= %d, time_in_msec= %f, nnz= %d\n", ntimes, time_in_sec*1000., nnz);
	double gflops = (double)nnz*2/(time_in_sec/(double)ntimes)/(double)1e9;
	if (sizeof(T) == sizeof(float)) {
		printf("\nELL float4 cpu time %lf ms GFLOPS %lf code %d \n\n",   time_in_sec / (double) ntimes * 1000, gflops, methodid);
	} else {
		printf("\nELL double4 cpu time %lf ms GFLOPS %lf code %d \n\n",   time_in_sec / (double) ntimes * 1000, gflops, methodid);
	}

	if (csrKernel) {
	    clReleaseKernel(csrKernel);
	}

	double onetime = time_in_sec / (double) ntimes;

	if (onetime < opttime) {
	    opttime = onetime;
	    optmethod = methodid;
	}
}
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

#if 1
template <typename  T>
void spmv_ell(char* oclfilename, coo_matrix<int, T>* coo_mat, int dim2Size, int ntimes, cl_device_type deviceType)
{

	printf("GORDON, spmv_ell\n");
	ELL<T> ell_ocl(coo_mat, dim2Size, oclfilename, deviceType, ntimes);

	//spmv_ell_ocl_T<T>(&ellmat, vec, res, dim2Size, opttime1, optmethod1, oclfilename, deviceType, coores, ntimes);
	
	ell_ocl.run();

	double opttime = ell_ocl.getOptTime();
	int optmethod = ell_ocl.getOptMethod();

	printf("\n------------------------------------------------------------------------\n");
	printf("ELL best time %f ms best method %d", opttime*1000.0, optmethod);
	printf("\n------------------------------------------------------------------------\n");
}
//----------------------------------------------------------------------
#endif

}; // namespace

#endif
