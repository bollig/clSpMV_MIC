#ifndef __CLASS_SPMV_BELL_H__
#define __CLASS_SPMV_BELL_H__

#include "util.h"
#include "CL/cl.h"
#include "matrix_storage.h"
#include <vector>
#include <string>
#include <algorithm>

#include "oclcommon.h"
#include "cl_base_class.h" // SuperBuffer still undefined
#include "class_base.h"

using namespace std;

namespace spmv {

#define USE(x) using BASE<T>::x
#define USECL(x) using CLBaseClass::x

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

	USE(supColid);
	USE(supData);
	USE(supVec);
	USE(supRes);

	USE(filename);

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
	USECL(loadKernel);
	USECL(enqueueKernel);

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
    double overallopttime;

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

    vec_v.resize(coo_mat->matinfo.width); // experimental use of vectors: xxx_v
	std::fill(vec_v.begin(), vec_v.end(), 1.);
    result_v.resize(coo_mat->matinfo.height);
	std::fill(result_v.begin(), result_v.end(), 0.);
    coores_v.resize(coo_mat->matinfo.height);
	std::fill(coores_v.begin(), coores_v.end(), 7.);
    spmv_only_T<T>(coo_mat, vec_v, coores_v);   // matrix*vector, serial
	for (int i=0; i < 10; i++) {
		printf("coores_v[%d]= %f\n", i, coores_v[i]);
	}
}
//----------------------------------------------------------------------
template <typename T>
void BELL<T>::run()
{
    overallopttime = 10000.0f;
    bestbw = 0;
    bestbh = 0;

	int count = -1;
    for (int bwidth = 4; bwidth < 9; bwidth += 4) {
    for (int bheight = 1; bheight < 9; bheight*=2) {
	// re-initialization Should not be required. Did not help solve the problem. 
	count++;


		//printf("++++++ for loop: bwidth= %d, bheight= %d ++++\n", bwidth, bheight);

		//bwidth = 4; // run same case twice
		//bheight = 1;

		//printf("\n\n** (before if) bwidth= %d, bheight= %d\n", bwidth, bheight);
		//if (count > 1) exit(0);

		bw = bwidth; // used in the method_i()
		bh = bheight;

		b4ell_matrix<int, T> mat;
		if (coo2b4ell<int, T>(coo_mat, &mat, bwidth, bheight, GPU_ALIGNMENT, 0) == false) {
	  		continue;
		}

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
		assert(padveclen == vecsize);

    	paddedvec_v.resize(padveclen);
		std::copy(vec_v.begin(), vec_v.end(), paddedvec_v.begin());

		supColid.setName("supColid");
		supColid.create(col_align*b4ellnum);
		std::copy(mat.b4ell_col_id.begin(), mat.b4ell_col_id.end(), supColid.host->begin());

		// Dangerous if used with size argument due to destructor. Would only work if 
		// all pointers were shared pointers. 
		//supData = CLBaseClass::SuperBuffer<T>(data_align*bheight*width4num*b4ellnum, "supData");
		supData.setName("supData");
		supData.create(data_align*bheight*width4num*b4ellnum);
		std::copy(mat.b4ell_data.begin(), mat.b4ell_data.end(), supData.host->begin());

		//supVec = CLBaseClass::SuperBuffer<T>(paddedvec_v, "supVec");
		supVec.create(paddedvec_v);
		supVec.setName("supVec");

		supVec.copyToDevice();
		supData.copyToDevice();
		supColid.copyToDevice();

    	int paddedres = findPaddedSize(rownum, 512);
		supRes.create(paddedres);
		supRes.setName("supRes");

		#if 0
		printf("\n\n** bw= %d, bh= %d\n", bw, bh);
		printf("col_align= %d\n", col_align);
		printf("data_align= %d\n", data_align);
		printf("blockrownum= %d\n", blockrownum);
		printf("vecsize= %d\n", vecsize);
		printf("b4ellnum= %d\n", b4ellnum);
		printf("width4num= %d\n", width4num);
		printf("padveclen= %d\n", padveclen);
		printf("paddedres= %d\n", paddedres);
		#endif

		//method_0(count); // works alone
		method_1();
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
    }}

    //freeObjects(devices, &context, &cmdQueue, &program); // should not be required
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

	// VERY INEFFICIENT. Should only compile each kernel once. 
	cl::Kernel kernel = loadKernel(kernel_name, filename);

	//printf("after load kernel\n");

	try {
		int i=0; 
		kernel.setArg(i++, supColid.dev);
		kernel.setArg(i++, supData.dev);
		kernel.setArg(i++, sizeof(int), &data_align4); // ERROR
		kernel.setArg(i++, sizeof(int), &col_align);
		kernel.setArg(i++, sizeof(int), &b4ellnum);
		kernel.setArg(i++, supVec.dev);
		kernel.setArg(i++, supRes.dev);
		kernel.setArg(i++, sizeof(int), &blockrownum);
    } catch (cl::Error er) {
        printf("[setKernelArg] ERROR: %s(%s)\n", er.what(), CLBaseClass::oclErrorString(er.err()));
		exit(0);
    }

	 printf(" globalsize= %d,%d, blocksize= %d,%d\n", globalsize[0], globalsize[1], blocksize[0], blocksize[1]); // <<REASON FOR ERROR

	enqueueKernel(kernel, cl::NDRange(globalsize[0],globalsize[1]), cl::NDRange(blocksize[0], blocksize[1]), true);
	supRes.copyToHost();

	for (int i=0; i < rownum; i++) {
		if (fabs(supRes[i]-coores_v[i]) > 1.e-4) 
		//printf("meth 0, (%d),  tmpresult= %f, coores= %f\n", i, tmpresult[i], coores[i]);
		;
	}
	two_vec_compare_T(coores_v, *supRes.host, rownum);

	for (int k = 0; k < 3; k++)
	{
		enqueueKernel(kernel, cl::NDRange(globalsize[0],globalsize[1]), cl::NDRange(blocksize[0], blocksize[1]), true);
	}

	double teststart = timestamp();
	for (int i = 0; i < ntimes; i++)
	{
		enqueueKernel(kernel, cl::NDRange(globalsize[0],globalsize[1]), cl::NDRange(blocksize[0], blocksize[1]), true);
	}

	double testend = timestamp();
	double time_in_sec = (testend - teststart)/(double)dim2;
	printf("ntimes= %d, time_in_msec= %f, nnz= %d\n", ntimes, time_in_sec*1000., nnz);
	double gflops = (double)nnz*2/(time_in_sec/(double)ntimes)/(double)1e9;
	printf("\nBELL %dx%d block cpu time %lf ms GFLOPS %lf code %d \n\n", bh, bw,  time_in_sec / (double) ntimes * 1000, gflops, methodid);


	double onetime = time_in_sec / (double) ntimes;
	if (onetime < opttime)
	{
	    opttime = onetime;
	    optmethod = methodid;
	}
}
//----------------------------------------------------------
template <typename T>
void BELL<T>::method_1()
{
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


	std::string kernel_name = getKernelName(kernelname);
	cl::Kernel kernel = loadKernel(kernel_name, filename);

	try {
		int i=0;
		kernel.setArg(i++, supColid.dev);
		kernel.setArg(i++, supData.dev);
		kernel.setArg(i++, sizeof(int), &data_align4); // ERROR
		kernel.setArg(i++, sizeof(int), &col_align);
		kernel.setArg(i++, sizeof(int), &b4ellnum);
		kernel.setArg(i++, supVec.dev);
		kernel.setArg(i++, supRes.dev);
		kernel.setArg(i++, sizeof(int), &blockrownum);
    } catch (cl::Error er) {
        printf("[setKernelArg] ERROR: %s(%s)\n", er.what(), CLBaseClass::oclErrorString(er.err()));
                exit(0);
    }

    enqueueKernel(kernel, cl::NDRange(globalsize[0],globalsize[1]), cl::NDRange(blocksize[0], blocksize[1]), true);
	supRes.copyToHost();
	two_vec_compare_T(coores_v, *supRes.host, rownum);

	for (int k = 0; k < 3; k++)
	{
		enqueueKernel(kernel, cl::NDRange(globalsize[0],globalsize[1]), cl::NDRange(blocksize[0], blocksize[1]), true);
	}

	double teststart = timestamp();
	for (int i = 0; i < ntimes; i++)
	{
		enqueueKernel(kernel, cl::NDRange(globalsize[0],globalsize[1]), cl::NDRange(blocksize[0], blocksize[1]), true);
	}
	double testend = timestamp();
	double time_in_sec = (testend - teststart)/(double)dim2;
	double gflops = (double)nnz*2/(time_in_sec/(double)ntimes)/(double)1e9;
	printf("\nBELL %dx%d block mad cpu time %lf ms GFLOPS %lf code %d \n\n", bh, bw,  time_in_sec / (double) ntimes * 1000, gflops, methodid);


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
#if 0
// BASED ON IMAGES
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
