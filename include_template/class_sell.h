//TODO (disable from Makefile)
//FIX	two_vec_compare_T(coores, tmpresult, rownum);

#ifndef __CLASS_SPMV_SELL_H__
#define __CLASS_SPMV_SELL_H__

#include "util.h"
#include "CL/cl.h"
#include "matrix_storage.h"
#include <vector>
#include <string>

#include "constant.h"
#include "oclcommon.h"
#include "cl_base_class.h" // SuperBuffer still undefined
#include "class_base.h"

#include "projectsettings.h"

namespace spmv {

#define USE(x) using BASE<T>::x
#define USECL(x) using CLBaseClass::x

template <typename T>
class SELL : public BASE<T>
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
	CLBaseClass::SuperBuffer<int> supSlicePtr;

	USE(ntimes);
	USE(filename);

    USE(aligned_length);
    USE(nnz);
    USE(rownum);
	USE(vecsize);
    //USE(ellnum);
	USE(coo_mat);

	USE(opttime);
	USE(optmethod);

    int sliceheight;
    int slicenum;
    int datasize;
	int totalnum;
	int maxwidth;
	int minwidth;

    double overallopttime;
	std::vector<T> paddedvec_v;
	std::vector<T> vec_v;
	std::vector<T> result_v;
	std::vector<T> coores_v;

    USE(dim2); // relates to workgroups

    //USE(vec);
    //USE(result);
    USE(coores);

	USE(getKernelName);
	USECL(loadKernel);
	USECL(enqueueKernel);

public:
	SELL(coo_matrix<int, T>* coo_mat, int dim2Size, char* oclfilename, cl_device_type deviceType, int ntimes);
	~SELL<T>() {}

	virtual void run();

protected:
	virtual void method_0_group();
	virtual void method_0_warp();
};

//----------------------------------------------------------------------
template <typename T>
SELL<T>::SELL(coo_matrix<int, T>* coo_mat, int dim2Size, char* oclfilename, cl_device_type deviceType, int ntimes) : 
   BASE<T>(coo_mat, dim2Size, oclfilename, deviceType, ntimes)
{
    printMatInfo_T(coo_mat);
    vec_v.resize(coo_mat->matinfo.width);
    result_v.resize(coo_mat->matinfo.height);
	std::fill(vec_v.begin(), vec_v.end(), 1.);
	std::fill(result_v.begin(), result_v.end(), 0.);
    coores_v.resize(coo_mat->matinfo.height);
    spmv_only_T<T>(coo_mat, vec_v, coores_v);

    //---------------

	opttime = 10000.0f;
	optmethod = 0;
}
//----------------------------------------------------------------------
template <typename T>
void SELL<T>::run()
{
	optmethod = 0;

	sliceheight = WARPSIZE;
	sell_matrix<int, T> mat;
	coo_mat->print();
	coo2sell<int, T>(coo_mat, &mat, sliceheight);
	mat.print();

    nnz = mat.matinfo.nnz;
	printf("nnz= %d\n", nnz);
	slicenum = mat.sell_slice_num;
	totalnum = mat.sell_slice_ptr[slicenum];
	maxwidth = 0;
	minwidth = 100000;

	printf("slicenum= %d\n", slicenum);

	for (int i = 0; i < slicenum; i++)
	{
	    int size = mat.sell_slice_ptr[i + 1] - mat.sell_slice_ptr[i];
	    size /= sliceheight;
	    if (size > maxwidth)
		maxwidth = size;
	    if (size < minwidth)
		minwidth = size;
	}

    //Initialize values
    nnz = mat.matinfo.nnz;
    rownum = mat.matinfo.height;
    vecsize = mat.matinfo.width;
    sliceheight = mat.sell_slice_height;
    slicenum = mat.sell_slice_num;
    datasize = mat.sell_slice_ptr[slicenum];

    //const cl_image_format floatFormat = { CL_R, CL_FLOAT, };

	supSlicePtr.create(slicenum+1);
	supSlicePtr.setName("slicePtr");
	std::copy(mat.sell_slice_ptr.begin(), mat.sell_slice_ptr.end(), supSlicePtr.host->begin());
	//exit(0);
    //ALLOCATE_GPU_READ(devSlicePtr, mat.sell_slice_ptr, sizeof(int)*(slicenum + 1));

	supColid.setName("supColid");
	supColid.create(datasize);
	std::copy(mat.sell_col_id.begin(), mat.sell_col_id.end(), supColid.host->begin());
    //ALLOCATE_GPU_READ(devColid, mat.sell_col_id, sizeof(int)*datasize);

	supData.setName("supData");
	supData.create(datasize);
	std::copy(mat.sell_data.begin(), mat.sell_data.end(), supData.host->begin());
    //ALLOCATE_GPU_READ(devData, mat.sell_data, sizeof(float)*datasize);

	supVec.create(vec_v);
	supVec.setName("supVec");
    //ALLOCATE_GPU_READ(devVec, vec, sizeof(float)*vecsize);

    int paddedres = findPaddedSize(rownum, 512);
	supRes.create(paddedres);
	supRes.setName("supRes");
    //devRes = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(T)*paddedres, NULL, &errorCode); CHECKERROR;
    //errorCode = clEnqueueWriteBuffer(cmdQueue, devRes, CL_TRUE, 0, sizeof(T)*rownum, result, 0, NULL, NULL); CHECKERROR;

	supSlicePtr.copyToDevice();
	supVec.copyToDevice();
	supData.copyToDevice();
	supColid.copyToDevice();

    //---------------------------------
    if (sliceheight == SELL_GROUP_SIZE) {
		method_0_group();
	}

    if (sliceheight == WARPSIZE) {
		method_0_warp(); 
	}

	double opttime = this->getOptTime();
	int optmethod  = this->getOptMethod();
	//double opttime = sell_ocl.getOptTime();
	//int optmethod = sell_ocl.getOptMethod();

	double gflops = (double)nnz*2/opttime/(double)1e9;
	const char* format = (sizeof(T) == sizeof(float)) ? "double" : "float";
	printf("\n------------------------------------------------------------------------\n");
	printf("SELL %s best time %f ms best method %d GFLOPS %f", format, opttime*1000.0, optmethod, gflops);
	printf("\n------------------------------------------------------------------------\n");

}
//----------------------------------------------------------------------
template <typename T>
void SELL<T>::method_0_warp()
{
	printf("================ METHOD_0_WARP =========================\n");

	int methodid = 0;
	cl_uint work_dim = 2;
	size_t blocksize[] = {SELL_GROUP_SIZE, 1};
	int gsize = ((rownum + SELL_GROUP_SIZE - 1)/SELL_GROUP_SIZE)*SELL_GROUP_SIZE;
	cl::NDRange globalsize(gsize, dim2);
	//size_t globalsize[] = {gsize, dim2};
	printf("gsize %d rownum %d slicenum %d sliceheight %d datasize %d nnz %d vecsize %d \n", gsize, rownum, slicenum, sliceheight, datasize, nnz, vecsize);
	//int warpnum = SELL_GROUP_SIZE / WARPSIZE;

	std::string kernel_name = getKernelName("gpu_sell_warp");
	printf("****** kernel_name: %s ******\n", kernel_name.c_str());
	cl::Kernel kernel = loadKernel(kernel_name, filename);

	try {
		int i=0; 
		kernel.setArg(i++, supSlicePtr.dev);
		kernel.setArg(i++, supColid.dev);
		kernel.setArg(i++, supData.dev);
		kernel.setArg(i++, supVec.dev);
		kernel.setArg(i++, supRes.dev);
		kernel.setArg(i++, sizeof(int), &slicenum);
    } catch (cl::Error er) {
        printf("[setKernelArg] ERROR: %s(%s)\n", er.what(), CLBaseClass::oclErrorString(er.err()));
		exit(0);
    }

	enqueueKernel(kernel, globalsize, cl::NDRange(blocksize[0], blocksize[1]), true);
	supRes.copyToHost();
	two_vec_compare_T(coores_v, *supRes.host, rownum);

	for (int k = 0; k < 3; k++) {
		enqueueKernel(kernel, globalsize, cl::NDRange(blocksize[0], blocksize[1]), true);
	}

	double teststart = timestamp();
	for (int i = 0; i < ntimes; i++) {
		enqueueKernel(kernel, globalsize, cl::NDRange(blocksize[0], blocksize[1]), true);
	}

	double testend = timestamp();
	double time_in_sec = (testend - teststart)/(double)dim2;
	double gflops = (double)nnz*2/(time_in_sec/(double)ntimes)/(double)1e9;
	const char* format = (sizeof(T) == sizeof(float)) ? "double" : "float";
	printf("\nSELL %s cpu warp time %lf ms GFLOPS %lf code %d \n\n",   format, time_in_sec / (double) ntimes * 1000, gflops, methodid);

	double onetime = time_in_sec / (double) ntimes;
	if (onetime < opttime)
	{
	    opttime = onetime;
	    optmethod = methodid;
	}
}   
//----------------------------------------------------------------------
template <typename T>
void SELL<T>::method_0_group()
{
//    if (sliceheight == SELL_GROUP_SIZE)

	printf("================ METHOD_0_GROUP =========================\n");
	int methodid = 1;
	cl_uint work_dim = 2;
	size_t blocksize[] = {SELL_GROUP_SIZE, 1};
	int gsize = slicenum * SELL_GROUP_SIZE;
	//size_t globalsize[] = {gsize, dim2};
	cl::NDRange globalsize(gsize, dim2);

	std::string kernel_name = getKernelName("gpu_sell_warp");
	printf("****** kernel_name: %s ******\n", kernel_name.c_str());
	cl::Kernel kernel = loadKernel(kernel_name, filename);


	try {
		int i=0; 
		kernel.setArg(i++, supSlicePtr.dev);
		kernel.setArg(i++, supColid.dev);
		kernel.setArg(i++, supData.dev);
		kernel.setArg(i++, supVec.dev);
		kernel.setArg(i++, supRes.dev);
		kernel.setArg(i++, sizeof(int), &slicenum);
    } catch (cl::Error er) {
        printf("[setKernelArg] ERROR: %s(%s)\n", er.what(), CLBaseClass::oclErrorString(er.err()));
		exit(0);
    }

	enqueueKernel(kernel, globalsize, cl::NDRange(blocksize[0], blocksize[1]), true);
	supRes.copyToHost();
	two_vec_compare_T(coores_v, *supRes.host, rownum);


	for (int k = 0; k < 3; k++) {
		enqueueKernel(kernel, globalsize, cl::NDRange(blocksize[0], blocksize[1]), true);
	}

	double teststart = timestamp();
	for (int i = 0; i < ntimes; i++) {
		enqueueKernel(kernel, globalsize, cl::NDRange(blocksize[0], blocksize[1]), true);
	}

	double testend = timestamp();
	double time_in_sec = (testend - teststart)/(double)dim2;
	double gflops = (double)nnz*2/(time_in_sec/(double)ntimes)/(double)1e9;
	const char* format = (sizeof(T) == sizeof(float)) ? "double" : "float";
	printf("\nSELL %s cpu group time %lf ms GFLOPS %lf code %d \n\n",   format, time_in_sec / (double) ntimes * 1000, gflops, methodid);

	double onetime = time_in_sec / (double) ntimes;
	if (onetime < opttime)
	{
	    opttime = onetime;
	    optmethod = methodid;
	}
}
//----------------------------------------------------------------------
template <typename  T>
void spmv_sell(char* oclfilename, coo_matrix<int, T>* coo_mat, int dim2Size, int ntimes, cl_device_type deviceType)
{

	printf("GORDON, spmv_sell\n");
	SELL<T> sell_ocl(coo_mat, dim2Size, oclfilename, deviceType, ntimes);

	sell_ocl.run();

	printf("GORDON: after ell_ocl.run\n");

}
//----------------------------------------------------------------------

}; // namespace

#endif
