#ifndef __CLASS_SPMV_SBELL_H__
#define __CLASS_SPMV_SBELL_H__

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
class SBELL : public BASE<T>
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

	//int col_align;
   	//int data_align;
    int blockrownum;
    int bwidth, bw;
    int bheight, bh;
    int width4num;
    int padveclen;

    int slicenum;
    int sliceheight;
    int totalsize;

    T* paddedvec;
    double overallopttime;

	vector<T> paddedvec_v;
	vector<T> vec_v;
	vector<T> result_v;
	vector<T> coores_v;

public:
	SBELL(coo_matrix<int, T>* mat, int dim2Size, char* oclfilename, cl_device_type deviceType, int ntimes);
	~SBELL<T>() { }
	virtual void run();
	virtual void method_0();
};

//----------------------------------------------------------------------
template <typename T>
SBELL<T>::SBELL(coo_matrix<int, T>* coo_mat, int dim2Size, char* oclfilename, cl_device_type deviceType, int ntimes) :
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
	#if 0
	for (int i=0; i < 10; i++) {
		printf("coores_v[%d]= %f\n", i, coores_v[i]);
	}
	#endif
}
//----------------------------------------------------------------------
template <typename T>
void SBELL<T>::run()
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

		sbell_matrix<int, T> mat;
		if (coo2sbell<int, T>(coo_mat, &mat, bwidth, bheight, WARPSIZE) == false) {
	  		continue;
		}

		opttime = 10000.0f;
		optmethod = 0;

    	//col_align = mat.sbell_height_aligned;
    	//data_align = mat.sbell_float4_aligned;
    	nnz = mat.matinfo.nnz;
    	rownum = mat.matinfo.height;
    	blockrownum = mat.sbell_row_num;
    	vecsize = mat.matinfo.width;
    	bwidth = mat.sbell_bwidth;
    	bheight = mat.sbell_bheight;
    	width4num = bwidth / 4;   // 8 in double precision?
    	padveclen = findPaddedSize(vecsize, 8);   // change for double precision?
		assert(padveclen == vecsize);
    	slicenum = mat.sbell_slice_num;
    	sliceheight = mat.sbell_slice_height;
    	totalsize = mat.sbell_slice_ptr[slicenum];

		
	// initialize
    //Initialize values

	supSlicePtr.create(slicenum+1);
	supSlicePtr.setName("slicePtr");
	std::copy(mat.sbell_slice_ptr.begin(), mat.sbell_slice_ptr.end(), supSlicePtr.host->begin());
    //ALLOCATE_GPU_READ(devSlicePtr, mat->sbell_slice_ptr, sizeof(int)*(slicenum + 1));


    	paddedvec_v.resize(padveclen);
		std::copy(vec_v.begin(), vec_v.end(), paddedvec_v.begin());

		supColid.setName("supColid");
		supColid.create(totalsize);
		std::copy(mat.sbell_col_id.begin(), mat.sbell_col_id.end(), supColid.host->begin());
    	//ALLOCATE_GPU_READ(devColid, mat->sbell_col_id, sizeof(int)*totalsize);

		// Dangerous if used with size argument due to destructor. Would only work if 
		// all pointers were shared pointers. 
		supData.setName("supData");
		supData.create(totalsize*bwidth*bheight);
		std::copy(mat.sbell_data.begin(), mat.sbell_data.end(), supData.host->begin());
    	//ALLOCATE_GPU_READ(devData, mat->sbell_data, sizeof(float)*totalsize*bwidth*bheight);

		//supVec = CLBaseClass::SuperBuffer<T>(paddedvec_v, "supVec");
		supVec.create(paddedvec_v);
		supVec.setName("supVec");
    	//ALLOCATE_GPU_READ(devVec, paddedvec, sizeof(float)*padveclen);

		supVec.copyToDevice();
		supData.copyToDevice();
		supColid.copyToDevice();
		supSlicePtr.copyToDevice();

    	int paddedres = findPaddedSize(rownum, SELL_GROUP_SIZE * bheight);
		supRes.create(paddedres);
		supRes.setName("supRes");

    	//devRes = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*paddedres, NULL, &errorCode); CHECKERROR;

		// I am probably copying more data than I need to since I copy entire array and not just the part with data.
		// end initialize

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

		method_0();
		//method_2(); // images. Not support on mic
		//method_3(); // images

		if (opttime < overallopttime) {
	    	overallopttime = opttime;
	    	bestbw = bwidth;
	    	bestbh = bheight;
		}
		double gflops = (double)nnz*2/opttime/(double)1e9;
		char* format = (sizeof(T) == sizeof(float)) ? "double" : "float";
		printf("SBELL info: block row num %d slice num %d total block num %d \n", mat.sbell_row_num, mat.sbell_slice_num, mat.sbell_slice_ptr[mat.sbell_slice_num]);
		printf("\n------------------------------------------------------------------------\n");
		printf("SBELL %s best time %f ms best method %d GFLOPS %f", format, opttime*1000.0, optmethod, gflops);
		printf("\n------------------------------------------------------------------------\n");
    }}

    //freeObjects(devices, &context, &cmdQueue, &program); // should not be required
	//double opttime = getOptTime();
}
//----------------------------------------------------------------------
template <typename T>
void SBELL<T>::method_0() // FOR SBELL
{
	printf("======== METHOD 0 ======================================================\n");
	int methodid = 0;
	cl_uint work_dim = 2;
	size_t blocksize[] = {SELL_GROUP_SIZE, 1};
	int gsize = ((blockrownum + SELL_GROUP_SIZE - 1)/SELL_GROUP_SIZE)*SELL_GROUP_SIZE;
	size_t globalsize[] = {gsize, dim2};
	char kernelname[100] = "gpu_sbell00";
	kernelname[9] += bh;
	kernelname[10] += bw;

	std::string kernel_name = getKernelName(kernelname);
	printf("****** kernel_name: %s ******\n", kernel_name.c_str());

	// VERY INEFFICIENT. Should only compile each kernel once. 
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
	char* format = (sizeof(T) == sizeof(float)) ? "double" : "float";
	printf("\nBELL %s  %dx%d block cpu time %lf ms GFLOPS %lf code %d \n\n", format, bh, bw,  time_in_sec / (double) ntimes * 1000, gflops, methodid);


	double onetime = time_in_sec / (double) ntimes;
	if (onetime < opttime)
	{
	    opttime = onetime;
	    optmethod = methodid;
	}
}
//----------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------

template <typename  T>
void spmv_sbell(char* oclfilename, coo_matrix<int, T>* coo_mat, int dim2Size, int ntimes, cl_device_type deviceType)
{
	SBELL<T> sbell_ocl(coo_mat, dim2Size, oclfilename, deviceType, ntimes);
	sbell_ocl.run();

	double opttime = sbell_ocl.getOptTime();
	int optmethod = sbell_ocl.getOptMethod();

	printf("\n------------------------------------------------------------------------\n");
	printf("SBELL best time %f ms best method %d", opttime*1000.0, optmethod);
	printf("\n------------------------------------------------------------------------\n");
}
//----------------------------------------------------------------------

// namespace
}; 

#endif
