#ifndef __OPENMP_SPMV_BASE_H__
#define __OPENMP_SPMV_BASE_H__

#include "util.h"
//#include "CL/cl.h"
#include "matrix_storage.h"
#include <vector>
#include <string>

//#include "cl_base_class.h"
//#include "oclcommon.h"

namespace spmv {

//----------------------------------------------------------------------
template <typename T>
class OPENMP_BASE //  : public CLBaseClass
{
protected:
    //cl_device_id* devices;
    //cl_context context;
	//cl_command_queue cmdQueue;
    //cl_program program;
	//cl_int errorCode;

    //Create device memory objects
    //cl_mem devColid;
    //cl_mem devData;
    //cl_mem devVec;
    //cl_mem devRes;
    //cl_mem devTexVec;

	//SuperBuffer<int> supColid;
	//SuperBuffer<T> supData;
	//SuperBuffer<T> supVec;
	//SuperBuffer<T> supRes;

	int ntimes;

    int aligned_length;
    int nnz;
    int rownum;
    int vecsize;
    int ellnum;

	double opttime;
	int optmethod;

    int dim2; // relates to workgroups
	std::string filename;

public:
    T* vec;
    T* result;
    T* coores;
	coo_matrix<int, T>* coo_mat;

	//ell_matrix<int, T>* mat;


public:
	OPENMP_BASE(coo_matrix<int, T>* mat, int dim2Size, int ntimes);
	~OPENMP_BASE() {
		//printf("inside base destructor\n");
    	//free_ell_matrix(mat);
    	//free(vec);
    	//free(result);
    	//free(coores);
	}
	virtual void method_0() { printf("method 0 not implemented\n"); }
	virtual void method_1() { printf("method 1 not implemented\n"); }
	virtual void method_2() { printf("method 2 not implemented\n"); }
	virtual void method_3() { printf("method 3 not implemented\n"); }
	virtual void method_4() { printf("method 4 not implemented\n"); }
	virtual void method_5() { printf("method 5 not implemented\n"); }
	virtual double getOptTime() {return opttime;}
	virtual std::string getKernelName(std::string kernel_name);
	int getOptMethod() {return optmethod;}
};


//template <typename T>
//void spmv_ell(char* oclfilename, coo_matrix<int, T>* mat, int dim2Size, int ntimes, cl_device_type deviceType);
//----------------------------------------------------------------------
//template <typename T>
//void spmv_ell_ocl_T(ell_matrix<int, T>* mat, T* vec, T* result, int dim2Size, double& opttime, int& optmethod, char* oclfilename, cl_device_type deviceType, T* coores, int ntimes)
template <typename T>
OPENMP_BASE<T>::OPENMP_BASE(coo_matrix<int, T>* coo_mat, int dim2Size, int ntimes) 
    //: CLBaseClass()
{
    //devices = NULL;
    //context = NULL;
    //cmdQueue = NULL;
    //program = NULL;

	//if (oclfilename) filename = oclfilename;

	this->coo_mat = coo_mat;

    //assert(initialization(deviceType, devices, &context, &cmdQueue, &program, oclfilename) == 1);

    //errorCode = CL_SUCCESS;
    //errorCode = 0;
	this->ntimes = ntimes;

    opttime = 10000.0f;
    optmethod = 0;
    dim2 = dim2Size;
}
//----------------------------------------------------------------------
template <typename T>
std::string OPENMP_BASE<T>::getKernelName(std::string kernel_name)
{
	std::string name = (sizeof(T) == sizeof(float)) ? kernel_name : kernel_name + "_d";
	return(name);
}
//----------------------------------------------------------------------
}; // namespace

#endif
