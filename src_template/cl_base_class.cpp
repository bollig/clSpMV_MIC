#include <iostream>
#include <fstream>
#include <stdio.h>
#include <vector>
#include <stdlib.h>
#include <string.h>

#include <limits.h>

#include "cl_base_class.h"

using namespace std;

// Static definition.
cl::Context CLBaseClass::context;
cl::CommandQueue CLBaseClass::queue;
int CLBaseClass::contextCreated = 0;

//----------------------------------------------------------------------
CLBaseClass::CLBaseClass(int rank) {
    printf("Initialize OpenCL object and context\n");

    if (!contextCreated) {
        //setup devices and context
        std::vector<cl::Platform> platforms;
        std::cout << "Getting the platform" << std::endl;
        err = cl::Platform::get(&platforms);
        std::cout << "GOT PLATFORM" << std::endl;
        printf("cl::Platform::get(): %s\n", oclErrorString(err));
        if (platforms.size() == 0) {
            printf("Platform size 0\n");
        }

        //create the context
        cl_context_properties properties[] =
        { CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[0])(), 0};

		char* device_type = getenv("CL_DEVICE_TYPE");
		printf("device_type= %s\n", device_type);
        std::cout << "Creating cl::Context (only selecting GPU devices)" << std::endl;

		if (!strcmp(device_type, "CL_DEVICE_TYPE_GPU")) {
        	context = cl::Context(CL_DEVICE_TYPE_GPU, properties);
			printf("GPU device\n");
		} else if (!strcmp(device_type, "CL_DEVICE_TYPE_ACCELERATOR")) {
        	context = cl::Context(CL_DEVICE_TYPE_ACCELERATOR, properties);
			printf("ACCELERATOR device\n");
		} else {
			printf("device_type %s not valid\n", device_type);
			exit(1);
		}
        // This prevents the context from being created again
        contextCreated++;
    }

    devices = context.getInfo<CL_CONTEXT_DEVICES>();

    //create the command queue we will use to execute OpenCL commands
    try{
        deviceUsed = rank % devices.size();
        queue = cl::CommandQueue(context, devices[deviceUsed], CL_QUEUE_PROFILING_ENABLE, &err);
        printf("[initialize] Using CL device: %d\n", deviceUsed);
        std::cout << "\tDevice Name: " <<
            devices[deviceUsed].getInfo<CL_DEVICE_NAME>().c_str() << std::endl;
        std::cout << "\tDriver Version: " <<
            devices[deviceUsed].getInfo<CL_DRIVER_VERSION>() << std::endl;
        std::cout << "\tVendor: " <<
            devices[deviceUsed].getInfo<CL_DEVICE_VENDOR_ID>() << std::endl;
        std::cout << "\tMax Compute Units: " <<
            devices[deviceUsed].getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() <<
            std::endl;
        std::cout << "\tMax Work Item Dimensions: " <<
            devices[deviceUsed].getInfo<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>()
            << std::endl;
    }
    catch (cl::Error er) {
        printf("[initialize] ERROR: %s(%d)\n", er.what(), er.err());
    }
    std::cout << "Done with cl::Context setup..." << std::endl;
}

//----------------------------------------------------------------------
std::string CLBaseClass::addExtension(std::string& source, std::string ext,
        bool enabled) {
    std::ostringstream oss;
    oss << "#pragma OPENCL EXTENSION " << ext << ": ";
    if (enabled)
        oss << "enable\n";
    else
        oss << "disable\n";
#if 0
    if (enabled)
            oss << "typedef double FLOAT;\n";
    else
            oss << "typedef float FLOAT;\n";
#endif

    oss << "\n" << source;

    return oss.str();
}

//----------------------------------------------------------------------
std::string CLBaseClass::getDeviceFP64Extension(int device_id) {
    if (device_id < 0) {
        device_id = deviceUsed;
    }

    std::vector<std::string> d_exts =
        split(devices[device_id].getInfo<CL_DEVICE_EXTENSIONS>(), ' ');

    std::vector<std::string>::iterator d;
    int count = 0;

    std::string ext = "";
    for (d = d_exts.begin(); d != d_exts.end(); d++) {
        if ((*d).find("fp64") != std::string::npos) {
            std::cout << "FOUND MATCHING FP64 EXTENSION: " << *d << std::endl;
            ext = *d;
            count ++;
        }
    }
    if (count > 1) {
        // If we find multiple extensions ending in fp64 then we
        // want to return the API standard extension:
        return "cl_khr_fp64";
    }
    return ext;
}

//----------------------------------------------------------------------
void CLBaseClass::loadProgram(std::string& kernel_source, bool enable_fp64)
{
    //Program Setup
    int pl;
    //unsigned int program_length;

    std::string updated_source(kernel_source);

//    if (enable_fp64)
    {
        updated_source = addExtension(kernel_source, getDeviceFP64Extension(deviceUsed), enable_fp64);
    }

    // std::cout << updated_source << std::endl;
    pl = updated_source.size();
    printf("[CLBaseClass] building kernel source of size: %d\n", pl);
//    printf("KERNEL: \n %s\n", updated_source.c_str());
    try
    {
        cl::Program::Sources source(1,
                std::make_pair(updated_source.c_str(), pl));
        program = cl::Program(context, source);
    }
    catch (cl::Error er) {
        printf("[CLBaseClass::loadProgram] ERROR: %s(%s)\n", er.what(), oclErrorString(er.err()));
    }

    try
    {
            char* kernel_dir = getenv("CL_KERNELS");
            char* cwd = getenv("PWD");

            if (kernel_dir == NULL) {
                printf("\n**** [CLBaseClass] Error: You must set CL_KERNELS in your environment to run on the GPU!\n\n");
                exit(EXIT_FAILURE);
            }
     //       printf("Loading kernels from the directory: %s and .\n **** Specified by environment variable: CL_KERNELS\n", kernel_dir);
            char includes[PATH_MAX];
            sprintf(includes, "-I%s -I%s", kernel_dir, cwd);
            //sprintf(includes, "-cl-opt-disable -I%s -I%s", kernel_dir, cwd);
            err = program.build(devices, includes);
    }
    catch (cl::Error er) {
        printf("program.build: %s\n", oclErrorString(er.err()));
        std::cout << "Build Status: " <<
            program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(devices[deviceUsed])
            << std::endl;
        std::cout << "Build Options:\t" <<
            program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(devices[deviceUsed])
            << std::endl;
        std::cout << "Build Log:\t " <<
            program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[deviceUsed]) <<
            std::endl;
        exit(EXIT_FAILURE);
    }
    printf("[CLBaseClass] done building program\n");

#if 0
    std::cout << "Build Status: " <<
        program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(devices[deviceUsed]) <<
        std::endl;
    std::cout << "Build Options:\t" <<
        program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(devices[deviceUsed]) <<
        std::endl;
    std::cout << "Build Log:\t " <<
        program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[deviceUsed]) <<
        std::endl;
#endif
}


//----------------------------------------------------------------------
std::string CLBaseClass::loadFileContents(const char* filename, bool searchEnvDir)
{
	// The filename should contain NO path information
//printf("** loadFileContents\n");
        if (searchEnvDir) {
                char* kernel_dir = getenv("CL_KERNELS");

                if (kernel_dir == NULL) {
                        printf("\n**** [CLBaseClass] Error: You must set CL_KERNELS in your environment to load kernel file contents!\n\n");
                        exit(EXIT_FAILURE);
                }
                char kernel_path[PATH_MAX];
                sprintf(kernel_path, "%s/%s", kernel_dir, filename);
				//printf("kernel_path= %s\n", kernel_path);

                std::ifstream ifs(kernel_path);
                std::string str((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
				//printf("1. SOURCE: %s\n", str.c_str());  // THE SOURCE IS NOT BEING READ!!!
				//printf("after 1. source, exit\n");exit(0);
                return str;
        } else {
                // Grab the whole file in one go using the iterators
                std::ifstream ifs(filename);
                std::string str((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
				//printf("2. SOURCE: %s\n", str.c_str());
                return str;
        }
}


//----------------------------------------------------------------------
// Split a string (for example, a list of extensions, given a character
// deliminator)
std::vector<std::string>& CLBaseClass::split(const std::string &s, char delim, std::vector<std::string> &elems) {
    std::stringstream ss(s);
    std::string item;
    while(std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}

std::vector<std::string> CLBaseClass::split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    return split(s, delim, elems);
}


// Split a string (for example, a list of extensions, given a string deliminator)
// NOTE: if keep_substr=false we discard the delim wherever it is matched
// if keep_substr=true we keep the delim in the substrings
std::vector<std::string>& CLBaseClass::split(const std::string &s, const std::string delim, std::vector<std::string> &elems, bool keep_substr) {
    size_t last_found = 0;
    size_t find_start = last_found;
    size_t found = 0;
    while(found != std::string::npos)
    {
        found = s.find(delim, find_start);
        std::string sub = s.substr(last_found, found - last_found);
        if (sub.length() > 0)
            elems.push_back(sub);
        if (keep_substr) {
            last_found = found;
            find_start = found + delim.length();
        } else {
            last_found = found + delim.length();
            find_start = found + delim.length();
        }
    }
    return elems;
}

//----------------------------------------------------------------------
std::vector<std::string> CLBaseClass::split(const std::string &s, const std::string delim, bool keep_substr) {
    std::vector<std::string> elems;
    return split(s, delim, elems, keep_substr);
}


//----------------------------------------------------------------------
cl::Kernel CLBaseClass::loadKernel(const std::string& kernel_name, const std::string& kernel_source_file)
{
    //tm["loadAttach"]->start();

	cl::Kernel kernel;
	bool useDouble = false; // FOR NOW


	#if 0
    if (!this->getDeviceFP64Extension().compare("")){
        useDouble = false;
    }
    if ((sizeof(FLOAT) == sizeof(float)) || !useDouble) {
        useDouble = false;
    }
	#endif

	cout << "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n";
	cout << "++++++       LOAD KERNEL                           +++++++++\n";
	cout << "kernel_name= " << kernel_name << endl;
	cout << "kernel_source_file = " << kernel_source_file << endl;
	cout << "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n";


    // The true here specifies we search throught the dir specified by environment variable CL_KERNELS
    std::string my_source = this->loadFileContents(kernel_source_file.c_str(), true);


    //std::cout << "This is my kernel source: ...\n" << my_source << "\n...END\n";
	//std::cout  << my_source  << std::endl;
    this->loadProgram(my_source, useDouble);

    try{
        //std::cout << "Loading kernel \""<< kernel_name << "\" with double precision = " << useDouble << "\n";
        kernel = cl::Kernel(program, kernel_name.c_str(), &err);
        //std::cout << "Done attaching kernels!" << std::endl;
    }
    catch (cl::Error er) {
        printf("[AttachKernel] ERROR: %s(%d)\n", er.what(), er.err());
    }
	//printf("gordon exit\n");exit(0);

    //tm["loadAttach"]->end();

	return(kernel);
}
//----------------------------------------------------------------------
void CLBaseClass::enqueueKernel(const cl::Kernel& kernel, const cl::NDRange& tot_work_items, const cl::NDRange& items_per_workgroup, bool is_finish)
{
	cl_int err; // already defined in base opencl class
	//printf("before queue.enqueueNDRangeKernal\n");
    err = queue.enqueueNDRangeKernel(kernel, /* offset */ cl::NullRange,
            tot_work_items, items_per_workgroup, NULL, &event);

	//printf("after queue.enqueueNDRangeKernal\n");
 
//END-START gives you hints on kind of “pure HW execution time”
//10
////the resolution of the events is 1e-09 sec
//11
//g_NDRangePureExecTimeMs = (cl_double)(end - start)*(cl_double)(1e-06); 
//
	std::vector<cl::Event> ve;
	ve.push_back(event);

    if (err != CL_SUCCESS) {
        std::cerr << "CommandQueue::enqueueNDRangeKernel()" \
            " failed (" << err << ")\n";
        std::cout << "FAILED TO ENQUEUE KERNEL" << std::endl;
        exit(EXIT_FAILURE);
    }

	if (is_finish) {
		try {
    		err = queue.finish();
        	//queue.flush(); // needed? (done by clwaitForEvents);
    	} catch (cl::Error er) {
        	printf("[enqueueKernel] ERROR: %s(%s)\n", er.what(), oclErrorString(er.err()));
			exit(0);
    	}
	}

	try {
		//printf("before waitForEvents\n");
		cl::Event::waitForEvents(ve);
		//printf("after waitForEvents\n");
    } catch (cl::Error er) {
        printf("[enqueueKernel] ERROR: %s(%s)\n", er.what(), oclErrorString(er.err()));
		exit(0);
    }
	cl_ulong start = 0, end = 0;
	event.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
	event.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
	printf("GPU execution time = %.2f ms\n", (float) (end-start)*1.e-6);
	return; // TEMPORARY
}
//----------------------------------------------------------------------
