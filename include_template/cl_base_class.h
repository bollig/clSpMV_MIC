#ifndef __CL_BASE_CLASS_H__
#define __CL_BASE_CLASS_H__
#define __CL_ENABLE_EXCEPTIONS
#include <iostream> 
#include <string>
#include <sstream>
#include <CL/cl.hpp>
#include <stdio.h>
#include <vector>
#include <boost/shared_ptr.hpp>

// A bunch of util methods borrowed from Ian's tutorial on opencl
class CLBaseClass 
{
public:
	// We use static so all of our inheriting classes can share buffers
	// across the context
	static cl::Context context;
	static cl::CommandQueue queue;

	cl::Kernel loadKernel(const std::string& kernel_name, const std::string& kernel_source_file);
	void enqueueKernel(const cl::Kernel& kernel, const cl::NDRange& tot_work_items, 
				const cl::NDRange& items_per_workgroup, bool is_finish);

    protected:
        unsigned int deviceUsed;
        std::vector<cl::Device> devices;

        // Track if context was created so we dont accidentally make a new one
        static int contextCreated; 

        cl::Program program;
        cl::Kernel kernel;

        //debugging variables
        cl_int err;
        cl::Event event;
        cl::Event event2;
        cl::Event event3;

    public:

        //----------------------------------------------------------------------
        CLBaseClass(int rank=0);

        //----------------------------------------------------------------------
        std::string addExtension(std::string& source, std::string ext, bool
                enabled = true);

        //----------------------------------------------------------------------
        std::string getDeviceFP64Extension(int device_id=-1);

        //----------------------------------------------------------------------
        void loadProgram(std::string& kernel_source, bool enable_fp64 = false);


        std::string loadFileContents(const char* filename, bool searchEnvDir);

        //----------------------------------------------------------------------
        // Helper function to get error string
        // *********************************************************************
        static const char* oclErrorString(cl_int error)
        {
            static const char* errorString[] = {
                "CL_SUCCESS",
                "CL_DEVICE_NOT_FOUND",
                "CL_DEVICE_NOT_AVAILABLE",
                "CL_COMPILER_NOT_AVAILABLE",
                "CL_MEM_OBJECT_ALLOCATION_FAILURE",
                "CL_OUT_OF_RESOURCES",
                "CL_OUT_OF_HOST_MEMORY",
                "CL_PROFILING_INFO_NOT_AVAILABLE",
                "CL_MEM_COPY_OVERLAP",
                "CL_IMAGE_FORMAT_MISMATCH",
                "CL_IMAGE_FORMAT_NOT_SUPPORTED",
                "CL_BUILD_PROGRAM_FAILURE",
                "CL_MAP_FAILURE",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "CL_INVALID_VALUE",
                "CL_INVALID_DEVICE_TYPE",
                "CL_INVALID_PLATFORM",
                "CL_INVALID_DEVICE",
                "CL_INVALID_CONTEXT",
                "CL_INVALID_QUEUE_PROPERTIES",
                "CL_INVALID_COMMAND_QUEUE",
                "CL_INVALID_HOST_PTR",
                "CL_INVALID_MEM_OBJECT",
                "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR",
                "CL_INVALID_IMAGE_SIZE",
                "CL_INVALID_SAMPLER",
                "CL_INVALID_BINARY",
                "CL_INVALID_BUILD_OPTIONS",
                "CL_INVALID_PROGRAM",
                "CL_INVALID_PROGRAM_EXECUTABLE",
                "CL_INVALID_KERNEL_NAME",
                "CL_INVALID_KERNEL_DEFINITION",
                "CL_INVALID_KERNEL",
                "CL_INVALID_ARG_INDEX",
                "CL_INVALID_ARG_VALUE",
                "CL_INVALID_ARG_SIZE",
                "CL_INVALID_KERNEL_ARGS",
                "CL_INVALID_WORK_DIMENSION",
                "CL_INVALID_WORK_GROUP_SIZE",
                "CL_INVALID_WORK_ITEM_SIZE",
                "CL_INVALID_GLOBAL_OFFSET",
                "CL_INVALID_EVENT_WAIT_LIST",
                "CL_INVALID_EVENT",
                "CL_INVALID_OPERATION",
                "CL_INVALID_GL_OBJECT",
                "CL_INVALID_BUFFER_SIZE",
                "CL_INVALID_MIP_LEVEL",
                "CL_INVALID_GLOBAL_WORK_SIZE",
            };

            const int errorCount = sizeof(errorString) / sizeof(errorString[0]);
            const int index = -error;
            return (index >= 0 && index < errorCount) ? errorString[index] : "";
        }

        // Split a string (for example, a list of extensions, given a character
        // deliminator)
        std::vector<std::string>& split(const std::string &s, char delim,
                std::vector<std::string> &elems);

        std::vector<std::string> split(const std::string &s, char delim);


        std::vector<std::string>& split(const std::string &s, const
                std::string delim, std::vector<std::string> &elems, bool
                keep_substr=false);
        std::vector<std::string> split(const std::string &s, const std::string delim, bool keep_substr=false); 

public:
	

template <typename T> 
class SuperBuffer {
public:
	cl::Buffer dev;
	//boost:shared_ptr<std::vector<T> >* host;
	std::vector<T>* host;
	T* host_ptr;
	int size;
	int error;
	bool host_changed;
	bool dev_changed;
	bool delete_host; // true if responsible for deleting host;
	std::string name;

	// I cannot change pointer to host (cpu) data after creation
	// // cost of CLBaseClass is high. Should only be called once. 
	~SuperBuffer() {
		printf("inside SuperBuffer(%s) destructor\n", name.c_str());
		// host is managed by a boost shared pointer. In this way, when a SuperBuffer
		// is copied to another, memory is managed correctly
		if (delete_host && host) {
			delete host; 
			host = 0;
			printf("delete host ptr\n");
		}
	}

	SuperBuffer(std::string name="") {
		this->name = name;
		host = 0;
		host_ptr = 0;
		delete_host = false;
		printf("++++ Created empty SuperBuffer ++++ \n\n");
	}
	void create(std::vector<T>& host_) { // std::string name="") : host(&host_) {
		dev_changed = false;
		host_changed = true;
		try {
			host = &host_;
			host_ptr = 0;
			delete_host = false;
			dev = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(T)*host->size(), NULL, &error);
		} catch (cl::Error er) {
	    	printf("[cl::Buffer] ERROR: %s(%s)\n", er.what(), oclErrorString(er.err()));
			exit(0);
		}
	}
	SuperBuffer(std::vector<T>& host_, std::string name="") : host(&host_) {
		printf("SuperBuffer(std::vector<T>& host_, std::string name=\n");
		this->name = name;
		dev_changed = false;
		host_changed = true;
		host_ptr = 0;
		delete_host = false;
		try {
			printf("sizof(T)e: %d\n", sizeof(T));
			printf("size: %d\n", sizeof(T)*host->size());
			printf("host size: %d\n", host->size());
			dev = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(T)*host->size(), NULL, &error);
			printf("Created SuperBuffer *** %s (size: %d bytes) ***\n\n", name.c_str(), host->size()*sizeof(T));
		} catch (cl::Error er) {
	    	printf("[cl::Buffer] ERROR: %s(%s)\n", er.what(), oclErrorString(er.err()));
			exit(0);
		}
	}
	void create(std::vector<T>* host_) { // std::string name="") : host(&host_) {
		dev_changed = false;
		host_changed = true;
		host_ptr = 0;
		delete_host = false;
		try {
			host = host_;
			dev = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(T)*host->size(), NULL, &error);
		} catch (cl::Error er) {
	    	printf("[cl::Buffer] ERROR: %s(%s)\n", er.what(), oclErrorString(er.err()));
			exit(0);
		}
	}
	SuperBuffer(std::vector<T>* host_, std::string name="") : host(host_) {
		printf("SuperBuffer(std::vector<T>* host_, std::string name=\n");
		this->name = name;
		dev_changed = false;
		host_changed = true;
		delete_host = false;
		host_ptr = 0;
		try {
			dev = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(T)*host->size(), NULL, &error);
			printf("Created SuperBuffer *** %s (size: %d bytes) ***\n\n", name.c_str(), host->size()*sizeof(T));
		} catch (cl::Error er) {
	    	printf("[cl::Buffer] ERROR: %s(%s)\n", er.what(), oclErrorString(er.err()));
			exit(0);
		}
	}
	void create(int size) { // std::string name="") : host(&host_) {
		dev_changed = false;
		host_changed = true;
		host = new std::vector<T>(size, 0); 
		size = this->size;    // assert(host.size() == size)
		host_ptr = 0;
		delete_host = false;
		try {
			dev = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(T)*host->size(), NULL, &error);
		} catch (cl::Error er) {
	    	printf("[cl::Buffer] ERROR: %s(%s)\n", er.what(), oclErrorString(er.err()));
			exit(0);
		}
	}
	// SuperBuffer allocates the space
	SuperBuffer(int size, std::string name="") {
		printf("SuperBuffer(int size, std::string name=\n");
		this->name = name;
		dev_changed = false;
		host_changed = true;
		host = new std::vector<T>(size, 0); // implicitly convert from int to double if necesary
		host_ptr = 0;
		delete_host = true;
		try {
			dev = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(T)*host->size(), NULL, &error);
			printf("Created SuperBuffer *** %s (size: %d bytes) ***\n\n", name.c_str(), size*sizeof(T));
		} catch (cl::Error er) {
	    	printf("[cl::Buffer] ERROR: %s(%s)\n", er.what(), oclErrorString(er.err()));
			exit(0);
		}
	}
	// SuperBuffer allocates the space
	SuperBuffer(int size, T* ptr, std::string name="") {
	// NOT DEBUGGED
		printf("SuperBuffer(int size, std::string name=\n");
		this->name = name;
		dev_changed = false;
		host_changed = true;
		host = 0;
		host_ptr = ptr;
		this->size = size; // number elements
		try {
			dev = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(T)*host->size(), NULL, &error);
			printf("Created SuperBuffer *** %s (size: %d bytes) ***\n\n", name.c_str(), size*sizeof(T));
		} catch (cl::Error er) {
	    	printf("[cl::Buffer] ERROR: %s(%s)\n", er.what(), oclErrorString(er.err()));
			exit(0);
		}
	}
	
	inline T operator[](int i) {
		return (*host)[i];  // efficiency is iffy
	}
	inline T& operator[](int i) const {
		return (*host)[i];  // efficiency is iffy
	}
	int devSizeBytes() {
		int mem_size;
		try {
			mem_size = dev.getInfo<CL_MEM_SIZE>();
	 	} catch (cl::Error er) {
	    	printf("[cl::Buffer] ERROR: %s(%s)\n", er.what(), CLBaseClass::oclErrorString(er.err()));
	 		mem_size = -1; // invalid object
		}
		return(mem_size);
	}
	int hostSize() {
		return(host->size());
	}
	int typeSize() {
		return(sizeof(T));
	}

	int devSize()  { return( devSizeBytes()/typeSize() ); }
	int hostSizeBytes() { return(hostSize()*typeSize()); }

	void copyToHost(int nb_elements=-1, int start_index=0) {
		//if (gpu_changed == false) return;
		//gpu_changed = false;
		int nb_elements_bytes = nb_elements*sizeof(T);
		int offset_bytes = start_index * sizeof(T);
		int mem_size_bytes = devSizeBytes(); 
		int transfer_bytes = mem_size_bytes - offset_bytes;
		if (mem_size_bytes < 1) return;
		if (nb_elements > -1 && transfer_bytes > nb_elements_bytes) {
			transfer_bytes = nb_elements_bytes;
		}
		// do not use monitoring events
    	error = queue.enqueueReadBuffer(dev, CL_TRUE, offset_bytes, transfer_bytes, &(*host)[0], NULL, NULL);
		if (error != CL_SUCCESS) {
			std::cerr << " enqueueRead ERROR: " << error << std::endl;
		}
	}
	// nb_bytes and start_index not yet used
	void copyToDevice(int nb_elements=-1, int start_index=0) {
		//if (host_changed == false) return;
		//host_changed = false;
		int nb_elements_bytes = nb_elements*sizeof(T);
		int offset_bytes = start_index * sizeof(T);
		int mem_size_bytes = devSizeBytes(); 
		int transfer_bytes = mem_size_bytes - offset_bytes;
		if (mem_size_bytes < 1) return;
		if (nb_elements > -1 && transfer_bytes > nb_elements_bytes) {
			transfer_bytes = nb_elements_bytes;
		}
		// do not use monitoring events
    	error = queue.enqueueWriteBuffer(dev, CL_TRUE, offset_bytes, transfer_bytes, &(*host)[0], NULL, NULL);
		if (error != CL_SUCCESS) {
			std::cerr << " enqueueWrite ERROR: " << error << std::endl;
		}
	}
	void setName(std::string name) {
		this->name = name;
	}
}; // end SuperBuffer subclass



};

#endif 