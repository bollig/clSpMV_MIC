
template <typename T> 
class SuperBuffer {
public:
	cl::Buffer dev;
	std::vector<T>* host;
	int error;
	bool host_changed;
	bool dev_changed;
	std::string name;

	// I cannot change pointer to host (cpu) data after creation
	// // cost of CLBaseClass is high. Should only be called once. 
	SuperBuffer(std::string name="") {
		this->name = name;
		host = 0;
		printf("++++ Created empty SuperBuffer ++++ \n\n");
	}
	void create(std::vector<T>& host_) { // std::string name="") : host(&host_) 
		dev_changed = false;
		host_changed = true;
		try {
			host = &host_;
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
	void create(std::vector<T>* host_) { // std::string name="") : host(&host_) 
		dev_changed = false;
		host_changed = true;
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
}; // end SuperBuffer subclass



};

#endif 
