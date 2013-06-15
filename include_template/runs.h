#ifndef __RUNDATA_H__
#define __RUNDATA_H__

#include <string>

class RunData
{
public:
	std::string data_filename;
	std::string kernel_path;
	std::string kernel_name;
	std::string asci_binary;
	std::string type; // float/double
	std::string cpu_processor; 
	std::string co_processor; // MIC, K20, TESLA, CPU
	std::string sparsity; // COMPACT, RANDOM
	std::string reordering; // Cuthill-McGee, space-filling (which type)
	std::string sparse_format; // ELL, SeLL, SBELL, BELL, CVR
	int nb_coprocessors; 
	int nb_cores;
	int nb_sockets;
	int nb_nodes_per_stencil;
	int nb_nodes; // grid size: square matrices)
	double kernel_exec_time;

public:
	void print();
};
#endif

#if 0
	std::kernel
  - kernel file/name
  - float or double
  - work group and total number of threads
  - nb nodes in template
  - grid size
  - type of process (MIC, K20, TESLA, AMD)
  - type of CPU (how to get this)
  - register analysis of rows (NEED CODE FOR THIS)
  - sparsity type (COMPACT, RANDOM)
  - whether or not matrix data has been reordered and the type
     (Cut-Hill-McGee, space-filling curve)
  - sparse matrix format: (ELL, SELL, SBELL, BELL, CVR)
#endif
//--------------------------------------------------------
