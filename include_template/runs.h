#ifndef __RUNDATA_H__
#define __RUNDATA_H__

#include <string>

class RunData
{
public:
    enum stencilType {COMPACT=0, RANDOM, RANDOMWITHDIAG, RANDOMDIAGS};
    enum nonzeroStats {UNIFORM=0, NORMAL};

	std::string data_filename;
	std::string kernel_path;
	std::string kernel_name;
	std::string asci_binary;
	std::string type; // float/double
	std::string cpu_processor; 
	std::string co_processor; // MIC, K20, TESLA, CPU
	//std::string sparsity; // COMPACT, RANDOM
	std::string reordering; // Cuthill-McGee, space-filling (which type)
	std::string sparse_format; // ELL, SeLL, SBELL, BELL, CVR
	int nb_coprocessors; 
	int nb_cores;
	int nb_sockets;
	int nb_nodes_per_stencil;
	int nb_nodes; // grid size: square matrices)
	double kernel_exec_time;
    int use_subdomains;

    int nb_rows;
    int nb_mats;
    int nb_vecs;
    int stencil_size;
    stencilType sparsity; // **
    int diag_sep;
    int inner_bandwidth;
    int sort_col_indices;
    int random_seed;
    int nonzero_stats;  // random, Gaussian, etc.
    int n3d; // average size of 3d grid. nb_rows = n3d^3

public:
    RunData()
    {
        sort_col_indices = 1;
        random_seed = 0;
    }
	void print();
};
#endif

#if 0
   //newVars(int new_col_id_size, int new_nb_vec, int new_nb_mats, int new_nz, stencilType new_sparsity, int new_diag_step, int new_inner_bandwidth, int  new_nonzero_stats)
   // newVars(new_col_id_size, new_nb_vec, new_nb_mats, new_nz, new_sparsity, new_diag_step, new_inner_bandwidth, 
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
