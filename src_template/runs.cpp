#include <stdio.h>
#include <string>
#include "runs.h"

void RunData::print()
{
	printf("=====================================\n");
	printf("Input Node file: %s\n", data_filename.c_str());
	printf("kernel_path: %s\n", kernel_path.c_str());
	printf("kernel_name: %s\n", kernel_name.c_str());
	printf("asci_binary: %s\n", asci_binary.c_str());
	printf("data type: %s\n", type.c_str());
	printf("co_processor: %s\n", co_processor.c_str());
	printf("cpu_processor: %s\n", cpu_processor.c_str());
	printf("sparse_format: %s\n", sparse_format.c_str());
	printf("nb_sockets: %d\n", nb_sockets);
	printf("nb_cores: %d\n", nb_cores);
	printf("nb_coprocessors: %d\n", nb_coprocessors);
	printf("nb_nodes: %d\n", nb_nodes);
	printf("nb_nodes_per_stencil: %d\n", nb_nodes_per_stencil);
	printf("kernel_exec_time: %f (ms)\n", kernel_exec_time);
}
//----------------------------------------------------------------------
