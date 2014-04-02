#!/usr/bin/python
# Run on the mic, by launching from the host.

import os

#-------------------------------------------------------
# not clear the threads command is working
THREADS= "export OMP_NUM_THREADS=244"
SCHEDULE= "export OMP_SCHEDULE=static,64"
KMC= "export KMP_AFFINITY=granularity=fine,scatter"
KMC= "export KMP_AFFINITY=scatter"
KMC= "export KMP_AFFINITY=granularity=fine,compact"

SCHEDULE= "export OMP_SCHEDULE=dynamic,64"
KMC= "export KMP_AFFINITY=compact"
LIB = "export LD_LIBRARY_PATH=/opt/intel/composer_xe_2013/lib/mic/"
F="matrix/ell_ell_sym_1_x_weights_direct__no_hv_stsize_32_3d_8x_8y_8z.bmtx"
F= "matrix/ell_ell_x_weights_direct__no_hv_stsize_32_3d_64x_64y_64z.bmtx"
F="matrix/ell_kd-tree_x_weights_direct__no_hv_stsize_32_3d_128x_128y_128z.bmtx"
F="matrix/ell_kd-tree_x_weights_direct__no_hv_stsize_32_3d_192x_192y_192z.bmtx"
F="matrix/ell_kd-tree_rcm_sym_1_x_weights_direct__no_hv_stsize_32_3d_192x_192y_192z.bmtx"
F="matrix/ell_kd-tree_rcm_sym_1_x_weights_direct__no_hv_stsize_32_3d_128x_128y_128z.bmtx"
F="matrix/ell_x_weights_direct__no_hv_stsize_32_3d_64x_64y_64z.bmtx"
F="matrix/ell_kd-tree_rcm_sym_1_x_weights_direct__no_hv_stsize_32_3d_64x_64y_64z.bmtx"
F="matrix/ell_kd-tree_x_weights_direct__no_hv_stsize_32_3d_64x_64y_64z.bmtx"
F="matrix/ell_kd-tree_x_weights_direct__no_hv_stsize_32_3d_96x_96y_96z.bmtx"
F="matrix/ell_kd-tree_rcm_sym_1_x_weights_direct__no_hv_stsize_32_3d_96x_96y_96z.bmtx"


EXEC= "./linux/release/spmv_openmp_mic %s 1" % F
#-------------------------------------------------------
def make_cmd_from_host(outfile):
    OMP= "%s;%s;%s" % (THREADS, SCHEDULE, KMC)
    INNER= "\'cd mic; %s; %s; %s > %s \'" % (OMP, LIB, EXEC, outfile)
    #INNER= "\'cd mic; %s; %s; %s  \'" % (OMP, LIB, EXEC)
    CMD="(ssh S2-mic0 %s)" % INNER
    return(CMD)
#-------------------------------------------------------
#-------------------------------------------------------
#-------------------------------------------------------
#----------------------------------------------------------------------
def run_cases():

    files = [F]
    os.system("mkdir output")

    for f in files:
            out_file = "gordonxxx.out"
            CMD = make_cmd_from_host(out_file)
            print(CMD + "\n")
            os.system(CMD)
#----------------------------------------------------------------------
run_cases()
#----------------------------------------------------------------------
