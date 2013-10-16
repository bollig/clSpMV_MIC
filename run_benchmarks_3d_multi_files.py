#!/usr/bin/python
# Run on the mic, by launching from the host.

# run on the maximum number of threads, once per file, with and without RCM

import os

#-------------------------------------------------------
# not clear the threads command is working
THREADS= "export OMP_NUM_THREADS=244"  # on S1
THREADS= "export OMP_NUM_THREADS=240"  # on S2
SCHEDULE= "export OMP_SCHEDULE=static,64"
KMC= "export KMP_AFFINITY=granularity=fine,scatter"
KMC= "export KMP_AFFINITY=scatter"
KMC= "export KMP_AFFINITY=granularity=fine,compact"

SCHEDULE= "export OMP_SCHEDULE=dynamic,64"
KMC= "export KMP_AFFINITY=compact"
LIB = "export LD_LIBRARY_PATH=/opt/intel/composer_xe_2013/lib/mic/"
dir = "matrix1"
dir = "matrix_random"   # randomized Cartesian grids for more realistic matrix structures
prefixes = ["ell_kd-tree_x_weights_direct__no_hv_stsize_32_3d_",
            "ell_kd-tree_rcm_sym_1_x_weights_direct__no_hv_stsize_32_3d_"]
          #"ell_kd-tree_rcm_sym_1_x_weights_direct__no_hv_stsize_32_3d_64x_96y_96z.bmtx"
npts = [[32,32,32],[48,48,48],[48,48,64],[48,64,64],[64,64,64],[64,64,96],[64,96,96],[96,96,96],
      [96,96,128],[96,128,128]]
npts = [[48,48,64],[64,48,48],[48,64,48],[96,64,64],[64,96,64],[64,64,96]]


#-------------------------------------------------------
def make_cmd_from_host(outfile, execute_line):
    OMP= "%s;%s;%s" % (THREADS, SCHEDULE, KMC)
    INNER= "\'cd mic; %s; %s; %s > %s \'" % (OMP, LIB, execute_line, outfile)
    #INNER= "\'cd mic; %s; %s; %s  \'" % (OMP, LIB, EXEC)
    CMD="(ssh S3-mic0 %s)" % INNER  # frodo
    CMD="(ssh mic0 %s)" % INNER  # hpc15-35
    return(CMD)
#-------------------------------------------------------
#-------------------------------------------------------
#-------------------------------------------------------
def run_case(prefix,nx,ny,nz):
    in_file = "matrix1/"+prefix+str(nx)+"x_"+str(ny)+"y_"+str(nz)+"z.bmtx"
    out_file = prefix+str(nx)+"x_"+str(ny)+"y_"+str(nz)+"z.out"
    os.system("mkdir output")
    execute_line= "./linux/release/spmv_openmp_mic %s 1" % in_file

    CMD = make_cmd_from_host(out_file, execute_line)
    print(CMD + "\n")
    os.system(CMD)
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

for pref in prefixes:
    print "pref= ", pref
    for np in npts:
        nx,ny,nz= np
        run_case(pref,nx,ny,nz)
#----------------------------------------------------------------------
