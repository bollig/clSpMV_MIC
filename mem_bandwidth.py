#!/usr/bin/python
import os

# not clear the threads command is working
THREADS= "export OMP_NUM_THREADS=244"
SCHEDULE= "export OMP_SCHEDULE=dynamic,64"
SCHEDULE= "export OMP_SCHEDULE=static,64"
KMC= "export KMP_AFFINITY=granularity=fine,scatter"
KMC= "export KMP_AFFINITY=scatter"
KMC= "export KMP_AFFINITY=granularity=fine,compact"
KMC= "export KMP_AFFINITY=compact"
LIB = "export LD_LIBRARY_PATH=/opt/intel/composer_xe_2013/lib/mic/"
EXEC= "./linux/release/memory_tests"

def make_cmd(outfile):
    OMP= "%s;%s;%s" % (THREADS, SCHEDULE, KMC)
    INNER= "\'cd mic; %s; %s; %s > %s \'" % (OMP, LIB, EXEC, outfile)
    #CMD="(ssh S2-mic0 %s)" % INNER #frodo
    CMD="(ssh mic0 %s)" % INNER #fsu
    return(CMD)

bench=["read","write", "read_write", "read_write_cpp"]
bench=["read_write_cpp", "gather_cpp"]
bench=["read_write_cpp"]
bench=["write", "read", "read_write", "read_write_cpp", "read_write_cpp_alone"]
bench=["read_write_cpp_alone", "read_write"]
col=["compact"]
bench=["write", "read", "read_write", "gather", "unpack", "read_write_cpp", "gather_cpp", "read_write_cpp_alone"]
bench=["read", "read_cpp"]
bench=["write", "write_cpp"]
bench=["gather_cpp"]
bench=["read_write_cpp","write_cpp","read_cpp"]
bench=["gather", "unpack", "gather_cpp"]
bench=["gather"]
bench=["unpack", "gather_cpp"]
col=["compact", "reverse", "random"]
bench=["read", "read_cpp", "read_write_cpp", "read_write", "write", "write_cpp","gather","unpack","gather_cpp"]
nb_rows=[2,4,8,16,32,64,128]
nb_rows=[0]  # vary rows within the test bandwidth program
###
for b in bench:
    for c in col:
        for r in nb_rows:
           file_content="""
                bandwidth_experiment = %s
                col_id_type = %s
                nb_rows = %d
           """ % (b, c, r*128*128*128)
           #print(file_content)
           out_file = "bench=%s_col=%s_rows=%0.2d.out" % (b,c,r)
           fd = open('bench.conf', 'w')
           fd.write(file_content)
           fd.close()
           CMD = make_cmd(out_file)
           print CMD
           os.system(CMD)
#----------------------------------------------------------------------
