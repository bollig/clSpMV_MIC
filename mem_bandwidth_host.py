#!/usr/bin/python
import os

HOSTNAME = os.getenv('HOSTNAME')

# not clear the threads command is working
THREADS= "export OMP_NUM_THREADS=8"
THREADS= "export OMP_NUM_THREADS=1"
THREADS= "export OMP_NUM_THREADS=32"
THREADS= "export OMP_NUM_THREADS=16"
SCHEDULE= "export OMP_SCHEDULE=dynamic,64"
SCHEDULE= "export OMP_SCHEDULE=static,64"
SCHEDULE= "export OMP_SCHEDULE=dynamic,64"

KMC= "export KMP_AFFINITY=granularity=fine,scatter"
KMC= "export KMP_AFFINITY=scatter"
KMC= "export KMP_AFFINITY=granularity=fine,compact"
KMC= "export KMP_AFFINITY=compact"
#LIB= "export LD_LIBRARY_PATH=/opt/intel/composer_xe_2013/lib/mic/"
LIB= "export JUNK="
EXEC= "./linux/release/mem_test_host"

def make_cmd_host(outfile):
    OMP= "%s;%s;%s" % (THREADS, SCHEDULE, KMC)
    INNER= "%s; %s; %s > %s " % (OMP, LIB, EXEC, outfile)
    INNER= "%s; %s > %s " % (OMP, EXEC, outfile)
    #CMD="(ssh S2-mic0 %s)" % INNER
    CMD="(%s)" % INNER
    return(CMD)

bench=["read","write", "read_write", "read_write_cpp"]
bench=["read_write_cpp", "gather_cpp"]
bench=["read_write_cpp"]
bench=["write", "read", "read_write", "read_write_cpp", "read_write_cpp_alone"]
bench=["read_write_cpp_alone", "read_write"]
col=["compact", "reverse", "random"]
bench=["write", "read", "read_write", "gather", "unpack", "read_write_cpp", "gather_cpp", "read_write_cpp_alone"]
bench=["read", "read_cpp"]
bench=["write", "write_cpp"]
bench=["gather_cpp"]
bench=["read_write_cpp","write_cpp","read_cpp"]
bench=["gather", "unpack", "gather_cpp"]
bench=["gather"]
bench=["read", "read_cpp", "read_write_cpp", "read_write", "write", "write_cpp"]
col=["compact"]
nb_rows=[2,4,8,16,32,64,128,256]
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
           out_file = "bench=%s_col=%s_rows=%0.2d_host.out" % (b,c,r)
           fd = open('bench.conf', 'w')
           fd.write(file_content)
           fd.close()
           CMD = make_cmd_host(out_file)
           print CMD
           host = "hostname = %s" % HOSTNAME
           os.system(CMD)
           #CMD1 = "cat %s >> %s" % (host, out_file)
           #os.system(CMD1)
#----------------------------------------------------------------------
