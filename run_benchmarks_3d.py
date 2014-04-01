#!/usr/bin/python
import os

case = [5,7,9,10]
workgroup_size = [1, 32, 128]
dimensions = 2
sparsity = ["compact", "random"]
sparsity = ["random"]
#nb_nodes_per_stencil = [16,31,32,33,64]
nb_nodes_per_stencil = [32]
hostname = "%s" % os.getenv("HOSTNAME")

case = [5,7,9,10]
workgroup_size = [1]
#nb_nodes = [8,16,32,64,128]
nb_nodes = [128]

#random_x_weights_direct__no_hv_stsize_33_2d_8x_8y_1z.mtxb

def generateFileList():
    files = []
    print(sparsity)
    for s in sparsity:
        for ns in nb_nodes_per_stencil:
            for nn in nb_nodes:
                    #file = "%s_x_weights_direct__no_hv_stsize_%d_2d_%dx_%dy_1z.mtx" % (s,ns,nn,nn) # works
                    file = "%s_x_weights_direct__no_hv_stsize_%d_3d_%dx_%dy_%dz.mtxb" % (s,ns,nn,nn,nn)
                    files.append(file)
    return files
#----------------------------------------------------------------------
def run_cases():

    files = generateFileList()
    os.system("mkdir output")

    for f in files:
        for c in case:
            CMD="./linux/release/spmv_all matrix/%s %s 5 > output/%s_case%s_%s" % (f, c, f,c, hostname)
            print(CMD + "\n")
            os.system(CMD)
#----------------------------------------------------------------------
files = generateFileList()
print(files)

run_cases()


#----------------------------------------------------------------------
