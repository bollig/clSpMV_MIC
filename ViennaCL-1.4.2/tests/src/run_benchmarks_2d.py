#!/usr/bin/python
import os

workgroup_size = [1, 32, 128]
dimensions = 2
sparsity = ["compact", "random"]
#sparsity = ["random"]
#nb_nodes_per_stencil = [16,31,32,33,64]
nb_nodes_per_stencil = [16,32]

workgroup_size = [1]
#nb_nodes = [8,16,32,64,128]
nb_nodes = [64,128]

#random_x_weights_direct__no_hv_stsize_33_2d_8x_8y_1z.mtxb

def generateFileList():
    files = []
    print(sparsity)
    for s in sparsity:
        for ns in nb_nodes_per_stencil:
            for nn in nb_nodes:
                    file = "%s_x_weights_direct__no_hv_stsize_%d_2d_%dx_%dy_1z.mtxb" % (s,ns,nn,nn)
                    files.append(file)
    return files
#----------------------------------------------------------------------
def run_cases():

    files = generateFileList()
    print(files)
    #os.system("mkdir output")

    for file_name in files:
        print "filename = ", file_name
        fil = "matrix/" + file_name
        file_content = """
         coprocessor = MIC
         filename = %s
        """ % fil
        fd = open('test.conf', 'w')
        fd.write(file_content)
        fd.close()
        CMD="./gordon_sparse-test-opencl > output/%s_out" % file_name
        print(CMD + "\n")
        os.system(CMD)
#----------------------------------------------------------------------
#files = generateFileList()
#print(files)

run_cases()


#----------------------------------------------------------------------
