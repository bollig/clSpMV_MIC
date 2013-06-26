#!/usr/bin/python
import os

case = [5,7,9,10]
workgroup_size = [1, 32, 128]
dimensions = 2
sparsity = ["compact", "random"]
nb_nodes_per_stencil = [16,31,32,33,64]

case = [10]
workgroup_size = [1]
nb_nodes = [8,16,32,64,128]

#random_x_weights_direct__no_hv_stsize_33_2d_8x_8y_1z.mtxb

def generateFileList():
	files = []
	print(sparsity)
	for s in sparsity:
		for ns in nb_nodes_per_stencil:
			for nn in nb_nodes:
					#file = "%s_x_weights_direct__no_hv_stsize_%d_2d_%dx_%dy_1z.mtx" % (s,ns,nn,nn) # works
					file = "%s_x_weights_direct__no_hv_stsize_%d_2d_%dx_%dy_1z.mtxb" % (s,ns,nn,nn)
					files.append(file)
					#file_contents = """
					#sparsity = %s
					#workgroup_size = %s
					#nb_nodes_per_stencil = %d
					#case = %d
					#""" % (s, w, n, c)
					#print file_contents
					##CMD = "cat '%s' > in_file.txt" % file_contents
					#f = open('in_file.txt', 'w')
					#f.write(file_contents)
					#f.close()
	return files
#----------------------------------------------------------------------
def run_cases(file):

	files = generateFileList()

	for f in files:
		for c in case:
			CMD="./linux/release/spmv_all matrix/%s %s 5 > %s_case%s" % (f, c, f,c)
			print(CMD + "\n")
			os.system(CMD)
#----------------------------------------------------------------------
files = generateFileList()
print(files)

file = "accelerator.mtx"
run_cases(file)

file = "nod1e6_sten10.mtx"
run_cases(file)

file = "nod1e6_sten32.mtx"
run_cases(file)


#----------------------------------------------------------------------
