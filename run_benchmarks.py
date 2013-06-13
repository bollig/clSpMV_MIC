#!/usr/bin/python
import os


#----------------------------------------------------------------------
def run_cases(file):
	case = [5,7,9,10]

	for c in case:
		CMD="./linux/release/spmv_all matrix/%s %s 5 > %s_case%s" % (file,c, file,c)
		print(CMD + "\n")
		os.system(CMD)
#----------------------------------------------------------------------
file = "accelerator.mtx"
run_cases(file)

file = "one_million.mtx"
run_cases(file)

file = "thirtytwo_million.mtx"
run_cases(file)




