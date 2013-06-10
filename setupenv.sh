#!/bin/bash

if [ $HOSTNAME = "S2" ] ; then
	echo "Frodo"
	export CL_DEVICE_TYPE="CL_DEVICE_TYPE_ACCELERATOR"
	export CLSPMVPATH=/mnt/global/LCSE/gerlebacher/src/clSpMV_MIC
	export CL_KERNELS=${CLSPMVPATH}/kernels
elif [ $HOSTNAME = "Gordons-MacBook-Pro.local" ] ; then
	echo "Home Mac"
	export CL_DEVICE_TYPE="CL_DEVICE_TYPE_GPU"
	export CLSPMVPATH=/Users/erlebach/Documents/src/spmv_mic
	export CL_KERNELS=/Users/erlebach/Documents/src/spmv_mic/kernels
fi

