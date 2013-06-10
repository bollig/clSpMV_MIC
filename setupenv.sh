#!/bin/bash

if [ $HOSTNAME = "S2" ] ; then
	echo "Frodo"
	export CLSPMVPATH=/mnt/global/LCSE/gerlebacher/src/clSpMV_MIC
	export CL_KERNELS=${CLSPMVPATH}/kernels
elif [ $HOSTNAME = "Gordons-MacBook-Pro.local" ] ; then
	echo "Home Mac"
	export CLSPMVPATH=/Users/erlebach/Documents/src/spmv_mic
	export CL_KERNELS=/Users/erlebach/Documents/src/spmv_mic/kernels
fi

