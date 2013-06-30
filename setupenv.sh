#!/bin/sh

if [ $HOSTNAME = "S2" ] ; then
	echo "Frodo"
	export CL_DEVICE_TYPE="CL_DEVICE_TYPE_ACCELERATOR"
	export CLSPMVPATH=$HOME/src/clSpMV_MIC
	export CL_KERNELS=${CLSPMVPATH}/kernels
elif [ $HOSTNAME = "casiornis" -o $HOSTNAME = "case013" ] ; then
	echo "Casiornis"
	export CL_DEVICE_TYPE="CL_DEVICE_TYPE_ACCELERATOR"
	export CLSPMVPATH=$HOME/src/clSpMV_MIC
	export CL_KERNELS=${CLSPMVPATH}/kernels
elif [ $HOSTNAME = "Gordons-MacBook-Pro.local" ] ; then
	echo "Home Mac"
	export CL_DEVICE_TYPE="CL_DEVICE_TYPE_GPU"
	export CLSPMVPATH=$HOME/src/clSpMV_MIC
	export CL_KERNELS=${CLSPMVPATH}/kernels
fi

