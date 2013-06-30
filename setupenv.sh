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
elif [ $HOSTNAME = "sc" -o $HOSTNAME = "hpc-15-35" ]  ; then
	echo "FSU MIC/host node"
	module load intel-oc
	export CL_DEVICE_TYPE="CL_DEVICE_TYPE_ACCELERATOR"
	export CLSPMVPATH=/panfs/storage.local/scs/home/gerlebacher/src/clSpMV_MIC
	export CL_KERNELS=${CLSPMVPATH}/kernels
fi

