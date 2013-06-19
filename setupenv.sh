#!/bin/sh

if [ $HOSTNAME = "S2" ] ; then
	echo "Frodo"
	export CL_DEVICE_TYPE="CL_DEVICE_TYPE_ACCELERATOR"
	export CLSPMVPATH=/mnt/global/LCSE/gerlebacher/src/clSpMV_MIC
	export CL_KERNELS=${CLSPMVPATH}/kernels
elif [ $HOSTNAME = "cas013" ] ; then
	echo "cascade/cas013"
	export CL_DEVICE_TYPE="CL_DEVICE_TYPE_ACCELERATOR"
	export CLSPMVPATH=/home/bollige/gerlebac/src/clSpMV_MIC
	export CL_KERNELS=${CLSPMVPATH}/kernels
	#source /home/bollige/shared/cascade_env.sh 
elif [ $HOSTNAME = "casiornis" ] ; then
	echo "Casiornis"
	export CL_DEVICE_TYPE="CL_DEVICE_TYPE_ACCELERATOR"
	export CLSPMVPATH=/home/bollige/gerlebac/src/clSpMV_MIC
	export CL_KERNELS=${CLSPMVPATH}/kernels
	#source /home/bollige/shared/cascade_env.sh 
elif [ $HOSTNAME = "Gordons-MacBook-Pro.local" ] ; then
	echo "Home Mac"
	export CL_DEVICE_TYPE="CL_DEVICE_TYPE_GPU"
	export CLSPMVPATH=/Users/erlebach/Documents/src/spmv_mic
	export CL_KERNELS=${CLSPMVPATH}/kernels
elif [ $HOSTNAME = "hpc-15-35" ]  ; then
	echo "FSU MIC/host node"
	module load intel-oc
	export CL_DEVICE_TYPE="CL_DEVICE_TYPE_ACCELERATOR"
	export CLSPMVPATH=/panfs/storage.local/scs/home/gerlebacher/src/clSpMV_MIC
	export CL_KERNELS=${CLSPMVPATH}/kernels
fi

