#!/bin/sh

if [ $HOSTNAME = "S2" -o $HOSTNAME = "S3" ] ; then
	echo "Frodo"
	export CL_DEVICE_TYPE="CL_DEVICE_TYPE_ACCELERATOR"
	export CLSPMVPATH=$HOME/src/clSpMV_MIC
	export CL_KERNELS=${CLSPMVPATH}/kernels
	export OPENCL_ROOT=/opt/intel/opencl-1.2-3.0.67279/
	# preload does not work
	#export LD_PRELOAD=$OPENCL_ROOT/lib64/libtbb_preview.so
	export LD_PRELOAD=
    source /opt/intel/composer_xe_2013.5.192/bin/compilervars.sh intel64

elif [ $HOSTNAME = "casiornis" -o $HOSTNAME = "cas013"  -o $HOSTNAME = "cas014" ] ; then
	echo $HOSTNAME
	module load intel
    module load intel/cluster
	export CL_DEVICE_TYPE="CL_DEVICE_TYPE_ACCELERATOR"
	export CLSPMVPATH=$HOME/src/clSpMV_MIC
	export CL_KERNELS=${CLSPMVPATH}/kernels
	export OPENCL_ROOT=/opt/intel/opencl-1.2-3.0.67279/
	#export LD_PRELOAD=$OPENCL_ROOT/lib64/libtbb_preview.so
	export LD_LIBRARY_PATH=$OPENCL_ROOT/lib64:$LD_LIBRARY_PATH
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

