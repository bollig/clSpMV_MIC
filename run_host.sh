#!/bin/sh
F=random_x_weights_direct__no_hv_stsize_32_3d_64x_64y_64z.mtx
F=compact_x_weights_direct__no_hv_stsize_32_3d_64x_64y_64z.mtx
F=compact_x_weights_direct__no_hv_stsize_32_3d_48x_48y_48z.mtx
F=compact_x_weights_direct__no_hv_stsize_32_3d_64x_64y_64z.mtxb
F=random_x_weights_direct__no_hv_stsize_32_3d_64x_64y_64z.mtxb
F=random_x_weights_direct__no_hv_stsize_16_2d_8x_8y_1z.mtxb
F=ell_ell_x_weights_direct__no_hv_stsize_16_3d_8x_8y_1z.bmtx
F=ell_ell_x_weights_direct__no_hv_stsize_32_3d_64x_64y_64z.bmtx
F=ell_x_weights_direct__no_hv_stsize_32_3d_64x_64y_64z.bmtx
F=ell_rcm_sym_1_x_weights_direct__no_hv_stsize_32_3d_64x_64y_64z.bmtx

export F=$F
export OMP_THREAD=16
# 2nd argument must be as low as possible in random case
export OMP_SCHEDULE=dynamic,64
export OMP_SCHEDULE=guided,64
export KMP_AFFINITY=scatter
export KMP_AFFINITY=compact

#./linux/release/spmv_openmp matrix/compact_x_weights_direct__no_hv_stsize_32_3d_64x_64y_64z.mtxb 1 5
#./linux/release/spmv_openmp matrix/$F 1 5

#echo "1"
#CMD= "/opt/intel/mic/bin/micnativeloadex ./linux/release/spmv_openmp -a \"matrix/$F 1 5\"" 
#echo "2"
#echo $CMD
#echo "3"
export SINK_LD_LIBRARY_PATH=/opt/intel/composer_xe_2013.5.192/compiler/lib/mic/
#/opt/intel/mic/bin/micnativeloadex ./linux/release/memory_tests > aaa.gordon
#(ssh S2-mic0 'cd mic;  export LD_LIBRARY_PATH=/opt/intel/composer_xe_2013/lib/mic/; ./linux/release/memory_tests > aaa.gordon')

./linux/release/spmv_openmp_host  matrix/$F 1


