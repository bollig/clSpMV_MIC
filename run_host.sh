#!/bin/sh 

# run a single case, with parameters determined by the file name. 


F=random_x_weights_direct__no_hv_stsize_32_3d_64x_64y_64z.mtx
F=compact_x_weights_direct__no_hv_stsize_32_3d_64x_64y_64z.mtx
F=compact_x_weights_direct__no_hv_stsize_32_3d_48x_48y_48z.mtx
F=compact_x_weights_direct__no_hv_stsize_32_3d_64x_64y_64z.mtxb
F=compact_x_weights_direct__no_hv_stsize_32_3d_64x_64y_64z.mtxb
F=random_x_weights_direct__no_hv_stsize_32_3d_64x_64y_64z.mtxb
F=random_x_weights_direct__no_hv_stsize_16_2d_8x_8y_1z.mtxb
F=ell_ell_x_weights_direct__no_hv_stsize_16_3d_8x_8y_1z.bmtx
F=ell_ell_x_weights_direct__no_hv_stsize_32_3d_64x_64y_64z.bmtx
F=ell_x_weights_direct__no_hv_stsize_32_3d_64x_64y_64z.bmtx
F=ell_rcm_sym_1_x_weights_direct__no_hv_stsize_32_3d_64x_64y_64z.bmtx
F=ell_kd-tree_x_weights_direct__no_hv_stsize_32_3d_192x_192y_192z.bmtx
F=ell_kd-tree_rcm_sym_1_x_weights_direct__no_hv_stsize_32_3d_128x_128y_128z.bmtx
# core dump
F=ell_kd-tree_x_weights_direct__no_hv_stsize_32_3d_64x_64y_64z.bmtx
F=ell_kd-tree_rcm_sym_1_x_weights_direct__no_hv_stsize_32_3d_96x_96y_96z.bmtx
F=ell_kd-tree_x_weights_direct__no_hv_stsize_32_3d_128x_128y_128z.bmtx

export F=$F
#no speedup past 16
export OMP_THREAD=32  # TIMING IS independent of number processors!!!
export OMP_THREAD=1  # TIMING IS independent of number processors!!!
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
#export SINK_LD_LIBRARY_PATH=/opt/intel/composer_xe_2013.5.192/compiler/lib/mic/
#/opt/intel/mic/bin/micnativeloadex ./linux/release/memory_tests > aaa.gordon
#(ssh S2-mic0 'cd mic;  export LD_LIBRARY_PATH=/opt/intel/composer_xe_2013/lib/mic/; ./linux/release/memory_tests > aaa.gordon')

output=$F
output+="_"
output+=$HOSTNAME
output+="_"
output+=$KMP_AFFINITY
output+="_"
output+=$OMP_SCHEDULE
echo $output

#(export OMP_THREAD=10; ./linux/release/spmv_openmp_host  matrix/$F 1 )
(export OMP_THREAD=10; ./linux/release/spmv_openmp_host  matrix/$F 1 > $output)


