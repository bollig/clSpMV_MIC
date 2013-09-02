#!/bin/sh
F=random_x_weights_direct__no_hv_stsize_32_3d_64x_64y_64z.mtx
F=compact_x_weights_direct__no_hv_stsize_32_3d_64x_64y_64z.mtx
F=compact_x_weights_direct__no_hv_stsize_32_3d_48x_48y_48z.mtx
F=compact_x_weights_direct__no_hv_stsize_32_3d_64x_64y_64z.mtxb
F=random_x_weights_direct__no_hv_stsize_32_3d_64x_64y_64z.mtxb
export F=$F
export OMP_THREAD=1
export OMP_SCHEDULE=dynamic,1
# 2nd argument must be as low as possible in random case
export OMP_SCHEDULE=static,16
export OMP_SCHEDULE=guided,16 
export KMP_AFFINITY=compact
export KMP_AFFINITY=scatter

./linux/release/spmv_openmp_mic -a "matrix/$F 1 5"

# I get to 2 Gflops, but OMP_THREAD has no influence. What is going on? 

