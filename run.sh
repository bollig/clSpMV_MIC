#!/bin/sh
F=random_x_weights_direct__no_hv_stsize_32_3d_64x_64y_64z.mtx
F=compact_x_weights_direct__no_hv_stsize_32_3d_64x_64y_64z.mtx
F=compact_x_weights_direct__no_hv_stsize_32_3d_48x_48y_48z.mtx
F=compact_x_weights_direct__no_hv_stsize_32_3d_64x_64y_64z.mtxb
F=random_x_weights_direct__no_hv_stsize_32_3d_32x_32y_32z.mtxb
F=random_x_weights_direct__no_hv_stsize_32_3d_64x_64y_64z.mtxb
F=random_x_weights_direct__no_hv_stsize_32_2d_128x_128y_1z.mtxb
F=compact_x_weights_direct__no_hv_stsize_32_3d_64x_64y_64z.mtxb
F=random_x_weights_direct__no_hv_stsize_32_3d_32x_32y_32z.mtxb
# 12.5 Gflop, 14GF
F=compact_x_weights_direct__no_hv_stsize_4_2d_4x_4y_1z.mtxb
F=random_x_weights_direct__no_hv_stsize_4_2d_8x_8y_1z.mtxb
# method_5 will not work unless stencil is multiple of 32
F=compact_x_weights_direct__no_hv_stsize_32_3d_64x_64y_64z.mtxb
F=compact_x_weights_direct__no_hv_stsize_32_3d_32x_32y_32z.mtxb
# 1.7 Gflop
F=compact_x_weights_direct__no_hv_stsize_32_3d_64x_64y_64z.mtxb  
F=random_x_weights_direct__no_hv_stsize_32_3d_64x_64y_64z.mtxb
F=random_x_weights_direct__no_hv_stsize_32_2d_8x_4y_1z.mtxb
F=compact_x_weights_direct__no_hv_stsize_32_3d_64x_64y_64z.mtxb
F=random_x_weights_direct__no_hv_stsize_32_3d_64x_64y_64z.mtxb
F=compact_x_weights_direct__no_hv_stsize_32_2d_8x_4y_1z.mtxb
F=compact_x_weights_direct__no_hv_stsize_32_3d_48x_48y_48z.mtxb
F=random_x_weights_direct__no_hv_stsize_32_3d_48x_48y_48z.mtxb  
export F=$F
# export OMP_DYNAMIC=FALSE
export OMP_NUM_THREADS=1
export OMP_NUM_THREADS=180
# 2nd argument must be as low as possible in random case
# export OMP_SCHEDULE=guided,16  # 27 Gflop/random
# export OMP_SCHEDULE=guided,8  # 22 Gflop/random
export OMP_SCHEDULE=static,16  # no influence 2.5 Gf
export OMP_SCHEDULE=guided,8  # 15 GF (best results with mymethod_2. 
export OMP_SCHEDULE=dynamic,64  # 27 Gflop/random (and scatter). 22Gflop with copact
export KMP_AFFINITY=scatter  # 25 GF
export KMP_AFFINITY=compact  # 22 GF

#./linux/release/spmv_openmp matrix/compact_x_weights_direct__no_hv_stsize_32_3d_64x_64y_64z.mtxb 1 5
./linux/release/spmv_openmp matrix/$F 1 5


# I get to 2 Gflops, but OMP_THREAD has no influence. What is going on? 
#   for a contrived ccase, which is parallizable.
# I get 0.55 Gflops for SpMV with compact, 64^3. regardless of parameter settings

# random case: 19 Gflops 4x4 case. 
