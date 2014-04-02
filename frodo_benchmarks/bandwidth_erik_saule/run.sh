#!/bin/sh

export KMP_AFFINITY=granularity=fine,scatter
#export KMP_AFFINITY=granularity=compact
#export LD_LIBRARY_PATH=~/miclib

# The bandwidths are incrrect for the following four cases
#results/simplesum-char-61-4:aggregated:1.20762e+17
#results/simplesum-char-O1-61-4:aggregated:1.21786e+17
#results/simplesum-int-61-4:aggregated:1.02341e+17
#results/simplesum-int-O1-61-4:aggregated:1.12575e+17


#for code in  memset memcpy vectsum simplesum simplesum-char memset-vect memset-nrngo memset-nr 
for code in  memset memcpy vectsum memset-nrngo memset-nr vectsum-pref memcpy-vect
#for code in simplesum-char-O1 simplesum-int-O1
#for code in simplesum-int-O1
#for code in vectsum
do
    echo "code ", $code
    #for core in 1 2 4 8 16 24 32 40 48 56 61
    #for core in 1 2 4 8 16 32 61
    for core in 61
    do
	for thread in 4
	#for thread in 1 2 3 4
	do
	    ./bandwidth-${code} $core $thread > results/${code}-${core}-${thread}
	done
    done
done
