#!/bin/sh

export KMP_AFFINITY=granularity=fine,scatter
#export KMP_AFFINITY=granularity=compact
#export LD_LIBRARY_PATH=~/miclib


#for code in  memset memcpy vectsum simplesum simplesum-char memcpy memset memset-vect memset-nrngo memset-nr memset
#for code in simplesum-char-O1 simplesum-int-O1
#for code in simplesum-int-O1
for code in vectsum
do
    #for core in 1 2 4 8 16 24 32 40 48 56 61
    for core in 1 2 4 8 16 32 61
    do
	#for thread in 4
	for thread in 1 2 3 4
	do
	    ./bandwidth-${code} $core $thread > results/${code}-${core}-${thread}
	done
    done
done
