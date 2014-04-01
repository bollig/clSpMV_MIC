#qsub  -I -l walltime=24:00:00,nodes=1:ppn=16 -q phi 
#qsub -X -I -l walltime=24:00:00,nodes=1:ppn=16 -q phi 

qsub -I -l walltime=4:00:00,nodes=1:ppn=12:phi,pmem=200mb


