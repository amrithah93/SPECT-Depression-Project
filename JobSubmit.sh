#!/bin/bash
#SBATCH -p qTRDHM
#SBATCH --exclude=arctrdhm001
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 5
#SBATCH --mem=512GB
#SBATCH -t 120:00:00
#SBATCH -e error%A.err 
#SBATCH -o out%A.out
#SBATCH -A trends53c17
#SBATCH --oversubscribe
#SBATCH -J depression_ica_analyses

# a small delay at the start often helps
sleep 10s 

# print some message to the log
module load matlab

# CD into your directory
cd '/data/qneuromark/Data/Depression/amenclinic_depression_SPECT/Data/rest_max'
# run the matlab batch script
matlab -batch 'gigica_step2'

# a delay at the end is also good practice
sleep 10s
