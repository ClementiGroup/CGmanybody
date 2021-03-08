#!/bin/bash

rm valid_Error* 
rm slurm-*
rm Error*
rm CV_error.txt 

sbatch myjob_run1.slurm
sbatch myjob_run2.slurm
sbatch myjob_run3.slurm
sbatch myjob_run4.slurm
sbatch myjob_run5.slurm