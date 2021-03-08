#!/bin/bash
#SBATCH --job-name=Simulation
#SBATCH --account=commons
#SBATCH --partition=scavenge
##SBATCH --partition=serial_long
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1G
#SBATCH --threads-per-core=1
#SBATCH --time=00:30:00
##SBATCH --mail-user=cwjiang1989@gmail.com
##SBATCH --mail-type=ALL

# touch test.txt
# echo 'abcd' >> test.txt

#sgdml all Training_set/coordforce_6atom_4500random.npz 4000 10 10 -s 200 --gdml
# add srun
# module
# source

module load Anaconda3/5.0.0
source activate py3

echo "My job ran on:"
echo $SLURM_NODELIST
#if [[ -d $SHARED_SCRATCH/$USER && -w $SHARED_SCRATCH/$USER ]]
#then
#cd $SHARED_SCRATCH/$USER
    # srun /path/to/myprogram
#srun python Simulation3.py >> output.txt
# srun python3 NN.py
# srun python3 plotU.py
srun python3 Compute_avgE.py
# srun python3 Simulationb.py -m 1 -m2 1
# srun python3 Compute_Torsion.py
# srun python3 KL.py
#fi