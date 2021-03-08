#!/bin/bash

# rm slurm-*
rm -r SLURMs
rm -r TRAJs

mkdir SLURMs
mkdir TRAJs

# cd SLURMs


for i in $(seq 1 5)
do
for j in $(seq 1 10)
do

cp myjob_run.slurm SLURMs/myjob_run_$i\_$j.slurm

sed -i '34s/-m -m2/-m '$i' -m2 '$j'/' SLURMs/myjob_run_$i\_$j.slurm
sed -i '4s/commons/commons/' SLURMs/myjob_run_$i\_$j.slurm
# sed -i '4s/commons/scavenge	/' SLURMs/myjob_run_$i\_$j.slurm
# sed -i '4s/commons/interactive	/' SLURMs/myjob_run_$i\_$j.slurm
sed -i '10s/24:00:00/24:00:00/' SLURMs/myjob_run_$i\_$j.slurm



done
done

cd SLURMs


for i in $(seq 1 5)
do
for j in $(seq 1 10)
do
sbatch myjob_run_$i\_$j.slurm

done
done


# sbatch myjob_run_1_1.slurm
# sbatch myjob_run_1_2.slurm
# sbatch myjob_run_1_3.slurm
# sbatch myjob_run_1_4.slurm
# sbatch myjob_run_1_5.slurm
# sbatch myjob_run_1_6.slurm
# sbatch myjob_run_1_7.slurm
# sbatch myjob_run_1_8.slurm
# sbatch myjob_run_1_9.slurm
# sbatch myjob_run_1_10.slurm