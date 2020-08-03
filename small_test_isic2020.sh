#!/bin/bash
#SBATCH --job-name=ISIC
#SBATCH --output=/homes/sallegretti/standard_output/small_test%a_o.txt
#SBATCH --error=/homes/sallegretti/standard_error/small_test%a_e.txt
#SBATCH --partition=prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --exclude aimagelab-srv-10,softechict-nvidia
#SBATCH --array=1-4

module load anaconda3
#export PYTHONPATH="${PYTHONPATH}:/homes/fpollastri/code/pytorch_examples/"

if [ "$SLURM_ARRAY_TASK_ID" -eq "1" ]; then
srun python -u /homes/sallegretti/ISIC_2020/classification_net.py --network resnet50 --epochs 120 --batch_size 16 --save_dir /nas/softechict-nas-1/sallegretti/SUBMISSIONMODELS/ --learning_rate 0.01 --SRV --optimizer SGD --scheduler plateau --augm_config 16 --dataset isic2020 --copy_into_tmp
fi

if [ "$SLURM_ARRAY_TASK_ID" -eq "2" ]; then
srun python -u /homes/sallegretti/ISIC_2020/classification_net.py --network resnet50 --epochs 120 --batch_size 16 --save_dir /nas/softechict-nas-1/sallegretti/SUBMISSIONMODELS/ --learning_rate 0.001 --SRV --optimizer SGD --scheduler plateau --augm_config 16 --dataset isic2020 --copy_into_tmp
fi

if [ "$SLURM_ARRAY_TASK_ID" -eq "3" ]; then
srun python -u /homes/sallegretti/ISIC_2020/classification_net.py --network resnet50 --epochs 120 --batch_size 16 --save_dir /nas/softechict-nas-1/sallegretti/SUBMISSIONMODELS/ --learning_rate 0.0001 --SRV --optimizer SGD --scheduler plateau --augm_config 16 --dataset isic2020 --copy_into_tmp
fi

if [ "$SLURM_ARRAY_TASK_ID" -eq "4" ]; then
srun python -u /homes/sallegretti/ISIC_2020/classification_net.py --network resnet50 --epochs 120 --batch_size 16 --save_dir /nas/softechict-nas-1/sallegretti/SUBMISSIONMODELS/ --learning_rate 0.00001 --SRV --optimizer SGD --scheduler plateau --augm_config 16 --dataset isic2020 --copy_into_tmp
fi


if [ "$SLURM_ARRAY_TASK_ID" -eq "5" ]; then
srun python -u /homes/sallegretti/ISIC_2020/classification_net.py --network resnet50 --epochs 120 --batch_size 16 --save_dir /nas/softechict-nas-1/sallegretti/SUBMISSIONMODELS/ --learning_rate 0.001 --SRV --optimizer SGD --scheduler plateau --augm_config 16 --dataset isic2020 --copy_into_tmp
fi
