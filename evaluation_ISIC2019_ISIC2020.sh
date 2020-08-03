#!/bin/bash
#SBATCH --job-name=ISIC
#SBATCH --output=/homes/sallegretti/standard_output/evaluation_ISIC2019%a_o.txt
#SBATCH --error=/homes/sallegretti/standard_error/evaluation_ISIC2019%a_e.txt
#SBATCH --partition=prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --exclude aimagelab-srv-10,softechict-nvidia
#SBATCH --array=1

module load anaconda3
#export PYTHONPATH="${PYTHONPATH}:/homes/fpollastri/code/pytorch_examples/"

if [ "$SLURM_ARRAY_TASK_ID" -eq "1" ]; then
srun python -u /homes/sallegretti/ISIC_2020/classification_net.py --network densenet201 --batch_size 8 --save_dir /nas/softechict-nas-1/sallegretti/SUBMISSIONMODELS --augm_config 16 -c 0 -c 1 2 3 4 5 6 7 --load_epoch 102 --epochs 0 --dataset isic2020
fi

