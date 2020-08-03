#!/bin/bash
#SBATCH --job-name=ISIC
#SBATCH --output=/homes/sallegretti/standard_output/submission_baseline%a_o.txt
#SBATCH --error=/homes/sallegretti/standard_error/submission_baseline%a_e.txt
#SBATCH --partition=prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --exclude aimagelab-srv-10,softechict-nvidia
#SBATCH --array=101-107

module load anaconda3
#export PYTHONPATH="${PYTHONPATH}:/homes/fpollastri/code/pytorch_examples/"

if [ "$SLURM_ARRAY_TASK_ID" -eq "1" ]; then
srun python -u /homes/sallegretti/ISIC_2020/classification_net.py --network densenet201 --epochs 120 --batch_size 8 --save_dir /nas/softechict-nas-1/sallegretti/SUBMISSIONMODELS/ --learning_rate 0.001 --SRV --optimizer SGD --scheduler plateau --augm_config 16 --dataset isic2020 --copy_into_tmp
fi

if [ "$SLURM_ARRAY_TASK_ID" -eq "2" ]; then
srun python -u /homes/sallegretti/ISIC_2020/classification_net.py --network densenet201 --epochs 120 --batch_size 8 --save_dir /nas/softechict-nas-1/sallegretti/SUBMISSIONMODELS/ --learning_rate 0.001 --SRV --optimizer SGD --scheduler plateau --cutout_holes 1 2 3 --cutout_pad 20 50 100  --augm_config 84 --dataset isic2020 --copy_into_tmp
fi

if [ "$SLURM_ARRAY_TASK_ID" -eq "3" ]; then
srun python -u /homes/sallegretti/ISIC_2020/classification_net.py --network densenet201 --epochs 120 --batch_size 8 --save_dir /nas/softechict-nas-1/sallegretti/SUBMISSIONMODELS/ --learning_rate 0.001 --SRV --optimizer SGD --scheduler plateau --cutout_holes 0 0 1 2 3 --cutout_pad 20 50 100  --augm_config 116 --dataset isic2020 --copy_into_tmp
fi

if [ "$SLURM_ARRAY_TASK_ID" -eq "4" ]; then
srun python -u /homes/sallegretti/ISIC_2020/classification_net.py --network densenet201 --epochs 120 --batch_size 8 --save_dir /nas/softechict-nas-1/sallegretti/SUBMISSIONMODELS/ --learning_rate 0.001 --SRV --optimizer SGD --scheduler plateau --augm_config 84 --dataset isic2020 --copy_into_tmp
fi

if [ "$SLURM_ARRAY_TASK_ID" -eq "5" ]; then
srun python -u /homes/sallegretti/ISIC_2020/classification_net.py --network resnet152 --epochs 120 --batch_size 8 --save_dir /nas/softechict-nas-1/sallegretti/SUBMISSIONMODELS/ --learning_rate 0.001 --SRV --optimizer SGD --scheduler plateau --augm_config 16 --dataset isic2020 --copy_into_tmp
fi

if [ "$SLURM_ARRAY_TASK_ID" -eq "6" ]; then
srun python -u /homes/sallegretti/ISIC_2020/classification_net.py --network seresnext101 --epochs 120 --batch_size 8 --save_dir /nas/softechict-nas-1/sallegretti/SUBMISSIONMODELS/ --SRV --optimizer SGD --scheduler plateau --cutout_holes 0 1 2 3 --cutout_pad 20 50 100  --augm_config 84 --learning_rate 0.01 --dataset isic2020 --copy_into_tmp
fi

if [ "$SLURM_ARRAY_TASK_ID" -eq "7" ]; then
srun python -u /homes/sallegretti/ISIC_2020/classification_net.py --network seresnext101 --epochs 120 --batch_size 8 --save_dir /nas/softechict-nas-1/sallegretti/SUBMISSIONMODELS/ --SRV --optimizer SGD --scheduler plateau --cutout_holes 0 0 1 2 3 --cutout_pad 20 50 100  --augm_config 116 --learning_rate 0.01 --dataset isic2020 --copy_into_tmp
fi

if [ "$SLURM_ARRAY_TASK_ID" -eq "101" ]; then
srun python -u /homes/sallegretti/ISIC_2020/classification_net.py --network densenet201 --epochs 120 --batch_size 8 --save_dir /nas/softechict-nas-1/sallegretti/SUBMISSIONMODELS/ --SRV --optimizer SGD --scheduler plateau --augm_config 16 --dataset isic2020 --learning_rate 0.00001 --load_epoch 36
fi

if [ "$SLURM_ARRAY_TASK_ID" -eq "102" ]; then
srun python -u /homes/sallegretti/ISIC_2020/classification_net.py --network densenet201 --epochs 120 --batch_size 8 --save_dir /nas/softechict-nas-1/sallegretti/SUBMISSIONMODELS/ --SRV --optimizer SGD --scheduler plateau --cutout_holes 1 2 3 --cutout_pad 20 50 100  --augm_config 84 --dataset isic2020 --learning_rate 0.0001 --load_epoch 30
fi

if [ "$SLURM_ARRAY_TASK_ID" -eq "103" ]; then
srun python -u /homes/sallegretti/ISIC_2020/classification_net.py --network densenet201 --epochs 120 --batch_size 8 --save_dir /nas/softechict-nas-1/sallegretti/SUBMISSIONMODELS/ --SRV --optimizer SGD --scheduler plateau --cutout_holes 0 0 1 2 3 --cutout_pad 20 50 100  --augm_config 116 --dataset isic2020 --learning_rate 0.00001 --load_epoch 31
fi

if [ "$SLURM_ARRAY_TASK_ID" -eq "104" ]; then
srun python -u /homes/sallegretti/ISIC_2020/classification_net.py --network densenet201 --epochs 120 --batch_size 8 --save_dir /nas/softechict-nas-1/sallegretti/SUBMISSIONMODELS/ --SRV --optimizer SGD --scheduler plateau --augm_config 84 --dataset isic2020 --learning_rate 0.00001 --load_epoch 30
fi

if [ "$SLURM_ARRAY_TASK_ID" -eq "105" ]; then
srun python -u /homes/sallegretti/ISIC_2020/classification_net.py --network resnet152 --epochs 120 --batch_size 8 --save_dir /nas/softechict-nas-1/sallegretti/SUBMISSIONMODELS/ --SRV --optimizer SGD --scheduler plateau --augm_config 16 --dataset isic2020 --learning_rate 0.00001 --load_epoch 25
fi

if [ "$SLURM_ARRAY_TASK_ID" -eq "106" ]; then
srun python -u /homes/sallegretti/ISIC_2020/classification_net.py --network seresnext101 --epochs 120 --batch_size 8 --save_dir /nas/softechict-nas-1/sallegretti/SUBMISSIONMODELS/ --SRV --optimizer SGD --scheduler plateau --cutout_holes 0 1 2 3 --cutout_pad 20 50 100  --augm_config 84 --dataset isic2020 --learning_rate 0.01
fi

if [ "$SLURM_ARRAY_TASK_ID" -eq "107" ]; then
srun python -u /homes/sallegretti/ISIC_2020/classification_net.py --network seresnext101 --epochs 120 --batch_size 8 --save_dir /nas/softechict-nas-1/sallegretti/SUBMISSIONMODELS/ --SRV --optimizer SGD --scheduler plateau --cutout_holes 0 0 1 2 3 --cutout_pad 20 50 100  --augm_config 116 --dataset isic2020 --learning_rate 0.0001 --load_epoch 33
fi

if [ "$SLURM_ARRAY_TASK_ID" -eq "203" ]; then
srun python -u /homes/sallegretti/ISIC_2020/classification_net.py --network densenet201 --epochs 120 --batch_size 8 --save_dir /nas/softechict-nas-1/sallegretti/SUBMISSIONMODELS/ --SRV --optimizer SGD --scheduler plateau --cutout_holes 0 0 1 2 3 --cutout_pad 20 50 100  --augm_config 116 --dataset isic2020 --learning_rate 0.00001 --load_epoch 51
fi

if [ "$SLURM_ARRAY_TASK_ID" -eq "206" ]; then
srun python -u /homes/sallegretti/ISIC_2020/classification_net.py --network seresnext101 --epochs 120 --batch_size 8 --save_dir /nas/softechict-nas-1/sallegretti/SUBMISSIONMODELS/ --SRV --optimizer SGD --scheduler plateau --cutout_holes 0 1 2 3 --cutout_pad 20 50 100  --augm_config 84 --dataset isic2020 --learning_rate 0.001 --load_epoch 49
fi

if [ "$SLURM_ARRAY_TASK_ID" -eq "501" ]; then
srun python -u /homes/sallegretti/ISIC_2020/classification_net.py --network densenet201 --epochs 93 --batch_size 8 --save_dir /nas/softechict-nas-1/sallegretti/SUBMISSIONMODELS/ --SRV --optimizer SGD --scheduler plateau --augm_config 16 --dataset isic2019_wholeset --learning_rate 0.0000001 --load_epoch 82
fi

if [ "$SLURM_ARRAY_TASK_ID" -eq "502" ]; then
srun python -u /homes/sallegretti/ISIC_2020/classification_net.py --network densenet201 --epochs 105 --batch_size 8 --save_dir /nas/softechict-nas-1/sallegretti/SUBMISSIONMODELS/ --SRV --optimizer SGD --scheduler plateau --cutout_holes 1 2 3 --cutout_pad 20 50 100  --augm_config 84 --dataset isic2019_wholeset --learning_rate 0.0000001 --load_epoch 94
fi

if [ "$SLURM_ARRAY_TASK_ID" -eq "503" ]; then
srun python -u /homes/sallegretti/ISIC_2020/classification_net.py --network densenet201 --epochs 94 --batch_size 8 --save_dir /nas/softechict-nas-1/sallegretti/SUBMISSIONMODELS/ --SRV --optimizer SGD --scheduler plateau --cutout_holes 0 0 1 2 3 --cutout_pad 20 50 100  --augm_config 116 --dataset isic2019_wholeset --learning_rate 0.000001 --load_epoch 83
fi

if [ "$SLURM_ARRAY_TASK_ID" -eq "504" ]; then
srun python -u /homes/sallegretti/ISIC_2020/classification_net.py --network densenet201 --epochs 130 --batch_size 8 --save_dir /nas/softechict-nas-1/sallegretti/SUBMISSIONMODELS/ --SRV --optimizer SGD --scheduler plateau --augm_config 84 --dataset isic2019_wholeset --learning_rate 0.0000001 --load_epoch 119
fi

if [ "$SLURM_ARRAY_TASK_ID" -eq "505" ]; then
srun python -u /homes/sallegretti/ISIC_2020/classification_net.py --network resnet152 --epochs 97 --batch_size 8 --save_dir /nas/softechict-nas-1/sallegretti/SUBMISSIONMODELS/ --SRV --optimizer SGD --scheduler plateau --augm_config 16 --dataset isic2019_wholeset --learning_rate 0.00000001 --load_epoch 86
fi

if [ "$SLURM_ARRAY_TASK_ID" -eq "506" ]; then
srun python -u /homes/sallegretti/ISIC_2020/classification_net.py --network seresnext101 --epochs 118 --batch_size 8 --save_dir /nas/softechict-nas-1/sallegretti/SUBMISSIONMODELS/ --SRV --optimizer SGD --scheduler plateau --cutout_holes 0 1 2 3 --cutout_pad 20 50 100  --augm_config 84 --dataset isic2019_wholeset --learning_rate 0.000001 --load_epoch 107
fi

if [ "$SLURM_ARRAY_TASK_ID" -eq "507" ]; then
srun python -u /homes/sallegretti/ISIC_2020/classification_net.py --network seresnext101 --epochs 103 --batch_size 8 --save_dir /nas/softechict-nas-1/sallegretti/SUBMISSIONMODELS/ --SRV --optimizer SGD --scheduler plateau --cutout_holes 0 0 1 2 3 --cutout_pad 20 50 100  --augm_config 116 --dataset isic2019_wholeset --learning_rate 0.00001 --load_epoch 92
fi
