#!/bin/env bash

#SBATCH -J RFI_CNN
#SBATCH --partition=dpt-gpu
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:pascal:1
#SBATCH --mem=20000

#module load CUDA
module load cuDNN/6.0-CUDA-8.0.61

# see here for more samples:
# /opt/cudasample/NVIDIA_CUDA-8.0_Samples/bin/x86_64/linux/release/

# if you need to know the allocated CUDA device, you can obtain it here:
echo $CUDA_VISIBLE_DEVICES

srun python $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11}

