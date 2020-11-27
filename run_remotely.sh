#!/bin/bash
#SBATCH --job-name=deepLabv3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB
#SBATCH --gres=gpu:aquila
#SBATCH --time=50:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=zz1763@nyu.edu
#SBATCH --output=deeplab_%j.out
#SBATCH --error=deeplab_%j.err
#SBATCH --gres=gpu:3 # How much gpu need, n is the number
#SBATCH -p aquila

module purge
module load anaconda3 cuda/9.0 cudnn/7.0 
eval "$(conda shell.bash hook)"
conda deactivate
conda activate tfv2
echo "deepLabv2_train">>log
echo "python train.py --batch_size 16">>log
python train.py >log 2>& 1
echo "FINISH"

