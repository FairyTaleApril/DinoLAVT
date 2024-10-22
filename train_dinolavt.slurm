#!/bin/bash
#SBATCH --job-name=train_dinolavt
# Job name
#SBATCH --mail-type=END,FAIL
# Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=u7619568@anu.edu.au
# Where to send mail
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G # Job memory request
#SBATCH --time=48:00:00 # Time limit hrs:min:sec
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2080ti:1 # Request 1 GPU
#SBATCH --output=train_dinolavt_%j.log
# Standard output and error log

pwd; hostname; date
echo "--- training ---"

export SINGULARITYENV_CONDAENV="/home/remote/u7619568/.conda/envs/dinolavt/"
singularity exec --nv --bind /home/projects /opt/apps/containers/conda/conda-nvidia-22.04-latest.sif bash <<EOF
 source /.singularity_bash
 python train.py --img_dir 'data/train2014' --img_embed_model 'patch_embed' --save_freq 40 --eval --ckpt 'output/ckpt/lavt_base_40.pth'
EOF

echo "--- done ---"
date
