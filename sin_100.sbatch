#!/bin/bash

#SBATCH --partition=mi50
#SBATCH --gres=gpu:mi50:1
#SBATCH --job-name="SPR"
#SBATCH --mem=64G
#SBATCH -c 4
#SBATCH --time=100:00:00
#SBATCH --output=/scratch/wz1232/og_spr/logs/%j.out
#SBATCH --open-mode=append
#SBATCH --export=ALL

singularity exec --nv --overlay /scratch/wz1232/overlay/SPR_amd.ext3:rw /scratch/wz1232/overlay/base.sif /bin/bash -c "

source /ext3/env.sh
conda activate

cd /scratch/wz1232/og_spr
$1
$2
$3
$4
$5
$6
$7
$8
$9

shift

$9
"