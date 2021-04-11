#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --mem=16000
#SBATCH --time=24:00:00
#SBATCH --job-name="SPR"

source activate ssl

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