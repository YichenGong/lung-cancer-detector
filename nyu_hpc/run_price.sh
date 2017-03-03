#!/bin/bash
#
#SBATCH --nodes=2
##SBATCH --ntasks=1
##SBATCH --cpus-per-task=2

#SBATCH --time=5:00:00

#SBATCH --mem=16GB

#SBATCH --job-name=DSB17

#SBATCH --mail-type=END
##SBATCH --mail-user=chirag.m@nyu.edu

#SBATCH --output=DSB_%j.out

module purge
module load scikit-learn/intel/0.18.1
module load tensorflow/python2.7/20170218
module list

cd $SCRATCH
cd lung-cancer-detector

python run.py
