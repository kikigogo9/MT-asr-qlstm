#!/bin/sh

#SBATCH --job-name=gridsearch
#SBATCH --partition=memory
#SBATCH --account=Education-AS-MSc-QIST
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=20G

module load 2024r1
module load python
module load py-pip

pip install -r req.txt

QUBITS=$1
LR=$2
METHOD=$3
LAYERS=$4


python gridsearch.py $QUBITS $LR $METHOD $LAYERS