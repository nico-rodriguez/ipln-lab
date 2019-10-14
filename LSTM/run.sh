#!/bin/bash

#SBATCH --job-name=red_neuronal_1
#SBATCH --ntasks=4
#SBATCH --mem=16384
#SBATCH --time=48:00:00
#SBATCH --tmp=16G
#SBATCH --partition=normal
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ignacioferre@gmail.com

source ../../../python_env/bin/activate

cd ipln/ipln-lab/LSTM
python3 LSTM.py