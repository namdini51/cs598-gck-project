#!/bin/bash

#SBATCH --time=12:00:00                    	# Job run time (hh:mm:ss)
#SBATCH --mail-user="donginn2@illinois.edu"   	# Email address to alert when job starts/finishes
#SBATCH --nodes=1                          	# Number of nodes
#SBATCH --gres=gpu:0                       	# Number of GPUs, use --gres=gpu:A10 to specify GPU model or --gres=gpu:A10:2 to specify both model and number of GPUs
#SBATCH --ntasks-per-node=32                	# Number of cores per node
#SBATCH --job-name=overlap             # Name of josqb
#SBATCH --account=25sp-cs598gck-eng          	# Account
#SBATCH --partition=eng-instruction       	# Parititon
#SBATCH --output=adn_cluster_overlap_log%j          # output file name
# --mem=128G				# Request memory

# check memory status
#free -h

# activate conda env
source ~/.bashrc
conda activate cs598

# set system arguments and dataset directory
python get-cluster-overlap.py
