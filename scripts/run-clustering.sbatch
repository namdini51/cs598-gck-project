#!/bin/bash

#SBATCH --time=12:00:00                    	# Job run time (hh:mm:ss)
#SBATCH --mail-user="donginn2@illinois.edu"   	# Email address to alert when job starts/finishes
#SBATCH --nodes=2                          	# Number of nodes
#SBATCH --gres=gpu:0                       	# Number of GPUs, use --gres=gpu:A10 to specify GPU model or --gres=gpu:A10:2 to specify both model and number of GPUs
#SBATCH --ntasks-per-node=16                	# Number of cores per node
#SBATCH --job-name=adn-cs598-assig2             # Name of job
#SBATCH --account=25sp-cs598gck-eng          	# Account
#SBATCH --partition=eng-instruction       	# Parititon
#SBATCH --output=adn_clustering_res_log%j          # output file name
#SBATCH --mem=256G				# Request memory

# check memory status
#free -h

# activate conda env
source ~/.bashrc
conda activate cs598

# set system arguments and dataset directory
EDGELIST_PATH="/scratch/donginn2/data/opcitance/opcitance_v2/cleaned_dataset/pubmed_edgelist.tsv"
QUALITY_FUNCTIONS=("cpm_0.01" "cpm_0.001")

# run my script over all datasets and quality functions
echo "==============================Program Start=============================="
for QUALITY_FUNCTION in "${QUALITY_FUNCTIONS[@]}"; do
    python run-clustering.py "$EDGELIST_PATH" "$QUALITY_FUNCTION"
done
echo "===============================Program End==============================="
