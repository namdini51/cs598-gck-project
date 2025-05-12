#!/bin/bash

#SBATCH --time=12:00:00                    	# Job run time (hh:mm:ss)
#SBATCH --mail-user="donginn2@illinois.edu"   	# Email address to alert when job starts/finishes
#SBATCH --nodes=1                          	# Number of nodes
#SBATCH --gres=gpu:0                       	# Number of GPUs, use --gres=gpu:A10 to specify GPU model or --gres=gpu:A10:2 to specify both model and number of GPUs
#SBATCH --ntasks-per-node=32                	# Number of cores per node
#SBATCH --job-name=emb-similarity             # Name of job
#SBATCH --account=25sp-cs598gck-eng          	# Account
#SBATCH --partition=eng-instruction       	# Parititon
#SBATCH --output=adn_sim_res_log%j          # output file name
# --mem=256G				# Request memory

# check memory status
#free -h

# activate conda env
source ~/.bashrc
conda activate cs598

#python get-emb-similarity.py
#python get-mesh-similarity.py


# # === EXECUTION ===
# echo "=================== Program Start ==================="

# # Run baseline
# echo "Running baseline MedCPT cohesion..."
# python get-emb-similarity.py

# # Run 312 weight
# echo "Running weight 312 MedCPT cohesion..."
# python get-emb-similarity.py --weight_tag "_312"

# # Run 213 weight
# echo "Running weight 213 MedCPT cohesion..."
# python get-emb-similarity.py --weight_tag "_213"

# echo "=================== Program End ==================="

python get-emb-similarity.py
