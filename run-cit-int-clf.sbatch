#!/bin/bash

#SBATCH --time=12:00:00                    	# Job run time (hh:mm:ss)
#SBATCH --mail-user="donginn2@illinois.edu"   	# Email address to alert when job starts/finishes
#SBATCH --nodes=1                          	# Number of nodes
#SBATCH --gres=gpu:A10                       	# Number of GPUs, use --gres=gpu:A10 to specify GPU model or --gres=gpu:A10:2 to specify both model and number of GPUs
#SBATCH --cpus-per-task=16
#SBATCH --job-name=adn-cs598-clf             # Name of job
#SBATCH --account=25sp-cs598gck-eng          	# Account
#SBATCH --partition=eng-instruction       	# Parititon
#SBATCH --output=adn_cs598_clf_log%j          # output file name
#SBATCH --mem=128G				# Request memory

# GPU monitoring
nvidia-smi --query-gpu=timestamp,name,index,utilization.gpu,memory.used,memory.total --format=csv -l 60 > gpu_usage_log_${SLURM_JOB_ID}.csv &
MONITOR_PID=$!

# load python & cuda  module and activate venv
module load python/3.11.11
module load cuda/12.6
source /scratch/donginn2/IMRaD-project/temp-env/bin/activate

# run script
#python ft-scibert-scicite.py  # fine tuning script
# python infer-scibert-scicite.py 0  # inference script (MAKE SURE TO INCLUDE CHUNK INDEX)

#CUDA_VISIBLE_DEVICES=0 python infer-scibert-scicite.py 4 > log_chunk4.txt 2>&1 & 
#CUDA_VISIBLE_DEVICES=1 python infer-scibert-scicite.py 5 > log_chunk5.txt 2>&1 &
#CUDA_VISIBLE_DEVICES=2 python infer-scibert-scicite.py 6 > log_chunk6.txt 2>&1 &

python infer-scibert-scicite.py 4 > log_chunk4.txt 2>&1 & 
wait

# Stop monitoring after job ends
kill $MONITOR_PID
