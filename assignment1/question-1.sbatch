#!/bin/bash
#Sample Batch Job Script
#
#SBATCH --time=12:00:00                    	# Job run time (hh:mm:ss)
#SBATCH --mail-user="donginn2@illinois.edu"   	# Email address to alert when job starts/finishes
#SBATCH --nodes=2                          	# Number of nodes
#SBATCH --gres=gpu:0                       	# Number of GPUs, use --gres=gpu:A10 to specify GPU model or --gres=gpu:A10:2 to specify both model and number of GPUs
#SBATCH --ntasks-per-node=32                	# Number of cores per node
#SBATCH --job-name=adn-cs598-test              	# Name of job
#SBATCH --account=25sp-cs598gck-eng          	# Account
#SBATCH --partition=eng-instruction       	# Parititon
#SBATCH --output=adn_assignment1_log%j          # output file name
#SBATCH --mem=512G				# Request memory

#check GPU and memory of node
#nvidia-smi
free -h

#run my script
python question-1.py
