#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --mail-user="donginn2@illinois.edu"
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --ntasks-per-node=32
#SBATCH --job-name=cluster-var
#SBATCH --account=25sp-cs598gck-eng
#SBATCH --partition=eng-instruction
#SBATCH --output=adn_leiden_variance_log%j
#SBATCH --mem=128G

# activate conda env
source ~/.bashrc
conda activate cs598

EDGELIST_PATH="/scratch/donginn2/data/opcitance/opcitance_v2/cleaned_dataset/pubmed_edgelist_final.tsv"
QUALITY_FUNCTION="cpm_0.01"

echo "============================== Program Start =============================="

for run_id in {1..5}
do
    echo "Starting Run ${run_id}..."
    python run-clustering-iterative.py "$EDGELIST_PATH" "$QUALITY_FUNCTION" "$run_id"
done

echo "=============================== Program End ==============================="


# python run-clustering-iterative.py "$EDGELIST_PATH" "$QUALITY_FUNCTION" 5
