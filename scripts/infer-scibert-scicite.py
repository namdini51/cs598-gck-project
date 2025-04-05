import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.nn import DataParallel
import os
import sys
import time

chunk_index = int(sys.argv[1])

BATCH_SIZE = 384
MAX_LENGTH = 128
CHUNK_SIZE = 10000000  # TO FINISH WITHIN TIME WINDOW: set chunk size

# load dataset & drop NaNs
df = pd.read_csv('/scratch/donginn2/data/opcitance/opcitance_v2/cleaned_dataset/clean_merged_IntxtCit.tsv', sep='\t', dtype=str, low_memory=False)
df = df.dropna(subset=['citation']).reset_index(drop=True)
print("Data loaded!", flush=True)

# load fine-tuned SciBERT model and tokenizer
model_name = "./models/scibert_scicite_finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, use_safetensors=True)

# # data parallel
# model = DataParallel(model, device_ids=[0, 1, 2])
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# single gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Get unique citation sentences to reduce redundant computation
unique_citations = df[['citation']].drop_duplicates().reset_index(drop=True)
total = len(unique_citations)
print(f"Total unique citation sentences: {total}", flush=True)

# calculate chunk range
start = chunk_index * CHUNK_SIZE
end = min(start + CHUNK_SIZE, total)
chunk = unique_citations.iloc[start:end].copy()
print(f"Processing chunk {chunk_index} ({start} to {end})", flush=True)

# Perform inference on unique citations
def classify_citation(sentences):
    inputs = tokenizer(sentences, padding=True, truncation=True, 
                       max_length=MAX_LENGTH, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
    return predictions

# loop over the defined chunk
label_map = {0: 'Background', 1: 'Method', 2: 'Result Comparison'}
all_preds = []

start_time = time.time()

for i in range(0, len(chunk), BATCH_SIZE):
    batch_start_time = time.time()

    batch_sentences = chunk['citation'].iloc[i:i+BATCH_SIZE].tolist()
    batch_preds = classify_citation(batch_sentences)
    all_preds.extend(batch_preds)

    # Print progress and ETA every 100 batches
    if i % (100 * BATCH_SIZE) == 0 or i == 0:
        elapsed = time.time() - start_time
        processed = i + BATCH_SIZE
        total_batches = len(chunk)
        est_total_time = (elapsed / processed) * total_batches
        eta = est_total_time - elapsed
        print(f"[{i}/{total_batches}] Processed: {processed:,} | Elapsed: {elapsed/60:.2f} min | ETA: {eta/60:.2f} min", flush=True)

chunk['citation_intent_label'] = [label_map[pred] for pred in all_preds]

# save
output_dir = '/scratch/donginn2/data/opcitance/opcitance_v2/cleaned_dataset/chunks'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, f'intents_chunk_{start}_{end}.tsv')
chunk.to_csv(output_path, sep='\t', index=False)

print("All chunks processed")

# # Batch inference
# all_preds = []

# for i in range(0, len(unique_citations), BATCH_SIZE):
#     batch_sentences = unique_citations['citation'][i:i+BATCH_SIZE].tolist()
#     batch_preds = classify_citation(batch_sentences)
#     all_preds.extend(batch_preds)

# # add predicted labels back to unique citations dataframe
# label_map = {0: 'Background', 1: 'Method', 2: 'Result Comparison'}
# unique_citations['citation_intent_label'] = [label_map[pred] for pred in all_preds]
# # unique_citations['citation_intent_label'] = all_preds

# df_final = df.merge(unique_citations, on='citation', how='left') # merging back to original version

# # save final dataframe
# df_final.to_csv('./data/clean_merged_IntxtCit_with_intents.tsv', sep='\t', index=False)
