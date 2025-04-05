import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)
from datasets import load_dataset
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, classification_report

# ========= Hyperparameter & Configuration Section =========
BATCH_SIZE = 32 # also try 64
LEARNING_RATE = 2e-5
EPOCHS = 5
MAX_SEQ_LENGTH = 128
WARMUP_RATIO = 0.1
MODEL_NAME = "allenai/scibert_scivocab_uncased"
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ==========================================================

torch.manual_seed(SEED)

# Load SciCite dataset
dataset = load_dataset("scicite", trust_remote_code=True)

# load SciBERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3).to(DEVICE)

# Tokenization function
def tokenize_batch(batch):
    return tokenizer(batch["string"], padding='max_length', truncation=True, max_length=MAX_SEQ_LENGTH)

tokenized_dataset = dataset.map(tokenize_batch, batched=True)
tokenized_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

# DataLoaders
train_loader = DataLoader(tokenized_dataset["train"], batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(tokenized_dataset["validation"], batch_size=BATCH_SIZE)
test_loader = DataLoader(tokenized_dataset["test"], batch_size=BATCH_SIZE)

# SETUP
total_steps = EPOCHS * len(train_loader)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(WARMUP_RATIO * total_steps),
    num_training_steps=total_steps
)

# TRAINING
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    print(f"\nEpoch {epoch + 1}/{EPOCHS}")

    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        inputs = {k: v.to(DEVICE) for k, v in batch.items() if k in ['input_ids', 'attention_mask']}
        labels = batch['label'].to(DEVICE)
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Average Training Loss: {avg_loss:.4f}")

    # Validation
    model.eval()
    val_preds, val_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            inputs = {k: v.to(DEVICE) for k, v in batch.items() if k in ['input_ids', 'attention_mask']}
            labels = batch['label'].numpy()
            logits = model(**inputs).logits
            predictions = logits.argmax(dim=-1).cpu().numpy()
            val_preds.extend(predictions)
            val_labels.extend(labels)

    val_acc = accuracy_score(val_labels, val_preds)
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(classification_report(val_labels, val_preds, target_names=['Background', 'Method', 'Result']))

# test evaluation
model.eval()
test_preds, test_labels = [], []
with torch.no_grad():
    for batch in tqdm(test_loader):
        inputs = {k: v.to(DEVICE) for k, v in batch.items() if k in ['input_ids', 'attention_mask']}
        labels = batch['label'].numpy()
        logits = model(**inputs).logits
        predictions = logits.argmax(dim=-1).cpu().numpy()
        test_preds.extend(predictions)
        test_labels.extend(labels)

test_acc = accuracy_score(test_labels, test_preds)
print("\n--- Final Test Results ---")
print(f"Test Accuracy: {test_acc:.4f}")
print(classification_report(test_labels, test_preds, target_names=['Background', 'Method', 'Result']))

# Save fine-tuned model
model.save_pretrained("./models/scibert_scicite_finetuned")
tokenizer.save_pretrained("./models/scibert_scicite_finetuned")
