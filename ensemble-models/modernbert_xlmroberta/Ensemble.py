import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import os
from sklearn.model_selection import train_test_split

# ============================================================
# --- CONFIG ---
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_DIR_1 = "modern_bert_finetuned"     # path to your first fine-tuned model
MODEL_DIR_2 = "xlm_roberta_finetuned"     # path to your second fine-tuned model
TEXT_COL = "text"
LABEL_COL = "label"
MAX_LENGTH = 512
BATCH_SIZE = 16
OUTPUT_REPORT_DIR = "ensemble_reports"
SEED = 42

os.makedirs(OUTPUT_REPORT_DIR, exist_ok=True)

# ============================================================
# --- LOAD AND SPLIT DATA ---
# ============================================================
DATA_CSV = "/scratch/project_2011211/Fahim/a_nlp/dataset.csv"
df = pd.read_csv(DATA_CSV).dropna(subset=[TEXT_COL, LABEL_COL])

# 60/20/20 stratified split
train_df, temp_df = train_test_split(df, test_size=0.4, random_state=SEED, stratify=df[LABEL_COL])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=SEED, stratify=temp_df[LABEL_COL])

texts = test_df[TEXT_COL].tolist()

# ============================================================
# --- LOAD MODELS AND TOKENIZERS ---
# ============================================================
tokenizer1 = AutoTokenizer.from_pretrained(MODEL_DIR_1)
tokenizer2 = AutoTokenizer.from_pretrained(MODEL_DIR_2)
model1 = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR_1).to(device)
model2 = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR_2).to(device)

# Read label mapping
with open(f"{MODEL_DIR_1}/label_mapping.txt") as f:
    labels = [line.strip().split("\t")[1] for line in f.readlines()]
label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
true_labels = [label_to_idx[lbl] for lbl in test_df[LABEL_COL]]

# ============================================================
# --- FUNCTION TO GET LOGITS ---
# ============================================================
def get_logits(model, tokenizer, texts):
    model.eval()
    all_logits = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), BATCH_SIZE), desc=f"Running {model.name_or_path}"):
            batch_texts = texts[i:i + BATCH_SIZE]
            inputs = tokenizer(batch_texts, truncation=True, padding=True,
                               max_length=MAX_LENGTH, return_tensors="pt").to(device)
            outputs = model(**inputs)
            all_logits.append(outputs.logits.cpu())
    return torch.cat(all_logits, dim=0)

# ============================================================
# --- GET LOGITS FROM BOTH MODELS ---
# ============================================================
print("\n=== Getting logits from both models ===")
logits1 = get_logits(model1, tokenizer1, texts)
logits2 = get_logits(model2, tokenizer2, texts)

probs1 = torch.softmax(logits1, dim=-1).numpy()
probs2 = torch.softmax(logits2, dim=-1).numpy()

# ============================================================
# --- EVALUATION FUNCTION ---
# ============================================================
def evaluate_and_report(name, preds, true_labels):
    acc = accuracy_score(true_labels, preds)
    report = classification_report(true_labels, preds, target_names=labels, zero_division=0)
    print(f"\n=== [{name}] ===")
    print(f"Accuracy: {acc:.4f}")
    print(report)

    # Sanitize filename
    safe_name = name.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "").lower()
    report_path = os.path.join(OUTPUT_REPORT_DIR, f"{safe_name}_report.txt")

    # Save report
    with open(report_path, "w") as f:
        f.write(f"=== {name} ===\n")
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write(report)
    print(f"Saved report to {report_path}")


# ============================================================
# --- 1. Soft Voting ---
# ============================================================
avg_logits = (logits1 + logits2) / 2
preds_soft = torch.argmax(avg_logits, dim=-1).numpy()
evaluate_and_report("Soft Voting (Equal Averaging)", preds_soft, true_labels)

# ============================================================
# --- 2. Weighted Soft Voting ---
# ============================================================
weight1, weight2 = 0.6, 0.4
avg_logits_w = weight1 * logits1 + weight2 * logits2
preds_weighted = torch.argmax(avg_logits_w, dim=-1).numpy()
evaluate_and_report("Weighted Soft Voting (0.6/0.4)", preds_weighted, true_labels)

# ============================================================
# --- 3. Hard Voting ---
# ============================================================
preds1 = torch.argmax(logits1, dim=-1).numpy()
preds2 = torch.argmax(logits2, dim=-1).numpy()
preds_hard = []
for i in range(len(preds1)):
    votes = [preds1[i], preds2[i]]
    preds_hard.append(max(set(votes), key=votes.count))
preds_hard = np.array(preds_hard)
evaluate_and_report("Hard Voting (Majority Vote)", preds_hard, true_labels)

# ============================================================
# --- 4. Stacking (Logistic Regression Meta-Learner) ---
# ============================================================
X_stack = np.hstack([probs1, probs2])
y_stack = np.array(true_labels)

split_idx = int(0.8 * len(X_stack))
X_train, X_test_meta = X_stack[:split_idx], X_stack[split_idx:]
y_train, y_test_meta = y_stack[:split_idx], y_stack[split_idx:]

meta_model = LogisticRegression(max_iter=2000, random_state=SEED)
meta_model.fit(X_train, y_train)
meta_preds = meta_model.predict(X_stack)
evaluate_and_report("Stacking (Logistic Regression)", meta_preds, true_labels)

print("\n✅ Ensemble evaluation completed. Reports saved in:", OUTPUT_REPORT_DIR)
