# -*- coding: utf-8 -*-
from __future__ import print_function, division, unicode_literals
import csv
import json
import numpy as np
import gc
from sklearn.metrics import classification_report

from torchmoji.sentence_tokenizer import SentenceTokenizer
from torchmoji.model_def import torchmoji_emojis
from torchmoji.global_variables import PRETRAINED_PATH, VOCAB_PATH

INPUT_CSV = 'dataset.csv'      # Must contain 'text' and 'label' columns
OUTPUT_PATH = 'predictions.csv'

# -------------------------------
# Read 'text' and 'label' from CSV
# -------------------------------
texts = []
true_labels = []
with open(INPUT_CSV, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader)
    text_idx = header.index('text')
    label_idx = header.index('label')
    for row in reader:
        texts.append(row[text_idx])
        true_labels.append(row[label_idx])

# -------------------------------
# Top elements function
# -------------------------------
def top_elements(array, k):
    ind_sorted = np.argsort(array)[::-1]
    return ind_sorted[:k]

# -------------------------------
# Tokenization and model loading
# -------------------------------
maxlen = 30
with open(VOCAB_PATH, 'r', encoding='utf-8') as f:
    vocabulary = json.load(f)

st = SentenceTokenizer(vocabulary, maxlen)
model = torchmoji_emojis(PRETRAINED_PATH)

# -------------------------------
# Predict and write simplified CSV
# -------------------------------
pred_labels = []

with open(OUTPUT_PATH, 'w', encoding='utf-8', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Text', 'TrueLabel', 'PredLabel'])

    for i, sentence in enumerate(texts):
        print('Processing sentence {}: {}'.format(i, sentence))

        # Tokenize single sentence
        tokenized, _, _ = st.tokenize_sentences([sentence])

        # Run prediction
        prob = model(tokenized)[0]  # batch size 1

        # Top emoji index as predicted label
        pred_label = str(top_elements(prob, 1)[0])
        pred_labels.append(pred_label)

        # Write to CSV
        writer.writerow([sentence, true_labels[i], pred_label])

        # Release memory
        del tokenized
        del prob
        gc.collect()

# -------------------------------
# Print classification report
# -------------------------------
print('\nClassification Report:')
print(classification_report(true_labels, pred_labels))
