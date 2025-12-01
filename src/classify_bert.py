# bert classifier comparison
# uses distilbert for multilingual support

import json
import torch
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments

class MalayalamDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def load_data(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # use original text for bert if available
    texts = [item.get('original_text', item['text']) for item in data]
    
    # create labels based on content (Politics detection)
    labels = []
    for item in data:
        if "രാഷ്ട്രീയം" in item.get('original_text', '') or "രാഷ്ട്രീയ" in item.get('original_text', ''):
            labels.append(1)
        else:
            labels.append(0)
            
    print(f"Positive samples (Politics): {sum(labels)}")
    
    # Oversample positive class to balance data
    pos_indices = [i for i, x in enumerate(labels) if x == 1]
    neg_indices = [i for i, x in enumerate(labels) if x == 0]
    
    if pos_indices:
        target_count = len(neg_indices)
        current_pos_count = len(pos_indices)
        
        print(f"Oversampling positives from {current_pos_count} to {target_count}...")
        
        # Duplicate positives
        multipler = target_count // current_pos_count
        remainder = target_count % current_pos_count
        
        new_pos_indices = pos_indices * multipler + pos_indices[:remainder]
        
        all_indices = new_pos_indices + neg_indices
        random.shuffle(all_indices)
        
        texts = [texts[i] for i in all_indices]
        labels = [labels[i] for i in all_indices]
        
    print(f"Final dataset size: {len(texts)} (Pos: {sum(labels)})")
    
    return texts, labels

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

def main():
    print("loading data...")
    texts, labels = load_data('data/processed_corpus.json')
    
    # split data
    train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2)
    
    print("loading model...")
    model_name = 'distilbert-base-multilingual-cased'
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    print("tokenizing...")
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128)
    
    train_dataset = MalayalamDataset(train_encodings, train_labels)
    test_dataset = MalayalamDataset(test_encodings, test_labels)
    
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        logging_dir='./logs',
        logging_steps=10,
        eval_strategy="epoch",
        use_cpu=True # force cpu to avoid cuda errors if not set up
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )
    
    print("training bert (this may take a while)...")
    trainer.train()
    
    print("evaluating...")
    results = trainer.evaluate()
    
    print("\n--- bert results ---")
    print(f"precision: {results['eval_precision']:.3f}")
    print(f"recall: {results['eval_recall']:.3f}")
    print(f"f1: {results['eval_f1']:.3f}")
    
    print("saving model...")
    model.save_pretrained('models/bert_classifier')
    tokenizer.save_pretrained('models/bert_classifier')
    
    print("done")

if __name__ == '__main__':
    main()