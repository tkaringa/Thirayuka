import os
import shutil
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

def recover():
    checkpoint_path = "results/checkpoint-352"
    final_path = "models/bert_classifier"
    
    print(f"Recovering model from {checkpoint_path}...")
    
    if not os.path.exists(checkpoint_path):
        print("Error: Checkpoint not found!")
        return

    print("Loading model weights...")
    model = DistilBertForSequenceClassification.from_pretrained(checkpoint_path)
    
    print("Loading tokenizer...")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
    
    print(f"Saving to {final_path}...")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    
    print("Success! Model recovered.")

if __name__ == "__main__":
    recover()
