# interactive search tool

import pickle
import json
from retrieval import BM25, search
from preprocess import clean_malayalam_text, tokenize_malayalam

def load_system():
    print("loading system...")
    try:
        with open('models/bm25_index.pkl', 'rb') as f:
            bm25 = pickle.load(f)
        with open('data/processed_corpus.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        documents = [item['text'] for item in data]
        return bm25, documents
    except Exception as e:
        print(f"error loading system: {e}")
        return None, None

def main():
    bm25, documents = load_system()
    if not bm25:
        return

    print("\n--- Malayalam Search Engine ---")
    print("Type 'exit' to quit")

    while True:
        query = input("\nEnter query: ").strip()
        if query.lower() == 'exit':
            break
        
        if not query:
            continue

        results = search(query, bm25, documents, top_k=5)
        
        if not results:
            print("No results found.")
        
        for i, res in enumerate(results):
            print(f"\n{i+1}. Doc {res['doc_id']} (Score: {res['score']:.3f})")
            print(f"   {res['text'][:200]}...")

if __name__ == '__main__':
    main()