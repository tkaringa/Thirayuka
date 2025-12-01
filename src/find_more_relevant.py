# find extra relevant docs

import pickle
import json
from retrieval import search

# load system
with open('models/bm25_index.pkl', 'rb') as f:
    bm25 = pickle.load(f)
with open('data/processed_corpus.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

documents = [item['text'] for item in data]

# check ranks 11-30
print("checking ranks 11-30...")
results = search('വാർത്ത', bm25, documents, top_k=30)
for r in results[10:30]:
    print(f"\nDoc {r['doc_id']} (score: {r['score']:.3f}):")
    print(f"  {r['text'][:150]}")
