# Simple search using BM25

import json
import pickle
import math
import numpy as np
from collections import Counter
from preprocess import clean_malayalam_text, tokenize_malayalam

class BM25:
    def __init__(self, documents, k1=1.5, b=0.75):
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.doc_len = [len(d.split()) for d in documents]
        self.avg_len = sum(self.doc_len) / len(documents)
        self.doc_freqs = []
        self.idf = {}
        self.build_index()

    def build_index(self):
        print("building bm25 index...")
        df = Counter()
        for doc in self.documents:
            words = set(doc.split())
            df.update(words)
        
        N = len(self.documents)
        for word, freq in df.items():
            idf = math.log((N - freq + 0.5) / (freq + 0.5) + 1)
            self.idf[word] = idf

    def score(self, query):
        scores = np.zeros(len(self.documents))
        q_words = query.split()
        
        for i, doc in enumerate(self.documents):
            score = 0
            doc_words = doc.split()
            doc_counts = Counter(doc_words)
            
            for q in q_words:
                if q not in doc_counts:
                    continue
                
                f = doc_counts[q]
                idf = self.idf.get(q, 0)
                
                num = f * (self.k1 + 1)
                den = f + self.k1 * (1 - self.b + self.b * (self.doc_len[i] / self.avg_len))
                
                score += idf * (num / den)
            scores[i] = score
        return scores

def load_corpus(filename):
    # Load text documents
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Use processed text
    documents = [item['text'] for item in data]
    return documents

def search(query, bm25, documents, top_k=5):
    # Clean and prepare query
    cleaned = clean_malayalam_text(query)
    tokens = tokenize_malayalam(cleaned)
    processed_query = ' '.join(tokens)

    # Calculate document scores
    scores = bm25.score(processed_query)
    
    # Get top results
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append({
                'doc_id': int(idx),
                'score': float(scores[idx]),
                'text': documents[idx][:200]
            })
    
    return results

def calculate_map(queries, relevance_judgments, bm25, documents):
    # Calculate MAP score
    average_precisions = []
    
    for query in queries:
        # Get search results
        results = search(query, bm25, documents, top_k=10)
        result_ids = [r['doc_id'] for r in results]
        
        # Get relevant documents
        relevant_docs = relevance_judgments.get(query, [])
        
        if not relevant_docs:
            continue
        
        # calc avg precision
        num_relevant = 0
        precision_sum = 0.0
        
        for i, doc_id in enumerate(result_ids):
            if doc_id in relevant_docs:
                num_relevant += 1
                precision_at_k = num_relevant / (i + 1)
                precision_sum += precision_at_k
        
        if num_relevant > 0:
            avg_precision = precision_sum / len(relevant_docs)
            average_precisions.append(avg_precision)
    
    # return map
    if average_precisions:
        map_score = sum(average_precisions) / len(average_precisions)
        return map_score
    else:
        return 0.0

def main():
    # load corpus
    documents = load_corpus('data/processed_corpus.json')
    
    if not documents:
        print("no documents")
        return
    
    # build index
    bm25 = BM25(documents)
    
    # save the index
    print("saving index...")
    with open('models/bm25_index.pkl', 'wb') as f:
        pickle.dump(bm25, f)
    
    print("testing search...")
    
    # test query
    query = "വാർത്ത" # news
    print(f"\nsearch results for query: {query}\n")
    
    results = search(query, bm25, documents)
    
    for i, res in enumerate(results):
        print(f"{i+1}. score: {res['score']:.3f}")
        print(f"   text: {res['text']}...\n")
    
    print("done!")

if __name__ == '__main__':
    main()
