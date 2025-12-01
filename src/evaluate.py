# evaluation metrics

import numpy as np
import pickle
import json
from retrieval import BM25, search

def precision_at_k(retrieved, relevant, k):
    # precision at k
    retrieved_at_k = retrieved[:k]
    relevant_count = sum(1 for doc in retrieved_at_k if doc in relevant)
    return relevant_count / k if k > 0 else 0

def recall_at_k(retrieved, relevant, k):
    # recall at k
    if not relevant:
        return 0
    retrieved_at_k = retrieved[:k]
    relevant_count = sum(1 for doc in retrieved_at_k if doc in relevant)
    return relevant_count / len(relevant)

def average_precision(retrieved, relevant):
    # average precision
    if not relevant:
        return 0
    
    precision_sum = 0
    num_relevant = 0
    
    for i, doc in enumerate(retrieved):
        if doc in relevant:
            num_relevant += 1
            precision_sum += num_relevant / (i + 1)
    
    return precision_sum / len(relevant)

def mean_average_precision(queries_results, queries_relevant):
    # mean average precision
    aps = []
    
    for query in queries_results:
        retrieved = queries_results[query]
        relevant = queries_relevant.get(query, [])
        ap = average_precision(retrieved, relevant)
        aps.append(ap)
    
    return sum(aps) / len(aps) if aps else 0

def dcg_at_k(relevances, k):
    # discounted cumulative gain
    relevances = relevances[:k]
    dcg = 0
    
    for i, rel in enumerate(relevances):
        dcg += rel / np.log2(i + 2)
    
    return dcg

def ndcg_at_k(relevances, k):
    # normalized dcg
    dcg = dcg_at_k(relevances, k)
    
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = dcg_at_k(ideal_relevances, k)
    
    return dcg / idcg if idcg > 0 else 0

def evaluate_system(queries_results, queries_relevant, k=10):
    # evaluate system
    print(f"evaluating top-{k}...")
    
    precisions = []
    recalls = []
    aps = []
    
    for query in queries_results:
        retrieved = queries_results[query]
        relevant = queries_relevant.get(query, [])
        
        p = precision_at_k(retrieved, relevant, k)
        r = recall_at_k(retrieved, relevant, k)
        ap = average_precision(retrieved, relevant)
        
        precisions.append(p)
        recalls.append(r)
        aps.append(ap)
    
    # averages
    avg_precision = sum(precisions) / len(precisions) if precisions else 0
    avg_recall = sum(recalls) / len(recalls) if recalls else 0
    avg_f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
    map_score = sum(aps) / len(aps) if aps else 0
    
    print(f"\nprecision@{k}: {avg_precision:.3f}")
    print(f"recall@{k}: {avg_recall:.3f}")
    print(f"f1@{k}: {avg_f1:.3f}")
    print(f"MAP: {map_score:.3f}")
    
    return {
        'precision': avg_precision,
        'recall': avg_recall,
        'f1': avg_f1,
        'map': map_score
    }

def load_retrieval_system():
    # load bm25 index
    try:
        with open('models/bm25_index.pkl', 'rb') as f:
            bm25 = pickle.load(f)
        with open('data/processed_corpus.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        documents = [item['text'] for item in data]
        return bm25, documents
    except:
        return None, None

def save_results_for_labeling(bm25, documents):
    # save search results
    test_queries = [
        ("vaarttha", "വാർത്ത"),
        ("kaayikam", "കായികം"),
        ("rashtreeyam", "രാഷ്ട്രീയം"),
        ("cinema", "സിനിമ"),
        ("technology", "സാങ്കേതികവിദ്യ ടെക്നോളജി")
    ]
    
    results_for_labeling = {}
    
    print("saving results for labeling...")
    for query_id, query_text in test_queries:
        results = search(query_text, bm25, documents, top_k=10)
        
        results_for_labeling[query_id] = {
            'query': query_text,
            'results': []
        }
        
        print(f"\nQuery: {query_id} ({query_text})")
        for i, result in enumerate(results):
            print(f"  Doc {result['doc_id']}: score={result['score']:.3f}")
            print(f"    {result['text'][:100]}...")
            
            results_for_labeling[query_id]['results'].append({
                'doc_id': result['doc_id'],
                'score': result['score'],
                'text': result['text']
            })
    
    # save to file
    with open('data/results_to_label.json', 'w', encoding='utf-8') as f:
        json.dump(results_for_labeling, f, ensure_ascii=False, indent=2)
    
    print("\nresults saved to data/results_to_label.json")
    print("update data/relevance_judgments.json")
    
    # create template
    try:
        with open('data/relevance_judgments.json', 'r') as f:
            pass
    except:
        template = {query_id: [] for query_id, _ in test_queries}
        template['_instructions'] = 'list relevant doc IDs'
        with open('data/relevance_judgments.json', 'w', encoding='utf-8') as f:
            json.dump(template, f, ensure_ascii=False, indent=2)

def run_queries_and_evaluate(bm25, documents):
    # load judgments
    try:
        with open('data/relevance_judgments.json', 'r', encoding='utf-8') as f:
            manual_judgments = json.load(f)
        
        has_judgments = any(len(v) > 0 for k, v in manual_judgments.items() if k != '_instructions')
        
        if not has_judgments:
            print("no judgments found")
            save_results_for_labeling(bm25, documents)
            return None
    except:
        print("no judgments file")
        save_results_for_labeling(bm25, documents)
        return None
    
    test_queries = [
        ("vaarttha", "വാർത്ത"),
        ("kaayikam", "കായികം"),
        ("rashtreeyam", "രാഷ്ട്രീയം"),
        ("cinema", "സിനിമ"),
        ("technology", "സാങ്കേതികവിദ്യ ടെക്നോളജി")
    ]
    
    queries_results = {}
    queries_relevant = {}
    
    print("running queries...")
    for query_id, query_text in test_queries:
        if query_id not in manual_judgments or not manual_judgments[query_id]:
            continue
            
        results = search(query_text, bm25, documents, top_k=10)
        retrieved_ids = [r['doc_id'] for r in results]
        
        queries_results[query_id] = retrieved_ids
        queries_relevant[query_id] = manual_judgments[query_id]
        
        print(f"\nQuery: {query_id}")
        print(f"  Retrieved: {len(retrieved_ids)}")
        print(f"  Relevant: {len(manual_judgments[query_id])}")
    
    if not queries_results:
        print("\nno queries with judgments")
        save_results_for_labeling(bm25, documents)
        return None
    
    # evaluate
    print("\n--- evaluation ---")
    
    for query_id in queries_results:
        retrieved = queries_results[query_id]
        relevant = queries_relevant[query_id]
        print(f"\n{query_id}:")
        print(f"  retrieved: {retrieved}")
        print(f"  relevant: {relevant}")
        print(f"  matches: {[d for d in retrieved[:10] if d in relevant]}")
    
    metrics = evaluate_system(queries_results, queries_relevant, k=10)
    
    return metrics

if __name__ == '__main__':
    print("loading system...")
    
    bm25, documents = load_retrieval_system()
    
    if bm25 is None or not documents:
        print("system not found")
    else:
        print(f"loaded {len(documents)} docs")
        run_queries_and_evaluate(bm25, documents)
    
    print("\ndone")
