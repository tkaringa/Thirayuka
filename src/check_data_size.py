import json

with open('data/processed_corpus.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

lengths = [len(d.get('original_text', d['text'])) for d in data]
print(f"Max length: {max(lengths)}")
print(f"Avg length: {sum(lengths)/len(lengths)}")
print(f"Num docs: {len(data)}")

# Check for extremely long docs
long_docs = [i for i, l in enumerate(lengths) if l > 100000]
print(f"Docs > 100k chars: {len(long_docs)}")
if long_docs:
    print(f"Example long doc ID: {long_docs[0]}")
