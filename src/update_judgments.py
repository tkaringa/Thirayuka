import json
import re

def update_judgments():
    print("Updating relevance judgments based on new corpus...")
    
    with open('data/processed_corpus.json', 'r', encoding='utf-8') as f:
        documents = json.load(f)
    
    # Define queries and keywords
    queries = {
        "vaarttha": ["വാർത്ത"],
        "kaayikam": ["കായികം", "സ്പോർട്സ്", "ക്രിക്കറ്റ്", "ഫുട്ബോൾ"], # Sports related terms
        "rashtreeyam": ["രാഷ്ട്രീയം", "രാഷ്ട്രീയ"],
        "cinema": ["സിനിമ", "ചലച്ചിത്രം"],
        "technology": ["സാങ്കേതികവിദ്യ", "ടെക്നോളജി"]
    }
    
    new_judgments = {}
    
    for q_key, keywords in queries.items():
        relevant_ids = []
        for i, doc in enumerate(documents):
            # Check for keywords
            # Use original text
            text = doc['original_text']
            
            # Check if keyword exists
            # Split text into words
            words = set(text.split())
            
            is_relevant = False
            for kw in keywords:
                # Check for exact match
                if kw in words:
                    is_relevant = True
                    break
            
            if is_relevant:
                relevant_ids.append(i)
        
        print(f"Query '{q_key}': found {len(relevant_ids)} relevant docs")
        new_judgments[q_key] = relevant_ids

    # Save new judgments
    with open('data/relevance_judgments.json', 'w', encoding='utf-8') as f:
        json.dump(new_judgments, f, ensure_ascii=False, indent=2)
    
    print("Saved new relevance judgments.")

if __name__ == '__main__':
    update_judgments()
