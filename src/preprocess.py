# preprocess malayalam text

import re
import json

def clean_malayalam_text(text):
    # remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    
    # remove english chars
    text = re.sub(r'[a-zA-Z]', '', text)
    
    # remove numbers
    text = re.sub(r'\d+', '', text)
    
    # keep only malayalam
    text = re.sub(r'[^\u0D00-\u0D7F\s\u0964\u0965]', '', text)
    
    return text.strip()

def get_stopwords():
    # common malayalam stopwords
    return {
        'ഒരു', 'ഈ', 'ആ', 'ആണ്', 'അത്', 'ഇത്', 'എന്ന്', 'എന്ന', 
        'ഉള്ള', 'ആയി', 'മറ്റ', 'മറ്റു', 'അവ', 'അവർ', 'ഇവ', 
        'ഏത്', 'ഏത', 'എന്ത്', 'എങ്ങനെ', 'എപ്പോൾ', 'അല്ല', 'ഇല്ല',
        'ഉണ്ട്', 'വളരെ', 'ഏറ്റവും', 'കുറിച്ച്', 'ശേഷം', 'മുമ്പ്',
        'വരെ', 'മാത്രം', 'അല്ലെങ്കിൽ', 'എങ്കിലും', 'പക്ഷേ'
    }

def simple_stem(word):
    # basic suffix stripping
    suffixes = ['ിൽ', 'ുടെ', 'ാൽ', 'ക്ക്', 'നു', 'ടെ', 'ലെ', 'ക്കും', 'യി', 'വും', 'ം']
    for s in suffixes:
        if word.endswith(s):
            return word[:-len(s)]
    return word

def tokenize_malayalam(text):
    # split, remove stopwords, stem
    tokens = text.split()
    stopwords = get_stopwords()
    
    processed = []
    for t in tokens:
        if t not in stopwords:
            processed.append(simple_stem(t))
            
    return processed

def preprocess_corpus(input_file, output_file):
    print("loading data...")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        texts = json.load(f)
    
    print(f"processing {len(texts)} texts...")
    
    processed_data = []
    
    for text in texts:
        # clean text
        cleaned = clean_malayalam_text(text)
        
        # skip short texts
        if len(cleaned) < 20:
            continue
        
        # tokenize
        tokens = tokenize_malayalam(cleaned)
        
        processed_data.append({
            'text': ' '.join(tokens),
            'original_text': cleaned,
            'tokens': tokens,
            'num_tokens': len(tokens)
        })
    
    # save data
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    
    print(f"saved {len(processed_data)} texts")
    print("done")

if __name__ == '__main__':
    preprocess_corpus('data/malayalam_corpus.json', 'data/processed_corpus.json')
