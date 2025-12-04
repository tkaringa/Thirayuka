# Text classifier using SVM

import json
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import pickle

def load_data(filename):
    # Load processed data
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    texts = [item['text'] for item in data]
    
    # Create labels for politics
    # Check for political keywords
    political_keywords = [
        "രാഷ്ട്രീയം", "രാഷ്ട്രീയ", # Politics
        "തിരഞ്ഞെടുപ്പ്", # Election
        "സിപിഎം", "സി.പി.എം", # CPM
        "കോൺഗ്രസ്", # Congress
        "ബിജെപി", "ബി.ജെ.പി", # BJP
        "എൽഡിഎഫ്", "എൽ.ഡി.എഫ്", # LDF
        "യുഡിഎഫ്", "യു.ഡി.എഫ്", # UDF
        "സർക്കാർ", # Government
        "മന്ത്രി", # Minister
        "പാർട്ടി", # Party
        "നേതാവ്", # Leader
        "സ്ഥാനാർഥി", # Candidate
        "വോട്ട്" # Vote
    ]
    
    labels = []
    for item in data:
        text = item.get('original_text', item['text'])
        if any(k in text for k in political_keywords):
            labels.append(1)
        else:
            labels.append(0)
    
    print(f"Positive samples (Politics): {sum(labels)}")
    print(f"Negative samples: {len(labels) - sum(labels)}")
    
    return texts, labels

def train_classifier(texts, labels):
    print("splitting data...")
    
    # Split training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    
    print(f"train: {len(X_train)}, test: {len(X_test)}")
    
    # Create TF-IDF features
    print("vectorizing...")
    # Preserve Malayalam tokens
    vectorizer = TfidfVectorizer(max_features=5000, token_pattern=r"(?u)\S+")
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Train SVM classifier
    print("training svm...")
    classifier = SVC(kernel='linear', class_weight='balanced', random_state=42)
    classifier.fit(X_train_tfidf, y_train)
    
    # Make predictions
    print("predicting...")
    y_pred = classifier.predict(X_test_tfidf)
    
    # Calculate performance metrics
    print("\n--- results ---")
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"precision: {precision:.3f}")
    print(f"recall: {recall:.3f}")
    print(f"f1 score: {f1:.3f}")
    
    print("\nreport:")
    print(classification_report(y_test, y_pred))
    
    # Save the model
    print("\nsaving...")
    with open('models/classifier.pkl', 'wb') as f:
        pickle.dump(classifier, f)
    
    with open('models/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    print("done")
    
    return classifier, vectorizer

if __name__ == '__main__':
    # load data
    texts, labels = load_data('data/processed_corpus.json')
    
    print(f"loaded {len(texts)} docs")
    
    # check data size
    if len(texts) < 2:
        print("not enough data")
        sys.exit(0)
    
    # check classes
    if len(set(labels)) < 2:
        print("need 2 classes")
        sys.exit(0)
    
    # train
    train_classifier(texts, labels)
