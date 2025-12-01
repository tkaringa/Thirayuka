# text classifier with tfidf svm

import json
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import pickle

def load_data(filename):
    # load processed data
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    texts = [item['text'] for item in data]
    
    # create labels based on content (Politics detection)
    # Check original_text for "രാഷ്ട്രീയം"
    labels = []
    for item in data:
        if "രാഷ്ട്രീയം" in item['original_text'] or "രാഷ്ട്രീയ" in item['original_text']:
            labels.append(1)
        else:
            labels.append(0)
    
    print(f"Positive samples (Politics): {sum(labels)}")
    print(f"Negative samples: {len(labels) - sum(labels)}")
    
    return texts, labels

def train_classifier(texts, labels):
    print("splitting data...")
    
    # split train test
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    
    print(f"train: {len(X_train)}, test: {len(X_test)}")
    
    # tfidf features
    print("vectorizing...")
    vectorizer = TfidfVectorizer(max_features=1000)
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # train svm
    print("training svm...")
    classifier = SVC(kernel='linear', random_state=42)
    classifier.fit(X_train_tfidf, y_train)
    
    # predict
    print("predicting...")
    y_pred = classifier.predict(X_test_tfidf)
    
    # metrics
    print("\n--- results ---")
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"precision: {precision:.3f}")
    print(f"recall: {recall:.3f}")
    print(f"f1 score: {f1:.3f}")
    
    print("\nreport:")
    print(classification_report(y_test, y_pred))
    
    # save model
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
