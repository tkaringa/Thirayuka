import sys
import pickle
import os

# Add src to path to import preprocess
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from preprocess import clean_malayalam_text, tokenize_malayalam

def test_prediction():
    print("Loading model...")
    try:
        with open('models/classifier.pkl', 'rb') as f:
            svm = pickle.load(f)
        with open('models/vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    input_text = "നെയ്യാറ്റിന്കരയില് സിപിഎമ്മിന് അഞ്ചിടത്ത് വിമതര് യുഡിഎഫിന്റെ ഭാഗമാവാതെ ആര്എംപിയും ആര്എസ്പിയും നഗരസഭാ തിരഞ്ഞെടുപ്പിനുള്ള ചിത്രം വ്യക്തമായതോടെ എൽഡിഎഫിൽ അഞ്ചിടത്ത് സിപിഎം വിമതർ"
    
    print(f"\nInput Text: {input_text}")
    
    # Preprocess
    cleaned = clean_malayalam_text(input_text)
    tokens = tokenize_malayalam(cleaned)
    processed_text = ' '.join(tokens)
    
    print(f"Processed: {processed_text}")
    
    # Vectorize
    vec = vectorizer.transform([processed_text])
    
    # Check vocab and weights
    vocab = vectorizer.vocabulary_
    feature_names = vectorizer.get_feature_names_out()
    
    # Get coefficients
    if hasattr(svm, 'coef_'):
        coefs = svm.coef_.toarray()[0]
        
        # Print top positive features
        print("\nTop 20 Political Keywords (Positive Weights):")
        top_indices = coefs.argsort()[-20:][::-1]
        for idx in top_indices:
            print(f"  {feature_names[idx]}: {coefs[idx]:.4f}")
            
    else:
        print("SVM does not have coef_ attribute (maybe not linear kernel?)")
        coefs = None

    print("\nVocabulary Check:")
    for token in tokens:
        if token in vocab:
            idx = vocab[token]
            weight = coefs[idx] if coefs is not None else "N/A"
            print(f"  '{token}': Found (Index {idx}), Weight: {weight}")
        else:
            print(f"  '{token}': NOT Found")

    # Predict
    decision = svm.decision_function(vec)[0]
    pred = svm.predict(vec)[0]
    
    print(f"\nIntercept: {svm.intercept_[0]}")
    print(f"Decision Function Value: {decision}")
    
    result = "Politics (രാഷ്ട്രീയം)" if pred == 1 else "Other"
    print(f"\nPrediction: {result}")

if __name__ == "__main__":
    test_prediction()
