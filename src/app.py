import streamlit as st
import pickle
import json
import sys
import os
import numpy as np
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import re
from collections import Counter
import pandas as pd

import random

# Add src to path
sys.path.append(os.path.dirname(__file__))

from preprocess import clean_malayalam_text, tokenize_malayalam
from retrieval import BM25, search

st.set_page_config(page_title="Malayalam Search", page_icon="🔍", layout="wide")

st.markdown("""
<style>
    .stTextInput > div > div > input {
        border-radius: 24px;
        padding: 10px 20px;
        border: 1px solid #dfe1e5;
        box-shadow: none;
    }
    .stTextInput > div > div > input:hover, .stTextInput > div > div > input:focus {
        box-shadow: 0 1px 6px rgba(32,33,36,.28);
        border-color: rgba(223,225,229,0);
    }
    .stButton > button {
        background-color: #f8f9fa;
        border: 1px solid #f8f9fa;
        border-radius: 4px;
        color: #3c4043;
        font-family: arial,sans-serif;
        font-size: 14px;
        margin: 11px 4px;
        padding: 0 16px;
        line-height: 27px;
        height: 36px;
        min-width: 54px;
        text-align: center;
        cursor: pointer;
        user-select: none;
    }
    .stButton > button:hover {
        box-shadow: 0 1px 1px rgba(0,0,0,.1);
        background-color: #f8f9fa;
        border: 1px solid #dadce0;
        color: #202124;
    }
    div[data-testid="stImage"] {
        display: flex;
        justify_content: center;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_resources():
    resources = {}
    
    # Load BM25
    try:
        with open('models/bm25_index.pkl', 'rb') as f:
            resources['bm25'] = pickle.load(f)
        with open('data/processed_corpus.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        resources['documents'] = [item['text'] for item in data]
        resources['original_docs'] = [item.get('original_text', item['text']) for item in data]
    except Exception as e:
        st.error(f"Error loading BM25: {e}")
        
    # Load SVM Classifier
    try:
        with open('models/classifier.pkl', 'rb') as f:
            resources['svm'] = pickle.load(f)
        with open('models/vectorizer.pkl', 'rb') as f:
            resources['vectorizer'] = pickle.load(f)
    except Exception as e:
        st.error(f"Error loading SVM Classifier: {e}")

    # Load BERT Classifier
    try:
        model_path = 'models/bert_classifier'
        if os.path.exists(model_path):
            resources['bert_tokenizer'] = DistilBertTokenizer.from_pretrained(model_path)
            resources['bert_model'] = DistilBertForSequenceClassification.from_pretrained(model_path)
    except Exception as e:
        st.error(f"Error loading BERT Classifier: {e}")
        
    return resources

def highlight_text(text, query):
    if not query:
        return text
    
    # Simple highlighting of query terms
    query_terms = query.split()
    highlighted = text
    for term in query_terms:
        # Case-insensitive replacement
        pattern = re.compile(f"({re.escape(term)})", re.IGNORECASE)
        highlighted = pattern.sub(r'<span style="background-color: #FFFF00; color: black; font-weight: bold;">\1</span>', highlighted)
    return highlighted

resources = load_resources()

# Sidebar Navigation
page = st.sidebar.selectbox("Navigate", ["Search", "Classify", "Corpus Stats"])

if page == "Search":
    # Centered Layout
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Logo
        if os.path.exists("search.svg"):
            with open("search.svg", "r", encoding="utf-8") as f:
                svg_code = f.read()
            st.markdown(f'<div style="text-align: center; max-width: 300px; margin: 0 auto;">{svg_code}</div>', unsafe_allow_html=True)
        else:
            st.markdown("<h1 style='text-align: center; color: #4285F4;'>Malayalam Search</h1>", unsafe_allow_html=True)
            
        # Search Bar
        query = st.text_input("", placeholder="Search Malayalam Documents...", label_visibility="collapsed")
        
        # Buttons
        b_col1, b_col2, b_col3, b_col4 = st.columns([1, 2, 2, 1])
        with b_col2:
            search_clicked = st.button("Malayalam Search")
        with b_col3:
            lucky_clicked = st.button("Random Document")

    # Results Area
    if query: # Streamlit reruns on enter in text_input
        if 'bm25' in resources:
            results = search(query, resources['bm25'], resources['documents'], top_k=10)
            
            st.markdown(f"About {len(results)} results")
            
            for i, res in enumerate(results):
                doc_id = res['doc_id']
                score = res['score']
                text = resources['original_docs'][doc_id]
                
                # Create a snippet
                snippet = text[:300] + "..." if len(text) > 300 else text
                highlighted_snippet = highlight_text(snippet, query)
                
                st.markdown(f"### [{doc_id}] Document {doc_id}")
                st.markdown(f"<small style='color:green'>Score: {score:.4f}</small>", unsafe_allow_html=True)
                st.markdown(highlighted_snippet, unsafe_allow_html=True)
                with st.expander("View Full Text"):
                    st.markdown(highlight_text(text, query), unsafe_allow_html=True)
                st.markdown("---")
                
    elif lucky_clicked:
        if 'documents' in resources:
            doc_id = random.randint(0, len(resources['documents'])-1)
            text = resources['original_docs'][doc_id]
            st.markdown(f"### Random Document {doc_id}")
            st.write(text)

elif page == "Classify":
    st.header("Text Classification")
    st.write("Detect if text is related to **Politics (രാഷ്ട്രീയം)**")
    
    input_text = st.text_area("Enter Malayalam Text", height=150)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Classify with SVM"):
            if 'svm' in resources and input_text:
                # Preprocess
                cleaned = clean_malayalam_text(input_text)
                # Vectorize
                vec = resources['vectorizer'].transform([cleaned])
                # Predict
                pred = resources['svm'].predict(vec)[0]
                
                if pred == 1:
                    st.success("SVM Prediction: **Politics (രാഷ്ട്രീയം)**")
                else:
                    st.info("SVM Prediction: **Other**")
            elif not input_text:
                st.warning("Please enter text.")
            else:
                st.error("SVM Classifier not loaded.")

    with col2:
        if st.button("Classify with BERT"):
            if 'bert_model' in resources and input_text:
                # Tokenize
                inputs = resources['bert_tokenizer'](input_text, return_tensors="pt", truncation=True, padding=True, max_length=128)
                # Predict
                with torch.no_grad():
                    outputs = resources['bert_model'](**inputs)
                
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=-1)
                pred = torch.argmax(probs, dim=-1).item()
                confidence = probs[0][pred].item()
                
                if pred == 1:
                    st.success(f"BERT Prediction: **Politics (രാഷ്ട്രീയം)** (Conf: {confidence:.2f})")
                else:
                    st.info(f"BERT Prediction: **Other** (Conf: {confidence:.2f})")
            elif not input_text:
                st.warning("Please enter text.")
            else:
                st.error("BERT Classifier not loaded.")

elif page == "Corpus Stats":
    st.header("Corpus Statistics")
    if 'documents' in resources:
        col1, col2, col3 = st.columns(3)
        
        num_docs = len(resources['documents'])
        col1.metric("Total Documents", num_docs)
        
        # Calculate vocab size and word freqs
        all_tokens = []
        for doc in resources['documents']:
            all_tokens.extend(doc.split())
        
        vocab_size = len(set(all_tokens))
        col2.metric("Vocabulary Size", vocab_size)
        
        avg_len = len(all_tokens) / num_docs
        col3.metric("Avg Doc Length", f"{avg_len:.0f} words")
        
        st.subheader("Top 20 Frequent Words")
        word_counts = Counter(all_tokens)
        common_words = word_counts.most_common(20)
        
        df = pd.DataFrame(common_words, columns=['Word', 'Count'])
        st.bar_chart(df.set_index('Word'))
        
        st.subheader("Sample Documents")
        st.json(resources['original_docs'][:3])
