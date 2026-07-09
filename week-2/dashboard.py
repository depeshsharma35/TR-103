import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Configure page
st.set_page_config(page_title="Support Ticket ML Dashboard", layout="wide", page_icon="🎫")

# Ensure NLTK data is downloaded
@st.cache_resource
def load_nltk_data():
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    return set(stopwords.words('english'))

stop_words = load_nltk_data()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

@st.cache_resource
def load_data():
    with open('processed_data.pkl', 'rb') as f:
        processed = pickle.load(f)
    
    with open('models_and_preds.pkl', 'rb') as f:
        models = pickle.load(f)
        
    return processed, models

try:
    processed_data, model_data = load_data()
except FileNotFoundError:
    st.error("Model files not found. Please run the training pipeline first.")
    st.stop()

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Performance Overview", "Live Inference"])

if page == "Performance Overview":
    st.title("Model Performance Overview")
    
    # Unpack model data
    y_test_cat = model_data['y_test_cat']
    y_test_prio = model_data['y_test_prio']
    cat_preds = model_data['cat_preds']
    prio_preds = model_data['prio_preds']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Category Classification")
        st.metric("Overall Accuracy", f"{accuracy_score(y_test_cat, cat_preds):.2%}")
        
        st.subheader("Classification Report")
        cat_report = classification_report(y_test_cat, cat_preds, output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(cat_report).transpose(), use_container_width=True)
        
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots(figsize=(6,4))
        cm_cat = confusion_matrix(y_test_cat, cat_preds)
        sns.heatmap(cm_cat, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=sorted(y_test_cat.unique()), 
                    yticklabels=sorted(y_test_cat.unique()), ax=ax)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        st.pyplot(fig)
        
    with col2:
        st.header("Priority Classification")
        st.metric("Overall Accuracy", f"{accuracy_score(y_test_prio, prio_preds):.2%}")
        
        st.subheader("Classification Report")
        prio_report = classification_report(y_test_prio, prio_preds, output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(prio_report).transpose(), use_container_width=True)
        
        st.subheader("Confusion Matrix")
        fig2, ax2 = plt.subplots(figsize=(6,4))
        cm_prio = confusion_matrix(y_test_prio, prio_preds)
        sns.heatmap(cm_prio, annot=True, fmt='d', cmap='Greens', 
                    xticklabels=sorted(y_test_prio.unique()), 
                    yticklabels=sorted(y_test_prio.unique()), ax=ax2)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        st.pyplot(fig2)

elif page == "Live Inference":
    st.title("Live Ticket Inference")
    st.write("Test the trained models by entering a sample support ticket text below.")
    
    ticket_text = st.text_area("Support Ticket Text", height=150, placeholder="E.g., I cannot access my account. The server seems to be down and it's an urgent issue!")
    
    if st.button("Predict"):
        if ticket_text.strip():
            with st.spinner("Processing..."):
                # Clean text
                cleaned = clean_text(ticket_text)
                
                # Vectorize
                tfidf = processed_data['tfidf_vectorizer']
                vectorized = tfidf.transform([cleaned])
                
                # Predict
                cat_model = model_data['cat_model']
                prio_model = model_data['prio_model']
                
                cat_pred = cat_model.predict(vectorized)[0]
                prio_pred = prio_model.predict(vectorized)[0]
                
            st.success("Prediction complete!")
            
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**Predicted Category:**\n\n### {cat_pred}")
            with col2:
                # Color code priority
                if prio_pred == 'High':
                    st.error(f"**Predicted Priority:**\n\n### {prio_pred}")
                elif prio_pred == 'Medium':
                    st.warning(f"**Predicted Priority:**\n\n### {prio_pred}")
                else:
                    st.success(f"**Predicted Priority:**\n\n### {prio_pred}")
                    
            with st.expander("Show processed text"):
                st.write(cleaned)
        else:
            st.warning("Please enter some text to predict.")
