import streamlit as st
import pickle
from transformers import pipeline

# Load models with error handling
try:
    with open("tfd.pkl", "rb") as f:
        tfidf = pickle.load(f)
    with open("logis.pkl", "rb") as f:
        clf = pickle.load(f)
except:
    st.error("Model files not found!")
    st.stop()

# Load transformers with CPU device to avoid meta tensor error
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad", device=-1)
ner_pipeline = pipeline("ner", grouped_entities=True, device=-1)

st.title("📰 Smart News Analyze App")
st.write("This app performs classification, summarization, QA, and NER on news articles.")

user_text = st.text_area("Paste a news article here:", height=200)

if st.button("Analyze"):
    if user_text.strip():
        # Classification
        X_input = tfidf.transform([user_text])
        pred_class = clf.predict(X_input)[0]
        class_names = ["World", "Sports", "Business", "Sci/Tech"]
        st.subheader("Classification")
        st.write(f"Predicted Category: {class_names[pred_class]}")

        # Summarization
        if len(user_text.split()) > 10:
            st.subheader("Summarization")
            summary = summarizer(user_text, max_length=50, min_length=20, do_sample=False)
            st.write(summary[0]['summary_text'])

        # NER
        st.subheader("Named Entity Recognition (NER)")
        entities = ner_pipeline(user_text)
        for ent in entities:
            st.write(f"{ent['word']} → {ent['entity_group']} (score={ent['score']:.2f})")
    else:
        st.warning("Please enter some text to analyze.")

# Question Answering
st.subheader("❓ Ask Question")
if user_text.strip():
    question = st.text_input("Ask a question about the article:")
    if st.button("Get Answer"):
        if question.strip():
            answer = qa_pipeline({"context": user_text, "question": question})
            st.write(f"Answer: {answer['answer']} (score={answer['score']:.2f})")