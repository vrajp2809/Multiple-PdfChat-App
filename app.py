import streamlit as st
import pdfplumber
from transformers import pipeline

# Initialize question answering pipeline
nlp = pipeline("question-answering", model="distilbert-base-cased-distilled-squad", revision="626af31")

def convert_pdf_to_super(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
        return text

def convert_files_to_super(pdf_files):
    super_texts = []
    for pdf_file in pdf_files:
        super_texts.append(convert_pdf_to_super(pdf_file))
    return super_texts

def convert_super_to_text(super_texts):
    combined_text = ""
    for text in super_texts:
        combined_text += text + "\n"
    return combined_text

# Streamlit UI
def main():
    st.set_page_config(page_title="PDF Chat App", page_icon=":books:")
    st.header("PDF Chat App :books:")

    # Allow uploading multiple PDF files
    pdf_files = st.file_uploader("Upload PDF files", type=['pdf'], accept_multiple_files=True, key="pdf")
    
    if pdf_files:
        pdf_texts = convert_files_to_super(pdf_files)
        combined_text = convert_super_to_text(pdf_texts)
        st.session_state['pdf_texts'] = pdf_texts
        
        if 'questions' not in st.session_state:
            st.session_state['questions'] = []

        user_question = st.text_input("Ask a question:", key="question")

        if st.button("Get Answer"):
            if user_question:
                # Question answering using Hugging Face Transformers
                answer = nlp({
                    "question": user_question,
                    "context": combined_text
                })

                st.write("Chatbot:", answer["answer"])

                st.session_state['questions'].append(user_question)
            else:
                st.warning("Please enter a question.")

        if st.button("Ask Another Question"):
            st.session_state['questions'] = []

if __name__ == "__main__":
    main()
