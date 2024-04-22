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

# Display previous question-answer pairs
def display_previous_questions(chat_history):
    st.text("Previous Question-Answer Pairs:")
    for i, chat in enumerate(chat_history, start=1):
        st.text(f"{i}) Question: {chat['question']}")
        st.text(f"   Answer: {chat['answer']}")

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

        # Check if there are previous chat history
        if 'chat_history' not in st.session_state:
            st.session_state['chat_history'] = []

        question_counter = 0
        # Display previous question-answer pairs
        # display_previous_questions(st.session_state['chat_history'])
        
        # Loop to ask questions until the user decides to stop
        while True:
            # Calculate the index for the current question
            current_index = len(st.session_state['chat_history']) + 1
            # Text input for the next question
            user_question = st.text_input("Ask a question:", key=f"question_{question_counter}" ,value="" )
            get_answer_button_key = f"get_answer_button_{question_counter}"
            if st.button("Get Answer", key=get_answer_button_key):
                if user_question:
                    # Question answering using Hugging Face Transformers
                    answer = nlp({
                        "question": user_question,
                        "context": combined_text
                    })

                    # Add question and answer to chat history
                    st.session_state['chat_history'].append({"question": user_question, "answer": answer["answer"]})

                    # Clear the text input field
                    # question_counter += 1

                    # st.text_input("Ask a question:", key=f"question_{question_counter}", value="")
                    
                    # Display chat history
                    display_previous_questions(st.session_state['chat_history'])
                else:
                    st.warning("Please enter a question.")
            
            # Check if the user wants to stop asking questions
            question_counter += 1
            # Break the loop if user doesn't want to ask another question
            if not st.session_state['questions']:
                break

if __name__ == "__main__":
    main()
