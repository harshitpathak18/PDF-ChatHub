import os
import streamlit as st
from PyPDF2 import PdfReader
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

st.set_page_config(page_title="Chat with PDF", layout="wide")

# Hiding and cleaning UI
hide_st_style = """
        <style>
        [data-testid="stAppViewContainer"] {
            background-image: url('https://t4.ftcdn.net/jpg/03/31/36/99/240_F_331369932_bzr00q7qt8Fj2KcFZ3bObaNP5vA8Vn9w.jpg');
            background-size: cover;
            margin-top: 0px; /* Set margin-top to 0px to remove space */
        }

        #MainMenu {visibility: hidden;}
        footer{visibility: hidden;}
        .st-emotion-cache-18ni7ap {visibility: hidden;}
        .st-emotion-cache-1avcm0n.ezrtsby2{visibility: hidden;}
        .st-emotion-cache-z5fcl4 {width: 100%; padding: 0rem 2rem 2rem;} 
        .st-emotion-cache-10trblm{text-align: center;}
        </style>
        """
st.markdown(hide_st_style, unsafe_allow_html=True)

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = "YOUR_GOOGLE_API_KEY"

google_api_key = os.environ.get("GOOGLE_API_KEY")
genai.configure(api_key=google_api_key)


def get_text(file_upload):
    if file_upload is not None:
        file_extension = file_upload.name.split('.')[-1].lower()

        if file_extension == 'docx':
            doc = Document(file_upload)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        elif file_extension == 'txt':
            text = file_upload.read().decode('utf-8')
        elif file_extension == 'pdf':
            text = ''
            with file_upload as pdf_file:
                pdf_reader = PdfReader(pdf_file)
                for i in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[i]
                    text += page.extract_text()
            text = str.strip(text)
        return text
    return None


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


@st.cache_data()
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template = """
    Answer the question in detail as much as possible from the provided context, make sure to provide all the details. 
    If the answer is not in the provided context, just say, "Sorry, Answer is not available in the given context", only provide an overview answer\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    try:
        new_db = FAISS.load_local("faiss_index", embeddings)
    except FileNotFoundError:
        st.warning("Vector store not found. Please upload PDF files and process them first.")
        return

    docs = new_db.similarity_search(user_question)

    print("Documents:", docs)  # Add this line for debugging

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    print("Response:", response)  # Add this line for debugging

    with st.container(height=400):
        st.info(f'Response: {response["output_text"]}')


def main():
    st.header("🔯 PDF ChatHub 🔯")

    col1, col2 = st.columns([3, 1])

    with col2:
        pdf_docs = st.file_uploader("Upload PDF Files", accept_multiple_files=True)

    with col1:
        user_question = st.text_input("Ask a Question from the PDF Files")

        if st.button("Process Pdf"):
            if user_question:
                with st.spinner("Processing & Generating Output"):
                    if pdf_docs:
                        raw_text = get_pdf_text(pdf_docs)
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)

                        user_input(user_question)
                    else:
                        st.warning("Please upload PDF files before processing.")

if __name__ == "__main__":
    main()
