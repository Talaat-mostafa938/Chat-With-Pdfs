import streamlit as st
import os
import io
from dotenv import load_dotenv
from google.colab import userdata
import pdfplumber
from docx import Document

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.chains.question_answering import load_qa_chain 
from langchain_classic.prompts import PromptTemplate

from langchain_community.document_loaders import PyPDFLoader , TextLoader , UnstructuredWordDocumentLoader
from langchain_community.document_loaders import Docx2txtLoader

def get_file_extension(file_path):
  root , extension = os.path.splitext(file_path)
  return extension[1:]

def get_text_extraction(uploaded_file):
  
  ext = get_file_extension(uploaded_file.name)
  text = ""
    
  if ext == "pdf":
      with pdfplumber.open(io.BytesIO(uploaded_file.read())) as pdf:
          text = "".join([page.extract_text() for page in pdf.pages if page.extract_text()])
            
  elif ext == "docx":
      doc = Document(io.BytesIO(uploaded_file.read()))
      text = "\n".join([para.text for para in doc.paragraphs])
        
  elif ext == "txt":
      text = uploaded_file.read().decode("utf-8")
        
  return text

  documents = loader.load()
  os.remove(tmp_path) # حذف الملف المؤقت بعد القراءة
  return "".join([doc.page_content for doc in documents])


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    return text_splitter.split_text(text)

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(texts=text_chunks,embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

def get_conversation_chain():
  
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("API KEY Not Found")

    prompt_template = """
    Answer the question as detailed as possible using the context below.
    if the answer is not contained within the text below, say "I don't know".

    Context: \n{context}
    Question: \n{question}

    Answer:
    """
    model = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0.3
    )

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context","question"],
    )

    chain = load_qa_chain(
        llm=model,
        chain_type="stuff",
        prompt=prompt
    )

    return chain

def user_input(user_question , uploaded_file):
    if not uploaded_file :
        print("Please upload a PDF document.")
        return

    text_extraction = get_text_extraction(uploaded_file)
    text_chunks = get_text_chunks(text_extraction)
    get_vectorstore(text_chunks)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vector_db = FAISS.load_local("faiss_index", embeddings , allow_dangerous_deserialization=True)
    docs = vector_db.similarity_search(user_question)

    chain = get_conversation_chain()
    response = chain.invoke({"input_documents":docs , "question":user_question} , return_only_outputs=True)
    response_output = response['output_text']
    return response_output

st.set_page_config(page_title="Chat with PDFs", page_icon=":books:")
st.header("Chat with multiple PDFs :books:")

#st.markdown("Upload a file and ask questions instantly!")
#st.markdown("---")

st.sidebar.header("Uploading...")
upload_file = st.sidebar.file_uploader("Upload The File: ", type=["txt", "pdf", "docx"] )
#upload_file = st.file_uploader("Upload The File: ", type=["txt", "pdf", "docx"])
input_text = st.text_input("Enter Your Question: ")

if st.button("Get The Answer"):
    if not upload_file:
        st.error("Please upload a file first.")
    elif not input_text.strip():
        st.warning("Please Enter The Question.")
    else:
        with st.spinner("Answering..."):
            try:
                result = user_input(input_text, upload_file)
                st.success("✅ Success:")
                st.write(result) 
            except Exception as e:
                st.error(f"Failed to Answer: {e}")
