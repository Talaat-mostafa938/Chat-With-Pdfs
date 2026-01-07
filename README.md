# Advanced PDF Chatbot with Groq and LangChain

## 📖 Overview
This project provides an advanced, modular, and efficient Streamlit application for chatting with PDF documents. Users can upload (PDFs,Word,txt), provide a Groq AI API key, and ask questions about the document content. The backend leverages LangChain, Google's Gemini Pro model, and a FAISS vector store for retrieval-augmented generation (RAG).

## ✨ Features
Multiple PDF Upload: Upload and process several (PDFs,Word,txt). 
Modular Architecture: The code is separated into UI, utilities, and main application logic for better readability and maintenance.   
Uses Groq's powerful llama-3.3-70b model for generating answers.     

## 🛠️ Tech Stack
Frontend: Streamlit   
Backend: LangChain, Groq Generative AI   
Embeddings: Hugging Face sentence-transformers     
Vector Store: FAISS (Facebook AI Similarity Search)      
