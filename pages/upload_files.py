from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from langchain_pinecone import PineconeVectorStore

import pandas as pd
import streamlit as st
import tempfile

#<!--GLOBAL : pinecone spec-->
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
model_name = 'multilingual-e5-large'
embeddings = PineconeEmbeddings(
    model=model_name,
    pinecone_api_key=os.environ["PINECONE_API_KEY"]
)

#Function to extract PDF
def extract_text_from_pdf(pdf_file):
  #Save uploaded file to tmp loc
  with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
      tmp_file.write(pdf_file.read())
      tmp_pdf_path = tmp_file.name

  loader = PyPDFLoader(tmp_pdf_path)
  documents = loader.load()
  text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=1000,
      chunk_overlap=200
  )
  texts = text_splitter.split_documents(documents)
  return texts

#Create vector database
def create_pc(index_name):
    cloud = 'aws'
    region = 'us-east-1' 
    spec = ServerlessSpec(cloud=cloud, region=region)

    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=embeddings.dimension,
            metric="cosine",
            spec=spec
        )
    else:
        print("Index already exists")

#Upsert Document
def upsert_doc(index_name, all_chunks):

    if index_name not in pc.list_indexes().names():
        create_pc(index_name)
    else:
        vectorstore = PineconeVectorStore.from_documents(
            all_chunks,
            embeddings,
            index_name=index_name
        )
    return True

#<!--Implementation-->
st.title("Upload your file")

indexes = pc.list_indexes().names()

option = st.selectbox(
    "Select your database:",
    indexes,
)
st.write("You selected:", option)

uploaded_files = st.file_uploader(
    "Upload your pdf files", accept_multiple_files=True, type="pdf"
)
if st.button("Submit"):
    if not uploaded_files:
        st.warning("Please upload at least one PDF file before submitting")
    else:
        all_chunks = []
        for uploaded_file in uploaded_files:
            st.write("Uploading " + uploaded_file.name + " ...")
            chunks = extract_text_from_pdf(uploaded_file)
            all_chunks.extend(chunks)
        upsert_doc(option, all_chunks)
        st.write("Successfully uploaded!")