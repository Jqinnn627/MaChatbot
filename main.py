# -*- coding: utf-8 -*-
"""OpenAI/Pinecone/LangChain.ipynb

File is downloaded at
    https://colab.research.google.com/drive/1__r_Sgd_9-lekxLLdXEwaGR4f0cwICFV
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeEmbeddings
import os
from pinecone import Pinecone, ServerlessSpec
import time
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic import hub
from google.colab import userdata
#--- End of tutorial example---
from langchain_community.document_loaders import PyPDFLoader
import streamlit as st

#Set credential
os.environ["PINECONE_API_KEY"] = userdata.get('PINECONE_API_KEY')
os.environ["OPENAI_API_KEY"] = userdata.get('OPENAI_API_KEY')

"""# OpenAI/Pinecone/RAG"""

#pinecone spec
model_name = 'multilingual-e5-large'
embeddings = PineconeEmbeddings(
    model=model_name,
    pinecone_api_key=os.environ["PINECONE_API_KEY"]
)

#Function to extract PDF
def extract_text_from_pdf(pdf_path):
  loader = PyPDFLoader(pdf_path)
  documents = loader.load()
  text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=1000,
      chunk_overlap=200
  )
  texts = text_splitter.split_documents(documents)
  return texts

#Load all pdf files in folder
def load_all_pdfs(folder_path):
    all_chunks = []

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            print("Loading:", filename)
            chunks = extract_text_from_pdf(pdf_path)
            all_chunks.extend(chunks)

    return all_chunks

#Execution of extracting files
pdf_folder = "/content/pdfs"
all_chunks = load_all_pdfs(pdf_folder)

print("Total chunks:", len(all_chunks))

#Create vector database
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

cloud = 'aws' # or os.environ.get('PINECONE_CLOUD')
region = 'us-east-1' # or os.environ.get('PINECONE_REGION')
spec = ServerlessSpec(cloud=cloud, region=region)

index_name = "company-info"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=embeddings.dimension,
        metric="cosine",
        spec=spec
    )
else:
    print("Index already exists")

# See that it is empty
print("Index before upsert:")
print(pc.Index(index_name).describe_index_stats())
print("\n")

#Upsert Document
index_name = "etaarticles"

vectorstore = PineconeVectorStore.from_documents(
    all_chunks,
    embeddings,
    index_name=index_name
)

time.sleep(5)

# See how many vectors have been upserted
print("Index after upsert:")
print(pc.Index(index_name).describe_index_stats())
print("\n")
time.sleep(2)

#Read document
index_name = "etaarticles" #Change to read other db
vectorstore = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

#Build LLM
retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
retriever= vectorstore.as_retriever()

llm = ChatOpenAI(
    openai_api_key=os.environ.get('OPENAI_API_KEY'),
    model_name='gpt-4o-mini',
    temperature=0.0
)

combine_docs_chain = create_stuff_documents_chain(
    llm, retrieval_qa_chat_prompt
)
retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

query1 = "From these information, what method u would recommend me to calculate Bus ETA prediction? write in point form"
answer1_with_knowledge = retrieval_chain.invoke({"input": query1})

print("Answer with knowledge:\n\n", answer1_with_knowledge['answer'])
print("\nContext used:\n\n", answer1_with_knowledge['context'])
print("\n")

query2 = "What parameters should be collected to predict ETA?"
ans2_rag = retrieval_chain.invoke({"input": query2})

print("Answer with knowledge:\n\n", ans2_rag['answer'])
print("\nContext used:\n\n", ans2_rag['context'])
print("\n")
time.sleep(2)

answer2_llm = llm.invoke(query2)

print("Query 1:", query2)
print("\nAnswer without knowledge:\n\n", answer2_llm.content)
print("\n")
time.sleep(2)

query3 = "What is included in IET Intelligent Trans Sys articles, can u summarize in details for me?"
ans3_rag = retrieval_chain.invoke({"input": query3})

print("Answer with knowledge:\n\n", ans3_rag['answer'])
print("\nContext used:\n\n", ans3_rag['context'])
print("\n")

"""# UI - Streamlit

"""

