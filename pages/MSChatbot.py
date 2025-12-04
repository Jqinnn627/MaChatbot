# -*- coding: utf-8 -*-
"""OpenAI/Pinecone/LangChain.ipynb

File is downloaded at
    https://colab.research.google.com/drive/1__r_Sgd_9-lekxLLdXEwaGR4f0cwICFV
"""

from langchain_pinecone import PineconeEmbeddings
import os
import time
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic import hub
#--- End of tutorial example---
import streamlit as st
from dotenv import load_dotenv

# Load Credential
load_dotenv()

#Open AI/LangChain/Pinecone
#pinecone spec
model_name = 'multilingual-e5-large'
embeddings = PineconeEmbeddings(
    model=model_name,
    pinecone_api_key=os.getenv("PINECONE_API_KEY")
)

#Read document
index_name = "company-info" #Change to read other db
vectorstore = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

#Build LLM
retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
retriever= vectorstore.as_retriever()

#Original LLM without knowledge base
llm = ChatOpenAI(
    openai_api_key=os.getenv('OPENAI_API_KEY'),
    model_name='gpt-4o-mini',
    temperature=0.0
)

#Chain wiith LLM
combine_docs_chain = create_stuff_documents_chain(
    llm, retrieval_qa_chat_prompt
)
#Result
retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

#<!----- UI???----->
st.title("Mastertech Chatbot")

# Initialize chat history
if "company_messages" not in st.session_state:
    st.session_state.company_messages = [{"role": "assistant", "content": "Let's start chatting! 👇"}]

# Display chat messages from history on app rerun
for message in st.session_state.company_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask something about Mastertech!"):
    # Add user message to chat history
    st.session_state.company_messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
        ans3_rag = retrieval_chain.invoke({"input": prompt})

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        assistant_response = ans3_rag['answer']
        for chunk in assistant_response.split():
            full_response += chunk + " "
            time.sleep(0.05)
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.company_messages.append({"role": "assistant", "content": full_response})
