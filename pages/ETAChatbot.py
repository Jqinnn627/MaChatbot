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

#List to hold components
context_texts = []
source_metadata_list = []
source_counter = 1

#Open AI/LangChain/Pinecone
#pinecone spec
model_name = 'multilingual-e5-large'
embeddings = PineconeEmbeddings(
    model=model_name,
    pinecone_api_key=os.getenv("PINECONE_API_KEY")
)

#Read document
eta_index_name = "etaarticles" #Change to read other db
vectorstore = PineconeVectorStore.from_existing_index(
    index_name=eta_index_name,
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
eta_retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

#----- UI???-----
st.title("YOUR FYP IS COMING")

# Initialize chat history
if "eta_messages" not in st.session_state:
    st.session_state.eta_messages = [{"role": "assistant", "content": "How about your FYP?! "}]

# Display chat messages from history on app rerun
for message in st.session_state.eta_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("START NOW"):
    # Add user message to chat history
    st.session_state.eta_messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
        ans_rag = eta_retrieval_chain.invoke({"input": prompt})

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        # The AI answer
        assistant_response = ans_rag['answer']
        for chunk in assistant_response.split():
            full_response += chunk + " "
            time.sleep(0.05)
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)

        # The AI refer
        for doc in ans_rag['context']:
            metadata = doc.metadata
            citation_key = f"{metadata.get('title')}_{metadata.get('author')}"
            full_citation = (
                f"**{metadata.get('title', 'N/A')}** "
                f"Author(s): {metadata.get('author', 'N/A')}. "
                f"Source: *{metadata.get('source', 'N/A')}*."
            )
            
            source_metadata_list.append({
                "key": citation_key, 
                "full_citation": full_citation
            })
        unique_sources = {}
        for item in source_metadata_list:
            if item['key'] not in unique_sources:
                source_tag = f"[Source {chr(64 + source_counter)}]"
                unique_sources[item['key']] = {
                    "tag": source_tag,
                    "citation": item['full_citation']
                }
                source_counter += 1
        summary_answer = ans_rag['answer']
        citation_list = [f"{info['tag']} {info['citation']}" for info in unique_sources.values()]          
        for citation in citation_list:
            st.markdown(citation)
    st.session_state.eta_messages.append({"role": "assistant", "content": full_response})
    st.session_state.eta_messages.append({"role": "assistant", "content": citation_list})