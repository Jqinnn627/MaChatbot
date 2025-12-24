from langchain_pinecone import PineconeEmbeddings
import os
import time
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic import hub

import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM



# Load Credential
load_dotenv()

#List to hold components
context_texts = []
source_metadata_list = []
source_counter = 1

#OpenAI/LangChain/Pinecone
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

#!--Build LLM--
##prompt engineering
system_prompt = SystemMessagePromptTemplate.from_template('''
    You are a helpful assistance designed for the employee worked in Mastertech Solution Sdn. Bhd.
    You are an AI assistant that communicates in a Malaysian-style tone: casual, slightly local (Manglish), but still clear and professional.  
    Always give accurate answers and logical reasoning.
    If information comes from retrieved documents, rely on them strictly.
    If unsure, say you don't know and provide them the keyword to google search online.
'''
)
human_prompt = HumanMessagePromptTemplate.from_template('''
    "Answer the question using ONLY the following context:\n\n{context}\n\nQuestion: {input}"
'''
)
chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])

##Data from database
retriever= vectorstore.as_retriever()

##Original LLM without knowledge base
llm = ChatOpenAI(
    openai_api_key=os.getenv('OPENAI_API_KEY'),
    model_name='gpt-4o-mini',
    temperature=0.0
)

#Chain with OpenAI LLM
combine_docs_chain = create_stuff_documents_chain(
    llm, chat_prompt
)
#Result
retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

# Let the result has Malaysian slang, Manglish :)
# A fine tuned llama model made by mesolitica
# model_id = "mesolitica/Malaysian-Llama-3.2-1B-Instruct"

# @st.cache_resource
# def load_malaysian_llama():
#     tokenizer = AutoTokenizer.from_pretrained(model_id)
#     model = AutoModelForCausalLM.from_pretrained(
#         model_id,
#         device_map={"": "cpu"},
#         dtype="float32"
#     )
#     return tokenizer, model

# tokenizer, malaysian_model = load_malaysian_llama()


#<!----- UI???----->
st.title("Chatbot")

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
        ans_rag = retrieval_chain.invoke({"input": prompt})

        # inputs = tokenizer(ans_rag['answer'], return_tensors="pt")
        # outputs = malaysian_model.generate(
        #     **inputs,
        #     max_new_tokens=100,
        #     eos_token_id = tokenizer.eos_token_id,
        #     do_sample=True,
        #     temperature=0.7
        # )

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        # assistant_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        assistant_response = ans_rag['answer']
        for chunk in assistant_response.split():
            full_response += chunk + " "
            time.sleep(0.05)
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
        for doc in ans_rag['context']:
            metadata = doc.metadata
            citation_key = f"{metadata.get('title')}_{metadata.get('author')}"
            full_citation = (
                f"**{metadata.get('title', 'N/A')}** "
                f"({metadata.get('year', 'N/A')}). Author(s): {metadata.get('author', 'N/A')}. "
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
    st.session_state.company_messages.append({"role": "assistant", "content": full_response})
