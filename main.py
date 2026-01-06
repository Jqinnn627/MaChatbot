import os
import time
from langchain_pinecone import PineconeEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter

import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.documents import Document

# Library for Model in Hugging Face
# from transformers import AutoTokenizer, AutoModelForCausalLM

# PostgreSQL
import psycopg

# UUID
from streamlit_cookies_manager import EncryptedCookieManager
import uuid
from datetime import datetime, timedelta
import pytz

# Web Scrapping
from bs4 import BeautifulSoup
import requests
from urllib.parse import urlparse, urlencode

# Load Credential
load_dotenv()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)

# ini Database
def get_db_conn():
    return psycopg.connect(
        dbname="user_db",
        user="postgres",
        password="postgre",
        host="localhost",
        port=5432
    )

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
index_name = "machat" #Change to read other db
vectorstore = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

#!--Build LLM--
##prompt engineering
system_prompt = SystemMessagePromptTemplate.from_template('''
    You are a helpful assistance designed for Malaysia user
    You are an AI assistant that communicates in a Malaysian-style tone: casual, local (Manglish), but still clear and professional.  
    Always give accurate answers and logical reasoning.
    If information comes from retrieved documents, rely on them strictly.
    If unsure, you can provide your own opinion and provide them the keyword to google search online.
'''
)
human_prompt = HumanMessagePromptTemplate.from_template('''
    Chat history:
    {chat_history}
    
    Previous Chat summary:
    {chat_summary}

    Answer the question using the following context, if user ask about previous question, check for chat history and chat summary:
    {context}

    Question: {input}
    '''
)

#Storage prompt engineering
storage_system_prompt = SystemMessagePromptTemplate.from_template('''
    You are an assistant that help system to summarize the conversation from user.
    Should summarize what user talking about, anything that related to user.
    Ignore the greeting.
    The summary should not more than 2 sentences.
'''
)
storage_human_prompt = HumanMessagePromptTemplate.from_template('''
    Summarize the following conversation:

    {text}
''')

chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])
summary_prompt = ChatPromptTemplate.from_messages([storage_system_prompt, storage_human_prompt])

##Data from database
retriever= vectorstore.as_retriever()

##Original LLM without knowledge base
llm = ChatOpenAI(
    openai_api_key=os.getenv('OPENAI_API_KEY'),
    model_name='gpt-4o-mini',
    temperature=0.0
)

combine_docs_chain = create_stuff_documents_chain(
    llm, chat_prompt
)

# A prompted LLM for chat summary
summary_assistant = summary_prompt | llm

#Result
retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

###########################################
#           Stop because useless            
#                                     
###########################################
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

# A function to create user record
def ensure_user_exists(user_id):
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO user_profile (user_id, preferences, chat_summary)
                VALUES (%s, %s, %s)
                ON CONFLICT (user_id) DO NOTHING
            """, (user_id, [], ""))
# A function to get previous summary on user id
def fetch_chat_summary(user_id):
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT chat_summary
                FROM user_profile
                WHERE user_id = %s
            """, (user_id,))
            row = cur.fetchone()
            return row[0] if row and row[0] else ""
# Function to store chat summary every session
def generate_session_summary(messages):
    text = "\n".join(
        f"{m['role']}: {m['content']}"
        for m in messages
        if m["role"] in ("user", "assistant")
    )

    chat_summary = summary_assistant.invoke({"text": text})
    
    return f"- User discussed: {chat_summary.content}"
# Function to save summary to DB(Postgres)
def save_summary(user_id, session_messages):
    session_summary = generate_session_summary(session_messages)
    print("session_summary in Save Summary function.")

    with get_db_conn() as conn:
        with conn.cursor() as cur:
            print("Debug: Saving now")
            cur.execute("""
                UPDATE user_profile
                SET chat_summary = COALESCE(chat_summary, '') || E'\n' || %s,
                    last_seen = NOW()
                WHERE user_id = %s
            """, (session_summary, user_id))
# Use last 6 messages
def format_chat_history(messages, limit=6):
    history = messages[-limit:]
    return "\n".join(
        f"{m['role'].upper()}: {m['content']}"
        for m in history
        if m["role"] in ("user", "assistant")
    )
### Similarity Search ###
SIMILARITY_THRESHOLD = 0.8 # 0 -- 1  || bad -- good
def retrieve_with_score(query, k=3):
    results = vectorstore.similarity_search_with_score(query, k=k)

    good_docs = []
    for doc, score in results:
        print(f"[DEBUG] Similarity score: {score}")
        if score > SIMILARITY_THRESHOLD:
            good_docs.append(doc)

    return good_docs, results

def search_trusted_sources(query):
    url = f"https://s.jina.ai/?q={query}"
    headers = {
        "Accept": "application/json",
        "Authorization": "Bearer jina_fd1a6ae50f6a46bbb61b791afc92fb2eZfRzQuxVNzPm53X1vqXREqeDJqNB",
        "X-Respond-With": "no-content"
    }

    response = requests.get(url, headers=headers, timeout=10)
    data = response.json()
    print(f"response: {data}")

    links = []
    for a in data.get("data", [])[:5]:
        links.append(a["url"])

    print(links)
    return links

def fetch_document(url):
    try:
        r = requests.get(url, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")

        text = soup.get_text(separator=" ", strip=True)

        if len(text) < 500:
            return None
        
        chunks = splitter.split_text(text)
        docs = [
            Document(
                page_content=chunk,
                metadata={
                    "source": url,
                    "type": "trusted_web"
                }
            )
            for chunk in chunks
        ]

        return docs

    except Exception as e:
        print(f"Fetch failed {url}: {e}")
        return None

#Simplify searching
def rewrite_query(query):
    prompt = f"""
        Rewrite this question into a short factual search query.
        Remove filler words.
        Keep important nouns.
        
        Question:
        {query}
    """
    return llm.invoke(prompt).content.strip()


#<!----- UI???----->
st.title("Chatbot")

# Set cookie
cookies = EncryptedCookieManager(
    prefix="chatbot",
    password="uknowimjusttrying"
)

if not cookies.ready():
    st.stop()

#store user_id in cookie
if "user_id" not in cookies:
    cookies["user_id"] = str(uuid.uuid4())
    cookies.save()

user_id = cookies["user_id"]
ensure_user_exists(user_id)

#Fetch chat summary for current user
chat_summary = fetch_chat_summary(user_id)

if "last_summary_time" not in st.session_state:
    st.session_state.last_summary_time = datetime.now(
        pytz.timezone("Asia/Kuala_Lumpur")
    )

if "last_summarized_len" not in st.session_state:
    st.session_state.last_summarized_len = 0

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Let's start chatting! 👇"}]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

Messagenow = datetime.now(pytz.timezone('Asia/Kuala_Lumpur'))
# Accept user input
if prompt := st.chat_input("I want to..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    chat_history = format_chat_history(st.session_state.messages)
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
        
        # Simplify searching (trying, bad then bye)
        search_query = rewrite_query(prompt)
        print(search_query)
        # checking similarity score
        docs, raw_scores = retrieve_with_score(search_query)

        # If Pinecone has good knowledge
        if len(docs) > 0:
            ans_rag = combine_docs_chain.invoke({
                "input": search_query,
                "context": docs,
                "chat_history": chat_history,
                "chat_summary": chat_summary
            })

        # Pinecone empty / low confidence → web fallback
        else:
            st.warning("Getting from trusted sources…")

            urls = search_trusted_sources(search_query)
            web_texts = []

            for url in urls:
                doc = fetch_document(url)
                if doc:
                    web_texts.extend(doc)

                    # Store into Pinecone for future
                    vectorstore.add_documents(doc)
                    #Debug if info stored
                    print(vectorstore._index.describe_index_stats())


            if len(web_texts) == 0:
                ans_rag = {
                    "answer": (
                        "Sorry ah, I couldn't find reliable info. "
                        "Try search keywords like: " + search_query
                    ),
                    "context": []
                }
            else:
                ans_rag = combine_docs_chain.invoke({
                    "input": search_query,
                    "context": web_texts,
                    "chat_history": chat_history,
                    "chat_summary": chat_summary
                })


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
        if isinstance(ans_rag, dict):
            assistant_response = ans_rag.get('answer', str(ans_rag))
            context_docs = ans_rag.get('context', [])
        else:
            # If it's just a string, use it directly
            assistant_response = ans_rag
            context_docs = []
        for chunk in assistant_response.split():
            full_response += chunk + " "
            time.sleep(0.05)
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})

    print("End of process, now backend")
    new_messages = st.session_state.messages[st.session_state.last_summarized_len:]
    time_passed = Messagenow - st.session_state.last_summary_time

    if time_passed >= timedelta(minutes=30) and len(new_messages) > 0:
        print("Conditions met! Calling save_summary...")
        save_summary(user_id, new_messages)
        st.session_state.last_summary_time = Messagenow
        st.session_state.last_summarized_len = len(st.session_state.messages)
    else:
        print("Conditions NOT met. Skipping summary.")



