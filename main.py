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

# MySQL / local XAMPP
from sqlalchemy import text

# UUID
from streamlit_cookies_manager import EncryptedCookieManager
import uuid
from datetime import datetime, timedelta
import pytz

# Web Scrapping
from bs4 import BeautifulSoup
import requests

#Pretrained-Model
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

# Load Credential
load_dotenv()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)

GREETING_KW = {
    "hi", "hello", "hey", "yo", "sup", "morning",
    "evening", "afternoon", "hai", "helo", "hello ah"
}

LEAVING_KW = {
    "bye", "byebye", "goodbye", "see you",
    "later", "gtg", "exit", "quit"
}

QUESTION_WORDS = {"what", "why", "how", "when", "where", "who"}
ACTION_WORDS = {"recommend", "explain", "tell", "show", "find"}


# init connection mysql
conn = st.connection('mysql', type='sql')

# # Import fine tuned LLM [Try with another devices with GPU > 15Gb :)]
# @st.cache_resource
# def load_fine_tuned_model():
#     model = AutoPeftModelForCausalLM.from_pretrained(
#         "manglish_model",
#         load_in_4bit = True,
#     )
#     tokenizer = AutoTokenizer.from_pretrained("manglish_model")
#     return model, tokenizer

# model, tokenizer = load_fine_tuned_model()

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
    You are a helpful assistance.
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

    Question: 
    {input}
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

# A function to create user record
# Through ... ,:a/:b/:c (bind params) can be random alphabet(?)
def ensure_user_exists(user_id):
    queryChecker = "SELECT * FROM user WHERE user_id = :id" #check if user exist
    result = conn.query(queryChecker, params={"id": user_id}, ttl=0)
    if result.empty:
        query = text("INSERT INTO user (user_id, chat_summary) VALUES (:id, :summ)")
        with conn.session as session:
            session.execute(
                query,
                {
                    "id": user_id, 
                    "summ": ""
                }
            )
            session.commit()
# A function to get previous summary on user id
def fetch_chat_summary(user_id):
    query = "SELECT chat_summary FROM user WHERE user_id = :id"

    # return a dataframe
    result = conn.query(query, params={"id": user_id}, ttl=0)

    if not result.empty:
        summary = result.iloc[0]["chat_summary"]
        return summary if summary else ""
    
    return ""
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
def save_summary(user_id, session_messages, previous_chat_summary):
    session_summary = generate_session_summary(session_messages)
    summary = previous_chat_summary + session_summary
    query = text("UPDATE user SET chat_summary = :summ, last_seen = NOW() WHERE user_id = :id")

    with conn.session as session:
        print("Debug: Saving now")
        session.execute(query,
            {
                "summ": summary,
                "id": user_id
            }
        )
        session.commit()
# Use last 6 messages
def format_chat_history(messages, limit=6):
    history = messages[-limit:]
    return "\n".join(
        f"{m['role'].upper()}: {m['content']}"
        for m in history
        if m["role"] in ("user", "assistant")
    )
### Similarity Search ###
SIMILARITY_THRESHOLD = 0.8 # 0 -- 1  || irrelevant -- relevant
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
        "Authorization": f"Bearer {os.getenv("JIRA_AI_API_KEY")}",
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

def manglish_response(context):
    url = "https://tabatha-bribeable-gena.ngrok-free.dev/generate"

    payload = {
        "query": context,
    }

    try:
        resp = requests.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        return True, resp.json()["response"]
    except requests.exceptions.RequestException as e:
        return False, f"Request failed: {e}"
    
def keyword_intent(text):
    t = text.lower()
    if any(k in t for k in GREETING_KW):
        return "greeting"
    if any(k in t for k in LEAVING_KW):
        return "bye"
    return None

def heuristic_intent(text):
    words = text.lower().split()

    # Extremely short + no intent words → greeting
    if len(words) <= 3:
        if not any(w in QUESTION_WORDS for w in words):
            if not any(w in ACTION_WORDS for w in words):
                return "greeting"

    return None

def llm_intent_fallback(text):
    prompt = f"""
    Classify the message into ONE label:
    - greeting
    - goodbye
    - question

    Message: "{text}"
    Only output the label.
    """
    return llm.invoke(prompt).strip().lower()

def detect_intent(text):
    for fn in [keyword_intent, heuristic_intent]:
        intent = fn(text)
        if intent:
            return intent

    # last resort
    return llm_intent_fallback(text)


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
    used_reponse = ""
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
        
        intent = detect_intent(prompt)
        #If greeting no need to process(?)
        if intent == "greeting":
            used_reponse = llm.invoke(
                "Reply naturally in Manglish to a greeting."
            )
        elif intent == "goodbye":
            used_reponse = llm.invoke(
                "Reply naturally in Manglish to say goodbye."
            )
        else: # question
            # Simplify searching
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
                
                isFlag = False #Defalut assistant_response
                if isinstance(ans_rag, dict):
                    assistant_response = ans_rag.get('answer', str(ans_rag))
                    print("LLM1: ", assistant_response)
                    print("LLM2 on duty!")
                    isFlag, final_response = manglish_response(assistant_response) # if error return false
                else:
                    # If string, use it directly
                    assistant_response = ans_rag
                    print("LLM1: ", assistant_response)
                    print("LLM2 on duty!")
                    isFlag, final_response = manglish_response(assistant_response)
                
                if (isFlag == True): # true: final_response | false: assistant_response
                    used_response = final_response
                    print("LLM2: ", final_response)
                else:
                    used_response = assistant_response

        # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        for chunk in used_response.split():
            used_response += chunk + " "
            time.sleep(0.05)
            message_placeholder.markdown(used_response + "▌")

        message_placeholder.markdown(used_response)
        st.session_state.messages.append({"role": "assistant", "content": used_response})

    print("End of process, now backend")
    new_messages = st.session_state.messages[st.session_state.last_summarized_len:]
    time_passed = Messagenow - st.session_state.last_summary_time

    if time_passed >= timedelta(minutes=1) and len(new_messages) > 0:
        print("Conditions met! Calling save_summary...")
        save_summary(user_id, new_messages, chat_summary)
        st.session_state.last_summary_time = Messagenow
        st.session_state.last_summarized_len = len(st.session_state.messages)
    else:
        print("Conditions NOT met. Skipping summary.")