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

#Regular Expression
import re

#Image 
import base64

# Load Credential
load_dotenv()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)

# Greeting
GREETING_KW = {
    "hi", "hello", "hey", "yo", "sup", "morning",
    "evening", "afternoon", "hai", "helo", "hello ah"
}
# Goodbye
LEAVING_KW = {
    "bye", "byebye", "goodbye", "see you",
    "later", "gtg", "exit", "quit"
}
# Ok/ Oh/ Eh/...
ACK_PATTERNS = [
    r"^ok+$",        
    r"^ok\s*ok+$",   
    r"^oh+$",        
    r"^ah+$",        
    r"^eh+$",
    r"^mm+$",
    r"^hmm+$",
]

# init connection mysql
conn = st.connection('mysql', type='sql')

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
#You are an AI assistant that communicates in a Malaysian-style tone: casual, slightly local (Manglish), but still clear and professional.
system_prompt = SystemMessagePromptTemplate.from_template('''
    You are a helpful assistance.
    Always give accurate answers and logical reasoning.
    No need to ask user question unless require more information to search.
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

# Combine them
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
        session.execute(query,
            {
                "summ": summary,
                "id": user_id
            }
        )
        session.commit()
# Use last 6 messages as previous chat, parse into chatgpt
def format_chat_history(messages, limit=6):
    history = messages[-limit:]
    return "\n".join(
        f"{m['role'].upper()}: {m['content']}"
        for m in history
        if m["role"] in ("user", "assistant")
    )
# Similarity Search
SIMILARITY_THRESHOLD = 0.8 # 0 -- 1  || irrelevant -- relevant
def retrieve_with_score(query, k=3):
    results = vectorstore.similarity_search_with_score(query, k=k)

    good_docs = []
    for doc, score in results:
        if score > SIMILARITY_THRESHOLD:
            good_docs.append(doc)

    return good_docs, results
# Jina API for online searching, it return data from trusted website 
def search_trusted_sources(query):
    url = f"https://s.jina.ai/?q={query}"
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {os.getenv("JIRA_AI_API_KEY")}",
        "X-Respond-With": "no-content"
    }

    response = requests.get(url, headers=headers, timeout=10)
    data = response.json()

    links = []
    for a in data.get("data", [])[:5]: # use 1st 5 links
        links.append(a["url"])
    return links
# extract data one by one from Jina API
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
        return None
# Simplify searching (avoid too much token being used): Optional
def rewrite_query(query):
    prompt = f"""
        Rewrite this question into a short factual search query.
        Remove filler words.
        Keep important nouns.
        
        Question:
        {query}
    """
    return llm.invoke(prompt).content.strip()
# Temp solution: Run fastapi.py with fine tuned LLM in Google Colab, then get manglish response
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
# Detect question
def is_question(text):
    t = text.lower()
    if "?" in t:
        return True
    if t.startswith(("can", "could you", "please", "help", "what", "why", "when", "how", "where")):
        return True
    return False
# See if user is greeting or leaving
def keyword_intent(text):
    t = text.lower()
    for k in GREETING_KW:
        if re.search(rf"\b{k}\b", t):
            return "greeting"
    for k in LEAVING_KW:
        if re.search(rf"\b{k}\b", t):
            return "goodbye"
    return None
# Acknowledgement: Ohh/ Okok
def ack_intent(text: str) -> bool:
    t = text.lower().strip()
    for p in ACK_PATTERNS:
        if re.match(p, t):
            return "ack"
    return None
# Function to detect content; default "others"
def detect_intent(text):
    if is_question(text):
        return "others"
    
    for fn in [ack_intent, keyword_intent]:
        intent = fn(text)
        if intent:
            return intent

    return "others"
#Process image
def img_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()
# UI
st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="images/big-data.png",
)
img = img_to_base64("images/big-data.png")

st.markdown(
    f"""
    <div style="display:flex; justify-content:center; align-items:center; gap:12px;">
        <img src="data:image/png;base64,{img}" width="50">
        <h1 style="margin:0; color:#E6E6FA">Chatbot</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# Set cookie
cookies = EncryptedCookieManager(
    prefix="chatbot",
    password="uknowimjusttrying"
)

if not cookies.ready():
    st.stop()

#check if user_id exist in cookie
if "user_id" not in cookies:
    cookies["user_id"] = str(uuid.uuid4())
    cookies.save()

user_id = cookies["user_id"]
ensure_user_exists(user_id) # store to database(MySQL)

# Fetch chat summary for current user
chat_summary = fetch_chat_summary(user_id)

# Used to check whether to store chat summary
if "last_summary_time" not in st.session_state:
    st.session_state.last_summary_time = datetime.now(
        pytz.timezone("Asia/Kuala_Lumpur")
    )

# If there is new message from user, summarize and store
if "last_summarized_len" not in st.session_state:
    st.session_state.last_summarized_len = 0

# CSS
st.markdown("""
<style>
/*Chat container*/            
.chat-container {
    max-width: 900px;
    margin: auto;
}

/* Chat bubble */
.chat-bubble {
    padding: 12px 16px;
    margin-bottom: 10px;
    width: auto;
    font-size: 16px;
}

/* Assistant */
.assistant {
    display: flex;
    color: #E6E6FA;
    justify-self: start;
}
/*Assitant icon background*/
.st-emotion-cache-z68l0b{
    background-color: #ffffff;
}

/* User */
.user {
    display: flex;
    color: #E6E6FA;
    justify-self: end;
}

/* User chatbot */
.st-emotion-cache-khw9fs{
    order:1;
}
            
/* Body Background color*/
.st-emotion-cache-13k62yr {
    background: #310854;
}

/*User chat bubble background*/
.st-emotion-cache-1mph9ef{
    background-color:#1e0336;
    width:auto;
    max-width:70%;
    align-self:end;
}

/*Bottom bar background color*/     
.st-emotion-cache-1y34ygi{
    background: #9370dB;
}

/* Chat row */
.chat-row {
    display: grid;
}

/* Text input background */
.st-emotion-cache-1353z0o{
	background-color: #6A0DAD;
}
/* Text input */
.st-b1 {
    background-color: #6A0DAD;
}
.st-emotion-cache-h3jt4q{
    background-color: #6A0DAD;
}

.st-emotion-cache-1f3w014{
    color: white;
}
</style>
""", unsafe_allow_html=True)

# Initialize chat history in session
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello World."}]

st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        role_class = "assistant" if message["role"] == "assistant" else "user"
        st.markdown(
            f"""
            <div class="chat-row">
                <div class="chat-bubble {role_class}">
                    {message["content"]}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

st.markdown("</div>", unsafe_allow_html=True)

Messagenow = datetime.now(pytz.timezone('Asia/Kuala_Lumpur'))
# Accept user input
if prompt := st.chat_input("I would like to..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    chat_history = format_chat_history(st.session_state.messages)
    used_response = ""
    # Display user message in chat message container
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    with st.chat_message("user"):
        st.markdown(
            f"""
            <div class="chat-row">
                <div class="chat-bubble user">
                    {prompt}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Check content type (greeting/leaving/ack/others)
        intent = detect_intent(prompt)

        # If greeting/leaving/ack no need to process with full LLM [Add more in future if possible(?)]
        if intent == "greeting":
            response = llm.invoke(
                "Reply naturally in Manglish to a greeting."
            )
            used_response = response.content
        elif intent == "ack":
            used_response = "Yea"
        elif intent == "goodbye":
            response = llm.invoke(
                "Reply naturally in Manglish to say goodbye."
            )
            used_response = response.content
        else: # question
            # Simplify searching if word more than 6
            if len(prompt.split()) > 6:
                search_query = rewrite_query(prompt)
            else:
                search_query = prompt

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

            # Pinecone empty / low confidence → web scraping
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
                        # Debug if info stored


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
                # End of web scraping

            # Defalut assistant_response    
            isFlag = False 

            # Dictionary
            if isinstance(ans_rag, dict):
                assistant_response = ans_rag.get('answer', str(ans_rag))
                print(assistant_response)
                isFlag, final_response = manglish_response(assistant_response) # if error return false
            else:
                # If string, use it directly
                assistant_response = ans_rag
                print(assistant_response)
                isFlag, final_response = manglish_response(assistant_response)

            # true: final_response | false: assistant_response    
            if (isFlag == True):
                used_response = final_response
            else:
                used_response = assistant_response

    # Display assistant response in chat message container
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        streamed_text = ""

        st.markdown(
            f"""
            <div class="chat-row">
                <div class="chat-bubble assistant">
        """, unsafe_allow_html=True)

        for chunk in used_response.split():
            streamed_text += chunk + " "
            time.sleep(0.05)
            message_placeholder.markdown(streamed_text + "▌")
        message_placeholder.markdown(streamed_text)

        st.markdown(
            f""" </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.session_state.messages.append({"role": "assistant", "content": streamed_text})
    st.markdown("</div>", unsafe_allow_html=True)

    #Summarize conversation of user
    new_messages = st.session_state.messages[st.session_state.last_summarized_len:]
    time_passed = Messagenow - st.session_state.last_summary_time

    # If condition met, perform; else skip;
    if time_passed >= timedelta(minutes=5) and len(new_messages) > 0:
        save_summary(user_id, new_messages, chat_summary)
        st.session_state.last_summary_time = Messagenow
        st.session_state.last_summarized_len = len(st.session_state.messages)
    else:
        print("Skip")