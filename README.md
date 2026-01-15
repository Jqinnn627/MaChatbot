A Chatbot with Manglish

Current:
- Langchain for RAG
- Pinecone for VectorDB
- OpenAI for LLM
- Simple prompt engineering for Manglish response (temp until manglish model can be hosted 24/7)
- Session for chat history
- Cookie for UUID
- MySQL for Chat Summary
- Web Scrapping for knowledge
- Fine-tuning for Manglish (SFT/ORPO) [Requirement: High GPU needed]
- Collect data/ Web Scrapping for knowledge base and latest answer

Planning:
- None for now

Overall:
1. No login required (Anonymous User)
2. One chatroom for one user
3. New user record with UUID saved to DB
4. The LLM have chat summary memory and chat history(-6)

Requirement:
1. XAMPP for MySQL
2. manglish_model_retuned_3 for Manglish (https://drive.google.com/file/d/1r9U4mX8vVnsW_O192yxlagt94OwX0U5E/view?usp=sharing)
3. put fastapi.py and manglish model in Google Colab(for free GPU)

##### run pip install -r requirements.txt to get dependencies
##### fastapi.py is used to host Manglish LLM, require at least 6GB GPU or else CUDA out of memory

Dataset: 
