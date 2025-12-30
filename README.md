A LLM with Manglish

Current:
- Langchain for RAG
- Pinecone for VectorDB
- OpenAI for LLM
- Simple prompt engineering for Manglish response

Planning:
1. Fine-tuning for Manglish
2. Collect data/ Web Scrapping for knowledge base and latest answer
3. Postgres for User Preference, Chat Summary
4. Session for chat history
5. Cookie for UUID

Overall:
1. No login required (Anonymous User)
2. One chatroom for one user
3. New user record with UUID saved to DB
4. Close browser = Close session and save chat summary and user preference to DB, clear chat history
5. Clear cookie = Remove user from DB
