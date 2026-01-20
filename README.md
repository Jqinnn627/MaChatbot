# A Chatbot with Manglish
Python env: 3.12

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
##### fastapi.py/HostAPI.ipynb are used to host Manglish LLM, require at least 6GB GPU or else CUDA out of memory

Dataset: https://huggingface.co/datasets/mesolitica/MaLLaM-2.5-Small-Manglish-QA

### Step to host fine-tuned model: (For temp solution :)
1. Create an account in ngrok (https://dashboard.ngrok.com/login), then get access token (Sidebar -> "Your Authtoken) (Once only)
2. Create new notebook in google colab, and copy C:\Users\ms_88\Documents\MaChatbot\fastapi.py content to the notebook (Once only)
3. In top right corner, click arrow down, click "change runtime type" -> "Hardware accelerator" , choose "T4 GPU"
4. Upload the Manglish model (manglish_model_retuned_3.zip) to Google Colab, then run the first code to unzip it.
5. Run 3rd code to move the file to parent dir, then rename the new file to manglish_model_retuned_3.
6. Run all the remaining code(except 1st, 3rd), and wait for a few minutes
7. Done.

### Step to run the AI chatbot:
Requirement: create virtual environment for dependencies (refer to above requirements.txt)
1. Activate the virtual environment (To create: python -m venv venv)
    *Press ctrl + ` for terminal
    a. .\venv\Scripts\activate
2. Run the script
    a. streamlit run main.py
3. Done
    