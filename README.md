# AI Customer Support Agent

This repository contains a beginner-friendly Customer Support AI agent built using CrewAI, Streamlit, and Gemini.

## Features
- **Knowledge Base Search**: Uses FAISS embeddings to answer general company policy questions.
- **Order Database**: Reads from an Excel file (`orders.xlsx`) to retrieve dynamic order statuses.
- **Human Escalation**: Seamlessly creates escalation summaries and logs them in a pending dashboard if the AI cannot resolve the request.
- **Contextual Memory**: Remembers previous messages during the chat session.

## Files Needed for Deployment
Before deploying to Streamlit Cloud, ensure your repository has:
1. `app.py`
2. `tools.py`
3. `requirements.txt`
4. `faiss.index` (Your pre-built vector database)
5. `chunks.json` (Your chunk text metadata)
6. `orders.xlsx` (Your Excel database with Order details)

## Deployment on Streamlit Cloud
1. Upload this folder's contents to a GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io/) and connect your GitHub account.
3. Deploy the repository pointing to `app.py`.
4. In your Streamlit app dashboard, go to **Settings > Secrets** and paste your API key:
   ```toml
   GOOGLE_API_KEY = "your_actual_gemini_api_key_here"
   ```
5. Your app is now live!
