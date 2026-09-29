# 🤖 AI Customer Support Agent

A production-ready Customer Support AI agent built with **CrewAI**, **Streamlit**, and **Groq** (`openai/gpt-oss-120b`).

## Features

| Feature | Description |
|---|---|
| 🔍 **Knowledge Base Search** | FAISS + fastembed semantic search over company documents |
| 📦 **Order Lookup** | Reads live order data from an Excel sheet |
| 🚨 **Human Escalation** | Logs escalations to a live sidebar dashboard |
| 🧠 **Contextual Memory** | Maintains full chat history within the session |
| ⚡ **Groq-powered** | Uses `openai/gpt-oss-120b` via Groq's ultra-fast API |

---

## Project Structure

```
├── app.py                        # Main Streamlit app
├── tools.py                      # CrewAI tools (knowledge base, orders, escalation)
├── requirements.txt              # Python dependencies
├── Orders_Tracking_Sheet.xlsx    # Order database
├── .gitignore                    # Excludes secrets & cache from git
├── .streamlit/
│   └── secrets.toml              # ⚠️ Local only — DO NOT commit
└── fiass Indexes/
    ├── faiss.index               # Pre-built FAISS vector index
    ├── chunks.json               # Chunked text + metadata
    ├── build_real_index.py       # Script to rebuild the index
    └── search_test.py            # Index search tester
```

---

## 🚀 Deploy to Streamlit Cloud

### Step 1 — Push to GitHub

Make sure `.streamlit/secrets.toml` is **not** committed (it's in `.gitignore`).

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

### Step 2 — Connect to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io/) and sign in with GitHub.
2. Click **New app** → select your repository.
3. Set **Main file path** to `app.py`.
4. Click **Deploy**.

### Step 3 — Add your Groq API Key (Secrets)

In the Streamlit Cloud dashboard for your app:

1. Click **⋮ (three dots)** → **Settings** → **Secrets**
2. Paste the following and save:

```toml
GROQ_API_KEY = "gsk_...your_groq_api_key_here..."
```

> Get your free API key at [console.groq.com/keys](https://console.groq.com/keys)

### Step 4 — Done! 🎉

Your app will automatically reboot and be live.

---

## 🏃 Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Add your key to .streamlit/secrets.toml
echo 'GROQ_API_KEY = "gsk_..."' > .streamlit/secrets.toml

# Launch
python -m streamlit run app.py
```

---

## ⚙️ Configuration

| Variable | Where to set | Description |
|---|---|---|
| `GROQ_API_KEY` | Streamlit Secrets / `secrets.toml` | Your Groq API key |

### Model

The app uses **`openai/gpt-oss-120b`** via Groq's OpenAI-compatible API endpoint (`https://api.groq.com/openai/v1`). To change the model, edit `MODEL_NAME` in `app.py`.

---

## 🔄 Rebuilding the Knowledge Base

If you update your company documents, rebuild the FAISS index:

```bash
cd "fiass Indexes"
pip install sentence-transformers faiss-cpu
python build_real_index.py
```

Then commit the updated `faiss.index` and `chunks.json` files.
