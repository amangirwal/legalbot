# ⚖️ Indian Legal Chatbot — Setup Guide

## Project Structure
```
legal-chatbot/
├── app.py                    ← Streamlit chat app (deploy this)
├── ingest.py                 ← Run once locally to build index
├── requirements.txt          ← Python dependencies
├── .gitignore
├── .streamlit/
│   └── secrets.toml          ← Local secrets (never commit!)
├── data/
│   └── constitution.pdf      ← Your PDF goes here
└── faiss_index/              ← Auto-created by ingest.py, commit this!
    ├── index.faiss
    └── index.pkl
```

---

## STEP 1 — Set up Python environment on your Mac

```bash
# Open Terminal and run these commands one by one:

# 1. Install Python 3.11 via Homebrew
brew install python@3.11

# 2. Create project folder
mkdir legal-chatbot
cd legal-chatbot

# 3. Create a virtual environment
python3.11 -m venv venv
source venv/bin/activate

# 4. Install all dependencies
pip install langchain==0.2.16 langchain-community==0.2.17 \
            faiss-cpu==1.8.0 sentence-transformers==3.1.1 \
            huggingface-hub==0.25.1 pymupdf==1.24.11 streamlit==1.39.0
```

---

## STEP 2 — Add your PDF

```bash
mkdir data
# Copy your PDF into the data/ folder and name it constitution.pdf
# Or change PDF_PATH in ingest.py to match your filename
```

---

## STEP 3 — Run ingestion (builds the FAISS index)

```bash
# Make sure you're in the legal-chatbot/ folder with venv active
python ingest.py

# You'll see output like:
# 📄 Loading PDF... Loaded 448 pages.
# ✂️  Splitting into chunks... Created 1823 chunks.
# 🔢 Creating embeddings...
# ✅ Done! Index saved to ./faiss_index/
```

---

## STEP 4 — Get a FREE Hugging Face token

1. Go to https://huggingface.co and create a free account
2. Go to Settings → Access Tokens → New Token
3. Give it a name, select "Read" role, click Create
4. Copy the token (starts with `hf_...`)

---

## STEP 5 — Test locally

```bash
# Create the secrets file for local testing
mkdir .streamlit
echo 'HF_TOKEN = "hf_your_token_here"' > .streamlit/secrets.toml
# Replace hf_your_token_here with your actual token

# Run the app
streamlit run app.py
# Opens at http://localhost:8501
```

---

## STEP 6 — Push to GitHub

```bash
# Initialize git repo
git init
git add .
git commit -m "Initial commit — legal chatbot"

# Create a new repo on github.com, then:
git remote add origin https://github.com/YOUR_USERNAME/legal-chatbot.git
git push -u origin main
```

> Make sure faiss_index/ is committed — it's your knowledge base!
> Make sure .streamlit/secrets.toml is NOT committed (it's in .gitignore)

---

## STEP 7 — Deploy on Streamlit Cloud

1. Go to https://share.streamlit.io
2. Sign in with GitHub
3. Click "New app" → select your repo → set main file to `app.py`
4. Click "Advanced settings" → Secrets → paste:
   ```
   HF_TOKEN = "hf_your_actual_token_here"
   ```
5. Click Deploy!

Your app will be live at `https://your-app-name.streamlit.app` 🎉

---

## Adding more PDFs later

```bash
# Add new PDFs to data/ folder, then re-run:
python ingest.py
# Commit the updated faiss_index/ to GitHub
# Streamlit Cloud will auto-redeploy
```
