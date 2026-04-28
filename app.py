#major project on chatbot

import streamlit as st
import time
import csv
import os
import json
from datetime import datetime
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq

st.set_page_config(
    page_title="VidhaanBot — Indian Legal Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

INDEX_DIR     = "faiss_index"
EMBED_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL     = "llama-3.3-70b-versatile"
TOP_K         = 4
FEEDBACK_FILE = "feedback_log.csv"

ANSWER_PROMPTS = {
    "English": """You are VidhaanBot, a friendly and knowledgeable Indian legal assistant.

RULES:
1. For greetings, small talk, or casual messages (hi, hello, how are you, thanks etc.) — respond naturally and warmly in 1-2 sentences. Do NOT mention documents.
2. For legal questions — search the provided context. If found, answer from context and cite the Article/Section/Part.
3. If a legal question is NOT in the context — say you couldn't find it in the indexed documents, then give a helpful general answer from your knowledge of Indian law, clearly labelled as general knowledge.
4. Match answer length to question complexity. Simple = short. Detailed = structured with bullet points.
5. Use conversation history for follow-up questions.
6. Give ONLY the answer. No suggestions or extra lines at the end.""",

    "Hindi": """आप VidhaanBot हैं, एक मित्रवत और जानकार भारतीय कानूनी सहायक।

नियम:
1. अभिवादन या सामान्य बातचीत के लिए — स्वाभाविक और मित्रवत उत्तर दें। दस्तावेज़ों का उल्लेख न करें।
2. कानूनी प्रश्नों के लिए — पहले दिए गए संदर्भ में खोजें। मिले तो उत्तर दें और अनुच्छेद/धारा बताएं।
3. यदि संदर्भ में नहीं मिला — बताएं कि दस्तावेज़ों में नहीं मिला, फिर सामान्य ज्ञान से उत्तर दें।
4. उत्तर की लंबाई प्रश्न की जटिलता के अनुसार रखें।
5. बातचीत के इतिहास का उपयोग करें।
6. केवल उत्तर दें। अंत में कोई सुझाव न जोड़ें।"""
}

SUGGESTION_PROMPTS = {
    "English": """Based on this legal question and answer, generate exactly 3 short follow-up questions.
Return ONLY a JSON array of 3 strings, nothing else. No explanation, no numbering.
Example: ["What are the exceptions?", "How is this enforced?", "What does Article 22 say?"]""",

    "Hindi": """इस कानूनी प्रश्न और उत्तर के आधार पर, ठीक 3 छोटे अनुवर्ती प्रश्न बनाएं।
केवल 3 स्ट्रिंग का JSON array लौटाएं, कुछ और नहीं।
उदाहरण: ["अपवाद क्या हैं?", "यह कैसे लागू होता है?", "अनुच्छेद 22 क्या कहता है?"]"""
}

# ── LOAD RESOURCES ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"}
    )
    return FAISS.load_local(
        INDEX_DIR, embeddings,
        allow_dangerous_deserialization=True
    )

@st.cache_resource
def load_groq_client():
    return Groq(api_key=st.secrets["GROQ_API_KEY"])

# ── FEEDBACK ──────────────────────────────────────────────────────────────────
def log_feedback(question, answer, rating, language):
    file_exists = os.path.exists(FEEDBACK_FILE)
    with open(FEEDBACK_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "language", "question", "answer", "rating"])
        writer.writerow([
            datetime.now().isoformat(), language,
            question[:300], answer[:500], rating
        ])

# ── STREAM ANSWER ─────────────────────────────────────────────────────────────
def stream_answer(question, vectorstore, client, chat_history, language):
    docs = vectorstore.similarity_search(question, k=TOP_K)
    context = "\n\n---\n\n".join([doc.page_content for doc in docs])

    messages = [{"role": "system", "content": ANSWER_PROMPTS[language]}]
    for msg in chat_history[-6:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({
        "role": "user",
        "content": f"Context from legal documents:\n\n{context}\n\nQuestion: {question}"
    })

    stream = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        max_tokens=2048,
        temperature=0.4,
        stream=True,
    )
    for chunk in stream:
        token = chunk.choices[0].delta.content
        if token:
            time.sleep(0.035)
            yield token

# ── FETCH SUGGESTIONS ─────────────────────────────────────────────────────────
def fetch_suggestions(question, answer, client, language):
    try:
        prompt = f"Question: {question}\n\nAnswer: {answer[:400]}"
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": SUGGESTION_PROMPTS[language]},
                {"role": "user", "content": prompt}
            ],
            max_tokens=120,
            temperature=0.5,
            stream=False,
        )
        raw = resp.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        suggestions = json.loads(raw)
        if isinstance(suggestions, list):
            return [str(s).strip() for s in suggestions[:3]]
    except Exception:
        pass
    return []

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=DM+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.block-container { padding-top: 1.5rem !important; }
.hero-wrap {
    background: linear-gradient(135deg, #0f1923 0%, #1a2942 50%, #0f1923 100%);
    border: 1px solid rgba(196,160,90,0.25); border-radius: 16px;
    padding: 2.4rem 2.8rem 2rem; margin-bottom: 1.6rem;
    position: relative; overflow: hidden;
}
.hero-wrap::before {
    content: "⚖"; position: absolute; right: -20px; top: -30px;
    font-size: 180px; opacity: 0.04; line-height: 1; pointer-events: none;
}
.hero-title { font-family: 'Playfair Display', serif; font-size: 2.4rem;
    font-weight: 700; color: #e8d5a3; margin: 0 0 0.2rem 0; letter-spacing: -0.5px; }
.hero-sub { font-size: 0.95rem; color: rgba(232,213,163,0.6); font-weight: 300;
    letter-spacing: 3px; text-transform: uppercase; margin-bottom: 1rem; }
.hero-desc { font-size: 0.97rem; color: rgba(255,255,255,0.72); line-height: 1.7;
    max-width: 680px; margin-bottom: 1.4rem; }
.badge-row { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 0.2rem; }
.badge { background: rgba(196,160,90,0.12); border: 1px solid rgba(196,160,90,0.3);
    color: #c4a05a; border-radius: 20px; padding: 4px 14px;
    font-size: 0.78rem; font-weight: 500; letter-spacing: 0.5px; }
.stat-row { display: flex; gap: 12px; margin-bottom: 1.6rem; flex-wrap: wrap; }
.stat-card { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px; padding: 14px 22px; flex: 1; min-width: 130px; text-align: center; }
.stat-num { font-family: 'Playfair Display', serif; font-size: 1.6rem;
    color: #c4a05a; font-weight: 600; display: block; }
.stat-label { font-size: 0.75rem; color: rgba(255,255,255,0.45);
    text-transform: uppercase; letter-spacing: 1px; }
.disclaimer { background: rgba(196,160,90,0.07); border-left: 3px solid #c4a05a;
    border-radius: 0 8px 8px 0; padding: 10px 16px; font-size: 0.82rem;
    color: rgba(255,255,255,0.55); margin-bottom: 1.6rem; }
div[data-testid="column"] .stButton > button {
    background: rgba(196,160,90,0.08) !important;
    border: 1px solid rgba(196,160,90,0.25) !important;
    color: #c4a05a !important; border-radius: 20px !important;
    font-size: 0.8rem !important; padding: 6px 14px !important;
    transition: all 0.2s !important; white-space: normal !important;
    height: auto !important; text-align: left !important; }
div[data-testid="column"] .stButton > button:hover {
    background: rgba(196,160,90,0.18) !important;
    border-color: rgba(196,160,90,0.5) !important; }
[data-testid="stChatMessage"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 12px !important; margin-bottom: 8px !important; }
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.07) !important; border-radius: 8px !important; }
[data-testid="stSidebar"] { background: #0d1520 !important;
    border-right: 1px solid rgba(196,160,90,0.15) !important; }
[data-testid="stSidebar"] .stButton > button {
    background: rgba(196,160,90,0.07) !important;
    border: 1px solid rgba(196,160,90,0.2) !important;
    color: rgba(255,255,255,0.75) !important; border-radius: 8px !important;
    font-size: 0.85rem !important; transition: all 0.2s !important; }
[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(196,160,90,0.16) !important; color: #c4a05a !important; }
[data-testid="stChatInput"] textarea {
    border: 1px solid rgba(196,160,90,0.3) !important;
    border-radius: 12px !important; background: rgba(255,255,255,0.04) !important; }
[data-testid="stRadio"] label { font-size: 0.85rem !important; }
hr { border-color: rgba(196,160,90,0.15) !important; }
</style>
""", unsafe_allow_html=True)

# ── HERO ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
  <div class="hero-sub">Major Project · AI & Law</div>
  <div class="hero-title">VidhaanBot</div>
  <div class="hero-desc">
    An AI-powered legal research assistant trained on the Indian Constitution,
    IPC, CrPC, and other foundational legal texts. Ask any question in English or Hindi
    and get precise, cited answers grounded entirely in the source documents —
    no hallucinations, no opinions, just the law as written.
  </div>
  <div class="badge-row">
    <span class="badge">📜 Indian Constitution</span>
    <span class="badge">🔍 Consumer Protection</span>
    <span class="badge">🙋🏻‍♀️ Women Safety</span>
    <span class="badge">🇮🇳 English + Hindi</span>
    <span class="badge">🦹🏻 IPC (Indian Penal Code)</span>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="stat-row">
  <div class="stat-card"><span class="stat-num">448+</span><span class="stat-label">Pages Indexed</span></div>
  <div class="stat-card"><span class="stat-num">1800+</span><span class="stat-label">Knowledge Chunks</span></div>
  <div class="stat-card"><span class="stat-num">2</span><span class="stat-label">Languages</span></div>
  <div class="stat-card"><span class="stat-num">70B</span><span class="stat-label">Parameter Model</span></div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer">
  ⚠️ <strong>Disclaimer:</strong> VidhaanBot is an academic research tool.
  Answers are generated from indexed documents and are not a substitute for professional legal advice.
  Always consult a qualified advocate for legal matters.
</div>
""", unsafe_allow_html=True)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:0.4rem 0 1rem'>
      <div style='font-family:Playfair Display,serif;font-size:1.3rem;
                  color:#c4a05a;font-weight:600;margin-bottom:4px'>VidhaanBot</div>
      <div style='font-size:0.72rem;color:rgba(255,255,255,0.35);
                  letter-spacing:2px;text-transform:uppercase'>Indian Legal Assistant</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**🌐 Language**")
    language = st.radio(
        "Language", ["English", "Hindi"],
        horizontal=True, key="language", label_visibility="collapsed"
    )

    st.divider()
    st.markdown("<div style='font-size:0.8rem;color:rgba(255,255,255,0.45);letter-spacing:1px;text-transform:uppercase;margin-bottom:8px'>Quick Topics</div>", unsafe_allow_html=True)
    quick_topics = [
        ("📜", "Fundamental Rights"), ("🛡️", "Article 21A"),
        ("🇮🇳", "Preamble"),          ("⚖️", "Directive Principles"),
        ("📚", "Right to Education"), ("🗺️", "Article 370"),
        ("🚨", "Emergency Provisions"),
    ]
    for icon, topic in quick_topics:
        if st.button(f"{icon}  {topic}", use_container_width=True):
            st.session_state["pending_question"] = f"Tell me about {topic}"
            st.rerun()

    st.divider()
    st.markdown("""
    <div style='font-size:0.75rem;color:rgba(255,255,255,0.3);line-height:1.8'>
      <div style='margin-bottom:4px'>🏗️ <strong style='color:rgba(255,255,255,0.5)'>Architecture</strong></div>
      <div>• RAG (Retrieval-Augmented Generation)</div>
      <div>• FAISS vector database</div>
      <div>• sentence-transformers embeddings</div>
      <div>• Llama 3.3 70B via Groq</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    if st.button("🗑️  Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.suggestions = []
        st.rerun()

    st.markdown(f"""
    <div style='font-size:0.7rem;color:rgba(255,255,255,0.2);margin-top:12px;line-height:1.9'>
      <div>Model: {LLM_MODEL}</div>
      <div>Built with LangChain · FAISS · Groq</div>
      <div style='margin-top:6px;color:rgba(196,160,90,0.4)'>Major Project — 2025-26</div>
    </div>
    """, unsafe_allow_html=True)

# ── LOAD ──────────────────────────────────────────────────────────────────────
with st.spinner("Loading knowledge base..."):
    try:
        vectorstore = load_vectorstore()
        client = load_groq_client()
    except Exception as e:
        st.error(f"Startup error: {e}")
        st.stop()

# ── SESSION STATE ─────────────────────────────────────────────────────────────
if "messages"      not in st.session_state: st.session_state.messages      = []
if "suggestions"   not in st.session_state: st.session_state.suggestions   = []
st.session_state.last_streamed = -1  # reset every run — never skip history

# ── CHAT HISTORY ──────────────────────────────────────────────────────────────
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            sources = msg.get("sources", [])
            if sources:
                with st.expander("📎 Sources", expanded=False):
                    for j, src in enumerate(sources):
                        st.markdown(
                            f"<span style='background:rgba(255,255,255,0.07);"
                            f"border:1px solid rgba(255,255,255,0.15);"
                            f"border-radius:20px;padding:2px 10px;"
                            f"font-size:11px;margin-right:4px;display:inline-block'>"
                            f"📄 {src['file']} · p{src['page']}</span>",
                            unsafe_allow_html=True
                        )
                    st.write("")
                    for j, src in enumerate(sources):
                        st.caption(f"**{src['file']}** · page {src['page']}")
                        st.text(src["text"] + "...")
                        if j < len(sources) - 1:
                            st.divider()
            col1, col2, col3 = st.columns([1, 1, 8])
            with col1:
                if st.button("👍", key=f"up_{i}"):
                    user_q = st.session_state.messages[i-1]["content"] if i > 0 else ""
                    log_feedback(user_q, msg["content"], "positive", language)
                    st.toast("Thanks for the feedback!", icon="✅")
            with col2:
                if st.button("👎", key=f"down_{i}"):
                    user_q = st.session_state.messages[i-1]["content"] if i > 0 else ""
                    log_feedback(user_q, msg["content"], "negative", language)
                    st.toast("Noted, thank you!", icon="📝")

# ── SUGGESTIONS ───────────────────────────────────────────────────────────────
if st.session_state.suggestions:
    st.markdown("**💡 You might also ask:**")
    cols = st.columns(3)
    for idx, suggestion in enumerate(st.session_state.suggestions):
        with cols[idx]:
            if st.button(suggestion, key=f"sug_{idx}",
                         use_container_width=True, type="secondary"):
                st.session_state["pending_question"] = suggestion
                st.session_state.suggestions = []
                st.rerun()

# ── CHAT INPUT ────────────────────────────────────────────────────────────────
pending_q = st.session_state.pop("pending_question", None)
typed_q   = st.chat_input(
    "कानूनी प्रश्न पूछें..." if language == "Hindi" else "Ask a legal question..."
)
question = typed_q or pending_q

if question:
    st.session_state.suggestions = []
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            answer = st.write_stream(
                stream_answer(question, vectorstore, client,
                              st.session_state.messages, language)
            )
            # Single append with sources
            docs = vectorstore.similarity_search(question, k=TOP_K)
            sources = [
                {
                    "file": doc.metadata.get("source_file", "document"),
                    "page": doc.metadata.get("page", "N/A"),
                    "text": doc.page_content[:250]
                }
                for doc in docs
            ]
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })

        except Exception as e:
            st.error(f"Error: {e}")

    with st.spinner("Getting suggestions..."):
        suggestions = fetch_suggestions(question, answer, client, language)
        st.session_state.suggestions = suggestions

    st.rerun()