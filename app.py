"""
app.py — Hosted Legal Chatbot (Streamlit Cloud)
RAG + FAISS + Groq + Streaming + Conversation Memory
"""

import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Indian Legal Assistant",
    page_icon="⚖️",
    layout="centered"
)

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

INDEX_DIR   = "faiss_index"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL   = "llama3.2"
TOP_K       = 4

SYSTEM_PROMPT = """
You are a knowledgeable Indian legal assistant.

Answer strictly using the provided legal context.

Rules:
1. Use conversation history for follow-up questions.
2. If answer is not in context say:
   "I could not find this in the provided documents."
3. Cite Article, Section or Part whenever possible.
4. Explain in simple language.
5. If user asks follow-up like:
   "Explain more"
   "Give example"
   "What are exceptions"
Use previous chat history.
"""

# --------------------------------------------------
# LOAD RESOURCES
# --------------------------------------------------

@st.cache_resource
def load_vectorstore():

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device":"cpu"}
    )

    return FAISS.load_local(
        INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )


@st.cache_resource
def load_groq_client():

    return Groq(
        api_key=st.secrets["GROQ_API_KEY"]
    )


# --------------------------------------------------
# STREAM RESPONSE
# --------------------------------------------------

def stream_answer(question, vectorstore, client, chat_history):

    # Retrieve relevant chunks
    docs = vectorstore.similarity_search(
        question,
        k=TOP_K
    )

    context = "\n\n---\n\n".join(
        [d.page_content for d in docs]
    )

    # Build messages
    messages = [
        {
            "role":"system",
            "content":SYSTEM_PROMPT
        }
    ]

    # Add last 3 exchanges
    for msg in chat_history[-6:]:
        messages.append({
            "role":msg["role"],
            "content":msg["content"]
        })

    # Add new question with retrieved context
    messages.append({
        "role":"user",
        "content":
        f"""Context from legal documents:

{context}

Question:
{question}
"""
    })

    # Groq streaming
    stream = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0.3,
        max_tokens=600,
        stream=True
    )

    for chunk in stream:
        try:
            token = chunk.choices[0].delta.content
            if token:
                yield token
        except:
            continue


# --------------------------------------------------
# UI
# --------------------------------------------------

st.title("⚖️ Indian Legal Assistant")
st.caption(
    "RAG + Groq + Conversation Memory + Streaming"
)

with st.spinner("Loading knowledge base..."):

    try:
        vectorstore = load_vectorstore()
        client = load_groq_client()

        st.success(
            "Ready! Powered by Llama 3 via Groq",
            icon="✅"
        )

    except Exception as e:
        st.error(f"Startup error: {e}")
        st.stop()


# Session chat memory
if "messages" not in st.session_state:
    st.session_state.messages = []


# Show old messages
for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])



# --------------------------------------------------
# CHAT INPUT
# --------------------------------------------------

if question := st.chat_input(
    "Ask a legal question..."
):

    # show user msg
    st.session_state.messages.append(
        {
            "role":"user",
            "content":question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)


    # assistant response
    with st.chat_message("assistant"):

        try:

            # Stream token by token
            full_answer = st.write_stream(
                stream_answer(
                    question,
                    vectorstore,
                    client,
                    st.session_state.messages
                )
            )

            # Save assistant answer
            st.session_state.messages.append(
                {
                    "role":"assistant",
                    "content":full_answer
                }
            )


            # show source chunks
            docs = vectorstore.similarity_search(
                question,
                k=TOP_K
            )

            with st.expander(
                "📄 Source passages used"
            ):

                for i,doc in enumerate(docs,1):

                    st.markdown(
                        f"**Passage {i}** — Page {doc.metadata.get('page','N/A')}"
                    )

                    st.caption(
                        doc.page_content[:300] + "..."
                    )

                    st.divider()

        except Exception as e:
            st.error(
                f"Error: {e}"
            )


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.header("⚖️ About")

    st.markdown(f"""
**Model:** {LLM_MODEL}

Features:
- Retrieval Augmented Generation
- Streaming responses
- Follow-up question memory
- Source citations

Examples:
- What is Article 21?
- What are Fundamental Rights?
- Explain Article 14
- What are exceptions?
- Give example of this
""")

    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    st.caption(
        "LangChain · FAISS · Groq · Streamlit"
    )