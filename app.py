import os
import json
from pathlib import Path

import faiss
import fitz
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from google import genai

load_dotenv()

BASE = Path(__file__).parent
DATA = BASE / "data"
PDF_PATH = DATA / "HT67F4892v120(4).pdf"
INDEX_PATH = DATA / "faiss.index"
META_PATH = DATA / "faiss_metadata.json"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LLM = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

st.set_page_config(
    page_title="HT67F4892 Datasheet Expert",
    page_icon="",
    layout="wide"
)

SYSTEM_PROMPT = """You are a datasheet expert for the Holtek HT67F4892 MCU.

Your knowledge source for this answer is ONLY the retrieved context from the
uploaded HT67F4892 Rev. V1.20 datasheet.

Rules:
1. Answer only what is supported by the supplied datasheet context.
2. Do not invent register names, bit meanings, values, electrical limits,
   timing values, or code details.
3. Explain clearly and simply first, then give technical details when useful.
4. Preserve exact register names, bit names, formulas and numerical values.
5. Always include a "References" section with the PDF page number(s) used.
6. Use the PDF page number supplied in the context metadata, not a guessed
   page number.
7. If the retrieved context does not contain enough information, say:
   "I couldn't find enough information in the provided HT67F4892 datasheet
   to answer this reliably." Then mention what information is missing.
8. If the user asks for code, generate code only from the retrieved datasheet
   information. State any assumptions explicitly.
9. Do not use outside knowledge to fill missing datasheet information.
10. If multiple retrieved passages disagree, do not silently choose one;
    explain the conflict and cite both pages.

Answer format:
### Explanation
...

### Technical details
...

### References
- Page X — relevant section
- Page Y — relevant section
"""

@st.cache_resource(show_spinner="Loading embedding model...")
def get_embedder():
    return SentenceTransformer(EMBED_MODEL)

@st.cache_resource(show_spinner="Loading vector index...")
def get_index():
    if not INDEX_PATH.exists() or not META_PATH.exists():
        return None, None
    index = faiss.read_index(str(INDEX_PATH))
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    return index, meta

@st.cache_resource(show_spinner="Opening datasheet...")
def get_pdf():
    return fitz.open(str(PDF_PATH))

def retrieve(query, top_k=6):
    index, meta = get_index()
    if index is None:
        return []

    model = get_embedder()
    q = model.encode([query], normalize_embeddings=True)
    q = np.asarray(q, dtype="float32")
    scores, ids = index.search(q, top_k)

    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0:
            continue
        item = meta["chunks"][int(idx)]
        results.append({
            "score": float(score),
            "page": int(item["page"]),
            "id": item["id"],
            "text": item["text"]
        })
    return results

def build_context(results):
    blocks = []
    for i, r in enumerate(results, 1):
        blocks.append(
            f"[SOURCE {i} | PDF PAGE {r['page']} | SCORE {r['score']:.4f}]\n"
            f"{r['text']}"
        )
    return "\n\n---\n\n".join(blocks)

def ask_llm(question, results, history):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured. Add it to .env and restart Streamlit."
        )

    client = genai.Client(api_key=api_key)
    context = build_context(results)

    history_text = []
    for msg in history[-6:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_text.append(f"{role}: {msg['content']}")
    conversation = "\n\n".join(history_text)

    user_prompt = f"""You are answering a question about the Holtek HT67F4892 MCU using ONLY the retrieved context from its uploaded datasheet.

DATASHEET CONTEXT:
{context}

RECENT CONVERSATION:
{conversation}

CURRENT USER QUESTION:
{question}

Rules:
1. Answer only what is supported by the supplied datasheet context.
2. Do not invent register names, bit meanings, values, electrical limits, timing values, or code details.
3. Explain clearly and simply first, then give technical details when useful.
4. Preserve exact register names, bit names, formulas and numerical values.
5. Always include a References section with the PDF page number(s) used.
6. Use the PDF page number supplied in SOURCE metadata; never guess it.
7. If the context is insufficient, say: I couldn't find enough information in the provided HT67F4892 datasheet to answer this reliably.
8. If code is requested, generate it only from the retrieved datasheet information and state assumptions.
9. Do not use outside knowledge to fill missing datasheet information.
10. If sources conflict, explain the conflict and cite both pages.

Answer format:
### Explanation
...

### Technical details
...

### References
- Page X — relevant section
- Page Y — relevant section
"""

    response = client.models.generate_content(
        model=os.getenv("GEMINI_MODEL", DEFAULT_LLM),
        contents=user_prompt,
        config={"temperature": 0.1},
    )
    return response.text

def page_preview(page_no):
    doc = get_pdf()
    if page_no < 1 or page_no > len(doc):
        return None
    page = doc[page_no - 1]
    pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
    return pix.tobytes("png")

def answer_question(question):
    results = retrieve(question, top_k=6)
    if not results:
        return None, []
    return ask_llm(question, results, st.session_state.messages), results

# ---------- UI ----------
st.title(" HT67F4892 Datasheet Expert")
st.caption("Ask questions about the internally indexed HT67F4892 Rev. V1.20 datasheet.")

with st.sidebar:
    st.header("Knowledge Base")
    st.write("**Device:** HT67F4892")
    st.write("**Revision:** V1.20")
    st.write("**Datasheet:** 182 pages")
    st.write("**Indexed chunks:** page-aware")

    if not INDEX_PATH.exists():
        st.error("Vector index not found.")
        st.code("python build_index.py", language="bash")

    st.divider()
    st.header("Settings")
    top_k = st.slider("Retrieved passages", 3, 10, 6)
    st.caption("Lower values are faster; higher values can help broad questions.")

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render previous conversation.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            st.markdown("**Retrieved source pages**")
            cols = st.columns(min(len(message["sources"]), 4))
            for i, page in enumerate(message["sources"][:4]):
                with cols[i]:
                    if st.button(f" Page {page}", key=f"hist_{message['id']}_{page}"):
                        st.session_state["preview_page"] = page

question = st.chat_input(
    "Ask a doubt about the HT67F4892 datasheet..."
)

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the datasheet and preparing the explanation..."):
            try:
                results = retrieve(question, top_k=top_k)
                if not results:
                    answer = (
                        "I couldn't find enough information in the indexed "
                        "HT67F4892 datasheet to answer this reliably."
                    )
                else:
                    answer = ask_llm(
                        question,
                        results,
                        st.session_state.messages[:-1]
                    )

                st.markdown(answer)

                pages = sorted(set(r["page"] for r in results))
                if pages:
                    st.markdown("**Retrieved source pages**")
                    cols = st.columns(min(len(pages), 4))
                    for i, page in enumerate(pages[:4]):
                        with cols[i]:
                            if st.button(
                                f" Page {page}",
                                key=f"now_{len(st.session_state.messages)}_{page}"
                            ):
                                st.session_state["preview_page"] = page

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": pages,
                    "id": len(st.session_state.messages)
                })
            except Exception as e:
                error_text = f"**Error:** {e}"
                st.error(error_text)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_text,
                    "sources": [],
                    "id": len(st.session_state.messages)
                })

# Page preview panel.
preview = st.session_state.get("preview_page")
if preview:
    st.divider()
    st.subheader(f" Datasheet Page {preview}")
    img = page_preview(preview)
    if img:
        st.image(img, caption=f"Original PDF page {preview}", width="stretch")
    if st.button("Close page preview"):
        st.session_state.pop("preview_page", None)
        st.rerun()
