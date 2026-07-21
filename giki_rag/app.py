import streamlit as st

import config
from chat import answer

st.set_page_config(
    page_title="GIKI Multimodal RAG",
    page_icon="🎓",
    layout="wide"
)

# ------------------------------------------------------------
# Session memory (chat history)
# ------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------

with st.sidebar:
    st.title("⚙️ System Overview")

    st.markdown("""
### Model
- Qwen2.5-VL 3B
- Running locally with Ollama

### Vector Database
- ChromaDB

### Embeddings
- all-MiniLM-L6-v2
- CLIP ViT-B/32

### GIKI Archive
- 3,368 pages scraped
- 26,254 image references
- 7,704 files archived
- ~2.3 GB data
""")

    if st.button("🗑 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# ------------------------------------------------------------
# Header
# ------------------------------------------------------------

st.title("🎓 GIKI Multimodal RAG")

st.caption(
    "Multimodal Retrieval-Augmented Generation over the complete GIKI website archive."
)

# ------------------------------------------------------------
# Render existing chat history
# ------------------------------------------------------------

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        st.markdown(msg["content"])

        if msg["role"] == "assistant":

            if msg.get("sources"):

                with st.expander("📚 Retrieved Sources"):

                    for src in msg["sources"]:

                        title = src.get("title") or "Untitled"

                        st.markdown(
                            f"- **{title}**  \n{src['url']}"
                        )

            if msg.get("images"):

                with st.expander("🖼 Retrieved Images"):

                    for img in msg["images"]:

                        img_path = config.IMAGE_DIR / img["local_path"]

                        if img_path.exists():

                            st.image(
                                str(img_path),
                                caption=img.get("caption") or img.get("alt"),
                                width=350
                            )

# ------------------------------------------------------------
# User input
# ------------------------------------------------------------

prompt = st.chat_input("Ask anything about GIKI...")

if prompt:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        with st.spinner("Searching knowledge base..."):

            result = answer(
                prompt,
                stream_to_stdout=False,
                return_retrieval=True
            )

        st.markdown(result["answer"])

        retrieved = result["retrieved"]

        with st.expander("📚 Retrieved Sources"):

            for src in retrieved["text"]:

                title = src.get("title") or "Untitled"

                st.markdown(
                    f"- **{title}**  \n{src['url']}"
                )

        with st.expander("🖼 Retrieved Images"):

            for img in retrieved["images"]:

                img_path = config.IMAGE_DIR / img["local_path"]

                if img_path.exists():

                    st.image(
                        str(img_path),
                        caption=img.get("caption") or img.get("alt"),
                        width=350
                    )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": retrieved["text"],
            "images": retrieved["images"]
        }
    )

    st.rerun()