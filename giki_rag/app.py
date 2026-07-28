from pathlib import Path

import streamlit as st
from audio_recorder_streamlit import audio_recorder

import config
import stt
import tts
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

    st.markdown("### 🔊 Voice")
    _tts_options = ["edge", "piper", "pyttsx3"]
    tts_choice = st.selectbox(
        "TTS voice engine",
        _tts_options,
        index=_tts_options.index(config.TTS_BACKEND)
        if config.TTS_BACKEND in _tts_options else 0,
    )
    speak_answers = st.checkbox("Speak answers aloud", value=True)

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

            if msg.get("audio_path") and Path(msg["audio_path"]).exists():
                st.audio(msg["audio_path"])

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
# User input (typed or spoken)
# ------------------------------------------------------------

col_rec, col_hint = st.columns([1, 8])
with col_rec:
    audio_bytes = audio_recorder(text="", icon_size="2x", key="voice_recorder")
with col_hint:
    st.caption("🎙 Click the mic to ask by voice, or type below.")

typed_prompt = st.chat_input("Ask anything about GIKI...")

prompt = None
if typed_prompt:
    prompt = typed_prompt
elif audio_bytes and audio_bytes != st.session_state.get("_last_audio_bytes"):
    # audio_recorder keeps returning the same bytes on every rerun until a new
    # recording is made — without this guard the same question would be
    # re-submitted after every st.rerun() (e.g. right after the answer renders).
    st.session_state["_last_audio_bytes"] = audio_bytes
    with st.spinner("Transcribing..."):
        transcribed = stt.transcribe(audio_bytes)
    if transcribed.strip():
        prompt = transcribed
    else:
        st.warning("Could not transcribe any speech — try again.")

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

        answer_audio_path = None
        if speak_answers:
            with st.spinner("Synthesizing speech..."):
                answer_audio_path = tts.synthesize(result["answer"], backend=tts_choice)
            st.audio(str(answer_audio_path), autoplay=True)

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
            "images": retrieved["images"],
            "audio_path": str(answer_audio_path) if answer_audio_path else None
        }
    )

    st.rerun()