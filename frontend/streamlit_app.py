import streamlit as st
import requests
import json
import time
from pathlib import Path

# API Endpoint
API_URL = "http://localhost:8000/api"

st.set_page_config(
    page_title="ResearchCRAG",
    page_icon="🔬",
    layout="wide"
)

# Minimal clean style
st.markdown("""
<style>
    .reportview-container {
        margin-top: -2em;
    }
    .stChatInput {
        bottom: 20px;
    }
    .citation-box {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 3px solid #0284c7;
        padding: 10px 14px;
        border-radius: 6px;
        margin-top: 8px;
        margin-bottom: 8px;
        color: #334155;
    }
    @media (prefers-color-scheme: dark) {
        .citation-box {
            background-color: #1e293b;
            border-color: #334155;
            border-left: 3px solid #38bdf8;
            color: #cbd5e1;
        }
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
def get_health():
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None

def get_papers():
    try:
        r = requests.get(f"{API_URL}/papers", timeout=3)
        return r.json() if r.status_code == 200 else []
    except Exception:
        return []

# Sidebar
with st.sidebar:
    st.title("🔬 ResearchCRAG")
    st.caption("Corrective RAG for AI/ML Papers")

    health = get_health()
    if health and health.get("status") == "online":
        st.success("🟢 Backend Connected", icon="✅")
        chunks = health.get("vectorstore", {}).get("total_chunks", 0)
        st.caption(f"**Vector Store:** {chunks} chunks indexed")
    else:
        st.error("🔴 Backend Offline (Port 8000)", icon="⚠️")

    st.divider()
    st.subheader("Settings")
    model_name = st.selectbox("LLM Model", ["llama3.2", "olmo", "mistral"], index=0)
    enable_web = st.checkbox("Web search fallback", value=True)

    st.divider()
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Main Interface Tabs
tab_chat, tab_upload = st.tabs(["💬 Assistant Chat", "📤 Manage Papers"])

# ----------------- TAB 1: CHAT -----------------
with tab_chat:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant" and msg.get("rewritten"):
                st.caption(f"🔄 *Query rewritten for retrieval:* `{msg['rewritten']}`")

            st.markdown(msg.get("content", ""))
            
            # Show citations if any
            if msg.get("citations"):
                with st.expander(f"📚 Verified Sources ({len(msg['citations'])})", expanded=False):
                    for cit in msg["citations"]:
                        st.markdown(f"""
                        <div class="citation-box">
                            <strong>📄 {cit.get('paper_title', cit.get('paper'))}</strong> ({cit.get('paper')} | Page {cit.get('page')} | {cit.get('section')})<br>
                            <small>"{cit.get('snippet')}"</small>
                        </div>
                        """, unsafe_allow_html=True)

            # Show trace if any
            if msg.get("trace"):
                with st.expander("🔍 Architecture Pipeline Trace", expanded=False):
                    for step in msg["trace"]:
                        st.text(f"[{step.get('timestamp')}] {step.get('step')}: {json.dumps(step.get('details'))}")

    if not st.session_state.messages:
        st.markdown("""
        <div style="text-align: center; padding: 60px 20px; color: #64748b;">
            <h2 style="color: inherit; margin-bottom: 8px;">🔬 ResearchCRAG</h2>
            <p style="font-size: 1.1em; max-width: 600px; margin: 0 auto; color: #94a3b8;">
                Ask questions about your indexed AI/ML research papers. Grounded answers, self-correcting retrieval, and verified citations.
            </p>
        </div>
        """, unsafe_allow_html=True)

# ----------------- TAB 2: UPLOAD -----------------
with tab_upload:
    st.subheader("Add Research Papers")
    st.caption("Upload any AI/ML paper in PDF format to index into the vector store.")

    uploaded = st.file_uploader("Select PDF file", type=["pdf"])
    if uploaded and st.button("Upload & Index", type="primary"):
        with st.spinner(f"Ingesting {uploaded.name}..."):
            try:
                files = {"file": (uploaded.name, uploaded.getvalue(), "application/pdf")}
                r = requests.post(f"{API_URL}/upload", files=files, timeout=60)
                if r.status_code == 200:
                    d = r.json()
                    st.success(f"Successfully indexed '{uploaded.name}' ({d.get('chunks_created')} chunks, {d.get('total_pages')} pages).")
                    st.rerun()
                else:
                    st.error(f"Upload failed: {r.text}")
            except Exception as e:
                st.error(f"Error: {e}")

    st.divider()
    st.subheader("Currently Indexed Research Papers")
    current_papers = get_papers()
    if current_papers:
        for p in current_papers:
            col_info, col_btn = st.columns([5, 1])
            with col_info:
                st.markdown(f"**📄 {p.get('title', p.get('filename'))}**")
                st.caption(f"Filename: `{p.get('filename')}` | Pages: **{p.get('total_pages')}** | Chunks: **{p.get('chunk_count')}**")
                if p.get("sections"):
                    st.caption(f"Sections: *{', '.join(p.get('sections', []))}*")
            with col_btn:
                if st.button("🗑️ Delete", key=f"del_{p.get('filename')}", use_container_width=True):
                    requests.delete(f"{API_URL}/papers/{p.get('filename')}")
                    st.rerun()
            st.divider()
    else:
        st.info("No papers indexed yet. Upload a PDF above.")

# ----------------- FIXED BOTTOM CHAT INPUT -----------------
prompt = st.chat_input("Ask a question about the papers...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with tab_chat:
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response_placeholder = st.empty()

            full_response = ""
            citations = []
            trace = []
            rewritten_query = None
            fast_path = False
            conf_level = "LOW"

            try:
                resp = requests.post(
                    f"{API_URL}/query/stream",
                    json={
                        "question": prompt,
                        "llm_model": model_name,
                        "enable_web_search": enable_web
                    },
                    stream=True,
                    timeout=120
                )

                if resp.status_code == 200:
                    for line in resp.iter_lines(decode_unicode=True):
                        if line and line.startswith("data: "):
                            raw_json = line[6:]
                            try:
                                event = json.loads(raw_json)
                                ev_type = event.get("type")

                                if ev_type == "status":
                                    stage = event.get("stage")
                                    if stage == "confidence":
                                        fast_path = event.get("fast_path", False)
                                        conf_level = event.get("level", "LOW")

                                elif ev_type == "token":
                                    token = event.get("token", "")
                                    full_response += token
                                    response_placeholder.markdown(full_response + "▌")

                                elif ev_type == "done":
                                    full_response = event.get("answer", full_response)
                                    citations = event.get("citations", [])
                                    trace = event.get("trace", [])
                                    rewritten_query = event.get("rewritten_query")
                                    fast_path = event.get("fast_path_used", fast_path)
                                    conf_level = event.get("confidence_level", conf_level)

                            except Exception:
                                pass

                    if not full_response:
                        full_response = "No response received from generator."

                    response_placeholder.markdown(full_response)

                    if citations:
                        with st.expander(f"📚 Verified Sources ({len(citations)})", expanded=False):
                            for cit in citations:
                                st.markdown(f"""
                                <div class="citation-box">
                                    <strong>📄 {cit.get('paper_title', cit.get('paper'))}</strong> ({cit.get('paper')} | Page {cit.get('page')} | {cit.get('section')})<br>
                                    <small>"{cit.get('snippet')}"</small>
                                </div>
                                """, unsafe_allow_html=True)

                    if trace:
                        with st.expander("🔍 Architecture Pipeline Trace", expanded=False):
                            for step in trace:
                                st.text(f"[{step.get('timestamp')}] {step.get('step')}: {json.dumps(step.get('details'))}")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_response,
                        "citations": citations,
                        "trace": trace,
                        "rewritten": rewritten_query,
                        "fast_path": fast_path,
                        "conf_level": conf_level
                    })

                else:
                    st.error(f"Error {resp.status_code}: {resp.text}")

            except Exception as ex:
                st.error(f"Connection Error: Could not reach backend ({ex})")

    st.rerun()
