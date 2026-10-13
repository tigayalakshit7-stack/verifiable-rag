import time
import streamlit as st
from src.retrieval.retriever import retrieve
from src.generation.synthesizer import synthesize_answer
from src.evaluation.metrics import compute_metrics
from src.vector_store import get_qdrant_client, get_collection_info
from src.config import PDF_DIR
from src.ingestion.pdf_loader import load_all_pdfs
from src.ingestion.chunker import chunk_pages
from src.ingestion.indexer import index_chunks
from src.ui.history import HistoryEntry, add_entry, format_relative
from src.ui.exporter import answer_to_markdown
from scripts.bootstrap import bootstrap_if_empty

# ---------- Page config ----------
st.set_page_config(
    page_title="Verifiable RAG · Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Premium CSS ----------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .stApp {
        background: radial-gradient(circle at 20% 0%, #0f172a 0%, #0a0e14 60%);
    }
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1240px;
    }

    /* Hide Streamlit chrome — but keep sidebar toggle visible */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    /* Do NOT hide `header` — it contains the sidebar toggle button */

    /* Hero */
    .hero {
        padding: 1.5rem 0 1rem 0;
        border-bottom: 1px solid #1e293b;
        margin-bottom: 1.5rem;
    }
    .hero h1 {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #4ade80 0%, #22d3ee 55%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 0.3rem 0;
        letter-spacing: -0.7px;
    }
    .hero p {
        color: #94a3b8;
        font-size: 0.95rem;
        margin: 0;
    }
    .hero-badges {
        display: flex;
        gap: 8px;
        margin-top: 12px;
        flex-wrap: wrap;
    }
    .pill {
        font-size: 0.7rem;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 20px;
        background: #0f172a;
        border: 1px solid #1e293b;
        color: #94a3b8;
    }
    .pill.green { color: #4ade80; border-color: #14532d; }

    /* Suggested question cards */
    .suggested {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 18px;
        cursor: pointer;
        transition: all 0.2s ease;
        height: 100%;
    }
    .suggested:hover {
        border-color: #22d3ee;
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(34, 211, 238, 0.08);
    }
    .suggested .icon {
        font-size: 1.4rem;
        margin-bottom: 6px;
    }
    .suggested .title {
        font-weight: 600;
        color: #e2e8f0;
        font-size: 0.9rem;
        margin-bottom: 4px;
    }
    .suggested .hint {
        color: #64748b;
        font-size: 0.75rem;
    }

    /* Progress stages */
    .stage-row {
        display: flex;
        gap: 10px;
        margin: 1rem 0;
    }
    .stage {
        flex: 1;
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 12px;
        text-align: center;
        transition: all 0.3s ease;
    }
    .stage.active {
        border-color: #22d3ee;
        background: #0c1e26;
        box-shadow: 0 0 16px rgba(34, 211, 238, 0.15);
    }
    .stage.done {
        border-color: #4ade80;
        background: #051a0e;
    }
    .stage .st-icon { font-size: 1.2rem; }
    .stage .st-name {
        font-size: 0.72rem;
        font-weight: 600;
        color: #cbd5e1;
        margin-top: 4px;
        letter-spacing: 0.3px;
    }
    .stage.active .st-name { color: #22d3ee; }
    .stage.done .st-name { color: #4ade80; }

    /* Claim cards */
    .claim-card {
        background: linear-gradient(135deg, #0f172a 0%, #0c1424 100%);
        border: 1px solid #1e293b;
        border-left: 3px solid #334155;
        border-radius: 10px;
        padding: 16px 18px;
        margin-bottom: 12px;
        transition: all 0.2s ease;
    }
    .claim-card:hover {
        border-color: #334155;
        transform: translateX(2px);
    }
    .claim-card.high { border-left-color: #22c55e; }
    .claim-card.medium { border-left-color: #eab308; }
    .claim-card.low { border-left-color: #ef4444; }

    .claim-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 8px;
    }
    .claim-num {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 22px; height: 22px;
        background: #1e293b;
        border-radius: 6px;
        font-size: 0.72rem;
        font-weight: 700;
        color: #94a3b8;
    }
    .badge {
        display: inline-block;
        font-size: 0.65rem;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 20px;
        letter-spacing: 0.4px;
    }
    .badge.high { background: #052e16; color: #4ade80; }
    .badge.medium { background: #422006; color: #facc15; }
    .badge.low { background: #450a0a; color: #f87171; }

    .claim-text {
        color: #e2e8f0;
        font-size: 0.92rem;
        line-height: 1.55;
        margin-bottom: 10px;
    }
    .cite {
        display: inline-block;
        font-family: ui-monospace, SFMono-Regular, monospace;
        font-size: 0.7rem;
        font-weight: 600;
        color: #22d3ee;
        background: #0c1e26;
        border: 1px solid #164e63;
        padding: 2px 8px;
        border-radius: 6px;
        margin-right: 5px;
        text-decoration: none !important;
    }
    .cite:hover { background: #164e63; }

    /* Source cards */
    .source-card {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 14px 16px;
        margin-bottom: 12px;
        transition: all 0.2s ease;
    }
    .source-card:target {
        border-color: #22d3ee;
        box-shadow: 0 0 20px rgba(34, 211, 238, 0.2);
    }
    .source-header {
        font-family: ui-monospace, monospace;
        font-size: 0.72rem;
        color: #22d3ee;
        margin-bottom: 8px;
        display: flex;
        justify-content: space-between;
    }
    .source-snippet {
        color: #cbd5e1;
        font-size: 0.85rem;
        line-height: 1.5;
    }

    /* Metric tiles */
    .metric-tile {
        background: linear-gradient(135deg, #0f172a 0%, #0c1424 100%);
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 18px 12px;
        text-align: center;
        transition: all 0.2s ease;
    }
    .metric-tile:hover {
        border-color: #334155;
        transform: translateY(-2px);
    }
    .metric-value {
        font-size: 1.7rem;
        font-weight: 800;
        color: #4ade80;
        letter-spacing: -0.5px;
    }
    .metric-label {
        font-size: 0.68rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
        margin-top: 4px;
    }

    /* Empty state */
    .empty-state {
        text-align: center;
        padding: 2rem 1rem;
        color: #64748b;
    }
    .empty-state .icon { font-size: 2.5rem; opacity: 0.4; }

    /* Sidebar tweaks */
    section[data-testid="stSidebar"] {
        background: #080c14;
        border-right: 1px solid #1e293b;
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem;
    }
    .history-item {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 8px 10px;
        margin-bottom: 6px;
        font-size: 0.78rem;
        color: #cbd5e1;
        cursor: pointer;
    }
    .history-item:hover { border-color: #22d3ee; }
    .history-meta {
        font-size: 0.65rem;
        color: #64748b;
        margin-top: 2px;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #22d3ee 0%, #818cf8 100%);
        color: #0a0e14;
        font-weight: 700;
        border: none;
        border-radius: 8px;
        padding: 0.55rem 1.2rem;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        box-shadow: 0 6px 22px rgba(34, 211, 238, 0.4);
        transform: translateY(-1px);
    }
    .stDownloadButton>button {
        background: transparent;
        border: 1px solid #334155;
        color: #cbd5e1;
    }
    .stDownloadButton>button:hover {
        border-color: #22d3ee;
        color: #22d3ee;
    }

    /* Text input */
    .stTextInput input {
        background: #0f172a !important;
        border: 1px solid #1e293b !important;
        color: #e2e8f0 !important;
        padding: 12px 16px !important;
        font-size: 0.95rem !important;
        border-radius: 10px !important;
    }
    .stTextInput input:focus {
        border-color: #22d3ee !important;
        box-shadow: 0 0 0 3px rgba(34, 211, 238, 0.1) !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        border-bottom: 1px solid #1e293b;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px 8px 0 0;
        padding: 10px 18px;
        color: #64748b;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .stTabs [aria-selected="true"] {
        background: #0f172a !important;
        color: #22d3ee !important;
        border-bottom: 2px solid #22d3ee !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------- Bootstrap ----------
if "bootstrapped" not in st.session_state:
    with st.spinner("Preparing index (first-run only)..."):
        bootstrap_if_empty()
    st.session_state.bootstrapped = True

# ---------- Session state ----------
defaults = {
    "answer": None, "retrieval": None, "metrics": None,
    "baseline_answer": None, "baseline_retrieval": None,
    "question": "", "history": [], "stage": 0,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------- Suggested questions ----------
SUGGESTIONS = [
    ("🔀", "Compare two papers", "How does Self-RAG differ from Corrective RAG in handling retrieval errors?",
     "Cross-document reasoning"),
    ("📖", "Explain a concept", "What is Retrieval-Augmented Generation?", "Definition + mechanism"),
    ("🎯", "Find techniques", "What techniques do modern RAG systems use to improve factual accuracy?", "Survey"),
    ("⚡", "Test robustness", "Why is retrieval quality important for RAG performance?", "Reliability"),
]

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("### 🔬 Verifiable RAG")
    st.caption("Multi-hop retrieval · claim-level citations")
    st.divider()

    st.markdown("**📚 Corpus**")
    try:
        info = get_collection_info(get_qdrant_client())
        st.metric("Indexed chunks", info.get("vectors_count", 0))
    except Exception:
        st.metric("Indexed chunks", 0)

    st.divider()
    st.markdown("**📄 Add PDFs**")
    uploaded = st.file_uploader(
        "Upload", type=["pdf"], accept_multiple_files=True,
        label_visibility="collapsed"
    )
    if uploaded and st.button("📥 Ingest", use_container_width=True):
        PDF_DIR.mkdir(parents=True, exist_ok=True)
        for f in uploaded:
            (PDF_DIR / f.name).write_bytes(f.getbuffer())
        with st.spinner("Ingesting..."):
            pages = load_all_pdfs(PDF_DIR)
            chunks = chunk_pages(pages)
            n = index_chunks(chunks)
        st.success(f"Indexed {n} chunks")
        st.rerun()

    if st.session_state.history:
        st.divider()
        st.markdown("**🕒 Recent**")
        for entry in reversed(st.session_state.history[-5:]):
            meta = f"{entry.claim_count} claims · {format_relative(entry.timestamp)}"
            st.markdown(
                f'<div class="history-item">'
                f'{entry.question[:52]}{"..." if len(entry.question) > 52 else ""}'
                f'<div class="history-meta">{meta}</div></div>',
                unsafe_allow_html=True,
            )

    st.divider()
    st.caption("v1.0 · Qdrant · Groq · Streamlit")

# ---------- Hero ----------
st.markdown("""
<div class="hero">
    <h1>🔬 Verifiable RAG</h1>
    <p>Ask across research papers. Every claim traced to its source.</p>
    <div class="hero-badges">
        <span class="pill green">● Multi-hop retrieval</span>
        <span class="pill">Claim-level citations</span>
        <span class="pill">Custom eval harness</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------- Callback for suggestion clicks ----------
def set_question(q: str):
    """Set the text input value directly via session state."""
    st.session_state.q_input = q

# ---------- Initialize input state ----------
if "q_input" not in st.session_state:
    st.session_state.q_input = ""

# ---------- Suggested questions (shown before first answer) ----------
if not st.session_state.answer:
    st.markdown("#### 💡 Try one of these")
    cols = st.columns(4)
    for col, (icon, title, q, hint) in zip(cols, SUGGESTIONS):
        with col:
            st.markdown(
                f'<div class="suggested">'
                f'<div class="icon">{icon}</div>'
                f'<div class="title">{title}</div>'
                f'<div class="hint">{hint}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.button(
                "Ask",
                key=f"sug_{title}",
                use_container_width=True,
                on_click=set_question,
                args=(q,),
            )

# ---------- Input ----------
st.markdown("#### ❓ Your question")
col1, col2 = st.columns([5, 1])
with col1:
    question = st.text_input(
        "q",
        placeholder="Ask anything about the indexed papers...",
        label_visibility="collapsed",
        key="q_input",
    )
with col2:
    run = st.button("🔍 Ask", use_container_width=True, type="primary")

# ---------- Progress stages ----------
stage_slot = st.empty()

def render_stages(current: int):
    """current: 0=idle, 1=decompose, 2=retrieve, 3=synthesize, 4=done"""
    stages = [
        ("🧩", "Decompose"),
        ("🔍", "Retrieve"),
        ("✍️", "Synthesize"),
        ("✅", "Done"),
    ]
    html = '<div class="stage-row">'
    for i, (icon, name) in enumerate(stages, 1):
        cls = "stage"
        if current >= 4:
            cls += " done"
        elif current == i:
            cls += " active"
        elif current > i:
            cls += " done"
        html += f'<div class="{cls}"><div class="st-icon">{icon}</div><div class="st-name">{name}</div></div>'
    html += '</div>'
    stage_slot.markdown(html, unsafe_allow_html=True)

# ---------- Run pipeline ----------
if run and question:
    st.session_state.question = question
    
    st.session_state.answer = None

    render_stages(1)
    t0 = time.time()
    with st.spinner("Decomposing question into sub-queries..."):
        retrieval = retrieve(question, mode="multi_hop")
        baseline_retrieval = retrieve(question, mode="single")

    render_stages(2)
    with st.spinner("Synthesizing grounded answer..."):
        answer = synthesize_answer(question, retrieval)
        baseline_answer = synthesize_answer(question, baseline_retrieval)

    render_stages(3)
    latency = time.time() - t0
    metrics = compute_metrics(retrieval, answer, latency)

    render_stages(4)
    time.sleep(0.3)  # let user see the "done" state
    stage_slot.empty()

    st.session_state.answer = answer
    st.session_state.retrieval = retrieval
    st.session_state.baseline_answer = baseline_answer
    st.session_state.baseline_retrieval = baseline_retrieval
    st.session_state.metrics = metrics
    add_entry(st.session_state.history, question, answer, latency)

# ---------- Display answer ----------
if st.session_state.answer:
    answer = st.session_state.answer
    baseline = st.session_state.baseline_answer
    metrics = st.session_state.metrics

    st.markdown("---")

    # Metric tiles
    c1, c2, c3, c4 = st.columns(4)
    tiles = [
        (c1, "Claims", metrics.claim_count),
        (c2, "Grounding", f"{metrics.grounding_rate*100:.0f}%"),
        (c3, "High-Conf", f"{metrics.high_confidence_rate*100:.0f}%"),
        (c4, "Sources", metrics.source_diversity),
    ]
    for col, label, value in tiles:
        col.markdown(
            f'<div class="metric-tile"><div class="metric-value">{value}</div>'
            f'<div class="metric-label">{label}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📝 Answer", "📚 Sources", "⚖️ Compare", "📊 Metrics"
    ])

    with tab1:
        if answer.insufficient_evidence:
            st.warning(f"⚠️ Insufficient evidence. {answer.notes}")

        for i, claim in enumerate(answer.claims, 1):
            badge_text = {"high": "HIGH", "medium": "MEDIUM", "low": "LOW"}[claim.confidence]
            refs_html = "".join(
                f'<a class="cite" href="#src-{cid[:8]}">{cid[:8]}</a>'
                for cid in claim.cited_chunk_ids
            ) or '<span class="cite" style="color:#ef4444;">no source</span>'

            st.markdown(
                f'<div class="claim-card {claim.confidence}">'
                f'<div class="claim-header">'
                f'<span class="claim-num">{i}</span>'
                f'<span class="badge {claim.confidence}">{badge_text}</span>'
                f'</div>'
                f'<div class="claim-text">{claim.text}</div>'
                f'<div>{refs_html}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Download
        md = answer_to_markdown(answer)
        st.download_button(
            "📥 Download answer (Markdown)",
            data=md,
            file_name=f"answer_{int(time.time())}.md",
            mime="text/markdown",
        )

    with tab2:
        for i, c in enumerate(answer.citations, 1):
            st.markdown(
                f'<div class="source-card" id="src-{c.chunk_id[:8]}">'
                f'<div class="source-header">'
                f'<span>[{i}] {c.chunk_id[:8]}</span>'
                f'<span>{c.source} · p.{c.page_number}</span>'
                f'</div>'
                f'<div class="source-snippet">{c.snippet}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    with tab3:
        st.markdown("#### 🔀 Multi-Hop vs ➡️ Baseline")
        cc1, cc2 = st.columns(2)

        with cc1:
            st.markdown("**🔀 Multi-Hop (ours)**")
            for i, claim in enumerate(answer.claims, 1):
                st.markdown(
                    f'<div class="claim-card {claim.confidence}">'
                    f'<div class="claim-header">'
                    f'<span class="claim-num">{i}</span>'
                    f'<span class="badge {claim.confidence}">{claim.confidence.upper()}</span>'
                    f'</div>'
                    f'<div class="claim-text">{claim.text}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        with cc2:
            st.markdown("**➡️ Baseline (single query)**")
            for i, claim in enumerate(baseline.claims, 1):
                st.markdown(
                    f'<div class="claim-card {claim.confidence}">'
                    f'<div class="claim-header">'
                    f'<span class="claim-num">{i}</span>'
                    f'<span class="badge {claim.confidence}">{claim.confidence.upper()}</span>'
                    f'</div>'
                    f'<div class="claim-text">{claim.text}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    with tab4:
        st.markdown("#### 📊 This answer's metrics")
        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            st.markdown("**Grounding**")
            st.progress(metrics.grounding_rate)
            st.caption(f"{metrics.grounding_rate*100:.1f}% of claims cited")
        with cc2:
            st.markdown("**High confidence**")
            st.progress(metrics.high_confidence_rate)
            st.caption(f"{metrics.high_confidence_rate*100:.1f}% cited 2+ sources")
        with cc3:
            st.markdown("**Cross-source**")
            st.progress(metrics.cross_source_rate)
            st.caption(f"{metrics.cross_source_rate*100:.1f}% spanned 2+ PDFs")

        st.divider()
        st.markdown("**Raw values**")
        st.json({
            "claim_count": metrics.claim_count,
            "citation_count": metrics.citation_count,
            "grounding_rate": round(metrics.grounding_rate, 3),
            "high_confidence_rate": round(metrics.high_confidence_rate, 3),
            "cross_source_rate": round(metrics.cross_source_rate, 3),
            "citation_density": round(metrics.citation_density, 3),
            "source_diversity": metrics.source_diversity,
            "latency_seconds": round(metrics.latency_seconds, 2),
        })
        st.caption("Full benchmark across the test set: [`docs/evaluation.md`](docs/evaluation.md)")