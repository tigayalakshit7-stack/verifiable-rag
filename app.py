import time
import streamlit as st
from pathlib import Path
from src.retrieval.retriever import retrieve
from src.generation.synthesizer import synthesize_answer
from src.generation.renderer import CONFIDENCE_BADGE
from src.evaluation.metrics import compute_metrics
from src.vector_store import get_qdrant_client, get_collection_info
from src.config import PDF_DIR
from src.ingestion.pdf_loader import load_all_pdfs
from src.ingestion.chunker import chunk_pages
from src.ingestion.indexer import index_chunks

# Bootstrap demo data on first run
from scripts.bootstrap import bootstrap_if_empty
if "bootstrapped" not in st.session_state:
    with st.spinner("Preparing index (first-run only)..."):
        bootstrap_if_empty()
    st.session_state.bootstrapped = True

# ---------- Page config ----------
st.set_page_config(
    page_title="Verifiable RAG",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Custom CSS ----------
st.markdown("""
<style>
    .stApp {
        background: #0a0e14;
    }
    .main .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }
    .hero h1 {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(90deg, #4ade80 0%, #22d3ee 60%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .hero p {
        color: #94a3b8;
        font-size: 1rem;
    }
    .claim-card {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-left: 3px solid #334155;
        border-radius: 8px;
        padding: 14px 16px;
        margin-bottom: 10px;
    }
    .claim-card.high { border-left-color: #22c55e; }
    .claim-card.medium { border-left-color: #eab308; }
    .claim-card.low { border-left-color: #ef4444; }
    .badge {
        display: inline-block;
        font-size: 0.7rem;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 10px;
        margin-right: 8px;
    }
    .badge.high { background: #052e16; color: #4ade80; }
    .badge.medium { background: #422006; color: #facc15; }
    .badge.low { background: #450a0a; color: #f87171; }
    .cite {
        font-family: ui-monospace, SFMono-Regular, monospace;
        font-size: 0.75rem;
        color: #94a3b8;
        background: #1e293b;
        padding: 1px 6px;
        border-radius: 4px;
        margin-right: 4px;
    }
    .source-card {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 12px 14px;
        margin-bottom: 10px;
    }
    .source-header {
        font-family: ui-monospace, monospace;
        font-size: 0.75rem;
        color: #22d3ee;
        margin-bottom: 6px;
    }
    .source-snippet {
        color: #cbd5e1;
        font-size: 0.85rem;
        line-height: 1.4;
    }
    .metric-tile {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 14px;
        text-align: center;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #4ade80;
    }
    .metric-label {
        font-size: 0.72rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
</style>
""", unsafe_allow_html=True)

# ---------- Session state ----------
for k, v in {
    "answer": None, "retrieval": None, "metrics": None,
    "baseline_answer": None, "baseline_retrieval": None, "baseline_metrics": None,
    "question": "",
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("### 🔬 Verifiable RAG")
    st.caption("Multi-hop retrieval with claim-level citations.")

    st.divider()
    st.markdown("### 📚 Corpus")
    try:
        info = get_collection_info(get_qdrant_client())
        count = info.get("vectors_count", 0)
        st.metric("Indexed chunks", count)
    except Exception:
        st.metric("Indexed chunks", 0)

    st.divider()
    st.markdown("### 📄 Add Your Own PDFs")
    uploaded = st.file_uploader(
        "Upload PDFs (optional)",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
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

    st.divider()
    st.caption("v1.0 · Qdrant · Groq · Streamlit")

# ---------- Hero ----------
st.markdown("""
<div class="hero">
    <h1>🔬 Verifiable RAG</h1>
    <p>Ask questions across research papers. Every claim is traced to a source.</p>
</div>
""", unsafe_allow_html=True)

# ---------- Question input ----------
col1, col2 = st.columns([5, 1])
with col1:
    question = st.text_input(
        "Question",
        placeholder="e.g., How does Self-RAG differ from Corrective RAG in handling retrieval errors?",
        label_visibility="collapsed",
    )
with col2:
    run = st.button("🔍 Ask", use_container_width=True, type="primary")

# ---------- Run ----------
if run and question:
    st.session_state.question = question

    with st.spinner("Decomposing and retrieving..."):
        t0 = time.time()
        retrieval = retrieve(question, mode="multi_hop")
        baseline_retrieval = retrieve(question, mode="single")

    with st.spinner("Synthesizing grounded answer..."):
        answer = synthesize_answer(question, retrieval)
        baseline_answer = synthesize_answer(question, baseline_retrieval)
        latency = time.time() - t0

    st.session_state.answer = answer
    st.session_state.retrieval = retrieval
    st.session_state.baseline_answer = baseline_answer
    st.session_state.baseline_retrieval = baseline_retrieval
    st.session_state.metrics = compute_metrics(retrieval, answer, latency)

# ---------- Display ----------
if st.session_state.answer:
    answer = st.session_state.answer
    retrieval = st.session_state.retrieval
    metrics = st.session_state.metrics
    baseline = st.session_state.baseline_answer

    st.markdown("---")

    # Metric tiles
    c1, c2, c3, c4 = st.columns(4)
    for col, label, value in [
        (c1, "Claims", metrics.claim_count),
        (c2, "Grounding", f"{metrics.grounding_rate*100:.0f}%"),
        (c3, "High-Conf", f"{metrics.high_confidence_rate*100:.0f}%"),
        (c4, "Sources", metrics.source_diversity),
    ]:
        col.markdown(
            f'<div class="metric-tile"><div class="metric-value">{value}</div>'
            f'<div class="metric-label">{label}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("")

    tab1, tab2, tab3, tab4 = st.tabs(["📝 Answer", "📚 Sources", "⚖️ Compare", "📊 Metrics"])

    with tab1:
        if answer.insufficient_evidence:
            st.warning(f"⚠️ Insufficient evidence. {answer.notes}")

        for i, claim in enumerate(answer.claims, 1):
            badge = CONFIDENCE_BADGE[claim.confidence]
            refs = "".join(
                f'<span class="cite">{cid[:8]}</span>'
                for cid in claim.cited_chunk_ids
            ) or '<span class="cite">no source</span>'
            st.markdown(
                f'<div class="claim-card {claim.confidence}">'
                f'<span class="badge {claim.confidence}">{badge}</span>'
                f'<strong>{i}.</strong> {claim.text}<br><br>{refs}'
                f'</div>',
                unsafe_allow_html=True,
            )

    with tab2:
        for i, c in enumerate(answer.citations, 1):
            st.markdown(
                f'<div class="source-card">'
                f'<div class="source-header">[{i}] {c.chunk_id[:8]} · {c.source} · p.{c.page_number}</div>'
                f'<div class="source-snippet">{c.snippet}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    with tab3:
        st.markdown("#### Multi-Hop vs Baseline (single query)")
        cc1, cc2 = st.columns(2)
        with cc1:
            st.markdown("**🔀 Multi-Hop (ours)**")
            for i, claim in enumerate(answer.claims, 1):
                st.markdown(f"**{i}.** {claim.text}")
                st.caption(f"confidence: {claim.confidence} · {len(claim.cited_chunk_ids)} source(s)")
        with cc2:
            st.markdown("**➡️ Baseline (single)**")
            for i, claim in enumerate(baseline.claims, 1):
                st.markdown(f"**{i}.** {claim.text}")
                st.caption(f"confidence: {claim.confidence} · {len(claim.cited_chunk_ids)} source(s)")

    with tab4:
        st.markdown("#### Live metrics for this answer")
        st.json({
            "claim_count": metrics.claim_count,
            "grounding_rate": round(metrics.grounding_rate, 3),
            "high_confidence_rate": round(metrics.high_confidence_rate, 3),
            "cross_source_rate": round(metrics.cross_source_rate, 3),
            "citation_density": round(metrics.citation_density, 3),
            "source_diversity": metrics.source_diversity,
            "latency_seconds": round(metrics.latency_seconds, 2),
        })
        st.caption("Full benchmark across the test set: see docs/evaluation.md")