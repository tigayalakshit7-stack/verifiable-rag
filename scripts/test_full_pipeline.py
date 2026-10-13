"""
End-to-end test: retrieve -> synthesize -> render.

Compares baseline (single) vs multi-hop on the same questions.
"""
from src.retrieval.retriever import retrieve
from src.generation.synthesizer import synthesize_answer
from src.generation.renderer import render_answer_markdown


QUESTIONS = [
    "What is Retrieval-Augmented Generation?",
    "How does Self-RAG differ from Corrective RAG in handling retrieval errors?",
]


def run_one(question: str, mode: str):
    print(f"\n{'#'*72}")
    print(f"#  MODE: {mode.upper()}")
    print(f"{'#'*72}")
    retrieval = retrieve(question, mode=mode)
    answer = synthesize_answer(question, retrieval)
    print(render_answer_markdown(answer))


def main():
    for q in QUESTIONS:
        run_one(q, mode="single")
        run_one(q, mode="multi_hop")


if __name__ == "__main__":
    main()