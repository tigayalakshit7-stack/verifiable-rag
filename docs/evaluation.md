# RAG Evaluation Report

_Generated: 2026-10-13 12:21:24_

## Summary

| Metric | Baseline (single) | Multi-Hop (ours) | Δ |
| :--- | ---: | ---: | ---: |
| Questions evaluated | 5.0 | 5.0 | +0.0 |
| Avg claims per answer | 4.20 | 3.80 | -0.40 |
| Avg citations per answer | 2.60 | 2.60 | +0.00 |
| Citation density (citations/claim) | 0.61 | 0.69 | +0.08 |
| Grounding rate | 100.0% | 100.0% | +0.0% |
| High-confidence rate | 9.0% | 21.3% | +12.3% |
| **Cross-source rate** | 0.0% | 4.0% | +4.0% |
| Avg source diversity (PDFs cited) | 1.40 | 1.60 | +0.20 |
| Insufficient-evidence rate | 0.0% | 0.0% | +0.0% |
| Avg latency (s) | 7.52 | 11.92 | +4.39 |

## Per-Question Breakdown

| Question | Mode | Claims | Grounding | High-Conf | Cross-Src | Sources | Latency |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| What is Retrieval-Augmented Generation? | single | 4 | 100.0% | 0.0% | 0.0% | 1 | 3.23s |
| What is Retrieval-Augmented Generation? | multi_hop | 4 | 100.0% | 0.0% | 0.0% | 1 | 4.31s |
| How does Self-RAG differ from Corrective RAG in ha... | single | 5 | 100.0% | 20.0% | 0.0% | 1 | 3.44s |
| How does Self-RAG differ from Corrective RAG in ha... | multi_hop | 5 | 100.0% | 40.0% | 20.0% | 2 | 5.29s |
| What techniques do modern RAG systems use to impro... | single | 4 | 100.0% | 0.0% | 0.0% | 2 | 4.16s |
| What techniques do modern RAG systems use to impro... | multi_hop | 4 | 100.0% | 0.0% | 0.0% | 3 | 19.42s |
| Why is retrieval quality important for RAG perform... | single | 4 | 100.0% | 25.0% | 0.0% | 1 | 12.74s |
| Why is retrieval quality important for RAG perform... | multi_hop | 3 | 100.0% | 33.3% | 0.0% | 1 | 15.09s |
| What role does the retriever play in RAG? | single | 4 | 100.0% | 0.0% | 0.0% | 2 | 14.05s |
| What role does the retriever play in RAG? | multi_hop | 3 | 100.0% | 33.3% | 0.0% | 1 | 15.47s |