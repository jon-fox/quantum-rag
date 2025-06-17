# Quantum Reranker Test Queries

## 1. Simple Factual Retrieval
- What's the peak load recorded in ERCOT on June 15, 2025?
- Define Day-Ahead Market (DAM) HubAvg price.

## 2. Paraphrase Robustness
- How much load did ERCOT forecast versus actual generation on June 15, 2025?
- Difference between projected and real-time output in ERCOT mid-June 2025.

## 3. Ambiguous Queries
- June peak mismatch causes
- May renewable generation impact

## 4. Comparative / Analytical
- Compare renewable share influence on forecast errors in April vs. June 2025.
- Which month had a larger over-generation: January or March 2025?

## 5. Multi-Hop / Contextual
- Which days had > 45 000 MW load forecasts and also > 50 % renewables?
- On days when renewables exceeded 40 %, how close was actual to forecast?

## 6. Domain-Distractor Resilience
- Peak forecast error statistics  
- ERCOT SCED dispatchable headroom trends  
*(mix in docs about prices, temperatures or ancillary services as distractors)*

## 7. Long-Tail / Rare Events
- Describe load vs. generation behavior during the February 2025 cold snap.
- How did ECRSS max offers affect forecast accuracy on May 3–4, 2025?

## 8. Adversarial / Keyword-Stuffed
- forecast telemetry peak telemetry forecast peak telemetry forecast

## How to Use
1. **Batch-run** each query through your RAG + quantum reranker pipeline.  
2. **Inspect** the top 5 results: is the most relevant doc at #1?  
3. **Score** overall performance using NDCG@5 or MRR.
