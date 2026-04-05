# Phase 8 Architecture: Dense retrieval + reranker for merchant resolution

**Date:** 2026-04-04

## Motivation

Phase 7 showed that external merchant knowledge helps, but it also exposed the limits of the current lookup path.

The existing KB path uses:

- exact alias matching
- TF-IDF char-ngram nearest-neighbor fallback
- heuristic thresholds for direct category assignment versus metadata enrichment

This works when the merchant alias is already in the KB or only slightly different. It breaks when the merchant exists externally under a different canonical name.

Examples from the audit:

- `GOODLIFE CLUBS` exists externally as `GoodLife Fitness`
- `WENDY'S PF KANA` exists externally as `Wendy's`
- `ZARA BAYSHORE` exists externally as `Zara`
- `BAYSHORE SHOPPIN` exists externally as `Bayshore Centre`
- `SHELBYS 43` exists externally as `Shelby's Legendary Shawarma`
- `UNI OTT TUITION` exists externally as `University of Ottawa`

The question is no longer whether external knowledge helps. Phase 7 already answered that. The remaining issue is how to retrieve the correct merchant candidate when the bank string is abbreviated, truncated, location-heavy, or aliased.

That is primarily a retrieval problem, not a classification problem.

## Requirements for the next architecture

The next retrieval layer must:

- use external and public data only
- avoid any dependency on `codex_labeled.csv` labels
- run locally on consumer hardware
- stay within the repo's on-device privacy constraints
- improve recall without losing the precision of strong direct KB hits
- support English and at least tolerate Canadian and French merchant metadata
- fall back cleanly to the classifier when retrieval is uncertain

## Proposed direction

The proposed change is to move from the current JSON lookup to a local retrieve-and-rerank setup.

This is **not** classic generative RAG. Text generation is not required here. The retrieval layer is simply a better merchant resolver.

The proposed stack is:

1. Exact aliases and deterministic rules
2. Dense retrieval over merchant documents
3. Lexical retrieval in parallel
4. Candidate fusion
5. Reranking with a cross-encoder
6. A final decision step:
   - direct category if confidence is strong
   - metadata-enriched classifier fallback if the candidate is plausible but not decisive
   - no retrieval hit if uncertainty remains too high

## Why ChromaDB

Chroma is a good fit for the local persistent vector store because it:

- runs on-device
- supports persistent collections
- supports dense vector search
- stores documents and metadata together
- is simple enough for a single-machine pipeline

Chroma is not the full solution by itself. Dense similarity alone is still too weak for messy merchant aliases. The main architectural change is the retrieve-and-rerank pattern, not the database choice by itself.

## Retriever choice

### Dense retriever: `BAAI/bge-small-en-v1.5`

This model is a good fit because it is:

- strong for its size
- MIT licensed
- English-focused, which is acceptable here because bank descriptions and most place metadata are short and mostly English-heavy
- lighter than the larger BGE models
- realistic to run locally while embedding the Foursquare Canada subset

Compared with the current char-ngram fallback, this should perform better on cases such as:

- `GOODLIFE CLUBS` -> `GoodLife Fitness`
- `UNI OTT TUITION` -> `University of Ottawa`
- `SHELBYS 43` -> `Shelby's Legendary Shawarma`

## Reranker choice

### Cross-encoder reranker: `BAAI/bge-reranker-v2-m3`

This model is a good fit because it is:

- multilingual
- Apache 2.0 licensed
- built for reranking
- efficient enough to score a small candidate set on local hardware
- better suited than pure embedding similarity for short and ambiguous merchant strings

The reranker only needs to score a small set, such as the top 20 to 50 candidates from the retrievers. That should keep latency reasonable while still allowing a stronger final ranking step.

### Rejected option

`jinaai/jina-reranker-v2-base-multilingual` is a plausible option, but the model card uses `cc-by-nc-4.0`, which adds a non-commercial restriction. That is not an ideal default dependency for this repo.

## Hybrid retrieval strategy

Dense retrieval alone is not enough. Some merchant strings are short, noisy, or mostly lexical:

- `A & W KA`
- `NSLSC`
- `PAYPAL`
- `AMAZON.CA PRIME`

For that reason, the retriever should be hybrid.

### 1. Dense retrieval

Embed each merchant document and retrieve semantically similar candidates.

### 2. Lexical retrieval

Use BM25 or something similar over:

- canonical merchant name
- aliases
- location-free aliases
- metadata text
- Foursquare category labels

### 3. Candidate fusion

Take the union of the dense top-k and lexical top-k results and fuse them in application code.

The local version should not depend on Chroma Cloud-specific hybrid features. Candidate fusion should be handled directly in Python so the implementation remains portable and fully local.

## Merchant document design

Each merchant in the retrieval store should include:

- canonical name
- aliases
- normalized aliases
- stripped aliases with city, province, or trailing branch info removed where possible
- source such as `foursquare`, `curated_public`, and later other sources
- raw Foursquare category labels
- mapped project category if available
- mapping confidence
- locality, region, and country
- a short free-text metadata summary

Example document:

- canonical: `University of Ottawa`
- aliases: `uOttawa`, `University of Ottawa`, `University of Ottawa Medical Centre`
- metadata: `public university in Ottawa, Ontario; tuition, campus services, student fees`
- mapped category: maybe none for a direct route, but still useful for metadata enrichment

This is much richer than the current format of one canonical name, one normalized alias, and one metadata string.

## Full pipeline design

### Stage 1: Keep the existing deterministic tiers

- direction detection
- rules engine

These tiers should remain unchanged because they are inexpensive and highly precise.

### Stage 2: Merchant retrieval

Input: cleaned merchant string

1. Query the dense retriever
2. Query the lexical retriever
3. Fuse the top candidates
4. Rerank them with the cross-encoder

Output:

- top candidate
- reranker score
- margin over the next-best candidate
- candidate metadata

### Stage 3: Decision gate

If the top reranked candidate is very strong:

- return a direct category through `knowledge_base`

If the candidate looks plausible but not decisive:

- append the top candidate metadata to the transaction text
- send that text to the fine-tuned MiniLM
- return `finetune_metadata`

If retrieval is weak:

- do not use merchant memory
- fall back to the normal fine-tuned classifier

This keeps the same overall decision structure as Phase 7 while improving the retrieval step.

## Data ingestion changes

In Phase 7, Foursquare was filtered using the local transaction corpus as the candidate-name set. That created a transductive setup and still missed many merchants because the names did not normalize the same way.

For Phase 8, external data should be ingested differently:

1. Build the retrieval store directly from the external Canada subset
2. Do not prefilter it using benchmark merchant identities
3. Generate aliases and normalized forms from the external data itself
4. Add curated public merchants as extra documents instead of a separate matching path
5. Optionally keep separate collections for:
   - place merchants
   - financial or payment providers
   - digital subscriptions and online services

That should remove the main benchmark contamination issue from Phase 7.

## Expected benefits

This architecture should improve recall on the failure patterns already observed:

- merchants that exist externally under different canonical names
- merchants with shortened or branch-specific bank strings
- place names that need location or context for disambiguation
- short strings where exact matching is too brittle

It should also reduce dependence on hand-picked curated merchants by making broader use of the external place corpus.

## What it will not solve

Even with retrieve-and-rerank, a place-based KB will not solve everything.

Some transactions are not really place merchants at all:

- transfers
- bank artifacts
- card payments
- digital merchants such as Amazon Prime, AWS, Steam, or Hoyoverse

Those still need one or more of:

- dedicated rules
- separate provider collections
- external sources beyond Foursquare

Phase 8 should not be treated as a replacement for the classifier. It is a replacement for a brittle merchant lookup step with a stronger retrieval layer.

## Evaluation plan

This version should be evaluated more carefully than Phase 7.

### Retrieval metrics

- recall@1, recall@5, and recall@10 for merchant resolution
- MRR for merchant candidate ranking
- direct-route precision
- metadata-route precision

### Classification metrics

- overall unique accuracy
- weighted accuracy
- source breakdown
- category breakdown

### Holdout discipline

- do not build the retrieval store from `codex_labeled.csv`
- do not filter the external store using benchmark merchant identities
- keep a clean unseen-merchant evaluation split if possible

## Concrete implementation target

The first Phase 8 implementation should include:

1. A local Chroma persistent collection for merchant documents
2. Dense embeddings with `BAAI/bge-small-en-v1.5`
3. A local lexical retriever such as BM25
4. Manual candidate fusion
5. A cross-encoder reranker with `BAAI/bge-reranker-v2-m3`
6. The existing ensemble decision logic, but fed with reranked merchant candidates instead of the current nearest-match JSON lookup

## Conclusion

Phase 7 showed that merchant knowledge is useful. Phase 8 should treat merchant resolution as a retrieval problem rather than relying on approximate string matching.

The recommended architecture is:

- local Chroma dense index
- local lexical retrieval
- reranker over fused candidates
- the existing classifier as the final decision-maker when retrieval is uncertain

This is the most direct path to improving merchant recall without giving up the precision already achieved on strong direct hits.

## Reference links

- Chroma Python client / persistent collections: https://docs.trychroma.com/reference/python/client
- Chroma embedding functions: https://docs.trychroma.com/docs/embeddings/embedding-functions
- Chroma query docs: https://docs.trychroma.com/docs/querying-collections/query-and-get
- Chroma Search API / hybrid search notes: https://docs.trychroma.com/cloud/search-api/overview
- `BAAI/bge-small-en-v1.5`: https://huggingface.co/BAAI/bge-small-en-v1.5
- `BAAI/bge-reranker-v2-m3`: https://huggingface.co/BAAI/bge-reranker-v2-m3
- `jinaai/jina-reranker-v2-base-multilingual` (license caveat): https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual
