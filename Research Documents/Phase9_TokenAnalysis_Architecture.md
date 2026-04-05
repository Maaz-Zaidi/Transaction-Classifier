# Phase 9 Architecture: Token-aware query decomposition for merchant retrieval

**Date:** 2026-04-05

## Problem statement

Phase 8 established that the main remaining issues are in retrieval quality, not corpus coverage. The full Foursquare store has 1.8 million Canadian merchant entries. The system can find the right merchant family in most cases when given a clean canonical name. But real bank transaction strings are not clean canonical names.

Bank strings contain a mix of signal types compressed into a single flat text field. A string like "COSTCO GAS W1263 KANATA ON" contains four distinct kinds of information: COSTCO is the brand (the thing to look up in the KB), GAS is a category-descriptive word (the thing that tells you this was a fuel purchase, not groceries), W1263 is a branch or terminal identifier (noise), and KANATA ON is a location suffix (mostly noise for classification, though the preprocessing step already strips most of these).

The Phase 8 retrieval pipeline treated this string as a flat bag of tokens. The FTS query became `"COSTCO" OR "GAS" OR "W1263" OR "KANATA"`. The dense embedding was computed over the full string. Both of these injected noise into the retrieval and diluted the brand signal.

This is the specific mechanism behind several of the Phase 8 weaknesses:

- "ZARA BAYSHORE" drifted toward Bayshore venues in dense retrieval because BAYSHORE dominated the embedding.
- "WENDY'S PF KANA" drifted toward PF and KANA noise in lexical retrieval.
- "COSTCO GAS W1263" retrieved COSTCO correctly but the ML had no signal that GAS was a category-modifying descriptor.

## Design goals

I wanted a lightweight token decomposition layer that operates between preprocessing and retrieval. It should:

1. Classify each token in a cleaned transaction string as brand, descriptor, location, or noise.
2. Provide a brand-only query for KB lookup (exact alias and dense search).
3. Provide a cleaned FTS query that excludes noise and location.
4. Provide a descriptor context string that can be appended to the ML input for enrichment.
5. Work without any additional model loading. Pure heuristics, dictionaries, and regex.
6. Be fast enough to run per-transaction with no measurable latency impact.

The key design decision was to not use a model for token classification. The vocabulary of bank transaction tokens is small and well-structured. There are about 40 words that carry category-semantic signal (GAS, GROCERY, WHOLESALE, PHARMACY, COFFEE, RESTAURANT, DOWNLOADS, TUITION, and so on). There are about 50 Canadian locations that commonly appear in statements. Noise follows regular patterns (digits, branch codes, single characters). Everything else is a brand token.

A dictionary-based approach is deterministic, debuggable, and adds zero startup latency. A model-based approach would have been more flexible but harder to audit and would have added a dependency on another checkpoint.

## Token roles

I settled on four token roles.

Brand tokens are parts of the merchant name. These are the tokens that should drive KB lookup. COSTCO, TIM, HORTONS, AMAZON, ZARA. In a multi-token merchant like CANADIAN TIRE, both tokens are brand. The analyzer checks 2-token and 3-token windows against KB aliases before classifying individual tokens, so CANADIAN TIRE is recognized as a single brand rather than being split.

Descriptor tokens carry category-semantic signal. These are words that, when added to a brand, modify the expected category. GAS modifies COSTCO from generic to fuel. DOWNLOADS modifies AMAZON from shopping to entertainment. WHOLESALE is a shopping signal. GROCERY is a food signal. TUITION is a government or education signal. WEB SERVICES is a technology signal. The analyzer maintains two dictionaries: a single-token dictionary with about 40 entries and a multi-token dictionary with 6 entries (WEB SERVICES, WEB SERV, DRUG MART, GAS BAR, COIN WASH, FINE FOODS).

Each descriptor word maps to a hint string. GAS maps to "gas station, fuel". WHOLESALE maps to "wholesale warehouse shopping". DOWNLOADS maps to "digital content downloads". These hints are concatenated into a descriptor_context string that gets appended to the ML input.

Location tokens are Canadian cities and neighbourhoods that commonly appear in bank statements. OTTAWA, TORONTO, KANATA, BAYSHORE, NEPEAN, BARRHAVEN, and about 45 others. Most of these are already stripped by the preprocessing step, but some leak through (particularly when they appear in the middle of the string rather than as a trailing suffix). Location tokens are excluded from both FTS and brand queries.

Noise tokens are branch identifiers, terminal IDs, short codes, and numeric fragments. W1263, S123, P456, single characters, and pure digit sequences. These are excluded from all retrieval queries.

## Query construction

The decomposed tokens feed into three different query paths.

For exact alias lookup, the brand_query (brand tokens only) is tried in addition to the full cleaned text. If the full text "COSTCO GAS W1263" does not match any alias exactly, the brand query "COSTCO" is tried. This catches cases where the brand exists in the KB but the full bank string does not.

For FTS5 lexical search, the _fts_query_from_decomposed method builds the query from brand and descriptor tokens only. Noise and location tokens are excluded. This reduces false matches from branch codes and location terms.

For dense retrieval, the brand_query is used instead of the full text. This gives the dense embedder a cleaner input that is more likely to match the merchant's canonical name or aliases in the embedding space.

## Descriptor context injection

The descriptor context serves a dual purpose.

When the KB finds a match, the descriptor context is appended alongside the KB metadata. The ML receives something like "COSTCO GAS. merchant identity: Costco. external metadata: warehouse retailer. descriptor context: gas station, fuel". This gives the model both the general merchant type (warehouse retailer) and the transaction-specific signal (this particular visit was for gas).

When the KB does not find a match, the descriptor context is still injected. The ML receives "UNKNOWN MERCHANT GAS. descriptor context: gas station, fuel". This is a weaker signal than a full KB hit, but it is better than nothing. The ML model at least gets the hint that this transaction involves fuel.

The decision to inject descriptor context even without a KB match was intentional. There are 82 transactions in the test set where the KB cannot find a match. For any of those that contain descriptor words, the descriptor context is the only external signal available.

## Multi-token handling

The analyzer handles multi-token constructs at two levels.

Multi-token descriptors are checked before individual token classification. The full uppercase string is scanned for each phrase in the multi-token descriptor dictionary. When a phrase like WEB SERVICES is found, all constituent tokens are marked as descriptors and consumed. This prevents WEB from being classified as a brand and SERVICES from defaulting to brand.

Multi-token brands are checked against KB aliases. For window sizes of 3 and 2, the analyzer checks whether consecutive tokens match a known alias. If CANADIAN TIRE exists as an alias in the KB, both tokens are marked as brand. This prevents TIRE from being classified as something else and ensures the brand query includes the full merchant name.

The multi-token brand check only runs when a knowledge base is provided to the analyzer. When the analyzer runs without a KB (for example, in unit tests), multi-token brand detection is skipped and each token is classified individually.

## Interaction with the rules engine overhaul

The token analyzer was designed alongside the rules engine transition. The two changes are complementary.

In earlier phases, the rules engine carried merchant-identity rules as scaffolding while the KB and ML components were being built out. With those components now operational, the merchant-identity rules were retired and the rules engine returned to its intended scope: structural patterns (TRANSFER, MORTGAGE, REFUND) and generic descriptors (PARKING, PHARMACY, GYM). The KB now handles merchant identity.

The token analyzer handles a similar set of generic descriptors (GAS, GROCERY, WHOLESALE) but in a different way. The rules engine produces a final classification. The token analyzer produces a context signal that feeds into the ML. There is some overlap in vocabulary (PHARMACY appears in both the rules and the descriptor dictionary), but their roles are different. The rules engine catches PHARMACY as a standalone transaction and classifies it directly as Healthcare. The token analyzer catches PHARMACY as a token within a longer string and adds "pharmacy drugstore" to the descriptor context for the ML.

This overlap is by design. If a transaction is just "PHARMACY", the rules engine catches it and the token analyzer never runs (because the ensemble short-circuits after rules). If a transaction is "SHOPPERS PHARMACY OTTAWA", the rules engine does not match (SHOPPERS is not a structural pattern), the token analyzer runs and identifies PHARMACY as a descriptor, and the ML receives the descriptor context.

## Implementation choices

I chose to implement the analyzer as a class with a single analyze method rather than a set of standalone functions. This allows the constructor to be extended with configuration (for example, loading additional descriptor dictionaries from a file) without changing the call sites.

The dictionaries are module-level constants rather than instance variables. This keeps the common case fast and avoids per-instance memory allocation for data that does not change.

The DecomposedQuery dataclass is frozen and uses slots for memory efficiency. In a batch of 505 transactions, the decomposition adds negligible overhead (well under 1ms total).

The _CANADIAN_LOCATIONS set includes some entries that overlap with the preprocessing location stripper. This is intentional redundancy. The preprocessing step strips location suffixes (city followed by province code at end of string), but it does not catch location words in the middle of the string. The token analyzer catches those.

## Limitations

The descriptor dictionary is manually curated and covers about 40 words. There are certainly category-descriptive words that are not in the dictionary. However, the dictionary is easy to extend, and the current set covers the most common cases seen in Canadian bank statements.

The multi-token brand detection depends on the KB being loaded. In offline or testing scenarios where the KB is not available, multi-token brands like CANADIAN TIRE will not be detected and each token will be classified independently. This is acceptable for the current use case since the KB is always loaded in production.

The analyzer does not handle negation or modification. "NO GAS" would still classify GAS as a descriptor. This has not been an issue in practice because bank transaction strings rarely contain negation.

The location set is Canada-specific. If the system were deployed for transactions from other countries, the location set would need to be extended or made configurable.
