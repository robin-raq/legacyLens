# LegacyLens AI Cost Analysis

## Development & Testing Costs

Actual spend tracked during the full development cycle (MVP + Final).

| Item | Tokens | Cost |
|------|--------|------|
| OpenAI embeddings (ingestion) | ~100K tokens (169 chunks) | ~$0.002 |
| OpenAI embeddings (queries, ~80 test queries) | ~8K tokens | ~$0.0002 |
| Anthropic Claude (MVP answer gen, ~30 queries) | ~200K input + ~30K output | ~$1.00 |
| Google Gemini 2.5 Flash (final, ~50 queries + eval) | ~300K input + ~50K output | ~$0.10 |
| Pinecone | Free tier (169 vectors / 100K limit) | $0.00 |
| Railway | Hobby plan | $5.00/mo |
| **Total development** | | **~$6.10** |

**Key insight:** Switching from Claude Sonnet (~$3/1M input) to Gemini 2.5 Flash (~$0.15/1M input) cut LLM generation costs by ~95% with comparable answer quality (96% term recall vs 97% on Claude Haiku).

## Production Cost Projections

**Assumptions:**
- 5 queries per user per day
- ~500 tokens per query embedding
- ~2,000 tokens input + ~500 tokens output per LLM generation
- Pinecone free tier covers up to 100K vectors
- Gemini 2.5 Flash pricing: $0.15/1M input, $0.60/1M output (as of March 2026)

| Scale | Queries/mo | Embedding Cost | LLM Cost | Pinecone | Railway | **Total/month** |
|-------|-----------|---------------|----------|----------|---------|----------------|
| 100 users | 15,000 | $0.002 | $6.00 | $0 (free) | $5 | **~$11** |
| 1,000 users | 150,000 | $0.02 | $60 | $0 (free) | $20 | **~$80** |
| 10,000 users | 1,500,000 | $0.15 | $600 | $70 (starter) | $50 | **~$720** |
| 100,000 users | 15,000,000 | $1.50 | $6,000 | $230 (standard) | $200 | **~$6,430** |

## Cost Breakdown by Component

**LLM generation is the dominant cost** at every scale (>80% of total). Optimization strategies:

1. **Caching** -- cache common query embeddings and full responses for repeated questions
2. **Smaller models for simple queries** -- route EXPLAIN queries to a cheaper model; use Gemini Flash for complex features (Pattern, Translation)
3. **Client-side result caching** -- session memory prevents re-querying the same information
4. **Context window optimization** -- score-gap filtering reduces irrelevant chunks sent to the LLM, reducing input tokens

## Cost Comparison: Claude vs Gemini

The architecture switch from Claude Sonnet to Gemini 2.5 Flash had significant cost implications:

| Model | Input/1M tokens | Output/1M tokens | Cost per query (est.) | 100K queries/mo |
|-------|----------------|-------------------|----------------------|-----------------|
| Claude Sonnet | $3.00 | $15.00 | $0.014 | $1,400 |
| Claude Haiku | $0.25 | $1.25 | $0.001 | $100 |
| Gemini 2.5 Flash | $0.15 | $0.60 | $0.0006 | $60 |

Gemini 2.5 Flash provides the best cost-to-quality ratio for this use case. Answer quality (96% term recall) is comparable to Claude Haiku (97%) at roughly half the cost.

## Embedding Cost for New Code Additions

If the BLAS codebase were updated with new source files:

- **Current index:** 169 chunks, ~100K tokens embedded = $0.002
- **Incremental update (10 new files):** ~10K additional tokens = $0.0002
- **Full re-index:** Delete and re-upsert all vectors = $0.002

Embedding costs are negligible at any scale. The one-time ingestion cost is under a penny.
