You are GPT-5. Refactor this repo so users can enter their own topic labels and get interpretable recommendations.

# Requirements
- For each article (title+description), compute zero-shot multi-label scores vs user-entered labels (use HuggingFace `facebook/bart-large-mnli`, multi_label=True).
- Cache results keyed by hash of label set to avoid recomputing.
- Store topic vectors (aligned to labels) in DB.
- Build user profiles by averaging/summing vectors from liked articles.
- Recommend new articles by similarity (dot or cosine) between profile and candidate vectors.
- Each recommendation must return an explanation: top contributing labels with weights.

# API (FastAPI or Express)
1. `POST /topics/score_batch` → return topic scores per article.
2. `POST /profiles/from_interactions` → create/update a user profile vector.
3. `POST /reco/rank` → rank candidate articles for a user, include label explanations.

# Config
- MODEL_NAME (default bart-large-mnli)
- TOPK (default 10)
- MIN_SCORE (default 0.05)
- SIMILARITY (dot or cosine)

# Acceptance
Given ad-hoc user labels, system must:
- Score articles against them,
- Build user profile,
- Return ranked recommendations with top-K topic explanations,
- Use cache when possible.

Implement clean, typed code with docstrings and basic tests.
