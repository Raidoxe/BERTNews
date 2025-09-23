Modify my recommendation algorithm to use sparse, gated updates for user preference learning.

# Requirements
1. **Sparsify article vectors**
   - For each article vector s(t), keep only labels where s_i >= τ (typical τ in [0.03, 0.10]).
   - Set other entries to 0. Optionally also allow top-K pruning.

2. **Scoring**
   - User profile u ∈ [-1,1]^k, where positive values mean likes, negative values mean dislikes.
   - Article score = u ⋅ s̃(t), where s̃ is the sparsified vector.

3. **Feedback**
   - When user feedback is like/dislike, encode y ∈ {+1, -1}.
   - Update rule (per label i):
     - If s_i >= τ:
       u_i ← u_i + α * y * (s_i^γ)    # α = learning rate, γ ≥ 1 for confidence scaling
     - Else (s_i < τ):
       u_i ← u_i * (1 - decay)        # small decay toward 0 to prevent drift
   - Clip u_i into [-1,1].

4. **Parameters**
   - Learning rate α (default 0.1).
   - Threshold τ (default 0.05).
   - Decay (default 0.01).
   - Confidence exponent γ (default 1.0).

# Acceptance criteria
- Only meaningful labels (above threshold) change weights.
- Neutral/irrelevant labels stay neutral instead of drifting.
- Disliked labels decrease user weight if they appear in disliked articles.
- Positive labels increase weight on liked articles.
- User profile remains bounded in [-1,1].

Implement this as a reusable update function with clear docstrings and unit tests.
