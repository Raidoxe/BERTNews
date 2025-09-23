Modify the recommendation system so user preferences for each label are embedded directly in the scoring function.

# Requirements
- Each article has a topic vector s(t) ∈ [0,1]^k (NLI entailment probabilities per label).
- Each user has a preference vector u ∈ ℝ^k where:
  - u_i > 0 means the user likes label i (strength of preference).
  - u_i < 0 means the user dislikes label i (strength of avoidance).
  - u_i = 0 means neutral.

# Scoring
- Replace cosine similarity with a weighted dot product:
  score(t) = ∑ u_i * s_i(t)
- Articles with high scores on disliked labels (u_i < 0) are penalized automatically.

# Feedback update
- When user likes an article: u ← u + α * (s(t) - u)
- When user dislikes an article: u ← u - α * (s(t) + u)
- α is learning rate (e.g. 0.1). Normalize u to keep it bounded (e.g. in [-1,1]).

# Explanation
- For any recommendation, compute contributions:
  w_i(t) = u_i * s_i(t)
- Sort by |w_i(t)| and display top labels, noting positive = boosted score, negative = penalized.

# Acceptance criteria
- User profile directly encodes scale values per label.
- Disliked labels lower article scores during ranking, not just afterwards.
- System updates profile vector online as users like/dislike articles.
