You are GPT-5. Extend the recommendation system with online learning from user feedback.

# Setup
- Each article has a topic-score vector s(t) = [s1, s2, …, sk] from zero-shot NLI classification over user-defined labels.
- Each user has a profile vector u = [w1, w2, …, wk], initially the average of liked articles.

# Recommendation
- Recommend new articles by computing similarity(u, s(t)) (dot or cosine).
- Return top candidates with label explanations.

# Feedback loop
- When a user is shown an article and votes:
  - If "like": increase weights in u for labels where s(t) is high.
  - If "dislike": decrease weights in u for labels where s(t) is high.
- Apply a learning rate α (configurable, default 0.1):
  - u ← u + α * (± s(t) - u)   # + for like, – for dislike
- Normalize u to keep weights in [0,1].

# Goal
Over time, u should reflect the user’s evolving preferences on their chosen labels.
Store updated profile in DB after each interaction.
