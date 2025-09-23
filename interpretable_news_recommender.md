# Project Overview: Interpretable News Recommendation System

## 1. Project Goal
The aim of this project is to build an **interpretable, human‑in‑the‑loop news recommendation engine**. Unlike opaque, black‑box recommender systems, this engine will transparently present users with the features that influence their recommendations (e.g., *Donald Trump*, *Ukraine War*, *Climate Change*). Users can directly understand, adjust, and control how these features influence their personalized news feed.

---

## 2. Core Idea
- Each **news article** is represented by a set of **interpretable features** (keywords, named entities, topics).  
- Each **user** has a corresponding weight vector that reflects their interest in those features.  
- The **recommendation score** for an article is computed as a simple **linear combination** of article features and user weights.  
- The UI presents feature contributions clearly, allowing users to adjust preferences directly.

---

## 3. Workflow

### 3.1 Article Representation
- Extract features automatically using NLP techniques:
  - Named Entity Recognition (NER) (e.g., *Donald Trump*, *Ukraine*, *New York*)
  - Topic modeling (e.g., *US Politics*, *Economy*, *Climate Change*)
- Each article becomes a sparse feature vector:
  ```
  Article: "Trump meets Zelensky in New York"
  Features: {"Donald Trump": 1, "Zelensky": 1, "New York": 1, "US Politics": 0.8, "Ukraine War": 0.7}
  ```

### 3.2 User Representation
- Each user has a personalized weight vector initialized to neutral values (0).
- Example:
  ```
  User Weights: {"Donald Trump": +0.4, "Zelensky": -0.2, "Ukraine War": +0.3}
  ```

### 3.3 Recommendation Scoring
- Compute score as:
  ```
  score(article, user) = Σ (feature_value × user_weight)
  ```
- Transparent explanation for each recommendation:
  ```
  Why this article?
  +0.45 Donald Trump
  +0.30 Ukraine War
  -0.10 Sports
  = 0.65 total score
  ```

### 3.4 Feedback and Learning
- Users give **thumbs up / thumbs down** feedback.
- Everytime a new topic / relevant embedding is found it is added to the user's weight vector with an initial value of 0.
- Feedback updates the user’s feature weights:
  - Perceptron‑style: increase weights for liked features, decrease for disliked.
  - Or Logistic Regression (online SGD) for smoother updates.
- Users can also **manually adjust weights** (e.g., lower influence of *Donald Trump*).

---

## 4. Interpretability
- Every recommendation is backed by **visible, named features**.
- No hidden embeddings are exposed directly to users.
- Users can explicitly see and override how topics influence their feed.

---

## 5. Technology Stack
- **Feature Extraction:** NER, BERTopic for topic discovery.
- **Learning:**
  - Scikit‑Learn `SGDClassifier` for online logistic regression, or
  - Simple custom perceptron‑style updater.
- **Storage:** User profiles stored as weight vectors (JSON / database).
- **Frontend:** Sliders or toggles for each interpretable feature weight.
- **Backend:** Python API to score, update, and serve recommendations.

---

## 6. Expected Behavior
- After ~5–10 interactions, users see noticeable personalization.
- After ~30 interactions, recommendations align strongly with preferences.
- After ~100+ interactions, profiles are robust and stable.
- System adapts continuously to new feedback, allowing for **concept drift** (shifting user interests).

---

## 7. Benefits
- **Transparency:** Users see why articles are recommended.
- **Control:** Users can adjust preferences directly.
- **Simplicity:** Linear models keep computation efficient and explanations clear.
- **Adaptability:** Online updates allow personalization from very little data.

---

## 8. Next Steps
1. Build prototype with spaCy (NER) + scikit‑learn (logistic regression).
2. Implement feature extraction and scoring pipeline.
3. Design a frontend to visualize feature contributions and allow user control.
4. Test with small news dataset (e.g., 10k articles) and simulated feedback.
5. Evaluate convergence speed (how quickly personalization becomes meaningful).
6. Deploy MVP and refine based on user testing.

---

**In short:** This project is about making recommendations *interpretable, adjustable, and user‑centric*. It blends NLP for feature discovery, linear modeling for transparent scoring, and human‑in‑the‑loop learning for personalization.

