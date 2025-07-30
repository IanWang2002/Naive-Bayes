🧠 Naive Bayes, No Training Wheels
Built a Naive Bayes classifier from scratch (🚫 no NumPy, no scikit-learn) to classify animals in the classic Zoo dataset.
This isn't your plug-and-play model — it's manual probability modeling with:

🔢 Log-space math to avoid underflow

🧮 Laplace smoothing (pseudo-count = 0.1) for unseen feature handling

🧬 16 biological traits per animal, from 🦴 backbone to 🐾 legs

The classifier estimates priors, computes likelihoods, and predicts classes — all in a self-contained, explainable pipeline.
Great for learning how real machine learning works under the hood.

