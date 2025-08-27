# Bluesky Reposting Prediction

> End-to-end codebase for training and evaluating models to predict reposting behavior on Bluesky across topics. Primary metric: **binary F1 @ threshold 0.5**.

---

## TL;DR
- **Goal:** Predict whether a user will repost a given message on a decentralized platform (Bluesky).
- **Data:** 11 trending topics, raw API dumps (May 15–June 1, 2025) + a processed **1:5** (pos:neg) training set per topic.
- **Models:** In-distribution: 1. **MTX** 2. **UHM+UHU** 3. **UHU**
              Out-of-distribution: 1. **UHU** 2. **MTX**
- **Evaluation:** In-distribution (ID) per-topic and **leave-one-topic-out OOD**, reported as **F1 (positive class) at fixed 0.5**.
- **Repro:** Deterministic seeds, logged configs, saved checkpoints; minimal dependencies.

---

## Repository Structure
