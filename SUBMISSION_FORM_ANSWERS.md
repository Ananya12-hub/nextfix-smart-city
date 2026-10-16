# Answers for the Round 2 submission form

- **Product name:** NextFix
- **One-line pitch:** NextFix ranks city locations for repair and explains every rank, with confidence, gaps and a fairness check.
- **GitHub repository URL:** (your public repo URL, e.g. https://github.com/<user>/nextfix-smart-city)
- **Branch:** main
- **ML model type:** Trained from scratch
- **What does your model predict?** High/Critical incidents per location in the next window (90-day forecast with a predictive range).
- **Algorithm used:** Gamma-Poisson empirical-Bayes model (implemented from scratch in NumPy/SciPy); compared against Poisson regression and gradient boosting.
- **Training code file in your repo:** train.py (model in src/nextfix/model.py)
- **Result (optional):** On a time-based split no model beats the global-mean baseline (Poisson deviance about 1.08 for both); the dataset shows no learnable location-level signal (split-half correlation about 0). Details in reports/RESULTS.md.
- **APIs used:** None
- **Extra data used:** None
- **Dataset files your code uses:** locations.csv, incidents.csv, public_complaints.csv
- **How to run:** pip install -r requirements.txt && python train.py   (then open app/index.html)
- **Declaration:** tick both boxes (repo is public; every API/pretrained model listed, no keys committed).
