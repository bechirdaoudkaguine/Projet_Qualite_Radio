# Projet Qualité Radio — Streamlit + carte

L'application contient :
- prédiction avec la régression logistique ;
- carte interactive des 5000 mesures synthétiques ;
- filtres par classe de qualité ;
- popup avec RSRP, RSRQ, SINR, débits et latence.

Lancer :
```bash
pip install -r requirements.txt
streamlit run app.py
```

Structure :
```text
Projet_Qualite_Radio/
├── app.py
├── requirements.txt
├── dataset/dataset_radio.csv
├── models/logistic_regression.pkl
└── models/scaler.pkl
```
