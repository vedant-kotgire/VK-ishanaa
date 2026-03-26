# Ishanaa Analytics Dashboard

**Data-Driven Decision Making for a D2C Short Kurti Brand — UAE Market**

## 🚀 Deploy on Streamlit Cloud

1. Push this folder to a **GitHub repository**
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set **Main file path** to `app.py`
5. Deploy!

## 🖥️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📁 Project Structure

```
ishanaa_app/
├── app.py                          # Main Streamlit application (6 pages)
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── .streamlit/
│   └── config.toml                 # Theme & server config
└── data/
    ├── ishanaa_survey_raw.csv      # 2,000 rows × 29 cols (human-readable)
    ├── ishanaa_survey_encoded.csv  # 2,000 rows × 141 cols (ML-ready)
    └── ishanaa_data_dictionary.csv # 139 column definitions
```

## 📊 Dashboard Pages

| Page | What It Does |
|------|-------------|
| **📊 Descriptive** | Demographics, spending, preferences, pain points, pack interest |
| **🔍 Diagnostic** | K-Means/DBSCAN/Hierarchical clustering, Apriori association rules, correlation heatmap |
| **🔮 Predictive** | Classification (4 models compared), Regression (5 models), feature importance |
| **💡 Prescriptive** | Segment priority scorecard, discount matrix, auto-generated launch playbook |
| **🆕 Predict New** | Single customer form OR bulk CSV upload → instant predictions + downloadable results |
| **ℹ️ Dictionary** | Data dictionary, methodology, persona breakdown |

## 🔬 Algorithms Used

- **Classification:** Logistic Regression, Random Forest, XGBoost, SVM
- **Clustering:** K-Means (with Elbow + Silhouette), DBSCAN, Agglomerative Hierarchical
- **Association Rules:** Apriori (4 basket types: Style×Fabric×Color, Sleeve×Neck×Length, Occasion×Pack×Discount, Pain×Channel×Brand)
- **Regression:** Linear, Ridge, Lasso, Random Forest, Gradient Boosting

## 👥 Dataset: 2,000 Synthetic Respondents

- 6 latent personas with conditional probability distributions
- 5 noise types: persona bleed, contradictions, spending outliers, missing values, lazy respondents
- UAE-calibrated: AED pricing, local channels, South Asian expat demographics
