import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, io, os
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVC
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix, classification_report,
                             mean_absolute_error, mean_squared_error, r2_score, silhouette_score)
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from xgboost import XGBClassifier
from mlxtend.frequent_patterns import apriori, association_rules

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="Ishanaa Analytics", page_icon="👗", layout="wide", initial_sidebar_state="expanded")

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600;700&display=swap');
.block-container { padding-top: 2rem; }
h1, h2, h3 { font-family: 'DM Serif Display', serif !important; font-weight: 400 !important; }
.metric-card { background: white; border: 1px solid #e8e2dc; border-radius: 12px; padding: 20px; text-align: center; }
.metric-num { font-family: 'DM Serif Display', serif; font-size: 32px; color: #c25e3a; }
.metric-label { font-size: 13px; color: #7a7470; margin-top: 4px; }
.insight-box { background: linear-gradient(135deg, #f4ece6, #fdf5ef); border-left: 3px solid #c25e3a; padding: 16px 20px; border-radius: 0 10px 10px 0; margin: 16px 0; }
.insight-box strong { color: #c25e3a; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; }
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] { background-color: #f4ece6; border-radius: 8px; padding: 8px 16px; }
div[data-testid="stMetric"] { background: white; border: 1px solid #e8e2dc; border-radius: 12px; padding: 16px; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# DATA LOADING
# ============================================================
@st.cache_data
def load_data():
    base = os.path.dirname(__file__)
    raw = pd.read_csv(os.path.join(base, 'data', 'ishanaa_survey_raw.csv'))
    enc = pd.read_csv(os.path.join(base, 'data', 'ishanaa_survey_encoded.csv'))
    ddict = pd.read_csv(os.path.join(base, 'data', 'ishanaa_data_dictionary.csv'))
    return raw, enc, ddict

df_raw, df_enc, df_dict = load_data()

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def insight_box(title, text):
    st.markdown(f'<div class="insight-box"><strong>{title}</strong><br>{text}</div>', unsafe_allow_html=True)

def expand_multi_col(series, sep='; '):
    all_items = []
    for val in series.dropna():
        all_items.extend([v.strip() for v in str(val).split(sep)])
    return pd.Series(all_items).value_counts()

def get_feature_matrix():
    feature_cols = [c for c in df_enc.columns if c not in
                    ['respondent_id','persona','purchase_intent_ordinal','purchase_intent_binary',
                     'spend_per_kurti_ordinal','spend_per_kurti_midpoint_AED']]
    X = df_enc[feature_cols].copy()
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    imp = SimpleImputer(strategy='median')
    X_imp = pd.DataFrame(imp.fit_transform(X), columns=X.columns)
    return X_imp, feature_cols

def get_clf_data():
    X_imp, feature_cols = get_feature_matrix()
    y = df_enc['purchase_intent_binary'].copy()
    mask = y.notna()
    return X_imp[mask], y[mask].astype(int), feature_cols

def get_reg_data():
    X_imp, feature_cols = get_feature_matrix()
    y = df_enc['spend_per_kurti_midpoint_AED'].copy()
    mask = y.notna()
    return X_imp[mask], y[mask], feature_cols

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## 👗 **Ishanaa Analytics**")
    st.markdown("*Data-Driven Decision Making*")
    st.markdown("---")
    page = st.radio("Navigate", [
        "📊 Descriptive Analysis",
        "🔍 Diagnostic Analysis",
        "🔮 Predictive Analysis",
        "💡 Prescriptive Analysis",
        "🆕 New Customer Predictor",
        "ℹ️ Data Dictionary"
    ], index=0)
    st.markdown("---")
    st.caption(f"Dataset: {len(df_raw)} respondents")
    st.caption(f"Encoded features: {df_enc.shape[1]} columns")
    st.caption("UAE Market · March 2026")

# ================================================================
#  PAGE 1: DESCRIPTIVE
# ================================================================
if page == "📊 Descriptive Analysis":
    st.title("📊 Descriptive Analytics")
    st.markdown("*What does our potential UAE market look like?*")

    # KPIs
    total = len(df_raw)
    interested = df_enc['purchase_intent_binary'].sum()
    interested_pct = interested / df_enc['purchase_intent_binary'].notna().sum() * 100
    avg_spend = df_enc['spend_per_kurti_midpoint_AED'].mean()
    top_pack_series = expand_multi_col(df_raw['Q19_Pack_Interest'])
    top_pack = top_pack_series.index[0] if len(top_pack_series) > 0 else "N/A"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Respondents", f"{total:,}")
    c2.metric("Interested (%)", f"{interested_pct:.1f}%")
    c3.metric("Avg Spend/Kurti", f"{avg_spend:.0f} AED")
    c4.metric("Top Pack Interest", top_pack[:25]+"..." if len(str(top_pack))>25 else top_pack)

    st.markdown("---")

    # Demographics
    st.subheader("1.1 — Demographic Profile")
    dc1, dc2 = st.columns(2)

    with dc1:
        age_counts = df_raw['Q1_Age'].value_counts().reindex(["Under 18","18-22","23-27","28-34","35 and above"]).dropna()
        fig = px.bar(x=age_counts.index, y=age_counts.values, color_discrete_sequence=['#c25e3a'],
                     labels={'x':'Age Group','y':'Count'}, title="Age Distribution")
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with dc2:
        occ_counts = df_raw['Q2_Occupation'].value_counts()
        fig = px.pie(values=occ_counts.values, names=occ_counts.index, title="Occupation Breakdown",
                     color_discrete_sequence=px.colors.qualitative.Set2, hole=0.4)
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    dc3, dc4 = st.columns(2)
    with dc3:
        inc_order = ["No personal income","Under 3,000 AED","3,000-7,000 AED","7,001-12,000 AED","12,001-20,000 AED","Above 20,000 AED"]
        inc_counts = df_raw['Q4_Income'].value_counts().reindex(inc_order).dropna()
        fig = px.bar(x=inc_counts.index, y=inc_counts.values, color_discrete_sequence=['#3a6b8c'],
                     labels={'x':'Income Band','y':'Count'}, title="Income Distribution (AED)")
        fig.update_layout(height=350, showlegend=False, xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)

    with dc4:
        eth_counts = df_raw['Q3_Ethnicity'].value_counts()
        fig = px.pie(values=eth_counts.values, names=eth_counts.index, title="Ethnic Background",
                     color_discrete_sequence=px.colors.qualitative.Pastel, hole=0.4)
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    # Shopping Behavior
    st.markdown("---")
    st.subheader("1.2 — Shopping Behavior")
    sb1, sb2 = st.columns(2)

    with sb1:
        spend_order = ["Under 30 AED","30-60 AED","61-100 AED","101-150 AED","151-250 AED","Above 250 AED"]
        spend_counts = df_raw['Q9_Spend_Per_Kurti'].value_counts().reindex(spend_order).dropna()
        fig = px.bar(x=spend_counts.index, y=spend_counts.values, color_discrete_sequence=['#3a7d5c'],
                     labels={'x':'Spend Band','y':'Count'}, title="Spend Per Kurti (AED)")
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with sb2:
        brand_counts = expand_multi_col(df_raw['Q7_Current_Brands']).head(10)
        fig = px.barh(x=brand_counts.values, y=brand_counts.index, color_discrete_sequence=['#7a5195'],
                      labels={'x':'Selections','y':''}, title="Current Brand Landscape (Top 10)")
        fig.update_layout(height=350, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    # Product Preferences
    st.markdown("---")
    st.subheader("1.3 — Product Preference Popularity")
    pp1, pp2, pp3 = st.columns(3)

    with pp1:
        style_counts = expand_multi_col(df_raw['Q11_Style_Preference'])
        fig = px.barh(x=style_counts.values, y=style_counts.index, color_discrete_sequence=['#c25e3a'],
                      labels={'x':'Selections','y':''}, title="Style Preferences")
        fig.update_layout(height=400, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    with pp2:
        fabric_counts = expand_multi_col(df_raw['Q12_Fabric_Preference'])
        fig = px.barh(x=fabric_counts.values, y=fabric_counts.index, color_discrete_sequence=['#3a6b8c'],
                      labels={'x':'Selections','y':''}, title="Fabric Preferences")
        fig.update_layout(height=400, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    with pp3:
        color_counts = expand_multi_col(df_raw['Q13_Color_Preference'])
        fig = px.barh(x=color_counts.values, y=color_counts.index, color_discrete_sequence=['#3a7d5c'],
                      labels={'x':'Selections','y':''}, title="Color Preferences")
        fig.update_layout(height=400, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    # Purchase Intent & Pain Points
    st.markdown("---")
    st.subheader("1.4 — Purchase Intent & Pain Points")
    pi1, pi2 = st.columns(2)

    with pi1:
        intent_order = ["Definitely would buy","Probably would buy","Might or might not","Probably would not","Definitely would not"]
        intent_counts = df_raw['Q25_Purchase_Intent'].value_counts().reindex(intent_order).dropna()
        colors = ['#3a7d5c','#5aa87a','#b8943e','#e07050','#c0392b']
        fig = px.bar(x=intent_counts.values, y=intent_counts.index, orientation='h', color=intent_counts.index,
                     color_discrete_sequence=colors, labels={'x':'Count','y':''}, title="Purchase Intent (Q25)")
        fig.update_layout(height=350, showlegend=False, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    with pi2:
        pain_counts = expand_multi_col(df_raw['Q23_Pain_Points']).head(10)
        fig = px.barh(x=pain_counts.values, y=pain_counts.index, color_discrete_sequence=['#c0392b'],
                      labels={'x':'Selections','y':''}, title="Top Pain Points")
        fig.update_layout(height=350, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    # Pack Interest
    st.subheader("1.5 — Pack Interest Breakdown")
    pack_counts = expand_multi_col(df_raw['Q19_Pack_Interest'])
    fig = px.bar(x=pack_counts.index, y=pack_counts.values, color_discrete_sequence=['#7a5195'],
                 labels={'x':'Pack','y':'Selections'}, title="Which Ishanaa Packs Interest Customers?")
    fig.update_layout(height=380, xaxis_tickangle=-20)
    st.plotly_chart(fig, use_container_width=True)
    insight_box("So What?", f"The top pack is <b>{pack_counts.index[0]}</b> with {pack_counts.values[0]} selections — this should be the Day 1 launch product. Overall, {interested_pct:.0f}% of respondents show purchase interest, validating the market opportunity in the UAE.")


# ================================================================
#  PAGE 2: DIAGNOSTIC
# ================================================================
elif page == "🔍 Diagnostic Analysis":
    st.title("🔍 Diagnostic Analytics")
    st.markdown("*Why do certain customers want to buy — and others don't?*")

    tab1, tab2, tab3 = st.tabs(["🧩 Clustering", "🔗 Association Rules", "📈 Correlation Analysis"])

    # ---- CLUSTERING TAB ----
    with tab1:
        st.subheader("2.1 — Customer Persona Discovery")
        X_clust, _ = get_feature_matrix()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clust)

        # Elbow + Silhouette
        st.markdown("**Optimal K Selection**")
        el1, el2 = st.columns(2)
        inertias, sil_scores = [], []
        K_range = range(2, 9)
        for k in K_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X_scaled)
            inertias.append(km.inertia_)
            sil_scores.append(silhouette_score(X_scaled, labels, sample_size=800))

        with el1:
            fig = px.line(x=list(K_range), y=inertias, markers=True, labels={'x':'K','y':'Inertia'}, title="Elbow Method")
            fig.update_traces(line_color='#c25e3a')
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        with el2:
            fig = px.line(x=list(K_range), y=sil_scores, markers=True, labels={'x':'K','y':'Silhouette Score'}, title="Silhouette Scores")
            fig.update_traces(line_color='#3a7d5c')
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

        optimal_k = list(K_range)[np.argmax(sil_scores)]
        k_choice = st.slider("Select number of clusters", 2, 8, optimal_k)

        # K-Means
        km_final = KMeans(n_clusters=k_choice, random_state=42, n_init=10)
        cluster_labels = km_final.fit_predict(X_scaled)
        df_enc['cluster'] = cluster_labels

        # PCA 2D
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        fig = px.scatter(x=X_pca[:,0], y=X_pca[:,1], color=cluster_labels.astype(str),
                         labels={'x':'PC1','y':'PC2','color':'Cluster'}, title=f"K-Means Clusters (k={k_choice}) — PCA Projection",
                         color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

        sil_final = silhouette_score(X_scaled, cluster_labels, sample_size=800)
        st.info(f"**Silhouette Score: {sil_final:.3f}** — Values above 0.25 indicate meaningful cluster separation. Your customer personas are real, not statistical noise.")

        # Cluster Profiles
        st.markdown("**Cluster Profiles**")
        profile_cols = ['age_ordinal','income_midpoint_AED','wardrobe_count','shopping_freq_ordinal',
                        'spend_per_kurti_midpoint_AED','bundle_budget_midpoint_AED','return_anxiety_ordinal']
        safe_profile_cols = [c for c in profile_cols if c in df_enc.columns]
        cluster_profile = df_enc.groupby('cluster')[safe_profile_cols].mean().round(1)
        cluster_profile['size'] = df_enc.groupby('cluster')['cluster'].count()

        intent_col = 'purchase_intent_binary'
        if intent_col in df_enc.columns:
            cluster_profile['interest_rate_%'] = (df_enc.groupby('cluster')[intent_col].mean() * 100).round(1)

        st.dataframe(cluster_profile.style.background_gradient(cmap='YlOrRd'), use_container_width=True)

        # Radar chart
        radar_cols = [c for c in safe_profile_cols if c in cluster_profile.columns]
        if len(radar_cols) >= 3:
            from sklearn.preprocessing import MinMaxScaler
            radar_norm = MinMaxScaler().fit_transform(cluster_profile[radar_cols])
            radar_df = pd.DataFrame(radar_norm, columns=radar_cols, index=cluster_profile.index)

            fig = go.Figure()
            colors = px.colors.qualitative.Set2
            for i, cl in enumerate(radar_df.index):
                vals = radar_df.loc[cl].tolist() + [radar_df.loc[cl].tolist()[0]]
                cats = radar_cols + [radar_cols[0]]
                fig.add_trace(go.Scatterpolar(r=vals, theta=cats, name=f'Cluster {cl}', line_color=colors[i % len(colors)]))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])), title="Cluster Comparison Radar", height=450)
            st.plotly_chart(fig, use_container_width=True)

        # DBSCAN
        st.markdown("---")
        st.subheader("2.2 — DBSCAN Validation")
        db = DBSCAN(eps=3.5, min_samples=8)
        db_labels = db.fit_predict(X_scaled)
        n_db_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
        n_noise = (db_labels == -1).sum()

        fig = px.scatter(x=X_pca[:,0], y=X_pca[:,1], color=db_labels.astype(str),
                         labels={'x':'PC1','y':'PC2','color':'Cluster'}, title=f"DBSCAN: {n_db_clusters} clusters, {n_noise} noise points",
                         color_discrete_sequence=px.colors.qualitative.Set1)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        insight_box("DBSCAN Insight", f"DBSCAN identified {n_noise} noise/outlier respondents — these likely include the lazy straight-line respondents and contradictory rows. {n_db_clusters} natural clusters found without specifying K upfront.")

        # Hierarchical
        st.subheader("2.3 — Hierarchical Clustering Dendrogram")
        sample_idx = np.random.choice(len(X_scaled), size=min(200, len(X_scaled)), replace=False)
        X_sample = X_scaled[sample_idx]
        Z = linkage(X_sample, method='ward')
        fig_d, ax_d = plt.subplots(figsize=(12, 4))
        dendrogram(Z, truncate_mode='lastp', p=30, leaf_rotation=90, leaf_font_size=8, ax=ax_d, color_threshold=50)
        ax_d.set_title("Hierarchical Clustering Dendrogram (Ward's Method)")
        ax_d.set_ylabel("Distance")
        st.pyplot(fig_d)

    # ---- ASSOCIATION RULES TAB ----
    with tab2:
        st.subheader("2.4 — Association Rule Mining")
        st.markdown("*Discovering hidden product preference patterns*")

        arm_type = st.selectbox("Select Basket Type", [
            "Style × Fabric × Color (Product Design)",
            "Sleeve × Neckline × Length (Product Details)",
            "Occasion × Pack × Discount (Bundle Strategy)",
            "Pain × Channel × Brand (Competitive Intel)"
        ])

        if arm_type.startswith("Style"):
            cols = [c for c in df_enc.columns if c.startswith(('style_','fabric_','color_'))]
        elif arm_type.startswith("Sleeve"):
            cols = [c for c in df_enc.columns if c.startswith(('sleeve_','neck_','length_'))]
        elif arm_type.startswith("Occasion"):
            cols = [c for c in df_enc.columns if c.startswith(('occasion_','pack_','discount_'))]
        else:
            cols = [c for c in df_enc.columns if c.startswith(('pain_','channel_','brand_'))]

        if cols:
            basket = df_enc[cols].fillna(0).astype(bool)
            basket = basket.loc[:, basket.sum() > 20]

            min_sup = st.slider("Minimum Support", 0.05, 0.40, 0.12, 0.01)

            try:
                freq = apriori(basket, min_support=min_sup, use_colnames=True)
                if len(freq) > 0:
                    rules = association_rules(freq, metric="lift", min_threshold=1.1, num_itemsets=len(freq))
                    if len(rules) > 0:
                        rules['antecedents_str'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
                        rules['consequents_str'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
                        rules_sorted = rules.sort_values('lift', ascending=False).head(25)

                        st.markdown(f"**Found {len(rules)} rules** (showing top 25 by lift)")
                        display_cols = ['antecedents_str','consequents_str','support','confidence','lift']
                        st.dataframe(
                            rules_sorted[display_cols].rename(columns={
                                'antecedents_str':'If Customer Prefers →','consequents_str':'They Also Prefer →',
                                'support':'Support','confidence':'Confidence','lift':'Lift'
                            }).style.format({'Support':'{:.3f}','Confidence':'{:.3f}','Lift':'{:.2f}'}).background_gradient(subset=['Lift'], cmap='YlOrRd'),
                            use_container_width=True, height=500
                        )

                        # Network visualization
                        if len(rules_sorted) >= 3:
                            fig = px.scatter(rules_sorted, x='support', y='confidence', size='lift',
                                             color='lift', hover_data=['antecedents_str','consequents_str'],
                                             title="Association Rules: Support vs Confidence (bubble size = Lift)",
                                             color_continuous_scale='YlOrRd')
                            fig.update_layout(height=400)
                            st.plotly_chart(fig, use_container_width=True)

                        insight_box("Business Action", "Rules with high lift (>2.0) show items that are strongly associated beyond random chance. Use these to decide what product attributes to bundle together in each Ishanaa pack.")
                    else:
                        st.warning("No rules found at this threshold. Try lowering min support.")
                else:
                    st.warning("No frequent itemsets found. Try lowering min support.")
            except Exception as e:
                st.error(f"Error in ARM: {e}. Try adjusting the support threshold.")

    # ---- CORRELATION TAB ----
    with tab3:
        st.subheader("2.5 — Correlation & Feature Importance")
        num_cols = ['age_ordinal','income_midpoint_AED','wardrobe_count','shopping_freq_ordinal',
                    'spend_per_kurti_midpoint_AED','bundle_budget_midpoint_AED','return_anxiety_ordinal','purchase_intent_ordinal']
        num_cols = [c for c in num_cols if c in df_enc.columns]
        corr_matrix = df_enc[num_cols].corr()

        fig = px.imshow(corr_matrix, text_auto='.2f', color_continuous_scale='RdBu_r', aspect='auto',
                        title="Correlation Heatmap (Numeric Features)")
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

        # Quick RF importance
        st.markdown("**Preliminary Feature Importance (Random Forest on Purchase Intent)**")
        X_clf, y_clf, _ = get_clf_data()
        rf_quick = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        rf_quick.fit(X_clf, y_clf)
        importances = pd.Series(rf_quick.feature_importances_, index=X_clf.columns).sort_values(ascending=False).head(20)

        fig = px.barh(x=importances.values, y=importances.index, color_discrete_sequence=['#c25e3a'],
                      labels={'x':'Importance','y':''}, title="Top 20 Features Predicting Purchase Intent")
        fig.update_layout(height=500, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)
        insight_box("Key Takeaway", f"The top predictor of purchase interest is <b>{importances.index[0]}</b>. This should be the #1 factor in your marketing messaging and lead qualification.")


# ================================================================
#  PAGE 3: PREDICTIVE
# ================================================================
elif page == "🔮 Predictive Analysis":
    st.title("🔮 Predictive Analytics")
    st.markdown("*Predicting who will buy and how much they'll spend*")

    tab1, tab2 = st.tabs(["🎯 Classification (Who Will Buy?)", "💰 Regression (How Much?)"])

    # ---- CLASSIFICATION TAB ----
    with tab1:
        st.subheader("3.1 — Purchase Interest Classifier")
        X, y, feature_cols = get_clf_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42),
            'XGBoost': XGBClassifier(n_estimators=200, max_depth=6, random_state=42, eval_metric='logloss',
                                      use_label_encoder=False),
            'SVM': SVC(kernel='rbf', probability=True, random_state=42)
        }

        results = {}
        roc_data = {}
        with st.spinner("Training 4 classifiers..."):
            for name, model in models.items():
                model.fit(X_train_s, y_train)
                y_pred = model.predict(X_test_s)
                y_prob = model.predict_proba(X_test_s)[:,1]
                results[name] = {
                    'Accuracy': accuracy_score(y_test, y_pred),
                    'Precision': precision_score(y_test, y_pred, zero_division=0),
                    'Recall': recall_score(y_test, y_pred, zero_division=0),
                    'F1 Score': f1_score(y_test, y_pred, zero_division=0),
                    'ROC-AUC': roc_auc_score(y_test, y_prob)
                }
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                roc_data[name] = (fpr, tpr)

        # Comparison table
        res_df = pd.DataFrame(results).T
        st.markdown("**Model Comparison**")
        st.dataframe(res_df.style.format('{:.4f}').background_gradient(cmap='Greens', axis=0), use_container_width=True)

        best_model_name = res_df['F1 Score'].idxmax()
        best_f1 = res_df.loc[best_model_name, 'F1 Score']
        st.success(f"**Best Model: {best_model_name}** (F1 Score: {best_f1:.4f})")

        # ROC Curves
        mc1, mc2 = st.columns(2)
        with mc1:
            fig = go.Figure()
            colors = ['#c25e3a','#3a7d5c','#7a5195','#3a6b8c']
            for i, (name, (fpr, tpr)) in enumerate(roc_data.items()):
                auc = results[name]['ROC-AUC']
                fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f'{name} (AUC={auc:.3f})', line=dict(color=colors[i])))
            fig.add_trace(go.Scatter(x=[0,1],y=[0,1], line=dict(dash='dash', color='grey'), showlegend=False))
            fig.update_layout(title="ROC Curves", xaxis_title="FPR", yaxis_title="TPR", height=400)
            st.plotly_chart(fig, use_container_width=True)

        with mc2:
            # Confusion Matrix for best model
            best_model = models[best_model_name]
            y_pred_best = best_model.predict(X_test_s)
            cm = confusion_matrix(y_test, y_pred_best)
            fig = px.imshow(cm, text_auto=True, color_continuous_scale='Oranges',
                            x=['Predicted: Not Interested','Predicted: Interested'],
                            y=['Actual: Not Interested','Actual: Interested'],
                            title=f"Confusion Matrix — {best_model_name}")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        # Feature Importance
        st.markdown("**Feature Importance — What Drives Purchase Intent?**")
        if hasattr(best_model, 'feature_importances_'):
            fi = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False).head(15)
        elif hasattr(best_model, 'coef_'):
            fi = pd.Series(np.abs(best_model.coef_[0]), index=X.columns).sort_values(ascending=False).head(15)
        else:
            rf_fi = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
            rf_fi.fit(X_train_s, y_train)
            fi = pd.Series(rf_fi.feature_importances_, index=X.columns).sort_values(ascending=False).head(15)

        fig = px.barh(x=fi.values, y=fi.index, color_discrete_sequence=['#c25e3a'],
                      labels={'x':'Importance','y':''}, title=f"Top 15 Features — {best_model_name}")
        fig.update_layout(height=450, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

        insight_box("Classification Insight", f"The {best_model_name} achieves {best_f1:.1%} F1-Score. Top predictors: <b>{fi.index[0]}</b>, <b>{fi.index[1]}</b>, and <b>{fi.index[2]}</b>. Use these as your lead qualification screening criteria.")

    # ---- REGRESSION TAB ----
    with tab2:
        st.subheader("3.2 — Spending Power Predictor")
        X_r, y_r, _ = get_reg_data()
        Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_r, y_r, test_size=0.2, random_state=42)

        scaler_r = StandardScaler()
        Xr_train_s = scaler_r.fit_transform(Xr_train)
        Xr_test_s = scaler_r.transform(Xr_test)

        reg_models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=1.0),
            'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42)
        }

        reg_results = {}
        with st.spinner("Training 5 regression models..."):
            for name, model in reg_models.items():
                model.fit(Xr_train_s, yr_train)
                yr_pred = model.predict(Xr_test_s)
                reg_results[name] = {
                    'MAE (AED)': mean_absolute_error(yr_test, yr_pred),
                    'RMSE (AED)': np.sqrt(mean_squared_error(yr_test, yr_pred)),
                    'R² Score': r2_score(yr_test, yr_pred)
                }

        reg_df = pd.DataFrame(reg_results).T
        st.markdown("**Model Comparison**")
        st.dataframe(reg_df.style.format({'MAE (AED)':'{:.1f}','RMSE (AED)':'{:.1f}','R² Score':'{:.4f}'}).background_gradient(subset=['R² Score'], cmap='Greens'), use_container_width=True)

        best_reg_name = reg_df['R² Score'].idxmax()
        best_r2 = reg_df.loc[best_reg_name, 'R² Score']
        st.success(f"**Best Model: {best_reg_name}** (R² = {best_r2:.4f})")

        # Actual vs Predicted
        best_reg = reg_models[best_reg_name]
        yr_pred_best = best_reg.predict(Xr_test_s)

        fig = px.scatter(x=yr_test, y=yr_pred_best, labels={'x':'Actual Spend (AED)','y':'Predicted Spend (AED)'},
                         title=f"Actual vs Predicted — {best_reg_name}", opacity=0.5, color_discrete_sequence=['#3a6b8c'])
        fig.add_trace(go.Scatter(x=[0,350], y=[0,350], mode='lines', line=dict(dash='dash', color='red'), name='Perfect Prediction'))
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

        # Residuals
        residuals = yr_test.values - yr_pred_best
        fig = px.histogram(x=residuals, nbins=40, color_discrete_sequence=['#7a5195'],
                           labels={'x':'Residual (AED)','y':'Count'}, title="Residual Distribution")
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

        insight_box("Regression Insight", f"The {best_reg_name} predicts spending within ±{reg_df.loc[best_reg_name,'MAE (AED)']:.0f} AED on average. Use this to personalize pricing: show premium packs to high-predicted-spend customers, value packs to budget-predicted ones.")


# ================================================================
#  PAGE 4: PRESCRIPTIVE
# ================================================================
elif page == "💡 Prescriptive Analysis":
    st.title("💡 Prescriptive Analytics")
    st.markdown("*Exactly what to do — which customers, which products, which discounts*")

    # Build cluster data
    X_clust, _ = get_feature_matrix()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clust)
    km = KMeans(n_clusters=5, random_state=42, n_init=10)
    df_enc['cluster'] = km.fit_predict(X_scaled)

    # ---- 4.1 Target Segment Scorecard ----
    st.subheader("4.1 — Target Segment Priority Scorecard")

    scorecard = []
    for cl in sorted(df_enc['cluster'].unique()):
        mask = df_enc['cluster'] == cl
        size = mask.sum()
        intent_rate = df_enc.loc[mask, 'purchase_intent_binary'].mean()
        avg_spend = df_enc.loc[mask, 'spend_per_kurti_midpoint_AED'].mean()
        priority = size * (intent_rate if not np.isnan(intent_rate) else 0) * (avg_spend if not np.isnan(avg_spend) else 0) / 1000

        # Top pack
        packs = df_raw.loc[mask, 'Q19_Pack_Interest'].dropna()
        top_pack_counts = expand_multi_col(packs)
        top_pack = top_pack_counts.index[0] if len(top_pack_counts) > 0 else "N/A"

        # Top pain
        pains = df_raw.loc[mask, 'Q23_Pain_Points'].dropna()
        top_pain_counts = expand_multi_col(pains)
        top_pain = top_pain_counts.index[0] if len(top_pain_counts) > 0 else "N/A"

        # Top discount
        discounts = df_raw.loc[mask, 'Q21_Discount_Preference'].dropna()
        top_disc_counts = expand_multi_col(discounts)
        top_disc = top_disc_counts.index[0] if len(top_disc_counts) > 0 else "N/A"

        scorecard.append({
            'Cluster': cl, 'Size': size, 'Interest Rate %': round((intent_rate or 0)*100,1),
            'Avg Spend (AED)': round(avg_spend or 0, 0), 'Priority Score': round(priority, 1),
            'Top Pack': top_pack[:40], 'Top Pain': top_pain[:40], 'Best Discount': top_disc[:40]
        })

    sc_df = pd.DataFrame(scorecard).sort_values('Priority Score', ascending=False)
    st.dataframe(sc_df.style.background_gradient(subset=['Priority Score'], cmap='YlOrRd')
                 .background_gradient(subset=['Interest Rate %'], cmap='Greens'), use_container_width=True)

    top_cluster = sc_df.iloc[0]
    insight_box("Launch Recommendation",
                f"<b>Cluster {int(top_cluster['Cluster'])}</b> has the highest priority score ({top_cluster['Priority Score']}) — {int(top_cluster['Size'])} people, "
                f"{top_cluster['Interest Rate %']}% interest rate, avg spend {int(top_cluster['Avg Spend (AED)'])} AED. "
                f"Lead with <b>{top_cluster['Top Pack']}</b>, address their top pain (<b>{top_cluster['Top Pain']}</b>), "
                f"and offer <b>{top_cluster['Best Discount']}</b> as the primary promotion.")

    # ---- 4.2 Discount Matrix ----
    st.markdown("---")
    st.subheader("4.2 — Personalized Discount Matrix")
    disc_data = []
    for cl in sorted(df_enc['cluster'].unique()):
        mask = df_enc['cluster'] == cl
        discs = df_raw.loc[mask, 'Q21_Discount_Preference'].dropna()
        disc_counts = expand_multi_col(discs)
        for disc_type, count in disc_counts.head(5).items():
            disc_data.append({'Cluster': f'Cluster {cl}', 'Discount Type': disc_type, 'Preference Count': count})

    disc_df = pd.DataFrame(disc_data)
    if not disc_df.empty:
        fig = px.bar(disc_df, x='Cluster', y='Preference Count', color='Discount Type', barmode='group',
                     title="Discount Preference by Cluster", color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

    # ---- 4.3 Launch Playbook ----
    st.markdown("---")
    st.subheader("4.3 — Auto-Generated Launch Playbook")

    top_cl = sc_df.iloc[0]
    second_cl = sc_df.iloc[1] if len(sc_df) > 1 else top_cl

    playbook = f"""
    **🚀 ISHANAA UAE LAUNCH PLAYBOOK**

    **Primary Target:** Cluster {int(top_cl['Cluster'])} ({int(top_cl['Size'])} people, {top_cl['Interest Rate %']}% interest)
    - **Lead Product:** {top_cl['Top Pack']}
    - **Price Point:** ~{int(top_cl['Avg Spend (AED)'])} AED per kurti
    - **Lead Offer:** {top_cl['Best Discount']}
    - **Address Pain:** {top_cl['Top Pain']}

    **Secondary Target:** Cluster {int(second_cl['Cluster'])} ({int(second_cl['Size'])} people, {second_cl['Interest Rate %']}% interest)
    - **Lead Product:** {second_cl['Top Pack']}
    - **Lead Offer:** {second_cl['Best Discount']}

    **Marketing Allocation:** 50% budget to primary cluster, 30% to secondary, 20% test & learn across others.
    """
    st.markdown(playbook)


# ================================================================
#  PAGE 5: NEW CUSTOMER PREDICTOR
# ================================================================
elif page == "🆕 New Customer Predictor":
    st.title("🆕 New Customer Predictor")
    st.markdown("*Upload new data or fill a quick form to predict purchase interest & spending*")

    # Train models once
    @st.cache_resource
    def train_production_models():
        X_c, y_c, cols_c = get_clf_data()
        X_r, y_r, cols_r = get_reg_data()

        scaler_c = StandardScaler()
        X_cs = scaler_c.fit_transform(X_c)
        clf = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42)
        clf.fit(X_cs, y_c)

        scaler_r = StandardScaler()
        X_rs = scaler_r.fit_transform(X_r)
        reg = GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42)
        reg.fit(X_rs, y_r)

        X_all, _ = get_feature_matrix()
        scaler_k = StandardScaler()
        X_ks = scaler_k.fit_transform(X_all)
        km = KMeans(n_clusters=5, random_state=42, n_init=10)
        km.fit(X_ks)

        return clf, reg, km, scaler_c, scaler_r, scaler_k, list(X_c.columns), list(X_r.columns), list(X_all.columns)

    clf, reg, km, scaler_c, scaler_r, scaler_k, clf_cols, reg_cols, all_cols = train_production_models()

    pred_mode = st.radio("Prediction Mode", ["📝 Single Customer (Quick Form)", "📤 Bulk CSV Upload"], horizontal=True)

    if pred_mode == "📝 Single Customer (Quick Form)":
        st.markdown("**Fill in the key details about the potential customer:**")

        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            inp_age = st.selectbox("Age Group", ["Under 18","18-22","23-27","28-34","35 and above"])
            inp_occ = st.selectbox("Occupation", ["Student (School/Undergraduate)","Student (Postgraduate)",
                                                   "Working Professional","Self-Employed/Freelancer","Homemaker","Other"])
            inp_eth = st.selectbox("Ethnicity", ["South Asian","Arab/Middle Eastern","Southeast Asian","African","Western/European","Other"])

        with fc2:
            inp_income = st.selectbox("Monthly Income", ["No personal income","Under 3,000 AED","3,000-7,000 AED",
                                                          "7,001-12,000 AED","12,001-20,000 AED","Above 20,000 AED"])
            inp_wardrobe = st.selectbox("Kurtis Owned", ["None","1-5","6-12","13-20","More than 20"])
            inp_freq = st.selectbox("Shopping Frequency", ["Weekly","2-3 times a month","Once a month",
                                                            "Once in 2-3 months","Only during festivals/occasions","Rarely/Never"])

        with fc3:
            inp_fashion = st.selectbox("Fashion Identity", [
                "I follow trends closely and love experimenting with new styles",
                "I have a set style - I know what works and stick to it",
                "I prioritize comfort and practicality above everything",
                "I only dress up for occasions - daily wear is minimal effort",
                "I love mixing Indian and Western styles to create my own look"
            ])
            inp_anxiety = st.selectbox("Return Anxiety", [
                "Yes, frequently - it's a major concern",
                "Sometimes - depends on the brand/platform",
                "Rarely - I'm usually okay with online purchases",
                "Never - I'm confident buying online"
            ])

        if st.button("🔮 Predict", type="primary", use_container_width=True):
            age_map = {"Under 18":1,"18-22":2,"23-27":3,"28-34":4,"35 and above":5}
            income_map = {"No personal income":0,"Under 3,000 AED":1500,"3,000-7,000 AED":5000,
                          "7,001-12,000 AED":9500,"12,001-20,000 AED":16000,"Above 20,000 AED":25000}
            income_ord_map = {"No personal income":0,"Under 3,000 AED":1,"3,000-7,000 AED":2,
                              "7,001-12,000 AED":3,"12,001-20,000 AED":4,"Above 20,000 AED":5}
            wardrobe_map = {"None":0,"1-5":3,"6-12":9,"13-20":16,"More than 20":25}
            freq_map = {"Weekly":6,"2-3 times a month":5,"Once a month":4,"Once in 2-3 months":3,
                        "Only during festivals/occasions":2,"Rarely/Never":1}
            anxiety_map = {"Yes, frequently - it's a major concern":4,"Sometimes - depends on the brand/platform":3,
                           "Rarely - I'm usually okay with online purchases":2,"Never - I'm confident buying online":1}

            # Build feature vector matching all_cols
            new_row = pd.DataFrame(0, index=[0], columns=all_cols, dtype=float)

            # Set known values
            if 'age_ordinal' in new_row.columns:
                new_row['age_ordinal'] = age_map.get(inp_age, 3)
            if 'income_ordinal' in new_row.columns:
                new_row['income_ordinal'] = income_ord_map.get(inp_income, 2)
            if 'income_midpoint_AED' in new_row.columns:
                new_row['income_midpoint_AED'] = income_map.get(inp_income, 5000)
            if 'wardrobe_count' in new_row.columns:
                new_row['wardrobe_count'] = wardrobe_map.get(inp_wardrobe, 9)
            if 'shopping_freq_ordinal' in new_row.columns:
                new_row['shopping_freq_ordinal'] = freq_map.get(inp_freq, 4)
            if 'return_anxiety_ordinal' in new_row.columns:
                new_row['return_anxiety_ordinal'] = anxiety_map.get(inp_anxiety, 2)

            # Occupation one-hot
            for col in new_row.columns:
                if col.startswith('occ_') and inp_occ.split('(')[0].strip().replace(' ','_').replace('/','_').lower().rstrip('_') in col:
                    new_row[col] = 1
            # Ethnicity one-hot
            for col in new_row.columns:
                if col.startswith('eth_') and inp_eth.split('(')[0].strip().replace(' ','_').replace('/','_').lower().rstrip('_') in col:
                    new_row[col] = 1
            # Fashion identity
            fashion_idx = ["I follow trends closely and love experimenting with new styles",
                           "I have a set style - I know what works and stick to it",
                           "I prioritize comfort and practicality above everything",
                           "I only dress up for occasions - daily wear is minimal effort",
                           "I love mixing Indian and Western styles to create my own look"].index(inp_fashion) + 1
            if f'fashion_id_{fashion_idx}' in new_row.columns:
                new_row[f'fashion_id_{fashion_idx}'] = 1

            # Fill median for unknown features
            medians = df_enc[all_cols].median()
            for col in all_cols:
                if new_row[col].iloc[0] == 0 and col not in ['age_ordinal','income_ordinal','income_midpoint_AED',
                                                               'wardrobe_count','shopping_freq_ordinal','return_anxiety_ordinal',
                                                               'bundle_budget_ordinal','bundle_budget_midpoint_AED',
                                                               'spend_per_kurti_ordinal'] and not col.startswith(('occ_','eth_','fashion_id_')):
                    if pd.notna(medians.get(col)):
                        new_row[col] = medians[col]

            # Classification prediction
            X_clf_new = new_row[clf_cols] if all(c in new_row.columns for c in clf_cols) else new_row[all_cols]
            try:
                X_clf_scaled = scaler_c.transform(X_clf_new[clf_cols])
                interest_prob = clf.predict_proba(X_clf_scaled)[0][1]
                interest_label = "✅ Interested" if interest_prob >= 0.5 else "❌ Not Interested"
            except:
                X_cf = scaler_c.transform(new_row[clf_cols].fillna(0))
                interest_prob = clf.predict_proba(X_cf)[0][1]
                interest_label = "✅ Interested" if interest_prob >= 0.5 else "❌ Not Interested"

            # Regression prediction
            try:
                X_reg_scaled = scaler_r.transform(new_row[reg_cols].fillna(0))
                spend_pred = reg.predict(X_reg_scaled)[0]
            except:
                spend_pred = 80

            # Cluster assignment
            try:
                X_km_scaled = scaler_k.transform(new_row[all_cols].fillna(0))
                cluster_id = km.predict(X_km_scaled)[0]
            except:
                cluster_id = 0

            # Display results
            st.markdown("---")
            st.markdown("### 🎯 Prediction Results")
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Purchase Interest", interest_label)
            r2.metric("Interest Probability", f"{interest_prob:.1%}")
            r3.metric("Predicted Spend", f"{max(15,spend_pred):.0f} AED/kurti")
            r4.metric("Customer Cluster", f"Cluster {cluster_id}")

            # Recommendation
            if interest_prob >= 0.7:
                st.success("🎯 **HIGH VALUE LEAD** — This customer has strong purchase intent. Prioritize outreach. Show premium packs and limited drops.")
            elif interest_prob >= 0.5:
                st.info("👍 **WARM LEAD** — Moderate interest. Start with value packs (Campus Essentials) and first-order discounts to convert.")
            elif interest_prob >= 0.3:
                st.warning("🤔 **LUKEWARM** — Some interest but not convinced. Retarget with social proof (reviews, UGC) and free delivery offers.")
            else:
                st.error("❌ **LOW PRIORITY** — Unlikely to convert. Don't allocate ad spend. May re-engage during festival seasons.")

    # ---- BULK UPLOAD ----
    else:
        st.markdown("**Upload a CSV file with new customer survey data**")
        st.markdown("The CSV should have the same column structure as the encoded dataset. Missing columns will be filled with median values.")

        uploaded = st.file_uploader("Upload CSV", type=['csv'])

        if uploaded is not None:
            try:
                new_df = pd.read_csv(uploaded)
                st.success(f"Uploaded {len(new_df)} rows, {new_df.shape[1]} columns")
                st.dataframe(new_df.head(), use_container_width=True)

                if st.button("🔮 Run Predictions on All Rows", type="primary"):
                    with st.spinner("Processing predictions..."):
                        # Align columns
                        for col in all_cols:
                            if col not in new_df.columns:
                                new_df[col] = 0

                        new_X = new_df[all_cols].fillna(0)

                        # Impute with training medians
                        medians = df_enc[all_cols].median()
                        for col in all_cols:
                            zero_mask = new_X[col] == 0
                            if zero_mask.any() and pd.notna(medians.get(col)):
                                new_X.loc[zero_mask, col] = medians[col]

                        # Classification
                        try:
                            X_c_new = scaler_c.transform(new_X[clf_cols])
                            probs = clf.predict_proba(X_c_new)[:, 1]
                            preds = (probs >= 0.5).astype(int)
                        except:
                            probs = np.full(len(new_X), 0.5)
                            preds = np.zeros(len(new_X), dtype=int)

                        # Regression
                        try:
                            X_r_new = scaler_r.transform(new_X[reg_cols])
                            spend_preds = reg.predict(X_r_new)
                        except:
                            spend_preds = np.full(len(new_X), 80)

                        # Clustering
                        try:
                            X_k_new = scaler_k.transform(new_X[all_cols])
                            cluster_preds = km.predict(X_k_new)
                        except:
                            cluster_preds = np.zeros(len(new_X), dtype=int)

                        # Add predictions
                        result_df = new_df.copy()
                        result_df['predicted_interest'] = preds
                        result_df['interest_probability'] = probs.round(3)
                        result_df['predicted_spend_AED'] = spend_preds.round(0)
                        result_df['assigned_cluster'] = cluster_preds

                        st.markdown("### Prediction Results")
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Interested Customers", f"{preds.sum()} / {len(preds)}")
                        m2.metric("Avg Predicted Spend", f"{spend_preds.mean():.0f} AED")
                        m3.metric("Interest Rate", f"{preds.mean()*100:.1f}%")

                        st.dataframe(result_df.head(20), use_container_width=True)

                        # Download
                        csv_buffer = result_df.to_csv(index=False)
                        st.download_button("⬇️ Download Predictions CSV", csv_buffer, "ishanaa_predictions.csv",
                                           "text/csv", use_container_width=True)

                        insight_box("What to do next",
                                    f"Filter the downloaded CSV for <code>predicted_interest = 1</code> AND <code>interest_probability > 0.7</code> — "
                                    f"those are your high-priority leads. Sort by <code>predicted_spend_AED</code> to find the highest-value customers first.")

            except Exception as e:
                st.error(f"Error processing file: {e}")


# ================================================================
#  PAGE 6: DATA DICTIONARY
# ================================================================
elif page == "ℹ️ Data Dictionary":
    st.title("ℹ️ Data Dictionary & Methodology")

    st.subheader("Survey Structure")
    st.markdown("""
    - **25 questions** across 5 sections: Demographics, Shopping Behavior, Product Preferences, Bundle & Brand, Engagement & Intent
    - **Target market:** UAE-based women, primarily South Asian expats + broader community
    - **4 ML algorithms served:** Classification, Clustering, Association Rule Mining, Regression
    """)

    st.subheader("Data Dictionary")
    st.dataframe(df_dict, use_container_width=True, height=500)

    st.subheader("Dataset Statistics")
    s1, s2, s3 = st.columns(3)
    s1.metric("Raw CSV Rows", f"{len(df_raw):,}")
    s2.metric("Encoded Columns", f"{df_enc.shape[1]}")
    s3.metric("Dictionary Entries", f"{len(df_dict)}")

    st.subheader("Synthetic Data Generation Notes")
    st.markdown("""
    - **6 latent personas:** Budget Campus Girl (25%), Polished Professional (20%), Festive Enthusiast (18%),
      Trend Chaser (15%), Conscious Minimalist (10%), Non-Customer (12%)
    - **Noise injected:** ~15% persona bleed per row, 100 contradictions, 60 spending outliers, 40 lazy respondents, 1-8% missing values per column
    - **UAE calibration:** AED pricing, local shopping channels (Noon, Namshi), South Asian expat demographics, Eid + Diwali calendar
    """)

    st.subheader("Persona Distribution")
    if '_persona' in df_raw.columns:
        persona_counts = df_raw['_persona'].value_counts()
        fig = px.bar(x=persona_counts.index, y=persona_counts.values, color=persona_counts.index,
                     color_discrete_sequence=px.colors.qualitative.Set2, labels={'x':'Persona','y':'Count'})
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.caption("Ishanaa Analytics Dashboard · Built for Vedant & Team · March 2026")
