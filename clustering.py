import os
os.environ["OMP_NUM_THREADS"] = "1"
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

st.set_page_config(
    page_title="Income and Spending Clustering",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Clustering using Income and Spending")
st.write(
    "This dashboard uses K-Means clustering to group customers based on "
    "Annual Income and Spending Score."
)

st.sidebar.header("⚙️ Dashboard Settings")
uploaded_file = st.sidebar.file_uploader(
    "Upload Mall Customers CSV file", type=["csv"]
)
st.sidebar.info(
    "Required columns: Annual Income (k$) and Spending Score. "
    "CustomerID and Gender are optional."
)

@st.cache_data
def load_data(file):
    return pd.read_csv(file)

# Load the uploaded dataset
if uploaded_file is None:
    st.warning(
        "Please upload your customer CSV file to start the analysis. "
        "The original notebook output shows 200 customer records, but the "
        "CSV dataset itself was not included with the uploaded HTML file."
    )
    st.stop()

try:
    data = load_data(uploaded_file)
except Exception as e:
    st.error(f"Unable to read the CSV file: {e}")
    st.stop()

data.columns = data.columns.str.strip()
income_col = "Annual Income (k$)"
spending_col = "Spending Score"

if income_col not in data.columns or spending_col not in data.columns:
    st.error(
        "Required columns are missing. Your CSV must contain "
        "'Annual Income (k$)' and 'Spending Score'."
    )
    st.write("Available columns:", list(data.columns))
    st.stop()

# Convert clustering columns to numeric and remove invalid rows
data[income_col] = pd.to_numeric(data[income_col], errors="coerce")
data[spending_col] = pd.to_numeric(data[spending_col], errors="coerce")
clean_data = data.dropna(subset=[income_col, spending_col]).copy()

if len(clean_data) < 4:
    st.error("At least 4 valid customer rows are required for clustering.")
    st.stop()

# Sidebar cluster selection
k = st.sidebar.slider("Number of clusters (K)", min_value=2, max_value=10, value=4)
X = clean_data[[income_col, spending_col]]

# K-Means model
model = KMeans(n_clusters=k, random_state=0, n_init=10)
clean_data["Cluster"] = model.fit_predict(X)

# Dashboard metrics
st.subheader("📌 Dataset Overview")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Records", f"{len(data):,}")
c2.metric("Valid Records", f"{len(clean_data):,}")
c3.metric("Number of Clusters", k)
c4.metric("K-Means Inertia", f"{model.inertia_:,.2f}")

# Navigation
page = st.sidebar.radio(
    "Choose an Option", [
        "🏠 Home", 
        "📋 Dataset", 
        "📊 Data Analysis", 
        "📉 Elbow Method", 
        "🎯 Cluster Visualization", 
        "🔮 Prediction", 
        "💡 Insights"
    ]
)

if page == "🏠 Home":
    st.header("🏠 Income and Spending Analysis")
    st.markdown("""
    ### Project Objective
    To group customers according to their annual income and spending score using the **K-Means Unsupervised Learning algorithm**.
    
    ### Main Steps
    1. Upload the customer dataset.
    2. Select Annual Income and Spending Score.
    3. Find a suitable K value using the Elbow Method.
    4. Apply K-Means clustering.
    5. Visualize customer groups and view the prediction results.
    """)
    st.subheader("Preview of Customer Data")
    st.dataframe(clean_data.head(10), use_container_width=True)

elif page == "📋 Dataset":
    st.header("📋 Dataset")
    st.write("First five rows")
    st.dataframe(data.head(), use_container_width=True)
    st.write("Complete dataset")
    st.dataframe(data, use_container_width=True)
    
    st.subheader("Dataset Information")
    info_df = pd.DataFrame({
        "Column": data.columns,
        "Data Type": [str(dtype) for dtype in data.dtypes],
        "Missing Values": [int(data[col].isna().sum()) for col in data.columns]
    })
    st.dataframe(info_df, use_container_width=True)
    
    st.subheader("Statistical Summary")
    st.dataframe(data.describe(include="all").transpose(), use_container_width=True)

elif page == "📊 Data Analysis":
    st.header("📊 Data Analysis")
    left, right = st.columns(2)
    with left:
        st.subheader("Annual Income Distribution")
        fig, ax = plt.subplots()
        sns.histplot(clean_data[income_col], kde=True, ax=ax)
        ax.set_xlabel("Annual Income (k$)")
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.subheader("Spending Score Distribution")
        fig, ax = plt.subplots()
        sns.histplot(clean_data[spending_col], kde=True, ax=ax)
        ax.set_xlabel("Spending Score")
        st.pyplot(fig)
        plt.close(fig)
        
    st.subheader("Income vs Spending")
    fig, ax = plt.subplots()
    sns.scatterplot(data=clean_data, x=income_col, y=spending_col, ax=ax)
    ax.set_title("Income Spent Analysis")
    st.pyplot(fig)
    plt.close(fig)

elif page == "📉 Elbow Method":
    st.header("📉 Finding the Optimized K Value")
    max_k = min(10, len(X) - 1)
    inertias = []
    k_values = range(1, max_k + 1)
    for i in k_values:
        km = KMeans(n_clusters=i, random_state=0, n_init=10)
        km.fit(X)
        inertias.append(km.inertia_)
        
    fig, ax = plt.subplots()
    ax.plot(list(k_values), inertias, color="red", marker="o")
    ax.set_title("Optimal K Value - Elbow Method")
    ax.set_xlabel("Number of Clusters")
    ax.set_ylabel("Inertia")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)
    st.info(
        "Choose the K value near the point where the curve begins to flatten. "
        "The uploaded notebook uses K = 4 for its final clustering."
    )

elif page == "🎯 Cluster Visualization":
    st.header(f"🎯 Visualizing the Clusters for K = {k}")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=clean_data, x=income_col, y=spending_col, hue="Cluster", palette="viridis", s=80, ax=ax
    )
    ax.scatter(
        model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], marker="X", s=250, color="red", label="Centroids"
    )
    ax.set_title("Income Spent Analysis")
    ax.set_xlabel("Income")
    ax.set_ylabel("Spent")
    ax.legend(title="Cluster")
    st.pyplot(fig)
    plt.close(fig)
    
    st.subheader("Cluster-wise Customer Count")
    cluster_counts = clean_data["Cluster"].value_counts().sort_index()
    st.bar_chart(cluster_counts)
    
    st.subheader("Cluster Average Values")
    st.dataframe(
        clean_data.groupby("Cluster")[[income_col, spending_col]].mean().round(2),
        use_container_width=True
    )

elif page == "🔮 Prediction":
    st.header("🔮 Predict Customer Cluster")
    st.write("Enter a customer's annual income and spending score.")
    
    col1, col2 = st.columns(2)
    with col1:
        income_value = st.number_input(
            "Annual Income (k$)", min_value=0.0, value=float(clean_data[income_col].median()), step=1.0
        )
    with col2:
        spending_value = st.number_input(
            "Spending Score", min_value=0.0, max_value=100.0, value=float(clean_data[spending_col].median()), step=1.0
        )
        
    if st.button("Predict Cluster", type="primary"):
        # Wrap input into a DataFrame with identical column names to suppress scikit-learn warnings
        input_data = pd.DataFrame([[income_value, spending_value]], columns=[income_col, spending_col])
        prediction = model.predict(input_data)[0]
        st.success(f"Predicted Cluster: {prediction}")
        st.write(
            "This customer is assigned to the cluster whose centroid is "
            "closest to the entered income and spending values."
        )
        
    st.subheader("Final Prediction Result")
    result_columns = [
        col for col in ["CustomerID", income_col, spending_col, "Cluster"] if col in clean_data.columns
    ]
    result = clean_data[result_columns]
    st.dataframe(result, use_container_width=True)
    
    csv = result.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Cluster Results", data=csv, file_name="customer_clustering_results.csv", mime="text/csv"
    )

elif page == "💡 Insights":
    st.header("💡 Customer Cluster Insights")
    summary = clean_data.groupby("Cluster").agg(
        Customer_Count=(income_col, "size"),
        Average_Income=(income_col, "mean"),
        Average_Spending=(spending_col, "mean")
    ).round(2)
    st.dataframe(summary, use_container_width=True)
    st.markdown("""
    ### How to interpret the clusters
    - **Low income / low spending:** Customers with lower income and lower spending scores.
    - **Low income / high spending:** Customers with lower income and higher spending scores.
    - **High income / low spending:** Customers with higher income and lower spending scores.
    - **High income / high spending:** Customers with higher income and higher spending scores.
    
    Cluster numbers are labels assigned by K-Means. Their numeric order does not represent a ranking.
    """)
    st.caption("Customer Clustering using Income and Spending | K-Means Unsupervised Learning")
