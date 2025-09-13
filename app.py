import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Title
st.title("Advanced EDA & Statistical Analysis App")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Dataset preview
    st.subheader("Dataset Preview")
    st.write(df.head())

    # Dataset info
    st.subheader("Dataset Info")
    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    st.write(df.dtypes)

    # Missing values
    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    # Summary statistics
    st.subheader("Summary Statistics")
    st.write(df.describe(include='all'))

    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    numeric_df = df.select_dtypes(include=np.number)
    if not numeric_df.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    # Column analysis
    st.subheader("Column Analysis")
    column = st.selectbox("Select a column", df.columns)

    if pd.api.types.is_numeric_dtype(df[column]):
        st.write(f"Summary of {column}")
        st.write(df[column].describe())

        # Histogram
        fig, ax = plt.subplots()
        sns.histplot(df[column], bins=30, kde=True, ax=ax)
        ax.set_title(f"Distribution of {column}")
        st.pyplot(fig)

        # Boxplot
        fig, ax = plt.subplots()
        sns.boxplot(x=df[column], ax=ax)
        ax.set_title(f"Boxplot of {column}")
        st.pyplot(fig)

    else:
        st.write(f"Value counts of {column}")
        st.write(df[column].value_counts())

        # Bar chart
        fig, ax = plt.subplots()
        df[column].value_counts().plot(kind='bar', ax=ax)
        ax.set_title(f"Count of {column}")
        st.pyplot(fig)

    # Pairplot
    if not numeric_df.empty:
        st.subheader("Pairplot (Numeric Columns)")
        fig = sns.pairplot(numeric_df)
        st.pyplot(fig)

    # Outlier detection
    st.subheader("Outlier Detection (IQR Method)")
    for col in numeric_df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)][col]
        st.write(f"{col}: {len(outliers)} outliers")
