import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from PIL import Image
def app():
    # Title of the app
    st.title('Loan Approval Prediction')

    st.text('This model predicts whether a loan application will be approved or not based on various features of the applicant.')

    st.title('Exploratory Data Analysis')

    st.write('This dataset have been collected from Kaggle.' \
    ' The dataset contains information about loan applicants and whether their loans were approved or not.')

    df = pd.read_csv('loan_approval_dataset.csv')

    st.write('## Dataset Overview')
    st.write('This dataset contains information about loan applicants and whether their loans were approved or not.')

    st.write(df)

    df.columns = df.columns.str.strip()

    st.write('# Heatmap')

    # Select only numeric features
    numeric_features = df.select_dtypes(include='number')

    # Compute correlation matrix
    corr_matrix = numeric_features.corr()

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot heatmap
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt=".2f", linewidths=0.5, ax=ax)
    ax.set_title('Correlation Matrix Heatmap')

    # Display with Streamlit
    st.pyplot(fig)

    st.write('# Self Employed Distribution')

    # Count values in the 'self_employed' column
    self_employed_counts = df['self_employed'].value_counts()
    labels = self_employed_counts.index
    sizes = self_employed_counts.values
    total = sizes.sum()

    # Plot donut chart
    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        autopct='%1.1f%%',
        startangle=90,
        colors=['#66b3ff', '#ff9999'],
        wedgeprops={'width': 0.4}  # makes it a donut
    )

    # Equal aspect ratio ensures the pie is drawn as a circle
    ax.axis('equal')
    ax.set_title('Self Employed Distribution')

    # Add total count in the center
    ax.text(0, 0, f'Total: {total}', ha='center', va='center', fontsize=12, fontweight='bold')

    st.pyplot(fig)

    st.write('# Loan Status Distribution')

    # Count values in the 'loan_status' column
    default_counts = df['loan_status'].value_counts()
    labels = default_counts.index
    sizes = default_counts.values
    total = sizes.sum()

    # Plot donut chart
    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        autopct='%1.1f%%',
        startangle=90,
        colors=['#66b3ff', '#ff9999'],
        wedgeprops={'width': 0.4}  # makes it a donut
    )

    # Equal aspect ratio ensures the pie is drawn as a circle
    ax.axis('equal')
    ax.set_title('Loan Status Distribution')

    # Add total count in the center
    ax.text(0, 0, f'Total: {total}', ha='center', va='center', fontsize=12, fontweight='bold')

    st.pyplot(fig)

    st.write('# Boxplot for Cibil Score and Loan Amount')

    image = Image.open(r'C:\Users\ThinkPad\Downloads\cibil-loan-amount.png')
    st.image(image, use_container_width=True)

    st.write('# Loan Status by Self Employed')
    
    image2 = Image.open(r'C:\Users\ThinkPad\Downloads\loan-status-self-employed.png')
    st.image(image2, use_container_width=True)

    st.write('# Loan status and Loan Amount')
    image3 = Image.open(r'C:\Users\ThinkPad\Downloads\loan-status-loan-amount.png')
    st.image(image3, use_container_width=True)

    st.write('# Loan Status and Loan Term')
    image4 = Image.open(r'C:\Users\ThinkPad\Downloads\loan-status-loan-term.png')
    st.image(image4, use_container_width=True)

if __name__ == "__main__":
    app()