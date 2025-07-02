import streamlit as st
import pandas as pd
import joblib

def app():
    st.write('# Model Inference')

    model = joblib.load('log_reg.pkl')

    # Input form for user data
    df_inf = {
        'no_of_dependents': st.number_input('Number of Dependents', min_value=0, max_value=99, value=0),
        'education': st.selectbox('Graduation Status', ['Not Graduate', 'Graduate']),
        'self_employed': st.selectbox('Self Employed', ['No', 'Yes']),
        'loan_amount': st.number_input('Loan Amount', min_value=0, max_value=1000000, value=100000),
        'loan_term': st.number_input('Loan Term (in months)', min_value=0, max_value=600, value=360),
        'cibil_score': st.number_input('CIBIL Score', min_value=0, max_value=900, value=650),
        'residential_assets_value': st.number_input('Residential Assets Value', min_value=0, max_value=10000000, value=500000),
        'commercial_assets_value': st.number_input('Commercial Assets Value', min_value=0, max_value=10000000, value=500000),
    }

    df_inf = pd.DataFrame([df_inf])

    # Display user input
    st.write("### User Input:")
    st.dataframe(df_inf)

    # Predict button
    if st.button('Predict'):
        y_proba_inf = model.predict_proba(df_inf)[:, 1]
        y_pred_inf = (y_proba_inf >= 0.3).astype(int)

        for i, (proba, pred) in enumerate(zip(y_proba_inf, y_pred_inf)):
            st.write(f"**Probability of Rejection:** {proba:.2f}")
            if pred == 1:
                st.error(f"Prediction: Rejected ❌")
            else:
                st.success(f"Prediction: Approved ✅")
        

if __name__ == '__main__':
    app()