import streamlit as st
import numpy as np
import pandas as pd

# Load the dataset
df = pd.read_csv("Dataset.csv")
df.dropna(inplace=True)

# Extract column names and unique values
columnlist = df.columns
uniquevalues = {col: df[col].unique() for col in columnlist}

# Streamlit application
st.set_page_config(page_title="Heart Disease Prediction", layout="wide", initial_sidebar_state="expanded")

# Header section
st.title("💓 Heart Disease Prediction App")
st.markdown(
    """
    Welcome to the **Heart Disease Prediction App**!  
    This tool helps predict the likelihood of heart disease based on user inputs.  
    **Note**: This is not a replacement for professional medical advice.
    """
)
st.divider()

# Sidebar inputs with enhanced UI
st.sidebar.header("Enter Your Details")
st.sidebar.markdown("Please provide the following details to get your prediction.")

inputlist = []
for col, values in uniquevalues.items():
    if col != "HeartProblem":  # Exclude the target column
        input_text = st.sidebar.selectbox(
            f"Select {col}:", 
            options=values, 
            key=col, 
            help=f"Choose your {col} from the available options."
        )
        inputlist.append(input_text.lower())

# Add an image or visual
st.sidebar.image("images/custom_image.jpg", use_column_width=True, caption="Stay Healthy!")
st.sidebar.markdown("---")

# Prediction button
if st.sidebar.button("🔍 Predict"):
    # Calculate probabilities
    keys = list(uniquevalues.keys())
    prob = df.groupby(keys[-1])["HeartProblem"].count()

    try:
        probyes = prob["yes"]
        probno = prob["no"]
    except KeyError:
        st.error("Dataset does not contain sufficient 'yes' or 'no' data.")
        st.stop()

    predictyes = []
    predictno = []

    for i in range(len(keys) - 1):
        predictyes.append(sum(np.logical_and(df[keys[i]] == inputlist[i], df["HeartProblem"] == 'yes')))
        predictno.append(sum(np.logical_and(df[keys[i]] == inputlist[i], df["HeartProblem"] == 'no')))

    finalprobyes = 1
    finalprobno = 1

    for i in predictyes:
        finalprobyes *= i / probyes if probyes > 0 else 0

    for i in predictno:
        finalprobno *= i / probno if probno > 0 else 0

    # Display the prediction result
    st.subheader("Prediction Result")
    if finalprobyes > finalprobno:
        st.error(
            "⚠️ **Heart Disease Detected**: Please consult a healthcare professional immediately for further evaluation and advice."
        )
    else:
        st.success(
            "✅ **No Heart Disease Detected**: Great! Keep maintaining a healthy lifestyle and regular checkups."
        )
    
    # Add insights or tips
    st.markdown(
        """
        ### Tips for a Healthy Heart:
        - 🥗 Eat a balanced diet rich in fruits and vegetables.
        - 🏃 Engage in regular physical activity.
        - 🚭 Avoid smoking and limit alcohol intake.
        - 🛌 Prioritize adequate sleep and manage stress.
        """
    )
else:
    st.info("Please fill in the details on the left and click 'Predict' to get your result.")

# Footer section
st.markdown("---")
st.markdown(
    """
    **Disclaimer**: This prediction is based on statistical analysis and is not a substitute for professional medical advice.
    """
)
