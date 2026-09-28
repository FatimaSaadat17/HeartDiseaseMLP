import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from model import MLP
from ray.train import Checkpoint

# Page Configuration for better aesthetics
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (CSS)
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Header Card */
    .header-card {
        background: linear-gradient(135deg, #ff4b4b 0%, #ff7676 100%);
        padding: 2.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(255, 75, 75, 0.2);
    }
    .header-card h1 {
        color: white !important;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    /* Form styling */
    .stForm {
        background-color: white;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    
    /* Prediction Cards */
    .result-card-positive {
        background-color: #ffeef0;
        border-left: 6px solid #ff4b4b;
        padding: 1.5rem;
        border-radius: 10px;
        margin-top: 1rem;
    }
    .result-card-negative {
        background-color: #e8f8f0;
        border-left: 6px solid #00c853;
        padding: 1.5rem;
        border-radius: 10px;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar Info
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/heart-health.png", width=80)
    st.title("About the App")
    st.info("""
    This app uses a Deep Learning Multi-Layer Perceptron (MLP) trained with PyTorch and Ray Tune to predict the presence of heart disease based on 13 clinical parameters.
    """)
    st.markdown("---")
    st.caption("🔍 **Note:** Ensure all numerical inputs are filled out correctly before clicking Predict.")

# import the dataset
df = pd.read_csv('./heart.csv')

# import heart disease nn model
checkpoint_path = './model_params/train_08597_00000_0_beta1=0.8031,beta2=0.9890,epsilon=0.0000,lr=0.0018_2026-09-27_18-09-19/checkpoint_000039'

# restore ray tune checkpoint
checkpoint = Checkpoint.from_directory(checkpoint_path)

with checkpoint.as_directory() as checkpoint_dir:
    # instantiate pytorch model with 13 features
    model = MLP(13) 
    
    # load entire state dict saved in model_params directory
    state_dict = torch.load(f"{checkpoint_dir}/model.pth")
    model.load_state_dict(state_dict)
    # set mode to evaluation mode for predicting
    model.eval()

# Header Section
st.markdown("""
<div class="header-card">
    <h1>🫀 Predicting Presence of Heart Disease Using MLP</h1>
    <p style="font-size: 1.1rem; opacity: 0.95">Enter patient clinical indicators below for instant AI-assisted evaluation</p>
</div>
""", unsafe_allow_html=True)

with st.form("feature_form"):
    st.markdown("<h3 style='color: black;'>📋 Patient Clinical Parameters</h3>", unsafe_allow_html=True)
    st.write("Please fill in the input fields below:")
    
    with st.container():
        # define the number of columns for each feature text input
        columns = st.columns(7)
        # get all features of the df
        features = [c for c in df.columns if c != 'target']
        # initialize an empty dict with all features and values set to None
        features_dict = {f: None for f in features}
        # define text placeholders for each text input in the form per feature
        placeholders = ['e.g 56', 'M/F', '[0-4]', 'mm/Hg', 'mg/dl', '1/0', '1/0', 'max heart rate', '1/0', 'ST depression 1/0', '[0-2]', '[0-4]', '[0-3]']
        j = 7
        
        for i, c in enumerate(columns[:7]):
            with c:
                features_dict[features[i]] = st.text_input(str(features[i]), placeholder=placeholders[i], width=100, key=i)
                if (j != 13):
                    features_dict[features[j]] = st.text_input(str(features[j]), placeholder=placeholders[j], width=100, key=j)
                    j = j+1

    st.markdown("<br>", unsafe_allow_html=True)
    # define submit button for the form
    submitted = st.form_submit_button("🩺 Run Prediction Model", use_container_width=True, type="primary")
    if submitted:
        print(features_dict)

# convert male to 1 and female to 0
if (features_dict['sex'] == 'M' or features_dict['sex'] == 'm'):
    features_dict['sex'] = 1
elif (features_dict['sex'] == 'F' or features_dict['sex'] == 'f'):
    features_dict['sex'] = 0

# convert features dict to a dataframe
x = pd.DataFrame(data=features_dict, index=[0])

# scale the features using the standard scaler
scaler = StandardScaler()
# scale original dataset and fit scaler on it
X = df.iloc[:, :-1]
scaler.fit(X)
# transform x using that scaler
x_scaled = scaler.transform(x)

# convert x into a torch tensor
x_tensor =  torch.tensor(x_scaled, dtype=torch.float32)

# input tensor into model and show the output
with torch.no_grad():
    pred = model(x_tensor)

# --- REPLACED/ENHANCED LAST PREDICTION DISPLAY LINE ---
if submitted:
    prediction_value = (pred >= 0.5).float()[0].item()
    prob_percent = float(pred[0].item()) * 100

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📊 Diagnostic Result")

    col_res1, col_res2 = st.columns([1, 2])

    with col_res1:
        st.metric(
            label="Model Risk Score",
            value=f"{prob_percent:.1f}%",
            delta="High Risk" if prediction_value == 1.0 else "Low Risk",
            delta_color="inverse" if prediction_value == 1.0 else "normal"
        )

    with col_res2:
        if prediction_value == 1.0:
            st.error("⚠️ **Presence of Heart Disease Detected**\nThe MLP model indicates a positive indication for heart disease based on the provided inputs.")
        else:
            st.success("✅ **No Presence of Heart Disease Detected**\nThe MLP model indicates a low risk/negative presence for heart disease.")
else:
    st.info("💡 Fill out the patient parameters above and click **Run Prediction Model** to display results.")