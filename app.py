from pathlib import Path
import select

import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image

# --- PATH SETTINGS ---
current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
css_file = current_dir / "styles" / "main.css"
resume_file = current_dir / "assets" / "resume.pdf"
profile_pic = current_dir / "assets" / "profile_pic.jpg"

# --- Define general information ---
PAGE_TITLE = "Portfolio | Cristopher Delgado"
PAGE_ICON = ":wave:"
NAME = "Cristopher Delgado"
DESCRIPTION = """
Biomedical Engineer specializing in data analysis and machine learning.
Driven in pursuing research and development of biomedical biosensing applications and diagnostics.
"""
EMAIL = "cristopher.d.delgado@gmail.com"
SOCIAL_MEDIA = {
    "LinkedIn": "https://www.linkedin.com/in/cristopher-d-delgado",
    "GitHub": "https://github.com/cristopher-d-delgado", 
}
PROJECTS = {
    "🧠 Brain Lesion Classification": {
        "link": "https://github.com/cristopher-d-delgado/brain_tumor_classification",
        "description": [
            "Implemented a convolutional neural network to classify brain lesions from MRI images by developing a deep learning convolutional neural network utilizing Tensorflow Keras.",
            "The best model achieved 99% specificity and 99% sensitivity with good generalization to unknown data.",
            "Merged various datasets to create diverse training images for model training.",
            "Enhanced learning capability of the model through the use of data augmentation to combat the imbalanced learning dataset.",
            "Created web application actionable insight using LIME to represent concerning areas of the MRI image slice."
        ]
    },
    "🫁 Pneumonia Detection": {
        "link": "https://github.com/cristopher-d-delgado/image_classification_pneumonia",
        "description": [
            "Developed a deep learning model to detect pneumonia from chest X-ray images.",
            "Enhanced learning capability of the model through the use of data augmentation to combat the imbalanced learning dataset.",
            "Conducted hyperparameter tuning to optimize the model's accuracy and reduce overfitting.",
            "The best model achieved 88% specificity and 94% sensitivity with good generalization to unknown data",
            "Provided actionable insight into the hypothetical scenario by recommending model implementation into existing software incorporated in medical devices"
        ]
    },
    "🫀 Heart Disease Prediction": {
        "link": "https://github.com/cristopher-d-delgado/heart_failure",
        "description": [
            "Created a logistic regression model to predict the likelihood of heart disease.",
            "Performed feature engineering to select the most relevant clinical parameters.",
            "Evaluated the model using precision, recall, and ROC-AUC metrics.",
            "Built models iteratively to evaluate their baselines in comparison to their optimized versions with hyperparameter tuning.",
            "Interpreted the model coefficients to determine the most influential factors for heart disease.",
            "The best model achieved a specificity of 86% and a sensitivity of 87% after hyperparameter tuning.",
            "Attempted logistic strategies including logistic regression, random forests, gradient boost, XGBoost, AdaBoost, and K-Nearest Neighbors in order to determine the best-performing logistic classifier."
        ]
    }
}


# Define page configuration 
st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON)

# Define portfolio title
st.title("#")

# --- Create sidebar that contains usable projects ---
# --- Add Sidebar for Model Selection ---
with st.sidebar:
    selected = option_menu(
        menu_title = "Navigation",
        options = ["Home", "Brain Lesion Classification", 
                   "Pneumonia Detection", "Heart Disease Prediction"], 
        icons = ["house-heart-fill", "activity", "lungs-fill", "heart-pulse-fill"], 
        default_index=0
    )

if selected == "Home": 
    # --- LOAD CSS, PDF, & PROFILE PIC ---
    with open(css_file) as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

    with open(resume_file, "rb") as pdf_file:
        PDFbyte = pdf_file.read()

    profile_pic = Image.open(profile_pic)

    # --- HERO SECTION ---
    col1, col2 = st.columns(2, gap="small")

    with col1:
        st.image(profile_pic, width=230)

    with col2:
        st.title(NAME)
        st.write(DESCRIPTION)
        st.download_button(
            label=":page_facing_up: Download Resume",
            data=PDFbyte,
            file_name=resume_file.name,
            mime="application/octet-stream"
        )
        st.write(":e-mail:", EMAIL)

    # --- SOCIAL LINKS ---
    st.write("#")
    cols = st.columns(len(SOCIAL_MEDIA))

    for index, (platform, link) in enumerate(SOCIAL_MEDIA.items()):
        cols[index].write(f"[{platform}]({link})")

    # --- EXPERIENCE AND QUALIFICTIONS ---
    st.write('#')
    st.subheader("Experience & Qualifications")
    st.write(
        """
        - Something
        """
    )

    # --- SKILLS ---
    st.write("#")
    st.subheader("Hard Skills")
    st.write(
        """
        - 🧑‍💻Programming: Python (Scikit-learn, Pandas, Tensorflow, SciPy, Numpy)
        - :bar_chart: Data Visualization: Matplotlib, Seaborn, Streamlit
        - 📈 Modeling: Logistic Regression, Linear Regression, Decision Trees, Deep Neural Networks
        - 🏢 Databases: SQL
        """
    )

    # --- Work History ---
    st.write('#')
    st.subheader("Work History")
    st.write("---")

    # --- JOB 1
    st.write("🧬️", "**Intern - Bioinformatics Data Analyst | Orlucent**")
    st.write("07/2023 - Present")
    st.write(
        """
        - Conducted comprehensive data analysis, developed machine learning models, generated detailed reports, and provided
        actionable insights to support informed decision-making
        - Played a key role in data understanding and data-driven solutions by drawing insights from machine learning models and
        making suggestions and evaluations for lesion assessment purposes
        - Develop a deep understanding of startup industry dynamics and challenges while working on a confidential project
        - Collaborate with cross-functional teams to meet project objectives
        """
    )

    # --- JOB 2
    st.write("#")
    st.write("🧫️", "**Yun Wang Lab - Lab Member | San Jose State University**")
    st.write("01/2022 - 12/2023")
    st.write(
        """
        - Collaborated with an interdisciplinary team to design a biosensor prototype with a microfluidic lab on chip technology in
        conjunction with Quantum Dots resulting in successful trials detection of Botulinum Neurotoxin Serotype A.
        - Conducted extensive testing of Quantum Dots for their stability and sensitivity for Botulinum Neurotoxin Detection Serotype A
        resulting in statistical evidence of the biosensors functionality.
        - Performed Reconstitution of Peptide in 10% Dimethyl Sulfoxide & Botulinum Neurotoxin in HEPES buffer required to perform
        biosensor testing.
        """
    )

    # --- Projects & Accomplishments ---
    st.write("#")
    st.subheader("Projects & Accomplishments")
    st.write("---")

    for project, details in PROJECTS.items():
        st.write(f"[{project}]({details['link']})")
        for desc in details['description']:
            st.write(f"- {desc}")
        st.write("#")

if selected == "Brain Lesion Classification":
    st.write("Currently Under Contruction!")