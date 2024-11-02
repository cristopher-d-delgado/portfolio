from pathlib import Path
import sys
from home import EDUCATION, PAGE_TITLE, PAGE_ICON, PROJECTS, NAME, DESCRIPTION, EMAIL, SOCIAL_MEDIA, WORK_HISTORY
import streamlit as st
from streamlit_brain.streamlit_app import brain_classification_app
#from streamlit_brain.predict import classify, preprocess_image
from streamlit_option_menu import option_menu
from PIL import Image

# --- PATH SETTINGS ---
current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
css_file = current_dir / "styles" / "main.css"
resume_file = current_dir / "assets" / "resume.pdf"
profile_pic = current_dir / "assets" / "profile_pic.jpg"

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

    # --- EDUCATION --
    st.write("#")
    st.subheader("Education")
    # Iterate over education dictionary
    for degree, details in EDUCATION.items():
        st.write(f"- {degree}{details}")
    
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
    
    for experience, details in WORK_HISTORY.items():
        st.write(f"{experience} ({details['dates']})") # Print work title and date range
        for desc in details['description']:
            st.write(f"- {desc}") # Bullet point description
        st.write("#")
    
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
    brain_classification_app()
else:
    st.write("Oops did not lauch!")