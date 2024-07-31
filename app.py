from pathlib import Path

from altair import Description
import streamlit as st
from PIL import Image

# --- PATH SETTINGS ---
current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
css_file = current_dir / "styles" / "main.css"
resume_file = current_dir / "assests" / "resume.pdf"
profile_pic = current_dir / "assests" / "profile_pic.png"

# --- Define general information ---
PAGE_TITLE = "Portfolio | Cristopher Delgado"
PAGE_ICON = ":wave:"
NAME = "Cristopher Delgado"
DESCRIPTION = """
Data Scientist, assisting in resaerch and development of medical resaerch.
"""
EMAIL = "cristopher.d.delgado@gmail.com"
SOCIAL_MEDIA = {
    "LinkedIn": "",
    "GitHub": "", 
}
PROJECTS = {
    ""
}

# Define page configuration 
st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON)

# Define portfolio title
st.title("YES!")

# --- LOAD CSS, PDF, & PROFILE PIC ---
with open(css_file) as f:
    st.markdown(f"<style>{}</style>".format(f.read()), unsafe_allow_html=True)

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
        label=":page_facing_up: Download Resume"
        data=PDFbyte
        file_name=resume_file.name,
        mime="application/octet-stream"
    )
    st.write(":e-mail:", EMAIL)
    