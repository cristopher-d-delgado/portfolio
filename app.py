from pathlib import Path

import streamlit as st
from PIL import Image

# --- PATH SETTINGS ---
current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
css_file = current_dir / "styles" / "main.css"
resume_file = current_dir / "assests" / "resume.pdf"
profile_pic = current_dir / "assests" / "profile_pic.png"

# --- Define general information ---
PAGE_TITLE = ""
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
