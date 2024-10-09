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

EDUCATION = {
    "MS Biomedical Engineer ": "San Jose State University, Dec 2024",
    "BS Biomedical Engineer ": "San Jose State University, May 2023",
    "Flatiron Data Science Certificate ": "Virtual"
}

WORK_HISTORY = {
    "🧬️ Intern - Bioinformatics Data Analyst | Orlucent": {
        "dates": "07/2023 - Present",
        "description": [
            "Conducted comprehensive data analysis, developed machine learning models, generated detailed reports, and provided actionable insights to support informed decision-making.",
            "Played a key role in data understanding and data-driven solutions by drawing insights from machine learning models and making suggestions and evaluations for lesion assessment purposes.",
            "Develop a deep understanding of startup industry dynamics and challenges while working on a confidential project."
            "Collaborate with cross-functional teams to meet project objectives."
        ]
    },
    "🧫️ Yun Wang Lab Diagnostics - Lab Member | San Jose State University": {
        "dates": "01/2022 - 12/2023",
        "description": [
            "Collaborated with an interdisciplinary team to design a biosensor prototype with a microfluidic lab on chip technology in conjunction with Quantum Dots resulting in successful trials detection of Botulinum Neurotoxin Serotype A.",
            "Conducted extensive testing of Quantum Dots for their stability and sensitivity for Botulinum Neurotoxin Detection Serotype A resulting in statistical evidence of the biosensors functionality.",
            "Performed Reconstitution of Peptide in 10% Dimethyl Sulfoxide & Botulinum Neurotoxin in HEPES buffer required to perform biosensor testing.",
        ]
    }
}