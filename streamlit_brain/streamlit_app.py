import streamlit as st
from tensorflow.python.keras.models import load_model
from .predict import classify, preprocess_image
from PIL import Image
from lime import lime_image
from skimage.segmentation import mark_boundaries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import requests
import os
import streamlit as st

def brain_classification_app():
    # Title app
    st.title("Brain Tumor Classification with Magnetic Resonance Imaging")

    # Set header for classification
    st.header('Please upload Brain MRI Slice Image')

    # Cache augemented model
    @st.cache_resource
    def load_keras_model(url, file_path):
        """Downloads the model from an AWS bucket URL and loads it."""
        try:
            # Attempt to load the model from the specified file path
            loaded_model = load_model(file_path)
            print("Model loaded successfully from local file.")
            return loaded_model
        except Exception as e:
            print(f"Failed to load model from local file: {e}. Attempting to download.")
            
            # If loading fails, try downloading the model from the AWS bucket
            try:
                response = requests.get(url)
                response.raise_for_status()  # Raise an error for bad responses
                print("Response status code:", response.status_code)  # Log the status code
                
                # Save the model to the file path
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                print("Model downloaded successfully!")
                
                # Load the model from the downloaded file
                loaded_model = load_model(file_path)
                print("Model loaded successfully from downloaded file.")
                return loaded_model
            
            except Exception as e:
                print(f"Error loading Keras model: {e}")
                return None
    
    # def load_keras_model(url, file_path):
    #     """Downloads the model from a AWS bucket URL and loads it."""
    #     if os.path.exists(file_path):
    #         try:
    #             # Load the saved model
    #             loaded_model = load_model(file_path)
    #             print("Model loaded successfully!")
    #             return loaded_model
    #         except Exception as e:
    #             print(f"Error loading Keras model from file: {e}")
    #             return None
    #     else:
    #         try:
    #             # Download the model from AWS bucket
    #             response = requests.get(url)
    #             response.raise_for_status()  # Raise an error for bad responses
    #             print("Response status code:", response.status_code)  # Log the status code
                
    #             # Save the model to the file path
    #             with open(file_path, 'wb') as f:
    #                 f.write(response.content)
    #             print("Model downloaded successfully!")
                
    #             # Load the saved model
    #             loaded_model = load_model(file_path)
    #             print("Model loaded successfully!")
    #             return loaded_model
            
    #         except Exception as e:
    #             print(f"Error loading Keras model: {e}")
    #             return None

    # Cache Lime explainer
    @st.cache_resource
    def load_lime_explainer(random_state=42):
        """
        Load LimeImageExplainer
        """
        explainer = lime_image.LimeImageExplainer(random_state=random_state)
        return explainer

    # Load classifier
    url = "https://braintumorclassificationcap.s3.us-west-1.amazonaws.com/op_model1_aug.keras"
    file_path = "op_model1_aug.keras"

    aug_model = load_keras_model(url, file_path)
    #print(type(aug_model))
    # Check if the model was loaded successfully
    if aug_model is not None:
        print("Keras model loaded successfully!")
    else:
        print("Failed to load Keras model. Check the model loading function and the file path.")

    class_names = [
        "glioma",
        "meningioma",
        "no_tumor",
        "pituitary"
    ]

    # Upload File
    file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

    # Create statement to predict
    if file is not None:
        image = Image.open(file).convert('RGB')
        
        # Make container with three columns 
        with st.container():
            # Divide container in three
            col1, col2, col3 = st.columns(3)
            
            with col2:
                # Display Image in app
                st.image(image, use_column_width=True)
                
        # Classify image
        class_name, prob = classify(image, aug_model, class_names)
        
        probability = round(prob*100, 2)
        
        # Write classification
        st.write(f"#### The Brain lesion is most likely a {class_name} case")
        st.write(f"#### The probability associated with {class_name} is: {probability}%")

        # Lime Explanation
        with st.expander("See Lime Explanation Mask and Importance Heatmap"):
            with st.container():
                # Divide container into 2 columns
                #col_1, col_2 = st.columns(2)
                
                # Load Lime explainer
                explainer = load_lime_explainer()
                
                # Define image we want to predict
                image = Image.open(file)
                
                # Preprocess the image for lime explanation and model prediction
                img = preprocess_image(image)
                
                # Develop local model explanation
                explanation = explainer.explain_instance(
                    image=img,
                    classifier_fn=aug_model.predict,
                    top_labels=4,
                    num_samples=2000,
                    hide_color=0,
                    random_seed=42
                )

                # Obtain mask and image from explainer
                temp, mask = explanation.get_image_and_mask(
                    explanation.top_labels[0],  # Using the top predicted label for visualization
                    positive_only=True,
                    num_features=5,
                    hide_rest=True,
                    min_weight=0.1
                )
                
                # Obtaining components to Diplay Heatmap on second subplot
                ind = explanation.top_labels[0]
                dict_heatmap = dict(explanation.local_exp[ind])
                heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)
                
                # Create the Lime Mask Figure with the Heatmap in a single Figure
                # Lime Mask
                fig, axes = plt.subplots(1, 2, figsize=(14,6), facecolor='white')
                axes[0].imshow(mark_boundaries(temp / 2 + 0.5, mask)) # Plots image
                axes[0].tick_params(axis='x', which='major', labelsize=12)
                axes[0].tick_params(axis='y', which='major', labelsize=12)
                axes[0].set_title("Concerning Area", fontsize=20)
                
                # Display heatmap on second subplot
                heatmap_plot = axes[1].imshow(heatmap, cmap='RdBu_r', vmin=-heatmap.max(), vmax=heatmap.max())
                axes[1].set_title("Red = More Concernig; Blue = Less Concerning", fontsize=20)
                axes[1].set_xlim(0, img.shape[1]) # Set x-axis to equal the image width
                axes[1].set_ylim(img.shape[0], 0) # Set y-axis to equal the image height
                # Adjust tick marks size for both x and y axes
                axes[1].tick_params(axis='x', which='major', labelsize=12)
                axes[1].tick_params(axis='y', which='major', labelsize=12)
                colorbar = plt.colorbar(heatmap_plot, ax=axes[1]) # Add colorbar
                colorbar.ax.tick_params(labelsize=12)
                
                # Create tight layout for figure
                plt.tight_layout()
                
                # Display the figure directly using Streamlit
                st.pyplot(fig)
                
    #### Make a section talking about the model 

    # Make Section Header
    st.header('Model Information')

    # Make Secondary Header
    st.write("## Model Architecture")

    # Make text explaining methodology
    st.write(
        "The final model architecture is found in the 'Model Architecture' section below. "
        "This model architecture is also the same model that is being used for the model classification that is utilized in this app for image predictions. "
        "The model uses 4 Convolutional layers, 4 Maxpooling layers, 2 Dropout layers, and 4 Fully Connected layers. The output layer is a 4 neuron output. In order to classify 'no_tumor', 'pituitary', 'meningioma', and 'glioma'. "
    )

    st.write("The detailed dive into the model training and development can be found in the following [repository]('https://github.com/cristopher-d-delgado/brain_tumor_classification').")

    # Display Model architecture plot
    st.image("streamlit_brain/images/model_arch.jpg", use_column_width=True)

    # Make Secondary Header 
    st.write("## Performance of Testing Data")

    # Make text explaining test data
    st.write(
        "The testing data used for the model development originates from Kaggle."
        "The first source is from a Kaggle dataset named [Brain Tumor Classification (MRI)](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri), "
        "the second dataset is named [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset), " 
        " and lastly the third dataset is named [Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection). "
    )
    st.write(
        "The testing data distribution consisted of 400 glioma, 421 meningioma, and 374 pituitary tumor images. "
        "There were 510 images that had no tumor present in the testing set. "
        "The data distribution is shown in the 'Merged Dataset Figure' below."
    )
    st.write(
        "The individual performance of each class is demonstrated in the confusion matrix figure below."
        "Adding on, the model performed with a 92% accuracy on unseen data."
    )

    # Display data distribution
    st.image("streamlit_brain/images/merged_dist.png", use_column_width=True)

    # Make text explaining confusion matrix 
    # Make a table with metrics on model
    data = {
        'Set': ['Training', 'Testing', 'Validation'],
        'Sensitivity': ['99.94%', '92.37%', '97.92%'],
        'Specificity': ['99.97%', '93.14%', '97.98%'],
        'Accuracy': ['99.97%', '92.37%', '97.92%'],
        'Validation Loss/Generalization': [0.006, 0.584, 0.081]
    }

    # Create a DataFrame from the sample data
    df = pd.DataFrame(data)

    # Make header for table
    st.write("## Model Metrics")

    # Display the table
    st.table(df)

    # Display Confusion matrix
    st.image("streamlit_brain/images/confusion_matrix_augmented.png", use_column_width=True)