# Road Signs Classification using CNN

A deep learning-based project to classify road signs into their respective categories using a Convolutional Neural Network (CNN).  
The model is trained on a dataset from **Kaggle** and deployed as an interactive web application using **Streamlit** on **Hugging Face Spaces**.


## Context
Traffic sign recognition is a crucial part of intelligent transportation systems, assisting in the development of autonomous driving and road safety solutions. This project aims to build a CNN-based model to classify various road signs accurately.


## Objective
To develop and deploy a CNN model capable of recognizing and classifying road signs from images, ensuring high accuracy and a user-friendly interface.


## Scope
- Train a CNN model on a labeled road signs dataset.
- Deploy the trained model for public use.
- Provide an accessible UI for real-time testing of road sign classification.


## Target Audience
- Students and researchers in Machine Learning & Computer Vision.
- Developers building intelligent transportation solutions.
- Enthusiasts exploring image classification projects.


## Dataset Overview
- **Source**: https://www.kaggle.com/datasets/harbhajansingh21/german-traffic-sign-dataset
- **Format**: `.pkl` (Pickle) files containing training, validation, and testing data.
- **Total Classes**: 43 road sign categories.


## Data Cleaning
- Loaded dataset from pickle files.
- Verified data integrity and removed any corrupted entries.
- Ensured all images have consistent dimensions.
- Encoded labels for model compatibility.


## Techniques Used
- **Python** for implementation.
- **NumPy & Pandas** for data handling.
- **Matplotlib** for visualization.
- **TensorFlow/Keras** for CNN model building.
- **Early Stopping** to prevent overfitting.
- **Streamlit** for UI development.
- **Hugging Face Spaces** for deployment.


## User Interface
The Streamlit-based UI allows users to upload road sign images and get:
- Predicted class name.
- Confidence score.
  
**Live Demo**: https://huggingface.co/spaces/Yahavarshini/road-signs-classifier


## Results & Conclusion
The CNN model achieved high accuracy on the test dataset, demonstrating strong generalization capabilities for unseen images.  
The system can be used as a base for advanced traffic sign recognition applications.


## Future Scope
- Improve model accuracy using transfer learning.
- Extend dataset for more diverse road signs.
- Optimize model for mobile deployment.
- Integrate into real-time traffic monitoring systems.


## ⚙ Installation & Usage
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/Road-Signs-Classification.git
cd Road-Signs-Classification

# Install dependencies
pip install -r requirements.txt

# Run the application locally
streamlit run app.py


