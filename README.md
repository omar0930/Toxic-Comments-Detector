# Toxic Comment Detection

## Overview
This project focuses on developing a machine learning model to detect toxic comments. Using a dataset from the Jigsaw Toxic Comment Classification Challenge, the model is designed to identify various types of toxicity, including threats, obscenity, insults, and identity hate.

## Features
- **Text Classification**: Identifies toxic comments and categorizes them into different types of toxicity.
- **Interactive Interface**: Provides a user-friendly interface using Gradio for real-time toxic comment detection.

## Technologies Used
- **Programming Languages**: Python
- **Libraries**: 
  - Data Processing: Pandas, NumPy
  - Machine Learning: TensorFlow, Keras
  - Visualization: Matplotlib, Seaborn
  - GUI: Gradio

## Dataset
The dataset used in this project is sourced from the [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/datasets/julian3833/jigsaw-toxic-comment-classification-challenge). It contains comments labeled for different types of toxicity.

## Project Structure
- toxic-comment-detector.ipynb: Jupyter Notebook containing the entire project workflow.
- data/: Directory to store the dataset.
- requirements.txt: File containing the list of required libraries.
## Model Architecture
- The model is built using TensorFlow and consists of:

- Embedding Layer: Converts text into dense vectors.
- Bidirectional LSTM Layers: Captures dependencies in both directions.
- Dense Layers: Final classification layers.
## Results
- The model's performance is evaluated using metrics such as accuracy, precision, recall, and F1-score. The interactive interface allows users to input comments and get real-time predictions on their toxicity levels.
