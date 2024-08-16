# Sentiment Analysis Project
## Overview
This project focuses on building a sentiment analysis system that classifies textual data, specifically movie reviews, into positive, negative, or neutral sentiments. Using advanced natural language processing (NLP) techniques and machine learning models like Recurrent Neural Networks (RNNs) with Long Short-Term Memory (LSTM) units, the system effectively handles sequential data and contextual understanding.

## Features
Text Preprocessing: Tokenization, stop-word removal, and vectorization.
Sentiment Classification: Accurate classification of sentiments using RNN and LSTM models.
Model Evaluation: Performance evaluation using metrics such as accuracy, precision, recall, and F1-score.
Meta-Learning Architectures: Combining multiple models for robust sentiment classification.

## Dataset
The project uses the IMDb movie review dataset, consisting of 50,000 labeled reviews (positive or negative). The dataset is split into training, validation, and test sets to ensure accurate model training and evaluation.

## Installation
Clone the repository:
```bash
git clone https://github.com/Sandar82/Sentiment-Analysis.git
```

Navigate to the project directory:
```bash
cd sentiment-analysis-project
```

## Usage
Data Preprocessing: Prepare the dataset using the provided preprocessing scripts.
Model Training: Train the sentiment analysis model using the training script.
Evaluation: Evaluate the model's performance on the test dataset.
Prediction: Use the model to predict sentiments on new text data.
Example:

```python
from predict import predict_sentiment

result = predict_sentiment("The movie was fantastic!")
print(result)  # Output: Positive
```

## Limitations
The model may struggle with detecting nuanced sentiments, such as sarcasm or mixed emotions.
Performance might not generalize well to texts outside the IMDb dataset without additional training.
Future Extensions
Enhance the model's ability to detect complex sentiments.
Expand the model to handle multiple languages and diverse text domains.
Explore transfer learning and advanced architectures like transformers for improved performance.

## Contributors 
* Maria Namitha Nelson
* Sandar Aung

## Acknowledgments
IMDb for providing the dataset.
PyTorch and SpaCy for their powerful libraries used in this project.
The open-source community for ongoing support and contributions.
This README file provides a comprehensive overview of the project, its usage, and how others can contribute or extend it.
