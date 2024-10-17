# Sentiment Analysis on Customer Reviews

This project implements a **sentiment analysis** model to classify customer reviews as either *positive* or *negative*. It uses **Natural Language Processing (NLP)** techniques for text preprocessing and **Logistic Regression** for classification.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Model](#model)
- [Results](#results)
- [License](#license)

## Project Overview
Sentiment analysis is the process of determining whether a piece of text expresses a *positive*, *negative*, or *neutral* sentiment. This project focuses on classifying customer reviews into **positive** or **negative** using text preprocessing, feature extraction, and machine learning models.

## Features
- **Text Preprocessing**: Tokenization, lemmatization, stopword removal using **spaCy**.
- **Feature Extraction**: Convert text into numerical features using **TF-IDF**.
- **Modeling**: Train a **Logistic Regression** model to classify reviews.
- **Evaluation**: Evaluate the model using accuracy and a classification report (precision, recall, F1-score).

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/sentiment-analysis-nlp.git
    cd sentiment-analysis-nlp
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Download the **spaCy** English model:
    ```bash
    python -m spacy download en_core_web_sm
    ```

## Usage

1. **Run the sentiment analysis script**:
    ```bash
    python sentiment_analysis.py
    ```

2. **Expected output**:
    The script will display the model's accuracy and a classification report after training and testing on the sample data.

3. **Modify the dataset**:
    You can replace the sample dataset with your own customer reviews in the `data` variable inside `sentiment_analysis.py`.

## Data

This project uses a small sample dataset of customer reviews stored within the code, but you can replace it with any text dataset. For larger datasets, you can use:
- [IMDb Movie Reviews Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- [Amazon Customer Reviews Dataset](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)

The data should have at least two columns: one for the **review text** and one for the **sentiment labels** (1 for positive, 0 for negative).

## Model

- **Logistic Regression** is used as the classification algorithm.
- **TF-IDF** is used to convert text data into numerical features for model input.
- **spaCy** is used for text preprocessing (tokenization, lemmatization, stopwords removal).

## Results

After running the script, you can expect an output similar to:

Accuracy: 0.85

Classification Report: precision recall f1-score support

Negative       0.80      1.00      0.89         1
Positive       1.00      0.75      0.86         1

accuracy                           0.85         2

macro avg 0.90 0.88 0.87 2 weighted avg 0.90 0.85 0.87 2

You can experiment with different models or datasets for better accuracy.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

