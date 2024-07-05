# Food Review Classification with Word Embeddings

This project demonstrates how to classify food reviews as positive or negative using neural networks implemented in TensorFlow/Keras. As a byproduct, it generates word embeddings using Keras' Embedding layer.

## Problem Statement

The task is to develop a machine learning model that can accurately classify food reviews into positive and negative sentiments based on the text content.

## Dataset

The dataset consists of example food reviews with corresponding sentiment labels:
- Positive reviews are labeled as `1`.
- Negative reviews are labeled as `0`.

## Implementation

### Requirements

- Python 3.x
- TensorFlow/Keras
- NumPy

### Data Preprocessing

1. **Text Encoding:** Each review is converted into a sequence of integers using one-hot encoding with a predefined vocabulary size.
   
2. **Padding Sequences:** The encoded sequences are padded to a maximum length to ensure uniform input size for the neural network.

### Model Architecture

- **Embedding Layer:** Converts input sequences into dense vectors of fixed size (embedding vectors).
  
- **Flatten Layer:** Converts the 2D embedded vectors into 1D vectors to feed into the Dense layer.
  
- **Dense Layer:** Output layer with sigmoid activation for binary classification.

### Training

- The model is trained on the padded review sequences and sentiment labels.
- Optimizer: Adam
- Loss function: Binary cross-entropy
- Metrics: Accuracy

### Evaluation

- The model's performance is evaluated on the training data.
- Achieved accuracy: 100% on the training set.

### Word Embeddings

- Extracted from the trained Embedding layer.
- Each word in the vocabulary has a corresponding embedding vector.

## Usage

1. Install the required dependencies:
   ```
   pip install tensorflow numpy
   ```

2. Run the training script:
   ```
   python food_review_classification.py
   ```

3. Evaluate the model and extract word embeddings as needed.

## Files Included

- `food_review_classification.py`: Main script containing the model implementation and training logic.
- `README.md`: This file, providing an overview of the project and instructions.

## Conclusion

This project demonstrates how neural networks can be used for sentiment analysis of food reviews using word embeddings. The provided model achieves perfect accuracy on the training data, indicating its effectiveness in classifying sentiment based on text data.

