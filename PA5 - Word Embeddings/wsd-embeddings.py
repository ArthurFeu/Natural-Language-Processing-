"""
Programming Assignment 5
Author: Arthur Mendonca Feu
Date: April 11, 2024
Class: CMSC-416-001 - INTRO TO NATURAL LANG PROCESS - Spring 2024

This script utilizes two different machine learning models, a Support Vector Machine (SVM) and a Neural Network (NN),
to perform Word Sense Disambiguation (WSD) by identifying the intended meaning of a word based on its context.
The feature representation used here is the average of GloVe embeddings, capturing the semantic essence of the context surrounding the target word.

Algorithm Steps:
	1. Extract the context AND the sense ID of the target word from the training data.
	2. Extract the context of the target word from the test data.
 
    3. Vectorize the context data from the training data:
        Utilizes GloVe embeddings to convert text documents (contexts) into numeric vectors.
        This transformation is necessary for machine learning models to process the text data.

    4. Train the classifier using the training data:
        Based on user's choice, initialize either an SVM with a linear kernel or a neural network model.
        If no model is specified, SVM is used by default.

    5. Make predictions using the classifier on the vectorized test data.

    6. Print the predictions to the standard output in the pseudo-XML format.
    
Usage:
    Run the script from the command line as follows:
    python wsd-embeddings.py line-train.txt line-test.txt <embedding file> [OPTIONS: SVM|NN] > my-line-answers.txt

_____________________________________________________________________________________________________________
Neural Network (NN)

Neural Networks are a set of algorithms modeled loosely after the human brain, designed to recognize patterns. They interpret sensory data through a kind of machine perception, labeling, or clustering raw input. The patterns they recognize are numerical, contained in vectors, into which all real-world data, be it images, sound, text, or time series, must be translated.

How It Works:

Input Layer: Receives the average GloVe embeddings as input. Each input neuron corresponds to a dimension in the GloVe vector used to represent words in the context.

Hidden Layers: My NN has two hidden layers with ReLU (rectified linear unit) activation functions. These layers are responsible for transforming the input into a space where the classes (senses of the word) become linearly separable. They learn increasingly abstract representations of the context.

Output Layer: The final layer uses a softmax activation function that outputs the probabilities of each class (sense). The number of neurons in this layer corresponds to the number of classes (senses).

Training: During training, the NN adjusts its weights through backpropagation, minimizing the loss function, which measures the difference between the predicted probabilities and the actual class.

Results with 300 dimensions GloVe embeddings:

    Confusion Matrix:
    Actual | Predicted      phone   product
    phone                   70      3
    product                 2       51

    Correct: 121
    Incorrect: 5
    Total: 126
    Accuracy: 96.03%

_____________________________________________________________________________________________________________
Support Vector Machine (SVM)

SVM with a linear kernel is utilized to classify words based on their contexts for word sense disambiguation. It finds the optimal dividing straight line that separates different senses of a word with the greatest margin, which is the distance to the nearest data points.

How It Works:

Maximizing the Margin: SVM in my script looks for the hyperplane with the maximum margin, i.e., the maximum distance between data points of both senses. This hyperplane acts as the decision boundary that separates different senses.

Support Vectors: The vectors (data points) that determine the hyperplane are the support vectors. They are the data points closest to the hyperplane.

Training: During training, the SVM model learns to assign new examples to one category or the other by calculating which side of the hyperplane they fall on, based on the features (average embeddings).

Results with 300 dimensions GloVe embeddings:

    Confusion Matrix:
    Actual | Predicted      phone   product
    phone                   69      1
    product                 3       53
    
    Correct: 122
    Incorrect: 4
    Total: 126
    Accuracy: 96.83%

_____________________________________________________________________________________________________________
All models compared:

Support Vector Machine (SVM) with 300 dimensions GloVe embeddings:
Accuracy: 96.83%

Neural Network (NN) with 300 dimensions GloVe embeddings:
Accuracy: 96.03%

Naive Bayes model:
Accuracy: 94.44%

Logistic Regression model:
Accuracy: 93.65%

Support Vector Machine model:
Accuracy: 92.86%

Decision List Classifier model:
Accuracy: 88.10%

most frequent sense:
Accuracy: 42.86%

The Support Vector Machine (SVM) model enhanced with 300 dimensions GloVe embeddings stands out with the highest accuracy of 96.83%, demonstrating superior performance. Following closely, the Neural Network (NN) model using the same high-dimensional GloVe embeddings achieves an accuracy of 96.03%.

In contrast, the Naive Bayes, Logistic Regression, and the traditional SVM models, which utilize a Bag of Words approach, show lower accuracies of 94.44%, 93.65%, and 92.86% respectively. The Decision List Classifier, also based on Bag of Words, records an accuracy of 88.10%.

The baseline 'most frequent sense' method, the simplest heuristic, significantly lags with only 42.86% accuracy, highlighting the enhanced capability of machine learning models with advanced feature representations over more basic text processing techniques.

The higher accuracies in the current assignment suggest better capabilities likely due to the more nuanced understanding of language provided by GloVe embeddings compared to indicators in a Bag of Words model.
"""

# Explanation over libraries used in the code:
# re: Regular expression operations
# sys: System-specific parameters and functions
# numpy: Fundamental package for scientific computing with Python
# SVC: Support Vector Classification
# keras: High-level neural networks API, written in Python and capable of running on top of TensorFlow
# Sequential: Linear stack of layers
# Dense: Regular densely-connected NN layer
# LabelEncoder: Encode target labels with value between 0 and n_classes-1 (used to label encode the sense IDs for the neural network model)

import re
import sys
import numpy
from sklearn.svm import SVC
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder

# Common functions for both models:

def clean_context(context):
    stopwords = {
        "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
        "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers",
        "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
        "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are",
        "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does",
        "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until",
        "while", "of", "at", "by", "for", "with", "about", "against", "between", "into",
        "through", "during", "before", "after", "above", "below", "to", "from", "up", "down",
        "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here",
        "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more",
        "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so",
        "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now", "d",
        "ll", "m", "o", "re", "ve", "y", "ain", "aren", "couldn", "didn", "doesn", "hadn",
        "hasn", "haven", "isn", "ma", "mightn", "mustn", "needn", "shan", "shouldn", "wasn",
        "weren", "won", "wouldn"
    }
    
    # Clean the context: remove HTML tags, apostrophes, digits, and non-alphanumeric characters

    re.sub(r'<head>.*?</head>', '', context, flags=re.DOTALL)
    cleaned_context = re.sub(r'<[^>]*>', ' ', context)
    cleaned_context = re.sub(r"'", ' ', cleaned_context) 
    cleaned_context = re.sub(r'\d+', ' ', cleaned_context)
    cleaned_context = re.sub(r'[^\w\s]', ' ', cleaned_context) 
    cleaned_context = re.sub(r'<(/?s|/?p|@)>', '', cleaned_context).strip()
    cleaned_context = cleaned_context.lower()
    
    # Splitting into words and filtering out stopwords
    words = cleaned_context.split()
    filtered_words = [word for word in words if word not in stopwords]
    
    # Joining the words back into a single string without stopwords
    cleaned_context = ' '.join(filtered_words)

    return cleaned_context

def extract_context_from_test(file_path):
    # Regex to match instance and context
    pattern = re.compile(r'<instance id="([^"]+)">.*?<context>(.*?)</context>', re.DOTALL)
    
    # Dictionary to hold instance ID as key and context as value
    context_dict = {}
    
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        matches = pattern.findall(content)
        
        for instance_id, context in matches:
            cleaned_context = clean_context(context)
            context_dict[instance_id] = cleaned_context
    
    return context_dict

def extract_contexts_and_senseid_vector(file_path):
    # Regex to match instance, senseid, and context
    pattern = re.compile(
        r'<instance id="([^"]+)">.*?<answer instance="[^"]+" senseid="([^"]+)"/>.*?<context>(.*?)</context>',
        re.DOTALL
    )
    # Dictionary to hold instance ID as key and context as value
    context_dict = {}
    senseid_vector = []
    
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        matches = pattern.findall(content)
        
        for instance_id, senseid, context in matches:
            cleaned_context = clean_context(context)
            context_dict[instance_id] = cleaned_context
            senseid_vector.append(senseid)
    
    return context_dict, senseid_vector

def load_glove_embeddings(embedding_file):
    embeddings_dict = {}
    with open(embedding_file, 'r', encoding='utf-8') as file:
        # For each line in the file split the line into a word and the embedding vector
        for line in file:
            values = line.split()
            word = values[0]
            vector = numpy.asarray(values[1:], dtype='float32')
            embeddings_dict[word] = vector
    return embeddings_dict

def average_embedding(context, embeddings_dict):
    words = context.split()
    embedding_vectors = []
    
    for word in words:
        # Retrieve the embedding vector for the word if it exists in the GloVe dictionary
        if word in embeddings_dict:
            word_embedding = embeddings_dict[word]
            embedding_vectors.append(word_embedding)
    
    # Check if we have collected any word embeddings
    if not embedding_vectors:
        # If no embeddings were found, return a zero vector with the same length as the GloVe vectors
        embedding_dimension = len(next(iter(embeddings_dict.values())))
        # Numpy provides a convenient way to create an array of zeros with a specified size
        return numpy.zeros(embedding_dimension, dtype='float32')
    
    # Calculate the mean across all collected embedding vectors
    mean_embedding_vector = numpy.mean(embedding_vectors, axis=0)
    return mean_embedding_vector

def print_answers(test_contexts, predictions):
    for instance_id, predicted_sense in zip(test_contexts.keys(), predictions):
        print(f'<answer instance="{instance_id}" senseid="{predicted_sense}"/>')
    
def vectorize_data(contexts, embeddings_dict):
    features = []
    for context in contexts.values():
        # Calculate the average embedding for each context and append it to the features list
        features.append(average_embedding(context, embeddings_dict))
    # Convert the list of features to a NumPy array for compatibility with scikit-learn
    return numpy.array(features)

# Specific functions for each model:
def predict_senseid_with_svm(train_contexts, corresponding_sense, test_contexts, glove_embeddings):
    train_features = vectorize_data(train_contexts, glove_embeddings)
    test_features = vectorize_data(test_contexts, glove_embeddings)

    # Initialize the Support Vector Classifier
    svm_classifier = SVC(kernel='linear')
    
    # Train the classifier with the training features and corresponding labels
    svm_classifier.fit(train_features, corresponding_sense)
    
    # Predict the senses for the test features
    predictions = svm_classifier.predict(test_features)
    return predictions

# Neural Network specific functions:
def build_neural_network_model(input_dim, num_classes):
    model = Sequential([
        keras.layers.Input(shape=(input_dim,)), # Input layer with the input dimension
        Dense(128, activation='relu'), # First hidden layer with 128 neurons and ReLU activation
        Dense(64, activation='relu'), # Second hidden layer with 64 neurons and ReLU activation
        Dense(num_classes, activation='softmax') # Output layer with the number of classes and softmax activation
    ])
    return model

def compile_neural_network_model(model):
    model.compile(
        optimizer='adam', # Adam optimizer for training
        loss='sparse_categorical_crossentropy', # Sparse categorical cross-entropy loss for multi-class classification
        metrics=['accuracy'] # Track accuracy during training
    )


def predict_senseid_with_neural_network(train_contexts, corresponding_sense, test_contexts, glove_embeddings):
    # Initialize and fit LabelEncoder
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(corresponding_sense)
    
    # Vectorize the training and test data
    x_train = vectorize_data(train_contexts, glove_embeddings)
    x_test = vectorize_data(test_contexts, glove_embeddings)

    # Get the number of classes (senses) in the training data (for the output layer of the NN
    num_classes = len(numpy.unique(y_train))
    
    # Build, compile, and train the neural network model
    model = build_neural_network_model(input_dim=x_train.shape[1], num_classes=num_classes)
    compile_neural_network_model(model)
    model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.1, verbose=0)

    # Get predictions for test data
    x_test_predictions = model.predict(x_test, verbose=0)
    x_test_predictions = numpy.argmax(x_test_predictions, axis=1)  # Convert probabilities to class labels
    predictions = label_encoder.inverse_transform(x_test_predictions)  # Use the same encoder instance to inverse transform
    return predictions

def main():
    if len(sys.argv) < 4:
        print("Usage: python3 wsd-embeddings.py line-train.txt line-test.txt <embedding file> [OPTIONS: SVM|NN] > my-line-answers.txt")
        sys.exit(1)

    train_file = sys.argv[1]
    test_file = sys.argv[2]
    embedding_file = sys.argv[3]
    model_choice = sys.argv[4] if len(sys.argv) == 5 else 'SVM'

    train_contexts, corresponding_sense = extract_contexts_and_senseid_vector(train_file)
    test_contexts = extract_context_from_test(test_file)

    glove_embeddings = load_glove_embeddings(embedding_file)

    if model_choice == 'NN':
        predictions = predict_senseid_with_neural_network(train_contexts, corresponding_sense, test_contexts, glove_embeddings)
    else:
        predictions = predict_senseid_with_svm(train_contexts, corresponding_sense, test_contexts, glove_embeddings)

    print_answers(test_contexts, predictions)

if __name__ == "__main__":
    main()
