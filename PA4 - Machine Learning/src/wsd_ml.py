"""
Programming Assignment 4
Author: Arthur Mendonca Feu
Date: March 27, 2024
Class: CMSC-416-001 - INTRO TO NATURAL LANG PROCESS - Spring 2024

This script implements three different machine learning models from scikit to perform Word Sense Disambiguation (WSD) by identifying the intended meaning of a word with multiple meanings based on its context. The classifier uses a Bag of Words feature representation to capture the context in which a target word appears. In the this approach, the context of the word is represented as a set of words (features) that occur disregarding the order in which these words appear. Each feature in the Bag of Words model corresponds to a word from the training data. For this program the target word is "line" and the possible senses are "phone" and "product".

Algorithm Steps:
	1. Extract the context AND the sense ID of the target word from the training data.
	2. Extract the context of the target word from the test data.

	3. Vectorize the context data from the training data.
		Utilizes CountVectorizer from scikit-learn to convert the text documents (contexts) into a matrix of token counts. This transformation is necessary for the machine learning models to process the text data. Each method is better explained in the main function.
  
	4. Train the classifier using the training data.
		Based on the user's choice initializes one of three models: Logistic Regression, Support Vector Machine with a linear kernel, or Multinomial Naive Bayes. If the user does not specify a model, the script uses the Naive Bayes model by default.
  
	5. Make predictions using the classifier.
	6. Print the predictions to the standard output in the pseudo-XML format.

You should run the script from the command line as follows:
py wsd-ml.py line-train.txt line-test.txt [OPTIONAL: ml-model] > my-line-answers.txt

	Avaliable models 		| Tag to write in the Command Line:
                        	|
	Naive Bayes (default)	| nb
	Logistic Regression		| logreg
	Support Vector Machine 	| svm

_____________________________________________________________________________________________________________
NAIVE BAYES MODEL

A Multinomial Naive Bayes classifier is used for classification with discrete features such as word counts in text classification. This model calculates the probability of each class and the conditional probability of each class given an input sample, making a prediction based on the class with the highest probability.

Results with the prediction based on the Naive Bayes model:

Confusion Matrix:

Actual | Predicted      phone   product
phone                   69      3
product                 4       50

Correct: 119
Incorrect: 7
Total: 126
Accuracy: 94.44%
_____________________________________________________________________________________________________________
LOGISTIC REGRESSION MODEL

This model estimates probabilities using a logistic function, which can map any real-valued number into a value between 0 and 1. For binary classification, if the predicted probability is greater than 0.5, the model predicts the class label as 1; otherwise, it predicts 0.

Results with the prediction based on the Logistic Regression model:

Confusion Matrix:

Actual | Predicted      phone   product
phone                   68      4
product                 4       50

Correct: 118
Incorrect: 8
Total: 126
Accuracy: 93.65%
_____________________________________________________________________________________________________________
SUPPORT VECTOR MACHINE MODEL

SVM with a linear kernel is utilized to classify words based on their contexts for word sense disambiguation. It finds the optimal dividing straight line that separates different senses of a word with the greatest margin, which is the distance to the nearest data points.

Results with the prediction based on the Support Vector Machine model:

Confusion Matrix:

Actual | Predicted      phone   product
phone                   68      4
product                 5       49

Correct: 117
Incorrect: 9
Total: 126
Accuracy: 92.86%
_____________________________________________________________________________________________________________

Results with the prediction based on the most frequent sense:

Actual | Predicted      phone   product
phone                   0       72
product                 0       54

Correct: 54
Incorrect: 72
Total: 126
Accuracy: 42.86%
_____________________________________________________________________________________________________________

My Decision List Classifier (wsd.py) results:

    Actual | Predicted      phone   product
    phone                   67      5
    product                 10      44

    Correct: 111
    Incorrect: 15
    Total: 126
    Accuracy: 88.10%
    
"""

import re
import sys
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

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

    cleaned_context = re.sub(r'<head>.*?</head>', '', context, flags=re.DOTALL)
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

def predict_and_print_answers(test_contexts, model, vectorizer):
    # loop through the sentences in the test set
    for instance_id, context in test_contexts.items():
        test_context = vectorizer.transform([context])
		
		# Make a prediction based on the model
        predicted_sense = model.predict(test_context)
		
        print(f'<answer instance="{instance_id}" senseid="{predicted_sense[0]}"/>')

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 wsd_ml.py line-train.txt line-test.txt [OPTIONAL: ml-model] > my-line-answers.txt")
        sys.exit(1)

    train_file = sys.argv[1]
    test_file = sys.argv[2]
    model_choice = sys.argv[3] if len(sys.argv) > 3 else 'nb'
    
    train_contexts, corresponding_sense = extract_contexts_and_senseid_vector(train_file)
    test_contexts = extract_context_from_test(test_file) 

    # Convert the contexts to a list
    training_data = list(train_contexts.values())
    
    # CountVectorizer converts a collection of text documents into a matrix of token counts: the Document-Term Matrix (DTM)
    # Each row of the matrix corresponds to a document (in this case, a context surrounding a target word), and each column represents a unique word from the entire training dataset. The value at a given row and column in the matrix represents the frequency of that word in the corresponding document.
    
    vectorizer = CountVectorizer()
    
    # Fit: This step learns the vocabulary (each unique word) of the entire training dataset
    # It determines the features (the columns in the Document-Term Matrix) that the model will learn to associate with different sense IDs (meanings) of the target word.
    # Transform: converts the training data into a Document-Term Matrix using the learned vocabulary. Each row in this matrix corresponds to a document from the training data, and each column represents a word from the vocabulary. 
    # The value in each cell of the matrix indicates how many times the word (column) appears in the document (row).
    # Now the machine learning models can learn patterns in the data.
    
    training_data_vectorized = vectorizer.fit_transform(training_data)

    # Initialize and train one of the models
    if model_choice == 'logreg':
        model = LogisticRegression(max_iter=1000)
    elif model_choice == 'svm':
        model = SVC(kernel='linear')
    else:
        model = MultinomialNB()
    
    # model.fit trains the chosen model using the vectorized contexts as the features and the corresponding senses as the targets.
    # Model learning from the training data how the frequency of words in the context relates to the sense of the target word.
    model.fit(training_data_vectorized, corresponding_sense)
    predict_and_print_answers(test_contexts, model, vectorizer)
    
    
if __name__ == "__main__":
    main()