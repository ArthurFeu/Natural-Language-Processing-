"""
Programming Assignment 3
Author: Arthur Mendonca Feu
Date: March 11, 2024
Class: CMSC-416-001 - INTRO TO NATURAL LANG PROCESS - Spring 2024

This script implements a Decision List classifier to perform Word Sense Disambiguation (WSD) by identifying the intended meaning of a word with multiple meanings based on its context. The classifier uses a Bag of Words feature representation to capture the context in which a target word appears. In the this approach, the context of the word is represented as a set of words (features) that occur disregarding the order in which these words appear. Each feature in the Bag of Words model corresponds to a word from the training data that is not part of a predefined list of stopwords, ensuring that only meaningful context words are considered as potential indicators of the word's sense. The Decision List classifier then ranks these features based on their log-likelihood ratio, a statistical measure that quantifies each feature's discriminative power in distinguishing between the different senses of the target word. In this case, the script is used to disambiguate the word "line" in the context of a sentence for the senses of "phone" and "product".

You should run the script from the command line as follows:
py wsd.py line-train.txt line-test.txt my-model.txt > my-line-answers.txt

Algorithm Steps:

    1. Read the training data file and count the occurrences of each word in the context of each sense and overall.
        - Function: process_file_and_count_words_with_sense(file_path, stopwords)
        
    2. Calculate the log-likelihood ratio for each word to determine its discriminative power for each sense.
    3. Rank the words by their log-likelihood ratio and select the most discriminative words.
        - Function: rank_features_by_log_likelihood(word_counts, sense_word_counts)

        Note: The final determiner of the most discriminative words is the score, which is the product of the log-likelihood ratio and the count of the word in the sense. In this way we give more weight to more frequent words and the accuracy is improved.

    4. Read the test data file and extract the context for each instance.
        - Function: extract_context_from_test(file_path)

    5. For each instance, predict the sense of the word based on the context using the most discriminative words (printed on the model).
        - Function: predict_sense(context, ranked_features)
        
My results:

    Actual | Predicted      phone   product
    phone                   67      5
    product                 10      44

    Correct: 111
    Incorrect: 15
    Total: 126
    Accuracy: 88.10%
    
Results with the prediction based on the most frequent sense:

    Actual | Predicted      phone   product
    phone                   0       72
    product                 0       54

    Correct: 54
    Incorrect: 72
    Total: 126
    Accuracy: 42.86%    
    
"""

import re
import sys
import math
from collections import defaultdict

def process_file_and_count_words_with_sense(file_path, stopwords):
    # Initialize dictionaries to store word counts and sense-specific word counts
    sense_word_counts = defaultdict(lambda: defaultdict(int))
    word_counts = defaultdict(int)
    
    # Read the contents of the file
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read()
    
    # Extract instances containing sense ids and contexts using regex
    instances = re.findall(r'<instance id=".*?">\n<answer instance=".*?" senseid="(.*?)"/>\n<context>(.*?)</context>', data, re.DOTALL)
    
    # Process each instance
    for senseid, context in instances:
        # Clean the context: remove HTML tags, apostrophes, digits, and non-alphanumeric characters
        cleaned_context = re.sub(r'<[^>]*>', ' ', context)
        cleaned_context = re.sub(r"'", ' ', cleaned_context) 
        cleaned_context = re.sub(r'\d+', ' ', cleaned_context)
        cleaned_context = re.sub(r'[^\w\s]', ' ', cleaned_context) 
        cleaned_context = cleaned_context.lower()
        
        # Split the cleaned context into words
        words = cleaned_context.split()
                
        # Iterate through each word in the context
        for word in words:
            if word not in stopwords:
                # Increment the count of the word in the sense-specific word counts dictionary
                sense_word_counts[senseid][word] += 1
                
                # Increment the count of the word in the overall word counts dictionary
                word_counts[word] += 1
                                
    # Convert defaultdicts to regular dictionaries and return them
    return dict(word_counts), {sense: dict(words) for sense, words in sense_word_counts.items()}


def extract_context_from_test(file_path):
    # Compile regex to match instance and context
    pattern = re.compile(r'<instance id="([^"]+)">.*?<context>(.*?)</context>', re.DOTALL)
    
    # Dictionary to hold instance ID as key and context as value
    context_dict = {}
    
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        matches = pattern.findall(content)
        
        for instance_id, context in matches:
            # Clean unnecessary tags 
            cleaned_context = re.sub(r'<head>.*?</head>', '', context, flags=re.DOTALL)
            cleaned_context = re.sub(r'<(/?s|/?p|@)>', '', cleaned_context).strip()
            context_dict[instance_id] = cleaned_context
    
    return context_dict

def rank_features_by_log_likelihood(word_counts, sense_word_counts):
    scores_and_senses = {}
    epsilon = 1e-10  # Small value to avoid division by zero

    # Get all features from both senses
    all_features_phone = set(sense_word_counts['phone'].keys())
    all_features_product = set(sense_word_counts['product'].keys())
    all_features_combined = all_features_phone | all_features_product

    # Remove features with less than 3 characters
    all_features = set()
    for feature in all_features_combined:
        if len(feature) >= 3:
            all_features.add(feature)

    for feature in all_features:
        count_sense1 = sense_word_counts['phone'].get(feature, 0)
        count_sense2 = sense_word_counts['product'].get(feature, 0)
        count_feature = word_counts.get(feature, 0)

        # Calculate probabilities for each sense given the feature
        prob_sense1_given_feature = count_sense1 / count_feature if count_sense1 > 0 else 0
        prob_sense2_given_feature = count_sense2 / count_feature if count_sense2 > 0 else 0

        # the feature predicts the sense for which it gives a higher probability
        if prob_sense1_given_feature > prob_sense2_given_feature:
            predicted_sense = 'phone'
            predict_sense_count = count_sense1
        else:
            predicted_sense = 'product'
            predict_sense_count = count_sense2
        
        # |log10((P(Sense1|Feature)) / (P(Sense2|Feature)))|
        log_likelihood = abs(math.log((prob_sense1_given_feature + epsilon) / (prob_sense2_given_feature + epsilon), 10))
        
        # Multiply by the count of the feature to give more weight to more frequent features
        # Without this, the accuracy is 72.22% and with this, the accuracy is 88.10%
        
        score = log_likelihood * predict_sense_count
        scores_and_senses[feature] = (score, log_likelihood, predicted_sense)

    # Rank features by score and alphabetically in case of ties
    # I sorted alfabetically in case of ties to ensure to have the same output every time
    # Without this, the output can vary from 86.51% to 89.68%
    ranked_features_and_senses = sorted(scores_and_senses.items(), key=lambda item: (-item[1][0], item[0]))

    
    # Remove words with log_likelihood = 0
    ranked_features_and_senses = [feature for feature in ranked_features_and_senses if feature[1][1] > 0]

    return ranked_features_and_senses

def predict_sense(context, ranked_features):
    # Split context into separeted words
    words_in_context = set(re.findall(r'\w+', context.lower()))
    
    sense_counts = {}
    for feature_data in ranked_features:
        feature_info = feature_data[1]
        sense = feature_info[2]
        
        if sense in sense_counts:
            sense_counts[sense] += 1
        else:
            sense_counts[sense] = 1
            
    most_common_sense = max(sense_counts, key=sense_counts.get)
        
    # Iterate through ranked features and check if any feature exists in the context
    for feature, (score, log, sense) in ranked_features:
        if feature in words_in_context:
            # If feature is found in context, return the associated sense
            return sense

    # If none of the features matched, return a default sense -> should be the most frequent sense
    return most_common_sense

def write_answers_to_file(test_contexts, ranked_features):
    for instance_id, context in test_contexts.items():
        predicted_sense = predict_sense(context, ranked_features)
        print("<answer instance=\"" + instance_id + "\" senseid=\"" + predicted_sense + "\"/>")
        
def write_model_to_file(ranked_features, model_file):
    with open(model_file, 'w', encoding='utf-8') as file:
        for feature, (score, log_likelihood, sense) in ranked_features:
            file.write(f"Feature: {feature}, Score: {score}, Log-Likelihood: {log_likelihood}, Predicted Sense: {sense}\n")


def main():
    if len(sys.argv) != 4:
        print("Usage: python3 wsd.py line-train.txt line-test.txt my-model.txt > my-line-answers.txt")
        sys.exit(1)

    train_file = sys.argv[1]
    test_file = sys.argv[2]
    model_file = sys.argv[3]
    
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
    
    word_counts, sense_word_counts = process_file_and_count_words_with_sense(train_file, stopwords)
    ranked_features = rank_features_by_log_likelihood(word_counts, sense_word_counts)
    test_contexts = extract_context_from_test(test_file)
    
    write_answers_to_file(test_contexts, ranked_features)
    write_model_to_file(ranked_features, model_file)

if __name__ == "__main__":
    main()
