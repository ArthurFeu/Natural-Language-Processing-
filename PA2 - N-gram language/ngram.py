"""
N-gram Language Model Generator
Author: Arthur Mendonça Feu
Date: February 2024
Email: mendoncafea@vcu.edu

This script generates random sentences based on an N-gram language model, 
suitable for text prediction and natural language processing tasks. It reads text from input files,
builds an N-gram model, and generates sentences with specified N-gram lengths.

Example Usage:
    python ngram.py 2 5 myTextFile1.txt myTextFile2.txt
    This will read myTextFile1.txt and myTextFile2.txt, build a 2-gram model, and generate 5 random sentences.

Algorithm:
1. Read and preprocess input text files to tokenize words into n-grams and add start (<st>) and end (<end>) tokens in the place of punctuation.
2. Build an N-gram model that maps N-gram sequences to their subsequent word frequencies.
3. Convert frequencies to probabilities to facilitate random word selection.
4. Generate sentences by starting with an N-gram of <st> tokens and randomly selecting the next word based on probabilities.
5. The sentence is finished when the next word is an <end> token.
"""

import random
import sys
import os
import re

def read_and_prepare_data(file_paths, n):
    all_tokens = []
    
    # make sure the file paths are valid and the files are .txt
    for file_path in file_paths:
        if not file_path.endswith('.txt'):
            print(f"Skipping file '{file_path}'. Only .txt files are supported.")
            continue

        if not os.path.exists(file_path):
            print(f"File '{file_path}' does not exist.")
            continue

        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read().lower()
            # Normalize whitespace and remove newline characters
            text = re.sub(r'\s+', ' ', text).strip()
            # Directly insert <end> after each sentence-ending punctuation
            text = re.sub(r'([.!?])', r'\1 <end>', text)
            # Split the text into sentences by <end> tags
            sentences = text.split('<end>')
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                tokens = re.findall(r'\b[\w]+|\b[\w]\'[\w\b]|[’]|[\.,!?]', sentence)
                # Prepend n <st> tags at the beginning of each token list for a sentence
                tokens_with_tags = ['<st>'] * n + tokens
                all_tokens.extend(tokens_with_tags)
                # Explicitly add <end> after each sentence's tokens
                all_tokens.append('<end>')

    # Remove every occurrence of '\n' (1 or more) from the array
    all_tokens = [item for item in all_tokens if not re.match(r'^\n+$', item)]

    return all_tokens



def build_ngram_model(tokens, n):
    ngram_model = {}

    # Loop through the tokens list, stopping n tokens short of the end to avoid index errors.
    for i in range(len(tokens) - n):  
        ngram_key = ' '.join(tokens[i:i+n])  # Create the N-gram key by joining n consecutive tokens with spaces.
        next_token = tokens[i+n]  # Identify the token following the current N-gram.

        if ngram_key not in ngram_model:  # Check if the N-gram key is not already in the model.
            ngram_model[ngram_key] = {next_token: 1}  # If not, add the N-gram key to the model with the next token as its first entry, occurrence count set to 1.
        else:  # If the N-gram key is already in the model,
            # Use setdefault to ensure the next token exists in the sub-dictionary with a default count of 0 if not already present.
            ngram_model[ngram_key].setdefault(next_token, 0)
            ngram_model[ngram_key][next_token] += 1  # Increment the occurrence count of the next token by 1.

    return ngram_model


def calculate_continuation_probabilities(ngram_model):
    probabilities_model = {}

    for ngram_key, continuations in ngram_model.items():
        total = sum(continuations.values())  # Calculate the total count of all continuations for the current N-gram.
        # Compute the probability of each continuation word by dividing its count by the total count of continuations.
        probabilities = {word: count / total for word, count in continuations.items()}
        probabilities_model[ngram_key] = probabilities  # Assign the calculated probabilities to the corresponding N-gram key in the probabilities model.

    return probabilities_model

def format_final_sentence(sentence):
     # Process the sentence for final output.
    final_sentence = ' '.join(sentence[:-1])  # Exclude <end> from the final sentence construction.
    final_sentence = re.sub(r'\s+([.,!?])', r'\1', final_sentence)  # Remove space before punctuation.
    final_sentence = re.sub(r"\b\s*’\s*", "’", final_sentence).capitalize() # Remove space after and before apostrophes and capitalize the first letter.
    return final_sentence

def generate_sentence(n, probabilities_model):
    # Initialize the current N-gram key with start tokens (<st>) repeated n times.
    current_key = ' '.join(['<st>'] * n)
    sentence = []

    while True:
        # Retrieve the probability distribution for the current N-gram key from the model.
        probabilities = probabilities_model.get(current_key, None)
                
        if probabilities is None or not probabilities:
            break  # Break if no probabilities are found to avoid infinite loop.
        
        # Unzip the probabilities dictionary into separate lists of words and their probabilities.
        words, probs = zip(*probabilities.items())
        
        # Randomly choose the next word based on the probability distribution.
        next_word = random.choices(words, weights=probs)[0]

        # Append the next word to the sentence.
        sentence.append(next_word)
        
        if next_word == '<end>':  # Check if the selected word is the end token and finish the sentence.
            break

        # Update the current N-gram key for the next iteration.
        current_key_parts = (current_key.split(' ')[1-n:] if n > 1 else []) + [next_word]
        current_key = ' '.join(current_key_parts)

   
    final_sentence = format_final_sentence(sentence)

    return final_sentence

def generate_unigram_sentence(tokens):
    sentence = []
    # Filter out <st> and <end> tokens since they are not part of the actual text
    filtered_tokens = [token for token in tokens if token not in ['<st>', '<end>']]
    
    # Filter out punctuation and tags for the initial word selection, so the sentence doesn't start with punctuation
    initial_filtered_tokens = [token for token in filtered_tokens if token not in ['!', '?', '.', ',', '’']]
    
    first_word = random.choice(initial_filtered_tokens)
    sentence.append(first_word)

    while True:
        next_word = random.choice(filtered_tokens)

        if next_word == '<end>':
            break
        if next_word in ['!', '?', '.']:
            sentence.append(next_word)
            break
        else:
            sentence.append(next_word)
    
    # Correct spacing for punctuation
    final_sentence = ' '.join(sentence)
    final_sentence = re.sub(r'\s+([.,!?])', r'\1', final_sentence)
    return final_sentence.capitalize()


def main():
    if len(sys.argv) < 4:
        print("Usage: python ngram.py n m input-file/s")
        return

    n = int(sys.argv[1]) - 1 # Subtract 1 to convert n-gram length to calculate the number of start tokens.
    m = int(sys.argv[2])
        
    input_files = sys.argv[3:]
    tokens = read_and_prepare_data(input_files, n)
    ngram_model = build_ngram_model(tokens, n)
    probabilities_model = calculate_continuation_probabilities(ngram_model)
        
    # Prints just for testing purposes:
    # print("\n\nTOKENS:" + str(tokens))
    # for key, value in ngram_model.items():
    #     print(f"KEY: {key}")
    #     print(f"VALUE: {value}")
    # print("\n\nPROBABILITIES_MODEL:" + str(probabilities_model))
        
    print("This program generates random sentences based on an Ngram model.\n")
    print("Author: Arthur Mendonça Feu\n")
    print(f"Command line settings: {' '.join(sys.argv[:3])}\n")

    if n > 0:
        for _ in range(m):
            print(generate_sentence(n, probabilities_model))
    else:
        for _ in range(m):
            print(generate_unigram_sentence(tokens))
        

if __name__ == "__main__":
    main()
