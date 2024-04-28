"""
Programming Assignment 4
Author: Arthur Mendonca Feu
Date: March 25, 2024
Class: CMSC-416-001 - INTRO TO NATURAL LANG PROCESS - Spring 2024

This script compares the predicted answers with the key answers to evaluate the performance of a word sense disambiguation script (wsd-ml.py). It calculates accuracy, creates a confusion matrix, and prints the results.

You should run the script from the command line as follows:
python3 scorer.py my-line-answers.txt line-key.txt

Algorithm Steps:

    1. Extract data from files containing predicted answers and key answers.
        - Function: extract_data(file_path)

    2. Compare the predicted answers with the key answers.
        - Function: compare_answers(my_answers, key_answers)

    3. Create a confusion matrix based on the comparison.
        - Function: create_confusion_matrix(my_answers, key_answers)
    
"""

import sys

def extract_data(file_path):
    data = {}
    # Try different encodings to read the file (I had some issues with the encoding of the files in the test cases)
    encodings = ['utf-8', 'utf-16', 'utf-8-sig']
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                for line in file:
                    if 'instance' in line and 'senseid' in line:
                        parts = line.strip().split('"')
                        if len(parts) >= 4:
                            instance = parts[1]
                            senseid = parts[3]
                            data[instance] = senseid
                return data 
        except UnicodeDecodeError:
            continue 
    print(f"Could not read file {file_path} with tried encodings: {encodings}")
    return data



def compare_answers(my_answers, key_answers):
    correct = 0
    incorrect = 0
    for instance, my_senseid in my_answers.items():
        key_senseid = key_answers.get(instance, None)
        if key_senseid is not None:
            if my_senseid == key_senseid:
                correct += 1
            else:
                incorrect += 1
    return correct, incorrect

def create_confusion_matrix(my_answers, key_answers):
    # Identify all unique senses from the answer keys
    senses = sorted(set(key_answers.values()))
    
    # Create a dictionary to hold the confusion matrix
    confusion_matrix = {actual: {predicted: 0 for predicted in senses} for actual in senses}
    
    # Populate the confusion matrix
    for instance, my_senseid in my_answers.items():
        key_senseid = key_answers.get(instance, None)
        if key_senseid is not None:
            confusion_matrix[key_senseid][my_senseid] += 1
    
    return confusion_matrix

def print_confusion_matrix(confusion_matrix):
    senses = sorted(confusion_matrix.keys())
    print('Confusion Matrix:')
    print('\nActual \\ Predicted', end='\t')
    print('\t'.join(senses))
    
    for actual_sense, predictions in confusion_matrix.items():
        print(actual_sense, end='\t\t\t')
        for predicted_sense in senses:
            print(confusion_matrix[actual_sense][predicted_sense], end='\t')
        print()

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 scorer.py my-line-answers.txt line-key.txt")
        sys.exit(1)

    my_answers_file = sys.argv[1]
    key_answers_file = sys.argv[2]

    my_answers = extract_data(my_answers_file)
    key_answers = extract_data(key_answers_file)

    if not my_answers:
        print("No answers extracted from my-line-answers.txt. Please check the file format.")
        sys.exit(2)

    correct, incorrect = compare_answers(my_answers, key_answers)
    confusion_matrix = create_confusion_matrix(my_answers, key_answers)
    print_confusion_matrix(confusion_matrix)

    total = correct + incorrect
    if total > 0:
        accuracy = (correct / total) * 100
        print(f"\nCorrect: {correct}")
        print(f"Incorrect: {incorrect}")
        print(f"Total: {total}")
        print(f"Accuracy: {accuracy:.2f}%")
    else:
        print("No valid comparisons made. Please check the formats of both answer files.")

if __name__ == "__main__":
    main()
