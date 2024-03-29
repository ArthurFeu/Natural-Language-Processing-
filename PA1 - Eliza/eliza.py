# Programming Assignment 1
# Author: Arthur Mendonca Feu
# Date: February 5, 2024
# Class: CMSC-416-001 - INTRO TO NATURAL LANG PROCESS - Spring 2024

# This program is a simple chatbot inspired by ELIZA, one of the earliest examples of natural language processing (NLP) in computing. The chatbot simulates a conversation with a therapist by recognizing certain patterns in the user's text input and responding with questions or statements that encourage further dialogue. The goal is not to provide actual therapy but to demonstrate the basics of pattern matching and text manipulation in NLP. The chatbot recognizes a variety of input types, including greetings, statements of feelings, questions, and more, and responds in a way that mirrors human conversation in a very basic form.

# Usage Instructions:

# In the folder of the file, run the following command in the terminal: py.\eliza.py

# 1 - Start the program.
# 2 - The program will greet you and ask for your name. Respond with your name.
# 3 - Continue the conversation by typing statements or questions.
# 4 - The chatbot will respond based on the patterns it recognizes in your input.
# 5 - If you want to end the conversation, type a phrase like "bye" or "I have to go."

# Conversation example:

# Eliza: Hello! My name is Eliza. What's your name?
# You: Hi
# Eliza: Hi? That seems more like a greeting. Please, tell me your name.
# You: Sorry, my name is arthur
# Nice to meet you, Arthur! How can I help you today?
# Arthur: I'm sad
# Eliza: How does that make you feel?
# Arthur: Like I can't do anything
# Eliza: Can you elaborate on that?
# Arthur: Yes, I just miss my friends
# Eliza: Do you often think of your friends?
# Arthur: every sunday
# Eliza: Please tell me more.
# Arthur: I just want to see them
# Eliza: Why do you just want to see them?
# Arthur: because they're very nice to me and I miss being around them
# Eliza: What other reasons come to mind?
# Arthur: because they're also loyal
# Eliza: Is that the real reason?
# Arthur: yes
# Eliza: You seem quite sure.
# Arthur: uhum. Also, I feel lonely
# Eliza: Can you elaborate on that?
# Arthur: maybe it's just a sunday haha
# Eliza: Glad to see you're in good spirits! What's on your mind?
# Arthur: nothing more, it's just the same feelings on sundays
# Eliza: Well, are you sure you want to end the conversation, Arthur?
# Arthur: yup
# Eliza: Goodbye, Arthur! Have a nice day!

import re
import random
import string

# Dictionary of responses, where the key is a regular expression matching the user's input,
# and the value is a list of possible responses for that type of input.

responses = {
    "good morning": ["Good morning! Let's begin?"],  # Greeting patterns and responses
    "good afternoon": ["Good afternoon! Let's begin?"],
    "good evening": ["Good evening! Let's begin?"],
    # Patterns to match statements about feelings and responses that encourage the user to explore those feelings further
    "i feel (.*)": ["Why do you feel %1?", "Well, why do you feel %1?"],
    # Patterns to match statements about the user's identity or state of being
    "i am (.*)": ["Why are you %1?", "Why do you think you are %1?"],
    "i'm (.*)": ["Why are you %1?", "Why do you think you are %1?"],
    "i need (.*)": ["Why do you need %1?", "Why do you think you need %1?"],
    # Patterns for other statements beginning with "I" and their reflective responses
    "i (.*) you": ["Why do you %1 me?", "Why do you think you %1 me?"],
    "i (.*) myself": ["Why do you %1 yourself?", "Why do you think you %1 yourself?"],
    "i (.*)": ["Why do you %1?", "Why do you think you %1?"],
    # Patterns for questions posed by the user to Eliza
    "why don't you (.*)": ["Why should I %1?", "Do you think I should %1?"],
    "why can't i (.*)": ["Do you think you should be able to %1?", "Why can't you %1?"],
    "are you (.*)": ["Why are you interested in whether I am %1?", "Would you prefer it if I were not %1?"],
    # Generic patterns for 'what' and 'how' questions
    "what (.*)": ["Why do you ask?", "How would an answer to that help you?", "What do you think?"],
    "how (.*)": ["How do you suppose?", "Perhaps you can answer your own question.", "What is it you're really asking?"],
    "because (.*)": ["Is that the real reason?", "What other reasons come to mind?", "Does that reason apply to anything else?"],
    "(.*) sorry (.*)": ["There are many times when no apology is needed.", "What feelings do you have when you apologize?"],
    # Specific patterns for discussing family members
    "(.*) my mother(.*)": [
        "Tell me more about your mother.",
        "What was your relationship with your mother like?",
        "How do you feel about your mother?",
        "How does this relate to your feelings today?",
        "Good family relations are important. Tell me more about your mother."
    ],
    "(.*) my father(.*)": [
        "Tell me more about your father.",
        "How did your father make you feel?",
        "How do you feel about your father?",
        "Does your relationship with your father relate to your feelings today?",
        "Do you have trouble showing affection with your family?"
    ],
    "(.*) my child(.*)": [
        "Did you have close friends as a child?",
        "What is your favorite childhood memory?",
        "Do you remember any dreams or nightmares from childhood?",
        "Did the other children sometimes tease you?",
        "How do you think your childhood experiences relate to your feelings today?"
    ],
    # Pattern for possession or relation, and responses encouraging further discussion
    "(.*) my (.*)": ["Do you often think of your %2?"],
    "(.*) you (.*)": ["We should be discussing you, not me.", "Why do you say that about me?", "Why do you care whether I %2?"],
    "(.*) hate (.*)": ["Why do you hate %2?", "Why do you think you hate %2?"],
    "(.*) love (.*)": ["Why do you love %2?", "Why do you think you love %2?"],
    # Simple responses to yes or no inputs
    "yes": ["You seem quite sure.", "OK, but can you elaborate a bit?"],
    "no": ["Why not?", "You are being a bit negative.", "Are you saying 'No' just to be negative?"],
    # Patterns for emotional states
    "(.*) bad (.*)": [
        "It sounds like you're dealing with some negative feelings. Would you like to talk more about it?",
        "It's tough to deal with negative situations. How are you coping with these feelings?",
        "Facing difficulties can be challenging. What do you think could improve your situation?"
    ],
    "(.*) good (.*)": [
        "It's great to hear that you're feeling positive. What's contributing to your good feelings?",
        "Hearing about good experiences is wonderful. What's been happening that's good?",
        "Good feelings are always welcome. Can you share more about what's been going well?"
    ],
    # Patterns for understated expressions or subtle hints
    "(.*) low[-]?key (.*)": [
        "It seems like you're hinting at something subtly. Would you like to share more about it?",
        "When you say 'low-key', it sounds like there's more to the story. What are you thinking?",
        "Sometimes we say things are 'low-key' when we're hesitant to share. Feel free to open up more if you'd like."
    ],
    # Patterns for expressions indicating there's more to say
    "(.*) much more (.*)": [
        "It sounds like there's a lot on your mind. Feel free to share more.",
        "When you say 'much more', it seems like there's a significant amount to discuss. I'm here to listen.",
        "You're considering a lot right now. Let's delve deeper, what's going through your mind?"
    ],
    # Patterns for various emotional or physical states
    "(.*) tired (.*)": [
        "Feeling tired can be a sign of many things. What do you think is contributing to your tiredness?",
        "Tiredness can really affect us. Have you been able to rest or find a way to relax?",
        "When you're feeling tired, it's important to take care of yourself. What do you think might help you recharge?"
    ],
    "(.*) excited (.*)": [
        "It's wonderful to feel excited! What's bringing you this positive energy?",
        "Excitement is such an uplifting emotion. What are you looking forward to?",
        "Feeling excited is great! What's happening that's making you feel this way?"
    ],
    "(.*) anxious (.*)": [
        "Dealing with anxiety can be tough. Would you like to talk about what's been on your mind?",
        "Anxiety can be overwhelming. What do you feel is causing these feelings?",
        "Feeling anxious is something many of us experience. What steps do you think you could take to feel more at ease?"
    ],
    "(.*) bored (.*)": [
        "Boredom can be a sign we need something new. What are you interested in exploring?",
        "Feeling bored can sometimes lead to new discoveries. What do you usually enjoy doing?",
        "Boredom strikes us all at times. What's something new you'd like to try or learn about?"
    ],
    # Patterns for laughts
    "(.*)(haha|lol|hehe|lmao)(.*)": [
        "Glad to see you're in good spirits! What's on your mind?",
        "It's nice to share a laugh. Do you want to talk more about that?"
    ],
    # Pattern for question about Eliza's affirmations
    "(.*) you sure(.*)": [
        "Sorry about that, please tell me more."
    ],
    # Catch-all pattern for statements not caught by other patterns
    "(.*)": [
        "Please tell me more.",
        "Well, let's change focus a bit... Tell me something else.",
        "Can you elaborate on that?",
        "Why do you say that %1?",
        "I see.",
        "Very interesting.",
        "I see. And what does that tell you?",
        "How does that make you feel?",
        "How do you feel when you say that?"
    ],
}

# List of phrases that, when matched, will end the conversation.
conversation_endings = [
    # Various ways the user might indicate they want to end the conversation.
    "bye", "goodbye", "exit", "quit", "see you later", "later", "cya", "ttyl", "talk to you later", 
    "adios", "farewell", "so long", "peace out", "take care", "have a nice day", "have a good day", 
    "have a great day", "have a good one", "have a great one", "have a good night", "have a great night", 
    "have a good evening", "have a great evening", "have a good morning", "have a great morning", 
    "have a good afternoon", "have a great afternoon", "i don't want to talk", "i don't want to talk anymore", 
    "i don't want to talk to you", "stop talking", "stop talking to me", "stop", "shut up", "shut up eliza", 
    "shut up eliza!", "please stop", "please stop talking", "please stop talking to me", "please stop talking to me eliza", 
    "that's all", "that's it", "that's everything", "that's all for now", "that's it for now", 
    "that's everything for now", "that's all, eliza", "that's it, eliza", "that's everything, eliza", 
    "that's all for now, eliza", "that's it for now, eliza", "that's everything for now, eliza", 
    "that's all eliza", "that's it eliza", "that's everything eliza", "that's all for now eliza", 
    "that's it for now eliza", "that's everything for now eliza", "that's all, eliza!", "that's it, eliza!", 
    "that's everything, eliza!", "that's all for now, eliza!", "that's it for now, eliza!", 
    "that's everything for now, eliza!", "that's all eliza!", "that's it eliza!", "that's everything eliza!", 
    "that's all for now eliza!", "that's it for now eliza!", "that's everything for now eliza!", 
    "nothing", "you are not helping", "you can't help me", "you can't", "you cant", "you can not", "just stop", "i have to go",
    "stop the conversation", "stop now"
]

# List of regular expression patterns to recognize and extract the user's name from their input.
name_patterns = [
    # Patterns cover various ways users may introduce themselves or be addressed.
    "(.*) my name(?: is|\'s|s)? (.*)",
    "(.*) i\'m (.*)",
    "(.*) im (.*)",
    "(.*) i am (.*)",
    "(.*) call me (.*)",
    "(.*) it\'s (.*)",
    "(.*) this is (.*)",
    "(.*) hi eliza, my name(?: is|\'s|s)? (.*)",
    "(.*) (?:hello|hi|hey) eliza,? i\'m (.*)",
    "(.*) you can call me (.*)",
    "(.*) people call me (.*)",
    "(.*) they call me (.*)",
    "(.*) i go by (.*)",
    "(.*) my friends call me (.*)",
    "(.*) everyone calls me (.*)",
    "(.*) my name\'s? (.*)",
    "(.*) name\'s? (.*)",
    "(.*) you may call me (.*)",
    "(.*) i prefer to be called (.*)",
    "(.*) you should know me by (.*)",
    "(.*) i like to be called (.*)",
    "(.*) just call me (.*)",
    "(.*) most people know me as (.*)"
]

def reflect(input):
    # Mapping for converting first-person pronouns to second-person and vice versa
    pronouns = {
        "i am": "you are", "i was": "you were", "i": "you", "me": "you",
        "my": "your", "mine": "yours", "you are": "I am", "you were": "I was",
        "you": "me", "your": "my", "yours": "mine"
    }
    words = input.lower().split()
    # Reflect pronouns found in the input input
    for i, word in enumerate(words):
        if word in pronouns:
            words[i] = pronouns[word]
    return ' '.join(words)

def clean_input(input):
    # Remove punctuation from the input to improve matching
    return input.lower().translate(str.maketrans('', '', string.punctuation))


def get_response(user_input, users_name, last_response, response_list):
    # Choose a random response from the list.
    response = random.choice(response_list)
    
    # Ensure the response is not the same as the last one.
    # If it is, choose another response from the list, trying up to 10 times.
    attempt_count = 0
    while response == last_response and len(response_list) > 1 and attempt_count < 10:
        response = random.choice(response_list)
        attempt_count += 1

    # If placeholders (%1, %2, ...) are present in the response, replace them.
    placeholders = re.findall(r'%\d+', response)
    for placeholder in placeholders:
        group_index = int(placeholder.strip('%')) - 1 
        if group_index < len(user_input.groups()):
            response = response.replace(placeholder, reflect(user_input.group(group_index + 1)))

    # Replace the placeholder %name% with the user's name.
    response = response.replace("%name%", users_name)

    return response

   
    
def match_response(user_input, users_name, last_response):
    user_input = clean_input(user_input)

    for pattern, response_list in responses.items():
        match = re.match(pattern, user_input)
        if match:
            return get_response(match, users_name, last_response, response_list), last_response

    return (user_input, users_name), last_response

def extract_name():
    name = clean_input(input("You: "))
    for pattern in name_patterns:
            match = re.match(pattern, clean_input(name))
            if match:
                name = match.group(2).split()[0]  # Return only the first word as the name
                if name.lower() in ['i', 'my', 'hi', 'hello', 'hey']:  # Exclude common misinterpretations
                    break
                return name.split()[0].capitalize()
    
    while not name.strip() or name in ['i', 'my', 'hi', 'hello', 'hey']:
        if not name.strip():
            print("I didn't catch your name. What's your name?")
        else:
            print(f"Eliza: {name.capitalize()}? That seems more like a greeting. Please, tell me your name.")
        name = input("You: ")

        for pattern in name_patterns:
            match = re.match(pattern, clean_input(name))
            if match:
                name = match.group(2).split()[0]  # Return only the first word as the name
                if name.lower() in ['i', 'my', 'hi', 'hello', 'hey']:  # Exclude common misinterpretations
                    break
                return name.capitalize()
    
    # If the loop exited due to valid input, but it's not caught by patterns
    return name.split()[0].capitalize()

def main():
    # Start the dialogue and ask for the user's name
    print("Eliza: Hello! My name is Eliza. What's your name?")
    name = extract_name()
    print(f"Eliza: Nice to meet you, {name}! How can I help you today?")
    
    last_response = None
    while True:
        user_input = input(f"{name}: ").strip()
        
        # Check if user_input is empty
        if not user_input:
            print("Eliza: You didn't type anything. Is everything okay? You can share everything with me.")
            continue
        
        # Check for conversation ending phrases with flexibility for text before the phrase
        if any(re.search(f"(.*){ending}", user_input.lower()) for ending in conversation_endings):
            print(f"Eliza: Well, do you want to finish the conversation, {name}?")
            confirmation = input(f"{name}: ").strip().lower()
            if confirmation in ["yes", "y", "yeah", "yep", "yup"]:
                print(f"Eliza: Goodbye, {name}! Have a nice day!")
                break
            else:
                print(f"Eliza: Okay, {name}. I'm here if you want to talk more.")
                continue
        
        response, last_response = match_response(user_input, name, last_response)
        print(f"Eliza: {response}")

if __name__ == "__main__":
    main()
