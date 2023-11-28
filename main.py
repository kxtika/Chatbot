import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model

import config

# Load WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

# Load preprocessed data
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model/model.h5')


def clean_up_sentence(sentence):
    """
        Tokenizes and lemmatizes the input sentence.

        Parameters:
        - sentence (str): The input sentence to be cleaned.

        Returns:
        - List[str]: A list of lemmatized words from the input sentence.
        """
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence):
    """
        Converts the input sentence into a bag of words representation.

        Parameters:
        - sentence (str): The input sentence.

        Returns:
        - np.array: A numpy array representing the bag of words.
        """
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence):
    """
        Predicts the intent class and probability of the input sentence.

        Parameters:
        - sentence (str): The input sentence.

        Returns:
        - List[dict]: A list containing dictionaries with 'intent' and 'probability'.
        """
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list


def get_response(intents_list, intents_json):
    """
        Retrieves a response based on the predicted intent.

        Parameters:
        - intents_list (List[dict]): List of intents and their probabilities.
        - intents_json (dict): JSON containing the available intents and responses.

        Returns:
        - str: The selected response based on the predicted intent.
        """
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result


print(config.STARTING_MESSAGE)

while True:
    message = input(config.USER)
    ints = predict_class(message)
    res = get_response(ints, intents)

    if float(ints[0]['probability']) < 0.7:
        print(config.BOT, config.UNKNOWN_INPUT)
    else:
        res = get_response(ints, intents)
        print(config.BOT, res)
