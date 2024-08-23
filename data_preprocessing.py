# Text Data Preprocessing Lib
import nltk
nltk.download('punkt')
nltk.download('wordnet')

# to stem words
from nltk.stem import PorterStemmer

# create an instance of class PorterStemmer
stemmer = PorterStemmer()

# importing json lib
import json
import pickle
import numpy as np

words=[] #list of unique roots words in the data
classes = [] #list of unique tags in the data
pattern_word_tags_list = [] #list of the pair of (['words', 'of', 'the', 'sentence'], 'tags')

# words to be ignored while creating Dataset
ignore_words = ['?', '!',',','.', "'s", "'m"]

# open the JSON file, load data from it.
train_data_file = open('intents.json')
data = json.load(train_data_file)
train_data_file.close()

# creating function to stem words
def get_stem_words(words, ignore_words):
    stem_words = []
    for word in words:

        # write stemming algorithm:
        '''
        Check if word is not a part of stop word:
        1) lowercase it 
        2) stem it
        3) append it to stem_words list
        4) return the list
        ''' 
        # Add code here #        

        
    

        if word.lower() not in ignore_words:
            # Stem the word and add to the list
            stem_word = PorterStemmer.stem(word.lower())
            stem_words.append(stem_word)
    return stem_words
   
    


'''
List of sorted stem words for our dataset : 

['all', 'ani', 'anyon', 'are', 'awesom', 'be', 'best', 'bluetooth', 'bye', 'camera', 'can', 'chat', 
'cool', 'could', 'digit', 'do', 'for', 'game', 'goodby', 'have', 'headphon', 'hello', 'help', 'hey', 
'hi', 'hola', 'how', 'is', 'later', 'latest', 'me', 'most', 'next', 'nice', 'phone', 'pleas', 'popular', 
'product', 'provid', 'see', 'sell', 'show', 'smartphon', 'tell', 'thank', 'that', 'the', 'there', 
'till', 'time', 'to', 'trend', 'video', 'what', 'which', 'you', 'your']

'''


# creating a function to make corpus
def create_bot_corpus(words, classes, pattern_word_tags_list, ignore_words):

    for intent in data['intents']:

        # Add all patterns and tags to a list
        for pattern in intent['patterns']:  

            # tokenize the pattern          
            pattern_words = nltk.word_tokenize(pattern)

            # add the tokenized words to the words list        
                 
   
            words = nltk.word_tokenize(words,ignore_words)
    
            stem_words = get_stem_words(words, ignore_words)
         
            # add the 'tokenized word list' along with the 'tag' to pattern_word_tags_list
            
            
        # Add all tags to the classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

            
    stem_words = get_stem_words(words, ignore_words) 

    # Remove duplicate words from stem_words
    stem_words = sorted(set(stem_words))
    # sort the stem_words list and classes list

    
    # print stem_words
    print('stem_words list : ' , stem_words)

    return stem_words, classes, pattern_word_tags_list


# Training Dataset: 
# Input Text----> as Bag of Words 
# Tags-----------> as Label

def bag_of_words_encoding(stem_words, pattern_word_tags_list):
    
    bag = []
    for word_tags in pattern_word_tags_list:
        # example: word_tags = (['hi', 'there'], 'greetings']

        pattern_words = word_tags[0] # ['Hi' , 'There]
        bag_of_words = []

        # stemming pattern words before creating Bag of words
        stemmed_pattern_word = get_stem_words(pattern_words, ignore_words)

        # Input data encoding 
        '''
        Write BOW algo :
        1) take a word from stem_words list
        2) check if that word is in stemmed_pattern_word
        3) append 1 in BOW, otherwise append 0
        '''
        # Define the Bag of Words algorithm
def create_bag_of_words(patterns, stem_words, ignore_words):
    bag_of_words = []
    
    for pattern in patterns:
        # Preprocess the pattern (tokenize, stem, and ignore specific words)
        pattern_words = get_stem_words(nltk.word_tokenize(pattern), ignore_words)
        
        # Initialize the BOW list for the current pattern
        bow = []
        
        for word in stem_words:
            # Check if the word is in the stemmed pattern words
            if word in pattern_words:
                bow.append(1)
            else:
                bow.append(0)
        
        # Append the BOW for the current pattern to the main list
        bag_of_words.append(bow)
    
    return bag_of_words
       
    
    return np.array(bag)

def class_label_encoding(classes, pattern_word_tags_list):
    
    labels = []

    for word_tags in pattern_word_tags_list:

        # Start with list of 0s 
        labels_encoding = list([0]*len(classes))  

        # example: word_tags = (['hi', 'there'], 'greetings']

        tag = word_tags[1]   # 'greetings'

        tag_index = classes.index(tag)

        # Labels Encoding
        labels_encoding[tag_index] = 1

        labels.append(labels_encoding)
        
    return np.array(labels)

def preprocess_train_data():
  
    stem_words, tag_classes, word_tags_list = create_bot_corpus(words, classes, pattern_word_tags_list, ignore_words)
    
    # Convert Stem words and Classes to Python pickel file format
    
with open("stem_words.pkl", "wb") as file:
    pickle.dump(stem_words, file)

# 2. Save the classes list to a file using pickle
with open("classes.pkl", "wb") as file:
    pickle.dump(classes, file)

# Later on, you can load these lists back as follows:
# Load the stem_words list
with open("stem_words.pkl", "rb") as file:
    loaded_stem_words = pickle.load(file)

# Load the classes list
with open("classes.pkl", "rb") as file:
    loaded_classes = pickle.load(file)

# Print to verify loaded data
print("Loaded Stem Words:", loaded_stem_words)
print("Loaded Classes:", loaded_classes)

  

# after completing the code, remove comment from print statements
 

