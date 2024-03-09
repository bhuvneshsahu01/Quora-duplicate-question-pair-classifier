from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from fuzzywuzzy import fuzz
from bs4 import BeautifulSoup
import re
import pickle as pkl
import fuzzywuzzy 

cv=pkl.load(open('cv.pkl','rb'))

# Initialize the PorterStemmer
stemmer = PorterStemmer()

def preprocess(text):
    text=text.lower().strip()
    # https://stackoverflow.com/a/19794953
    contractions = { 
        "ain't": "am not",
        "aren't": "are not",
        "can't": "can not",
        "can't've": "can not have",
        "'cause": "because",
        "could've": "could have",
        "couldn't": "could not",
        "couldn't've": "could not have",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hadn't've": "had not have",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'd've": "he would have",
        "he'll": "he will",
        "he'll've": "he will have",
        "he's": "he is",
        "how'd": "how did",
        "how'd'y": "how do you",
        "how'll": "how will",
        "how's": "how is",
        "i'd": "i would",
        "i'd've": "i would have",
        "i'll": "i will",
        "i'll've": "i will have",
        "i'm": "i am",
        "i've": "i have",
        "isn't": "is not",
        "it'd": "it would",
        "it'd've": "it would have",
        "it'll": "it will",
        "it'll've": "it will have",
        "it's": "it is",
        "let's": "let us",
        "ma'am": "madam",
        "mayn't": "may not",
        "might've": "might have",
        "mightn't": "might not",
        "mightn't've": "might not have",
        "must've": "must have",
        "mustn't": "must not",
        "mustn't've": "must not have",
        "needn't": "need not",
        "needn't've": "need not have",
        "o'clock": "of the clock",
        "oughtn't": "ought not",
        "oughtn't've": "ought not have",
        "shan't": "shall not",
        "sha'n't": "shall not",
        "shan't've": "shall not have",
        "she'd": "she would",
        "she'd've": "she would have",
        "she'll": "she will",
        "she'll've": "she will have",
        "she's": "she is",
        "should've": "should have",
        "shouldn't": "should not",
        "shouldn't've": "should not have",
        "so've": "so have",
        "so's": "so as",
        "that'd": "that would",
        "that'd've": "that would have",
        "that's": "that is",
        "there'd": "there would",
        "there'd've": "there would have",
        "there's": "there is",
        "they'd": "they would",
        "they'd've": "they would have",
        "they'll": "they will",
        "they'll've": "they will have",
        "they're": "they are",
        "they've": "they have",
        "to've": "to have",
        "wasn't": "was not",
        "we'd": "we would",
        "we'd've": "we would have",
        "we'll": "we will",
        "we'll've": "we will have",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what'll": "what will",
        "what'll've": "what will have",
        "what're": "what are",
        "what's": "what is",
        "what've": "what have",
        "when's": "when is",
        "when've": "when have",
        "where'd": "where did",
        "where's": "where is",
        "where've": "where have",
        "who'll": "who will",
        "who'll've": "who will have",
        "who's": "who is",
        "who've": "who have",
        "why's": "why is",
        "why've": "why have",
        "will've": "will have",
        "won't": "will not",
        "won't've": "will not have",
        "would've": "would have",
        "wouldn't": "would not",
        "wouldn't've": "would not have",
        "y'all": "you all",
        "y'all'd": "you all would",
        "y'all'd've": "you all would have",
        "y'all're": "you all are",
        "y'all've": "you all have",
        "you'd": "you would",
        "you'd've": "you would have",
        "you'll": "you will",
        "you'll've": "you will have",
        "you're": "you are",
        "you've": "you have"
        }

    q_decontracted = []

    for word in text.split():
        if word in contractions:
            word = contractions[word]

        q_decontracted.append(stemmer.stem(word))
   
    text = ' '.join(q_decontracted)
    text = BeautifulSoup(text)
    text = text.get_text()
    
    # Remove punctuations
    pattern = re.compile('\W')
    text = re.sub(pattern, ' ', text).strip()
    return text

def query_preproces(q1,q2):
    
    q1=preprocess(q1)
    q2=preprocess(q2)
    q1_tokens = q1.split()
    q2_tokens = q2.split()
    
    features=[]

    features.append(len(q1))
    features.append(len(q2))

    features.append(len(q1_tokens))
    features.append(len(q2_tokens))
    
    features.append(len( set(q1_tokens) & set(q2_tokens)))
                    
    features.append(round(features[-1]/(features[0]+features[1]),3))
    
    SAFE_DIV = 0.0001 

    STOP_WORDS = stopwords.words("english")
    
    
    

    # Get the non-stopwords in Questions
    q1_words = set([word for word in q1_tokens if word not in STOP_WORDS])
    q2_words = set([word for word in q2_tokens if word not in STOP_WORDS])
    
    #Get the stopwords in Questions
    q1_stops = set([word for word in q1_tokens if word in STOP_WORDS])
    q2_stops = set([word for word in q2_tokens if word in STOP_WORDS])
    
    # Get the common non-stopwords from Question pair
    common_word_count = len(q1_words.intersection(q2_words))
    
    # Get the common stopwords from Question pair
    common_stop_count = len(q1_stops.intersection(q2_stops))
    
    # Get the common Tokens from Question pair
    common_token_count = len(set(q1_tokens).intersection(set(q2_tokens)))
    
    
    features.append(common_word_count / (min(len(q1_words), len(q2_words)) + SAFE_DIV))
    features.append(common_word_count / (max(len(q1_words), len(q2_words)) + SAFE_DIV))
    features.append(common_stop_count / (min(len(q1_stops), len(q2_stops)) + SAFE_DIV))
    features.append(common_stop_count / (max(len(q1_stops), len(q2_stops)) + SAFE_DIV))
    features.append(common_token_count / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV))
    features.append(common_token_count / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV))
    
    # Last word of both question is same or not
    features.append(int(q1_tokens[-1] == q2_tokens[-1]))
    
    # First word of both question is same or not
    features.append(int(q1_tokens[0] == q2_tokens[0]))

    
    # Absolute length features
    features.append(abs(len(q1_tokens) - len(q2_tokens)))
    
    #Average Token Length of both Questions
    features.append((len(q1_tokens) + len(q2_tokens))/2)
    
    # Fuzzy Features


    
    # fuzz_ratio
    features.append(fuzz.QRatio(q1, q2))

    # fuzz_partial_ratio
    features.append(fuzz.partial_ratio(q1, q2))

    # token_sort_ratio
    features.append(fuzz.token_sort_ratio(q1, q2))

    # token_set_ratio
    features.append(fuzz.token_set_ratio(q1, q2))
    q1_bow=cv.transform([q1])
    q2_bow=cv.transform([q2])
    return np.hstack([q1_bow.toarray(),q2_bow.toarray(),np.array(features).reshape(1,20)])
    
    

    
    
