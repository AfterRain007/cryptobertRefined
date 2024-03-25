#Import all the module
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from html import unescape
import pandas as pd

def train_test_split(df):
    

def cleanString(text, url_pattern, ftp_pattern, punctuation_set):
    t = re.sub(url_pattern, ' ', text)  # remove urls if any
    t = re.sub(ftp_pattern, ' ', t)  # remove urls if any
    t = unescape(t)  # html entities fix

    # Convert text to lowercase
    text = t.lower()

    # Remove punctuation
    text = ''.join(char for char in text if char not in punctuation_set)

    # Tokenization
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    tokens = [token for token in tokens if token not in stop_words]

    # Remove special characters and numbers
    tokens = [re.sub(r"[^a-zA-Z]", "", token) for token in tokens]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Remove unnecessary spaces
    tokens = [token.strip() for token in tokens if token.strip()]

    # Join tokens back into a single string
    cleaned_text = " ".join(tokens)

    return cleaned_text

def cleanText(df):
    # Compile the regular expressions
    url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    ftp_pattern = re.compile(r'ftp[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    
    # Set of punctuation characters
    punctuation_set = set(string.punctuation)

    df['text'] = df['text'].apply(cleanString, args = (url_pattern, ftp_pattern, punctuation_set, ))

    df.loc[:,   'WC'] = df['text'].apply(lambda x: len(x.split()))
    df = df[df['WC']>=4]
    df.drop('WC', axis = 1, inplace = True)
    df.reset_index(inplace = True, drop = True)
    df.drop_duplicates(inplace = True)
    return df

def cleanRepeat(text, repetition_threshold=3):
    # Find all words in the text
    words = re.findall(r'\b\w+\b', text.lower())

    # Create a regular expression pattern for detecting repeating words
    pattern = r'\b(\w+)' + r'(\s+\1){%d,}\b' % (repetition_threshold - 1)

    # Search for the pattern in the text
    match = re.search(pattern, ' '.join(words))

    # Return True if a match is found, indicating repeating words
    return match is not None

def sentimentScore(review, maxLength = 128, device = "cpu"):
    # Tokenize the review outside the loop
    tokens = tokenizer(review, return_tensors='pt', max_length = maxLength, truncation = True).input_ids.to(device)

    # Pass the tokens directly to the model for batch processing
    result = model(tokens)

    # Convert the tensor to a numpy array and extract the predicted sentiment
    sentiment = int(torch.argmax(result.logits)) - 1

    return sentiment