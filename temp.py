import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from html import unescape
import pandas as pd

class TweetData:
    def __init__(self, data):
        self.data = data  # Pandas dataframe

    url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    ftp_pattern = re.compile(r'ftp[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    
    # Set of punctuation characters
    punctuation_set = set(string.punctuation)
    lemmatizer = WordNetLemmatizer()

    def cleanText2(self, url_pattern = url_pattern, ftp_pattern = ftp_pattern, punctuation_set = punctuation_set, lemmatizer = lemmatizer):
        import string
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
        # Remove URLs 
        self.data['text'] = self.data['text'].str.replace(url_pattern, '', regex=True)
        self.data['text'] = self.data['text'].str.replace(ftp_pattern, '', regex=True)
        self.data['text'] = unescape(self.data['text'])

        # Lowercase
        self.data['text'] = self.data['text'].str.lower()

        # Remove punctuation
        self.data['text'] = self.data['text'].str.replace('[{}]'.format(string.punctuation), "", regex=True)

        # Remove stopwords
        stop_words = set(stopwords.words("english"))
        self.data['text'] = self.data['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

        # Remove special characters and numbers (using str.replace)
        self.data['text'] = self.data['text'].str.replace(r"[^a-zA-Z\s]", "", regex=True)

        # Lemmatization (assuming lemmatizer is an instance)
        self.data['text'] = self.data['text'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))

        # Remove unnecessary spaces
        self.data['text'] = self.data['text'].str.strip()

        # No need to return anything as modifications are done in-place

# Load your CSV data
tweet_data = TweetData(pd.read_csv("./data/sentiment.csv"))

# print(tweet_data.data.head())
# Clean text data
tweet_data.cleanText2()

print(tweet_data.data.head())

# # Augment data
# tweet_data.augment_data()

# # Access cleaned and augmented data
# cleaned_data = tweet_data.get_data()[0]  # Access first dataframe (original)
# augmented_data_1 = tweet_data.get_data()[1]  # Access second dataframe (1st augmentation)

# # Use the cleaned and augmented data for your grid search and model training