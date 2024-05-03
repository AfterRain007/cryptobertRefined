import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from html import unescape

# Uncomment if you haven't download
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

def importData(fileName):
    df = pd.read_csv(f"./data/{fileName}", usecols=['text', 'sen'])

    # # Data from SurgeAI, I already emailed them to publish the dataset in here but they haven't reply to my email yet
    # SurgeAI = pd.read_csv(DIR+"Crypto Sentiment Dataset.csv", usecols=['Comment Text', 'Sentiment'])
    # sentiment_mapping = {'Negative': -1, 'Neutral': 0, 'Positive': 1}
    # SurgeAI['Sentiment'] = SurgeAI['Sentiment'].map(sentiment_mapping)
    # SurgeAI = SurgeAI.rename(columns={"Comment Text":"text", 'Sentiment':'sen'})
    # df = pd.concat([df, SurgeAI])
    # df.reset_index(inplace = True, drop = True)

    return df

def clean_text(text, url_pattern, ftp_pattern, punctuation_set):
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
    # Compile the regex and punctuation characters
    url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    ftp_pattern = re.compile(r'ftp[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    punctuation_set = set(string.punctuation)

    # Cleaning the text data
    df['text'] = df['text'].astype(str)
    df['text'] = df['text'].apply(clean_text, args = (url_pattern, ftp_pattern, punctuation_set, ))

    # Deleting text with lower than 4 word because it is deemed to have low context
    # And deleting text with 3 repeating words because it is deemed to be spam.
    df['WC'] = df['text'].apply(lambda x: len(x.split()))
    df['spam'] = df['text'].apply(hasRepeatingWord)
    df = df[(df['spam'] == False) & ~((df['WC'] < 4) & ((df['sen'] == 1) | (df['sen'] == 0)))]

    #Further cleaning (I goes through it manually)
    df = df[~((df['text'].str.contains('bitcoin going', flags=re.IGNORECASE)) &
        ~(df['text'].str.contains('bitcoin okay bro', flags=re.IGNORECASE)) &
        ~(df['text'].str.contains('explode', flags=re.IGNORECASE)) &
        ~(df['text'].str.contains('know going', flags=re.IGNORECASE)) &
        ~(df['text'].str.contains('bitcoin k loading', flags=re.IGNORECASE)) &
        ~(df['text'].str.contains('full audit usdc', flags=re.IGNORECASE)) &
        ~(df['text'].str.contains('record didnt need', flags=re.IGNORECASE)) &
        ~(df['text'].str.contains('squawkcnbc jerrymoran bitcoin', flags=re.IGNORECASE)) &
        ((df['sen'] != -1) | (df['WC'] > 4)))]

    return df[['text', 'sen']]

def hasRepeatingWord(text, repetition_threshold=3):
    # Find all words in the text
    words = re.findall(r'\b\w+\b', text.lower())

    # Create a regular expression pattern for detecting repeating words
    pattern = r'\b(\w+)' + r'(\s+\1){%d,}\b' % (repetition_threshold - 1)

    # Search for the pattern in the text
    match = re.search(pattern, ' '.join(words))

    # Return True if a match is found, indicating repeating words
    return match is not None

def partitioning(df, sample_size = 100):
    dfM1 = df[df['sen'] == -1].sample(sample_size)
    df0 = df[df['sen'] == 0].sample(sample_size)
    df1 = df[df['sen'] == 1].sample(sample_size)
    test = pd.concat([dfM1, df0, df1])
    df = df.drop(test.index)

    dfM1 = df[df['sen'] == -1].sample(sample_size)
    df0 = df[df['sen'] == 0].sample(sample_size)
    df1 = df[df['sen'] == 1].sample(sample_size)
    val = pd.concat([dfM1, df0, df1])
    df = df.drop(val.index)

    test.reset_index(inplace = True, drop = True)
    val.reset_index(inplace = True, drop = True)

    return df, test, val