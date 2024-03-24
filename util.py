#Import all the module
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from html import unescape

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Compile the regular expressions
url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
ftp_pattern = re.compile(r'ftp[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

# Set of punctuation characters
punctuation_set = set(string.punctuation)

def cleanString(text):
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
    df = df[(~df['text'].str.contains('Compared to the last tweet, the price has', flags=re.IGNORECASE)) &
            (~df['text'].str.contains('In the last 24 hours the price has', flags=re.IGNORECASE)) &
            (~df['text'].str.contains('transferred from Unknown Wallet to Unknown Wallet', flags=re.IGNORECASE)) &
            (~df['text'].str.contains('Market Cap. Swap', flags=re.IGNORECASE)) &
            (~df['text'].str.contains('Bitcoin BTC Current Price:', flags=re.IGNORECASE)) &
            (~df['text'].str.contains('Current #Bitcoin Price is', flags=re.IGNORECASE)) &
            (~df['text'].str.contains('Bitcoin Whale Alert:', flags=re.IGNORECASE)) &
            (~df['text'].str.contains('tx with a value of', flags=re.IGNORECASE)) &
            (~df['text'].str.contains('will be transfered from', flags=re.IGNORECASE)) &
            (~df['text'].str.contains('Someone just transfered', flags=re.IGNORECASE)) &
            (~df['text'].str.contains('is a super underrated bitcoiner I’ve been following', flags=re.IGNORECASE)) &
            (~df['text'].str.contains('has been transfered to an', flags=re.IGNORECASE)) &
            (~df['text'].str.contains('transferred from unknown wallet to', flags=re.IGNORECASE)) &
            (~df['text'].str.contains('Someone just transfered', flags=re.IGNORECASE)) &
            (~df['text'].str.contains('is a super underrated bitcoiner I’ve been following her tweets and tips', flags=re.IGNORECASE)) &
            (~df['text'].str.contains('Market Cap. Swap', flags=re.IGNORECASE)) &
            (~df['text'].str.contains('Bitcoin BTC Current Price:', flags=re.IGNORECASE)) &
            (~df['text'].str.contains('Current #Bitcoin Price is', flags=re.IGNORECASE)) &
            (~df['text'].str.contains('Bitcoin Whale Alert:', flags=re.IGNORECASE)) &
            (~df['text'].str.contains('will be transfered from', flags=re.IGNORECASE)) &
            (~df['text'].str.contains('#bitcoin SHORTED', flags=re.IGNORECASE)) &
            (~df['text'].str.contains('tx with a value of', flags=re.IGNORECASE)) &
            (~df['text'].str.contains('#bitcoin LONGED', flags=re.IGNORECASE)) &
            (~df['text'].str.contains('move from unknown wallet to', flags=re.IGNORECASE)) &
            (~df['text'].str.contains('BTC - short alert', flags=re.IGNORECASE)) &
            (~df['text'].str.contains('1 BTC Price: Bitstamp', flags=re.IGNORECASE)) &
            (~df['text'].str.contains('Trending Crypto Alert!', flags=re.IGNORECASE)) &
            (~df['text'].str.contains('#Bitcoin mempool Tx summary in the last 60 seconds', flags=re.IGNORECASE)) &
            (~df['text'].str.contains('BEARWHALE! just SHORTED', flags=re.IGNORECASE)) &
            (~df['text'].str.contains('Buyer alert:', flags=re.IGNORECASE)) &
            (~df['text'].str.contains('#Bitcoin Price:', flags=re.IGNORECASE)) &
            (~df['text'].str.contains('Based on #coindesk BPI #bitcoin', flags=re.IGNORECASE)) &
            (~df['text'].str.contains('BTCUSDT LONGED', flags=re.IGNORECASE)) &
            (~df['text'].str.contains('Everywhere should follow @', flags=re.IGNORECASE)) &
            (~df['text'].str.contains('BULLWHALE! just LONGED', flags=re.IGNORECASE)) &
            (~df['text'].str.contains('Long Position Taken On $', flags=re.IGNORECASE)) &
            (~df['text'].str.contains("A new block was found on the #Bitcoin network. We're at block height", flags=re.IGNORECASE)) &
            (~df['text'].str.contains('current #bitcoin price is', flags=re.IGNORECASE)) &
            (~df['text'].str.contains('today and watch your life turn around, start earning', flags=re.IGNORECASE)) &
            (~df['text'].str.contains('Symbol: BTCUSD (Binance)', flags=re.IGNORECASE, regex=False)) &
            (~df['text'].str.contains('Current  #Bitcoin Price:', flags=re.IGNORECASE)) &
            (~df['text'].str.contains('transferred from #Coinbase to unknown wallet', flags=re.IGNORECASE)) &
            (~df['text'].str.contains('SCAN RESULTS - 15m - #BTC PAIR', flags=re.IGNORECASE)) &
            (~df['text'].str.contains('$BTC Latest Block Info: Block', flags=re.IGNORECASE)) &
            (~df['text'].str.contains('Scan results - #Gateio - 15m', flags=re.IGNORECASE)) &
            (~df['text'].str.contains('BTCUSD LONGED @', flags=re.IGNORECASE)) &
            (~df['text'].str.contains('Follow for recent Bitcoin price updates.', flags=re.IGNORECASE)) &
            (~df['text'].str.contains('1 BTC Price: Bitstamp', flags=re.IGNORECASE)) &
            (~df['text'].str.contains('The current price of bitcoin is $', flags=re.IGNORECASE)) &
            (~df['text'].str.contains('Symbol:|Signal:|Price:|Volume:', flags=re.IGNORECASE)) &
            (~df['text'].str.contains('Deal Close:|Entry:|Entry Price:', flags=re.IGNORECASE)) &
            (~df['text'].str.contains('Scan results - | - 15m', flags=re.IGNORECASE))]
    
    df['WC'] = df['text'].apply(lambda x: len(x.split()))
    df
    return df