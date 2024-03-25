from util.transform import *
from util.augmentation import googleTranslateKey, googleTranslate

def main():
    # Loading Crypto Vocabulary
    f = open("./data/cryptoVocab.txt", "r")
    crypto_vocabulary = f.read().split(',')
    crypto_vocabulary = [term.replace('"', '') for term in crypto_vocabulary]

    # Language ISO-639 code for Back-Translation using Google Translate
    langList1 = ["it", "fr", "sv", "da", "pt", "id", "pl", "hr", "bg", "fi"]
    langList2 = ["no", "ru", "es", "nl", "af", "de", "sk", "cs", "lv", "sq"]

    # Language Code for Back-Translation using Transformers
    langList3 = ['zh', 'es', 'ru', 'jap', 'de', 'fr', 'it', 'id']

    key = "./data/keyME.json"
    translate_client = googleTranslateKey(key)
    # text = googleTranslate(text, 'id', translate_client)

    df = pd.read_csv("./data/sentiment.csv", usecols=['text', 'sen'])

    
    sen = df['sen']
    augmentedGoogleTranslate = pd.DataFrame()
    
    for lang in (langList1 + langList2):
        text = dfTrain['text'].apply(translateBby, args=(x,))
        temp = pd.DataFrame(text)
        temp['sen'] = sen
        temp['lang'] = lang
        augmentedGoogleTranslate = augmentedGoogleTranslate.append([augmentedGoogleTranslate, temp])

    augmentedTransformers = pd.DataFrame()
    for lang in (langList3):
        text = dfTrain['text'].apply(translateBby, args=(x,))
        temp = pd.DataFrame(text)
        temp['sen'] = sen
        temp['lang'] = lang
        augmentedTransformers = augmentedTransformers.append([augmentedTransformers, temp])




    
    # df = cleanText(df)
    # print(df['sen'].value_counts())

    
    

if __name__ == "__main__":
    main()