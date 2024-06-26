from util.preprocessing import *
from util.augmentGoogleTranslate import backTranslate as backTranslateGT
from util.augmentHelsinkiNLP import backTranslate as backTranslateHNLP
from util.augmentHelsinkiNLP import initializeModel

def augment():
    fileName = "Sentiment.csv"
    df = importData(fileName)
    dfClean = cleanText(df)
    
    train, test, val = partitioning(dfClean, 100)
    
    # We don't want to clean the training dataset yet because
    # we'll do data augmentation using back-translation
    train = df.loc[df.index.intersection(train.index)]
    train.to_csv("./data/dfTrain.csv")
    test.to_csv("./data/dfTest.csv")
    val.to_csv("./data/dfVal.csv")

    # Data augmentation using Transformers (Helsinki NLP)
    langCodeHNLP = ['zh', 'es', 'ru', 'jap', 
                    'de', 'fr', 'it', 'id']

    for lang in langCodeHNLP:
        model1, tokenizer1, model2, tokenizer2 = initializeModel(lang)
        temp = train['text'].apply(backTranslateHNLP, args=(model1, tokenizer1, model2, tokenizer2,))
        temp = pd.DataFrame(temp)
        temp['sen'] = train['sen']
        temp.to_csv(f'./augmented_data/dfTrain-{lang}HNLP.csv', index = False)

    # # Data augmentation using Google Translate
    # langCodeGT = ['it', 'fr', 'sv', 'da', 'pt',
    #               'id', 'pl', 'hr', 'bg', 'fi',
    #               'no', 'ru', 'es', 'nl', 'af',
    #               'de', 'sk', 'cs', 'lv', 'sq']

    # # ISO-639 code for Google Translate Language
    # # Italian, French, Swedish, Danish, Portuguese, 
    # # Indonesian, Polish, Croatioan, Bulgarian, Finnish, 
    # # Norwegian, Russian, Spanish, Dutch, Afrikaans,
    # # German, Slovak, Czech, Latvian, and Albanian
    
    # for lang in langCodeGT:
    #     temp = train['text'].apply(backTranslateGT, args=(lang, ))
    #     temp = pd.DataFrame(temp)
    #     temp['sen'] = train['sen']
    #     temp.to_csv(f'./augmented_data/dfTrain-{lang}GT.csv', index = False)