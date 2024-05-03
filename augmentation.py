from util.preprocessing import *
from util.augmentGoogleTranslate import backTranslate as backTranslateGT
from util.augmentHelsinkiNLP import backTranslate as backTranslateHNLP
from util.augmentHelsinkiNLP import initializeModel

def augment(check):
    if check == False:
        DIR = "./data/"
        df = importData(DIR)
        df = cleanText(df)

        train, test, val = partitioning(df, 100)
        train.to_csv("./data/dfTrain.csv")
        test.to_csv("./data/dfTest.csv")
        val.to_csv("./data/dfVal.csv")

        # langCodeGT = ['it', 'fr', 'sv', 'da', 'pt',
        #               'id', 'pl', 'hr', 'bg', 'fi',
        #               'no', 'ru', 'es', 'nl', 'af',
        #               'de', 'sk', 'cs', 'lv', 'sq']

        # for lang in langCodeGT:
        #     temp = train['text'].apply(backTranslateGT, args=(lang, ))
        #     temp = pd.DataFrame(temp)
        #     temp['sen'] = train['sen']
        #     temp.to_csv(f'./augmented_data/dfTrain-{lang}GT.csv', index = False)

        # langCodeHNLP = ['zh', 'es', 'ru', 'jap', 
        #                 'de', 'fr', 'it', 'id']

        # for lang in langCodeHNLP:
        #     model1, tokenizer1, model2, tokenizer2 = initializeModel(lang)
        #     temp = train['text'].apply(backTranslateHNLP, args=(model1, tokenizer1, model2, tokenizer2,))
        #     temp = pd.DataFrame(temp)
        #     temp['sen'] = train['sen']
        #     temp.to_csv(f'./augmented_data/dfTrain-{lang}HNLP.csv', index = False)