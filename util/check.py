import os

def checkFile():
    langCodeGT = ['it', 'fr', 'sv', 'da', 'pt',
                  'id', 'pl', 'hr', 'bg', 'fi',
                  'no', 'ru', 'es', 'nl', 'af',
                  'de', 'sk', 'cs', 'lv', 'sq']

    langCodeHNLP = ['zh', 'es', 'ru', 'jap', 
                    'de', 'fr', 'it', 'id']
    
    for lang in langCodeGT:
        if not os.path.exists(f"./augmented_data/dfTrain-{lang}GT.csv"):
            return False
        
    for lang in langCodeHNLP:
        if not os.path.exists(f"./augmented_data/dfTrain-{lang}HNLP.csv"):
            return False
    
    return True

def checkAugment():
    if checkFile == False:
        from augmentation import augment
        augment()