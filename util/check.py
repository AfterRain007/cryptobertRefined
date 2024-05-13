import os

def checkFile(choice):
    if choice == 0:
        langCodeHNLP = ['zh', 'es', 'ru', 'jap', 
                        'de', 'fr', 'it', 'id']
            
        for lang in langCodeHNLP:
            if not os.path.exists(f"./augmented_data/dfTrain-{lang}HNLP.csv"):
                return False

    elif choice == 1:
        langCodeGT = ['it', 'fr', 'sv', 'da', 'pt',
                    'id', 'pl', 'hr', 'bg', 'fi',
                    'no', 'ru', 'es', 'nl', 'af',
                    'de', 'sk', 'cs', 'lv', 'sq']

        for lang in langCodeGT:
            if not os.path.exists(f"./augmented_data/dfTrain-{lang}GT.csv"):
                return False

    elif choice == 2:
        langCodeHNLP = ['zh', 'es', 'ru', 'jap', 
                        'de', 'fr', 'it', 'id']
            
        for lang in langCodeHNLP:
            if not os.path.exists(f"./augmented_data/dfTrain-{lang}HNLP.csv"):
                return False

        langCodeGT = ['it', 'fr', 'sv', 'da', 'pt',
                    'id', 'pl', 'hr', 'bg', 'fi',
                    'no', 'ru', 'es', 'nl', 'af',
                    'de', 'sk', 'cs', 'lv', 'sq']

        for lang in langCodeGT:
            if not os.path.exists(f"./augmented_data/dfTrain-{lang}GT.csv"):
                return False
    
    return True

def checkAugment(choice):
    if checkFile(choice) == False:
        from augmentation import augment
        augment()