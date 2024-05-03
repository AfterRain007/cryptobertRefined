import os

langCodeGT = ['it', 'fr', 'sv', 'da', 'pt',
              'id', 'pl', 'hr', 'bg', 'fi',
              'no', 'ru', 'es', 'nl', 'af',
              'de', 'sk', 'cs', 'lv', 'sq']

langCodeHNLP = ['zh', 'es', 'ru', 'jap', 
                'de', 'fr', 'it', 'id']

def check_files():
    for lang in langCodeGT:
        if not os.path.exists(f"./augmented_data/dfTrain-{lang}GT.csv"):
            return False
        
    for lang in langCodeHNLP:
        if not os.path.exists(f"./augmented_data/dfTrain-{lang}HNLP.csv"):
            return False
    
    return True

print(check_files())