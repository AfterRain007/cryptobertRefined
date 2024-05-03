from util.check import checkAugment
from util.finetune import *
from util.preprocessing import cleanText

def main():
    # To check if data augmented file is there
    # If not then do augmentation.
    checkAugment()

    modelList = ["cardiffnlp/twitter-roberta-base-sentiment-latest",
                 "finiteautomata/bertweet-base-sentiment-analysis", 
                 "ElKulako/cryptobert",                             
                 "cardiffnlp/twitter-xlm-roberta-base-sentiment"    
                ]

    dfVal = pd.read_csv("./data/dfVal.csv")
    dfTest = pd.read_csv("./data/dfTest.csv")

    for dfTrain in importAugmentedData():
        for modelName in modelList:
            dfTrain, dfVal, tokenizer = initialize(dfTrain, dfVal, modelName)
            start(modelName, dfTrain, dfVal)

if __name__ == "__main__":
    main()