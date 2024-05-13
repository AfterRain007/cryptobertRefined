from util.check import checkAugment
from util.finetune import *
from util.preprocessing import cleanText

def main():
    # To check if data augmented file is there, if not then do augmentation.
    checkAugment()

    modelList = ["cardiffnlp/twitter-roberta-base-sentiment-latest",
                 "finiteautomata/bertweet-base-sentiment-analysis", 
                 "ElKulako/cryptobert",                             
                 "cardiffnlp/twitter-xlm-roberta-base-sentiment"    
                ]

    dataList = ["dfGT20",
                "dfGT10",
                "dfHNLP"]

    dfVal = pd.read_csv("./data/dfVal.csv")
    dfTest = pd.read_csv("./data/dfTest.csv")

    for i, dfTrain in enumerate(importAugmentedData()):
        for modelName in modelList:
            dfTrain, dfVal, tokenizer = initialize(dfTrain, dfVal, modelName)
            start(modelName, dfTrain, dfVal)
            saveResult(dataList[i], modelName)

if __name__ == "__main__":
    main()