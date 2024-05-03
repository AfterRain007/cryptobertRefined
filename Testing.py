from util.preprocessing import *

fileName = "Sentiment.csv"
df = importData(fileName)
dfClean = cleanText(df)

train, test, val = partitioning(dfClean, 100)

print(len())