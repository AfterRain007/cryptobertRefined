import pandas as pd
import re
df = pd.read_csv("./data/sentiment.csv", usecols=['text', 'sen'])
print(df['sen'].value_counts())

df = df[~((df['text'].str.contains('bitcoin going', flags=re.IGNORECASE)) |
    (df['text'].str.contains('bitcoin okay bro', flags=re.IGNORECASE)) |
    (df['text'].str.contains('explode', flags=re.IGNORECASE)) |
    (df['text'].str.contains('know going', flags=re.IGNORECASE)) |
    (df['text'].str.contains('bitcoin k loading', flags=re.IGNORECASE)) |
    (df['text'].str.contains('full audit usdc', flags=re.IGNORECASE)) |
    (df['text'].str.contains('record didnt need', flags=re.IGNORECASE)) |
    (df['text'].str.contains('squawkcnbc jerrymoran bitcoin', flags=re.IGNORECASE)))]

df.to_csv("./data/sentiment2.csv", index = False)