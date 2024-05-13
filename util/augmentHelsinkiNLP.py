# !pip install transformers -q
# !pip install sentencepiece -q
# !pip install sacremoses -q
# !pip install --upgrade tensorflow -q
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM #, pipeline
# import shutil
import torch
# import numpy as np
# import pandas as pd
import time as time

device = "cuda:0" if torch.cuda.is_available() else "cpu"
f = open("./data/cryptoVocab.txt", "r")
crypto_vocabulary = f.read().split(',')
crypto_vocabulary = [term.replace('"', '') for term in crypto_vocabulary]

def initializeModel(lan):
    tokenizer1  = AutoTokenizer.from_pretrained(f"Helsinki-NLP/opus-mt-en-{lan}")
    tokenizer1.additional_special_tokens = crypto_vocabulary
    model1      = AutoModelForSeq2SeqLM.from_pretrained(f"Helsinki-NLP/opus-mt-en-{lan}")
    model1      = model1.to(device)
    tokenizer2  = AutoTokenizer.from_pretrained(f"Helsinki-NLP/opus-mt-{lan}-en")
    tokenizer2.additional_special_tokens = crypto_vocabulary
    model2      = AutoModelForSeq2SeqLM.from_pretrained(f"Helsinki-NLP/opus-mt-{lan}-en")
    model2      = model2.to(device)
    torch.cuda.empty_cache()
    return model1, tokenizer1, model2, tokenizer2

def backTranslate(text, model1, tokenizer1, model2, tokenizer2):
    input_ids = tokenizer1(text, return_tensors="pt").input_ids.to(device)
    # print(len(input_ids))
    if len(input_ids[0]) > 512:
        return "Runtime 512 Error here baby!"
    else:
        outputs = model1.generate(input_ids=input_ids, num_beams=5)
        text1 = tokenizer1.batch_decode(outputs, skip_special_tokens=True)[0]

        input_ids = tokenizer2(text1, return_tensors="pt").input_ids.to(device)

    if len(input_ids[0]) < 512:
        outputs = model2.generate(input_ids=input_ids, num_beams=5)
        text2 = tokenizer2.batch_decode(outputs, skip_special_tokens=True)[0]
        return text2
    else:
        return "Runtime 512 Error here baby!" 