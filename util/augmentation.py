from google.cloud import translate_v2
import os
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
# import torch

def googleTranslateKey(key):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = f"{key}"
    translate_client = translate_v2.Client()
    return translate_client

def googleTranslate(text, lan, translate_client):
    temp = translate_client.translate(text,lan)['translatedText']

    return translate_client.translate(temp,'en')['translatedText']

# def initializeTransformers(lang, crypto_vocabulary, device="cuda" if torch.cuda.is_available() else "cpu"):

#     tokenizer1  = AutoTokenizer.from_pretrained(f"Helsinki-NLP/opus-mt-en-{lang}")
#     tokenizer1.additional_special_tokens = crypto_vocabulary
#     model1      = AutoModelForSeq2SeqLM.from_pretrained(f"Helsinki-NLP/opus-mt-en-{lang}")
#     model1      = model1.to(device)

#     tokenizer2  = AutoTokenizer.from_pretrained(f"Helsinki-NLP/opus-mt-{lang}-en")
#     tokenizer2.additional_special_tokens = crypto_vocabulary
#     model2      = AutoModelForSeq2SeqLM.from_pretrained(f"Helsinki-NLP/opus-mt-{lang}-en")
#     model2      = model2.to(device)

#     torch.cuda.empty_cache()
#     return model1, model2, tokenizer1, tokenizer2

# def transformersTranslate(text, numBeams = 5, device="cuda" if torch.cuda.is_available() else "cpu"):
#     input_ids = tokenizer1(text, return_tensors="pt").input_ids.to(device)
#     if len(input_ids[0]) < 512:
#         outputs = model1.generate(input_ids=input_ids, num_beams=numBeams)
#         text1 = tokenizer1.batch_decode(outputs, skip_special_tokens=True)[0]

#         input_ids = tokenizer2(text1, return_tensors="pt").input_ids.to(device)

#         if len(input_ids[0]) < 512:
#             outputs = model2.generate(input_ids=input_ids, num_beams=numBeams)
#             text2 = tokenizer2.batch_decode(outputs, skip_special_tokens=True)[0]
#             return text2
#     else:
#         return "Token Exceed 512"
