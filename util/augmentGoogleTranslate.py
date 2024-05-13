import pandas as pd
from google.cloud import translate_v2
import time as time
import os

# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r"Skripsi/Trash/keyME.json"
# translate_client = translate_v2.Client()

def backTranslate(text, lan):
  temp = translate_client.translate(text,lan)['translatedText']

  return translate_client.translate(temp,'en')['translatedText']