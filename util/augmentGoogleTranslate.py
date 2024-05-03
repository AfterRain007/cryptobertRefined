import pandas as pd
from google.cloud import translate_v2
import time as time
import os

# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r"Skripsi/Trash/keyME.json"
# translate_client = translate_v2.Client()

langCode = ['it', 'fr', 'sv', 'da', 'pt',
            'id', 'pl', 'hr', 'bg', 'fi',
            'no', 'ru', 'es', 'nl', 'af',
            'de', 'sk', 'cs', 'lv', 'sq']

# ISO-639 code for Google Translate Language
# Italian, French, Swedish, Danish, Portuguese, 
# Indonesian, Polish, Croatioan, Bulgarian, Finnish, 
# Norwegian, Russian, Spanish, Dutch, Afrikaans,
# German, Slovak, Czech, Latvian, and Albanian

def backTranslate(text, lan):
  temp = translate_client.translate(text,lan)['translatedText']

  return translate_client.translate(temp,'en')['translatedText']