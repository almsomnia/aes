import re

def load_dict(path):
   try:
      with open(path, 'r', encoding='utf-8') as f:
         return set(word.strip().lower() for word in f)
   except FileNotFoundError:
      print(f"File {path} not found")
      return set()

def check_spelling(text, dict_set):
   if not dict_set:
      return 0.0
   
   cleaned_text = re.sub(r'[^a-z\s]', '', text.lower())
   words = cleaned_text.split()

   if not words:
      return 0.0
   
   error_count = sum(1 for word in words if word not in dict_set)
   return error_count / len(words)