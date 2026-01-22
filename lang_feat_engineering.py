# import stanza
import pandas as pd

# pipeline = stanza.Pipeline("id", processors="tokenize,pos,lemma") 

def lang_feat_engineering(text, pipeline):
   if not isinstance(text, str):
      text = str(text) if text is not None else ""

   seriesIndex = ['total_words', 'total_sentences', 'avg_sentence_len', 'unique_word_ratio', 'pos_word_ratio']

   if not text.strip():
      return pd.Series(
         [0, 0, 0, 0, 0],
         index=seriesIndex
      )
   
   doc = pipeline(text)

   total_words = 0
   unique_words = set()
   pos_words = 0 # Noun, Verb, Adj

   total_sentences = len(doc.sentences)

   if total_sentences == 0:
      return pd.Series(
         [0, 0, 0, 0, 0],
         index=seriesIndex
      )
   
   for sentence in doc.sentences:
      for word in sentence.words:
         total_words += 1
         unique_words.add(word.lemma.lower())

         if word.upos in ('NOUN', 'VERB', 'ADJ'):
            pos_words += 1

   if total_words == 0:
      return pd.Series(
         [0, total_sentences, 0, 0, 0],
         index=seriesIndex
      )
   
   avg_sentence_len = total_words / total_sentences
   unique_word_ratio = len(unique_words) / total_words
   pos_word_ratio = pos_words / total_words

   return pd.Series(
      [
         total_words,
         total_sentences,
         avg_sentence_len,
         unique_word_ratio,
         pos_word_ratio
      ],
      index=seriesIndex
   )
