import pandas as pd

def lang_feat_engineering(text, pipeline):
   """
   Extract linguistic features from a given text using a Stanza pipeline.

   Features extracted:
   - total_words: Total number of tokens in the text.
   - total_sentences: Number of sentences identified.
   - avg_sentence_len: Average number of words per sentence.
   - unique_word_ratio: Ratio of unique lemmas to total words (lexical diversity).
   - pos_word_ratio: Ratio of Nouns, Verbs, and Adjectives to total words.

   Args:
      text (str): The input text to analyze.
      pipeline: A pre-loaded Stanza pipeline for Indonesian.

   Returns:
      pd.Series: A Series containing the 5 extracted linguistic features.
   """
   # Ensure input is a string
   if not isinstance(text, str):
      text = str(text) if text is not None else ""

   seriesIndex = ['total_words', 'total_sentences', 'avg_sentence_len', 'unique_word_ratio', 'pos_word_ratio']

   # Handle empty or whitespace-only strings
   if not text.strip():
      return pd.Series(
         [0, 0, 0, 0, 0],
         index=seriesIndex
      )
   
   # Process text with Stanza
   doc = pipeline(text)

   total_words = 0
   unique_words = set()
   pos_words = 0 # Count for Noun, Verb, Adj

   total_sentences = len(doc.sentences)

   # Handle cases where no sentences are detected
   if total_sentences == 0:
      return pd.Series(
         [0, 0, 0, 0, 0],
         index=seriesIndex
      )
   
   # Iterate through sentences and words to collect statistics
   for sentence in doc.sentences:
      for word in sentence.words:
         total_words += 1
         # Use lemma for uniqueness to ignore inflectional variations
         unique_words.add(word.lemma.lower())

         # Count POS tags for nouns, verbs, and adjectives
         if word.upos in ('NOUN', 'VERB', 'ADJ'):
            pos_words += 1

   # Handle division by zero if no words are found
   if total_words == 0:
      return pd.Series(
         [0, total_sentences, 0, 0, 0],
         index=seriesIndex
      )
   
   # Calculate feature ratios
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
