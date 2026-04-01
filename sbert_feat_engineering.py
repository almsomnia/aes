import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def sbert_feat_engineering(df: pd.DataFrame, stimulus_path, model, save_path=None):
   """
   Perform feature engineering using Sentence-BERT (SBERT) to calculate the relevance
   between student responses and the stimulus question.

   Args:
      df (pd.DataFrame): DataFrame containing a "RESPONSE" column.
      stimulus_path (str): Path to the stimulus text file.
      model: The loaded SBERT model (SentenceTransformer).
      save_path (str, optional): Path to save the resulting DataFrame as a CSV file.

   Returns:
      pd.DataFrame: The input DataFrame with an added "RELEVANCE_FEAT" column.
   """
   print("Extracting SBERT Features...")

   # Read stimulus file
   with(open(f"{stimulus_path}", "r", encoding="utf-8") as f):
      lines = f.readlines()

   # Clean lines and identify the question (assumed to be the last non-empty line)
   stimulus = [line.strip() for line in lines if line.strip()]
   question_text = stimulus[-1]

   print("DataFrame:")
   print(df)
   print("===================================\n")
   print("Stimulus:")
   print(stimulus)
   print("===================================\n")
   print(f"Question: {question_text}")

   # Encode the question and responses into vector embeddings
   vec_question = model.encode(question_text)
   vec_responses = model.encode(df["RESPONSE"].tolist(), batch_size=32, show_progress_bar=True)

   similarity_scores = []

   # Calculate cosine similarity between question vector and each response vector
   for i in range(len(vec_responses)):
      sim = cosine_similarity(
         [vec_question],
         [vec_responses[i]]
      )[0][0]
      similarity_scores.append(sim)

   # Add the similarity scores as a new feature column
   df["RELEVANCE_FEAT"] = similarity_scores

   # Optionally save the result to CSV
   if (save_path):
      df.to_csv(save_path)

   print("SBERT feature extracted")
   return df