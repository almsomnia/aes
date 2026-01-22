import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def sbert_feat_engineering(df: pd.DataFrame, stimulus_path, model, save_path=None):
   print("Extracting SBERT Features...")
   with(open(f"{stimulus_path}", "r", encoding="utf-8") as f):
      lines = f.readlines()

   stimulus = [line.strip() for line in lines if line.strip()]
   question_text = stimulus[-1]

   print("DataFrame:")
   print(df)
   print("===================================\n")
   print("Stimulus:")
   print(stimulus)
   print("===================================\n")
   print(f"Question: {question_text}")

   vec_question = model.encode(question_text)
   vec_responses = model.encode(df["RESPONSE"].tolist(), batch_size=32, show_progress_bar=True)

   similarity_scores = []

   for i in range(len(vec_responses)):
      sim = cosine_similarity(
         [vec_question],
         [vec_responses[i]]
      )[0][0]
      similarity_scores.append(sim)

   df["RELEVANCE_FEAT"] = similarity_scores
   
   if (save_path):
      df.to_csv(save_path)
   
   print("SBERT feature extracted")
   return df