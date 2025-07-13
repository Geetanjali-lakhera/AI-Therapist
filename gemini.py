import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai


load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")


df = pd.read_csv("combined_therapy_dataset.csv")
dialogues = df["dialogue"].tolist()[:2000]


model = SentenceTransformer("all-MiniLM-L6-v2", device='cpu')


print("🔍 Embedding top 2000 therapy samples...")
embeddings = model.encode(
    dialogues,
    convert_to_numpy=True,
    batch_size=32,
    show_progress_bar=True
)

print(" Ready! Start chatting with the AI Therapist (type 'exit' to quit)\n")


while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    user_embedding = model.encode(user_input, convert_to_numpy=True)
    sims = cosine_similarity([user_embedding], embeddings)[0]
    max_sim = np.max(sims)
    print(f"🔎 Similarity Score: {max_sim:.2f}")

    
    top_idx = np.argsort(sims)[-3:][::-1]
    similar_convos = "\n\n".join([dialogues[i] for i in top_idx])

    
    if max_sim < 0.4:
        prompt = f"""
You're an empathetic AI therapist.

The user says:
\"{user_input}\"

Please respond with emotional support and practical advice.
"""
    else:
        prompt = f"""
You're an empathetic AI therapist.

The user says:
\"{user_input}\"

Here are similar past conversations:
{similar_convos}

Based on both, respond with empathy and helpful advice.
"""

    try:
        response = gemini_model.generate_content(prompt)
        reply = response.text.strip()
    except Exception as e:
        reply = " Sorry, I had trouble responding just now."
        print(f"Gemini error: {e}")

    print(f"\n Therapist: {reply}\n")
