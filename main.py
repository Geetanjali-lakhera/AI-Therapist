from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import os
from dotenv import load_dotenv

app = Flask(__name__)

# --- Configure Gemini ---
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# --- Load Data ---
df = pd.read_csv("combined_therapy_dataset.csv")
dialogues = df["dialogue"].tolist()[:300]  # reduce for memory

# --- Load Model Once ---
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get_response", methods=["POST"])
def get_response():
    try:
        user_input = request.json["message"]
        print(f"\n📥 User: {user_input}")

        # Compute embeddings on demand
        all_embeddings = embedding_model.encode(dialogues, convert_to_numpy=True)
        user_embedding = embedding_model.encode(user_input, convert_to_numpy=True)

        similarities = cosine_similarity([user_embedding], all_embeddings)[0]
        top_idx = np.argmax(similarities)
        max_sim = similarities[top_idx]
        best_example = dialogues[top_idx]

        if max_sim < 0.4:
            prompt = f"""
You're an empathetic AI therapist.
The user says:
\"{user_input}\"

Respond with emotional support and practical advice.
"""
        else:
            prompt = f"""
You're an empathetic AI therapist.
The user says:
\"{user_input}\"

Here is a similar past conversation:
{best_example}

Respond supportively using both inputs.
"""

        try:
            gemini_reply = gemini_model.generate_content(prompt)
            reply_text = gemini_reply.text.strip()
        except Exception as gem_err:
            print("Gemini Error:", gem_err)
            reply_text = f"Sorry, Gemini failed: {str(gem_err)}"

        return jsonify({"response": reply_text})

    except Exception as e:
        print("❌ Server error:", e)
        return jsonify({"response": f"Server error: {str(e)}"})


if __name__ == "__main__":
    app.run(debug=True)
