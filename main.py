from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import os
from dotenv import load_dotenv

# --- Initialization ---
app = Flask(__name__)

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# --- Load Data & Model ---
print("📄 Loading data and model...")
df = pd.read_csv("combined_therapy_dataset.csv")
dialogues = df["dialogue"].tolist()[:500]  
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedding_model.encode(dialogues, convert_to_numpy=True)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get_response", methods=["POST"])
def get_response():
    try:
        user_input = request.json["message"]
        print(f"\n📥 User input: {user_input}")

        user_embedding = embedding_model.encode(user_input, convert_to_numpy=True)
        similarities = cosine_similarity([user_embedding], embeddings)[0]
        best_match_idx = np.argmax(similarities)
        max_similarity = similarities[best_match_idx]
        raw_dialogue = dialogues[best_match_idx]

        print(f"🔍 Best match similarity: {max_similarity:.2f}")

        if max_similarity < 0.4:
            prompt = f"""
You're an empathetic AI therapist.
The user says:
"{user_input}"

Respond with emotional support and practical advice.
"""
        else:
            prompt = f"""
You're an empathetic AI therapist.
The user says:
"{user_input}"

Here is a similar past conversation:
{raw_dialogue}

Based on both, reply to the user supportively.
"""

        print("📨 Prompt to Gemini:\n", prompt[:1000])  # log first 1000 chars

        # Call Gemini
        try:
            gemini_response = gemini_model.generate_content(prompt)
            response_text = gemini_response.text.strip()
            print("✅ Gemini Response:\n", response_text[:1000])  # log first 1000 chars
        except Exception as gemini_err:
            print(f"🔥 Gemini Error: {str(gemini_err)}")
            response_text = f"Sorry, Gemini failed: {str(gemini_err)}"

        return jsonify({"response": response_text})

    except Exception as e:
        print(f"❌ Server Error: {str(e)}")
        return jsonify({"response": f"Server error: {str(e)}"})

if __name__ == "__main__":
    app.run(debug=True)
