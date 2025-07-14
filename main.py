from flask import Flask, request, jsonify, render_template
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import os
import pickle
from dotenv import load_dotenv

app = Flask(__name__)

# --- Configure Gemini ---
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# --- Load precomputed dialogues + embeddings ---
print("📦 Loading cached embeddings...")
with open("embeddings_cache.pkl", "rb") as f:
    dialogues, all_embeddings = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get_response", methods=["POST"])
def get_response():
    try:
        user_input = request.json["message"]
        print(f"\n📥 User: {user_input}")

        # Use sentence-transformers only for user input
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        user_embedding = model.encode(user_input, convert_to_numpy=True)

        # Find best match
        similarities = cosine_similarity([user_embedding], all_embeddings)[0]
        top_idx = np.argmax(similarities)
        max_sim = similarities[top_idx]
        best_example = dialogues[top_idx]

        # Prompt building
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

        # Call Gemini
        try:
            gemini_reply = gemini_model.generate_content(prompt)
            reply_text = gemini_reply.text.strip()
        except Exception as gem_err:
            print("🔥 Gemini Error:", gem_err)
            reply_text = f"Sorry, Gemini failed: {str(gem_err)}"

        return jsonify({"response": reply_text})

    except Exception as e:
        print("❌ Server error:", e)
        return jsonify({"response": f"Server error: {str(e)}"})

if __name__ == "__main__":
    app.run(debug=True)
