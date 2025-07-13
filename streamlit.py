import os
import streamlit as st
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai


st.set_page_config(page_title="Therapist Bot", page_icon="💬", layout="centered")


st.markdown("""
    <style>
   body {
      margin: 0;
      font-family: "Segoe UI", sans-serif;
      background: #f4f6f9;
      display: flex;
      justify-content: center;
      align-items: center;
      height: 100vh;
    }

    .chat-container {
      width: 100%;
      max-width: 450px;
      background: white;
      border-radius: 15px;
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
      display: flex;
      flex-direction: column;
      padding: 20px;
    }

    .chat-box {
      flex: 1;
      overflow-y: auto;
      margin-bottom: 10px;
    }

    .bubble {
      padding: 10px 14px;
      margin: 8px 0;
      border-radius: 15px;
      max-width: 80%;
      line-height: 1.4;
    }

    .user {
      align-self: flex-end;
      background: #daf5dc;
      text-align: right;
    }

    .bot {
      align-self: flex-start;
      background: #e0f0ff;
    }

    .input-area {
      display: flex;
    }

    .input-area input {
      flex: 1;
      padding: 10px;
      border: 1px solid #ccc;
      border-radius: 20px;
      margin-right: 10px;
    }

    .input-area button {
      padding: 10px 16px;
      background: #2a9df4;
      border: none;
      color: white;
      border-radius: 20px;
      cursor: pointer;
    }

    .bot-avatar {
      width: 50px;
      height: 50px;
      border-radius: 50%;
      margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)


st.markdown("""
    <div class="center-text">
        <img class="avatar-img" src="https://cdn-icons-png.flaticon.com/512/4712/4712027.png">
        <h3>How can I help today?</h3>
    </div>
""", unsafe_allow_html=True)


load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

@st.cache_resource
def load_model_and_embeddings():
    df = pd.read_csv("combined_therapy_dataset.csv")
    dialogues = df["dialogue"].tolist()[:2000]
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(dialogues, convert_to_numpy=True, show_progress_bar=True)
    return model, embeddings, dialogues

embed_model, embeddings, dialogue_list = load_model_and_embeddings()


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


user_input = st.chat_input("Type something here...")


if user_input:
    st.session_state.chat_history.append(("user", user_input))
    user_emb = embed_model.encode(user_input, convert_to_numpy=True)
    sims = cosine_similarity([user_emb], embeddings)[0]
    max_sim = np.max(sims)
    top_idx = np.argsort(sims)[-3:][::-1]
    examples = "\n\n".join([dialogue_list[i] for i in top_idx])

    if max_sim < 0.4:
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
Here are similar past conversations:
{examples}
Based on both, reply to the user supportively.
"""

    try:
        reply = gemini_model.generate_content(prompt).text.strip()
    except Exception:
        reply = "Sorry, I had trouble responding just now."

    st.session_state.chat_history.append(("assistant", reply))


for role, msg in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(msg)
