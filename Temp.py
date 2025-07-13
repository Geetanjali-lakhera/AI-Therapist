from datasets import load_dataset
import pandas as pd

ds_empathetic = load_dataset("Estwld/empathetic_dialogues_llm")
ds_dailydialog = load_dataset("roskoN/dailydialog")
kaggle_hub = pd.read_csv("cleaned_counselchat_data.csv")

combined = []


for e in ds_empathetic["train"]:
    turns = e["conversations"]
    dialogue = " ".join([f"{c['role']}: {c['content']}" for c in turns])
    combined.append({
        "source": "empathetic",
        "emotion": e["emotion"],
        "context": e["situation"],
        "dialogue": dialogue
    })


for _, row in kaggle_hub.iterrows():
    combined.append({
        "source": "counselchat",
        "emotion": row.get("topic", "unspecified"),
        "context": row.get("question", ""),
        "dialogue": f"user: {row.get('question')} assistant: {row.get('answer')}"
    })


for item in ds_dailydialog["train"]:
    turns = item["utterances"]
    emotions = item.get("emotions", [])
    dialogue = " ".join([f"turn{i+1} (emotion={emotions[i] if i < len(emotions) else 'N/A'}): {t}" for i, t in enumerate(turns)])
    combined.append({
        "source": "dailydialog",
        "emotion": emotions[-1] if emotions else "unknown",  # You can change to majority or first if needed
        "context": "",
        "dialogue": dialogue
    })


df = pd.DataFrame(combined)
df.to_csv("combined_therapy_dataset.csv", index=False)

print("Dataset combined and saved successfully!")
