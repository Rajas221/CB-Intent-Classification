from sentence_transformers import SentenceTransformer
import joblib
import spacy
import numpy as np

# Load models
embedder = SentenceTransformer("all-MiniLM-L6-v2")
clf = joblib.load("intent_classifier.pkl")
label_encoder = joblib.load("intent_label_encoder.pkl")
nlp = spacy.load("en_core_web_sm")

# Slot memory
slots = {}

def predict_intent(text):
    embedding = embedder.encode([text])
    proba = clf.predict_proba(embedding)[0]
    intent_idx = np.argmax(proba)
    intent = label_encoder.inverse_transform([intent_idx])[0]
    confidence = proba[intent_idx]
    return intent, confidence

def extract_entities(text):
    doc = nlp(text)
    return {ent.label_.lower(): ent.text for ent in doc.ents if ent.label_ in ["MONEY", "PERSON", "GPE"]}

def generate_response(user_input):
    intent, confidence = predict_intent(user_input)
    entities = extract_entities(user_input)

    threshold = 0.6
    if confidence < threshold:
        return "🤖 I'm not sure I understood. Could you rephrase?"

    slots.update(entities)

    if intent == "transfer_money":
        amount = slots.get("money")
        recipient = slots.get("person")
        if amount and recipient:
            slots.clear()
            return f"✅ Transferring {amount} to {recipient}."
        elif not amount:
            return "💬 How much would you like to transfer?"
        else:
            return "💬 Who should I send the money to?"
    else:
        return f"Intent: {intent} (Confidence: {confidence:.2f})"

# CLI chat loop
print("🤖 Chatbot is ready. Type 'exit' to stop.")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("🤖 Goodbye!")
        break
    print("Bot:", generate_response(user_input))