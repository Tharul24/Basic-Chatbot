import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import random

# Load pre-trained model and tokenizer (DistilBERT fine-tuned on SST-2 for sentiment as an example)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

# Intent-response mapping
intents = {
    "greeting": ["Hi there!", "Hello!", "Hey, how can I help you?", "Hello! How can I help you today?", "Hi there! What's up?"],
    "goodbye": ["Goodbye!", "Bye", "See you later!", "Goodbye, have a great day!", "Take care!"],
    "thanks": ["You're welcome!", "Glad to help!", "Anytime!"],
    "help": ["Sure! Tell me what you need help with.", "I’m here to help! Ask me anything."],
    "sentiment_positive": ["That's great to hear!", "I'm glad you're feeling good!"],
    "sentiment_negative": ["I'm sorry to hear that. Want to talk about it?", "Stay strong. I'm here if you need to talk."],
    "unknown": ["I'm not sure how to respond to that. Can you rephrase?"]
}

# Keyword-based intent classification
def classify_intent(user_input):
    input_lower = user_input.lower()
    if any(word in input_lower for word in ["hi", "hello", "hey"]):
        return "greeting"
    elif any(word in input_lower for word in ["bye", "goodbye", "see you"]):
        return "goodbye"
    elif "thank" in input_lower:
        return "thanks"
    elif "help" in input_lower:
        return "help"
    else:
        return "unknown"

# Sentiment analysis
def detect_sentiment(user_input):
    inputs = tokenizer(user_input, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = F.softmax(logits, dim=1)
    sentiment = torch.argmax(probs).item()
    return "positive" if sentiment == 1 else "negative"

# Generate response
def generate_response(user_input):
    intent = classify_intent(user_input)

    # Check sentiment if intent unknown
    if intent == "unknown":
        sentiment = detect_sentiment(user_input)
        if sentiment == "positive":
            intent = "sentiment_positive"
        elif sentiment == "negative":
            intent = "sentiment_negative"

    responses = intents.get(intent, intents["unknown"])
    return random.choice(responses)

# Main chatbot loop
def chat():
    print("ChatBot: Hello! Ask me anything or type 'bye' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['bye', 'exit', 'quit']:
            print("ChatBot:", random.choice(intents["goodbye"]))
            break
        response = generate_response(user_input)
        print("ChatBot:", response)

# Run chatbot
if __name__ == "__main__":
    chat()
