from fastapi import FastAPI, Depends, HTTPException
from fastapi.security.api_key import APIKeyHeader
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI()
api_key_header = APIKeyHeader(name="X-API-Key")

# Replace with a secure key
VALID_API_KEYS = {"mysecretkey"}

def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key

# Load chatbot model
MODEL_NAME = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Store conversation history per user
user_conversations = {}

@app.get("/")
def home():
    return {"message": "Conversational chatbot is running!"}

@app.post("/chat")
def chat(user_id: str, message: str, api_key: str = Depends(verify_api_key)):
    if user_id not in user_conversations:
        user_conversations[user_id] = []

    # Encode the new message and append to conversation
    new_input_ids = tokenizer.encode(message + tokenizer.eos_token, return_tensors="pt")

    # Append past conversation context
    chat_history_ids = torch.cat([torch.tensor(user_conversations[user_id]), new_input_ids], dim=-1) if user_conversations[user_id] else new_input_ids

    # Generate response
    response_ids = model.generate(chat_history_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response_text = tokenizer.decode(response_ids[:, chat_history_ids.shape[-1]:][0], skip_special_tokens=True)

    # Store updated conversation history
    user_conversations[user_id] = response_ids.tolist()

    return {"response": response_text}
