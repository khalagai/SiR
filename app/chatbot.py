from transformers import pipeline

chatbot = pipeline("conversational", model="microsoft/DialoGPT-medium")

def get_chatbot_response(user_input: str) -> str:
    response = chatbot(user_input)
    return response["generated_text"]
