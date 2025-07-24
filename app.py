import os
import logging
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import google.generativeai as genai
from dotenv import load_dotenv
from retry import retry

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("API key not found. Please set the GOOGLE_API_KEY environment variable.")
genai.configure(api_key=api_key)

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

chat = None
model = None

def initialize_chat():
    """Initializes or re-initializes the chat model."""
    global chat, model
    try:
        if model is None:
            model = genai.GenerativeModel('gemini-1.5-flash')
        chat = model.start_chat(history=[])
        logging.info("Gemini chat session started successfully.")
        return True
    except Exception as e:
        logging.error(f"Error initializing Gemini model: {e}")
        chat = None
        return False


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/reset", methods=["POST"])
def reset_chat():
    """Endpoint to reset the conversation."""
    if initialize_chat():
        return jsonify({"message": "Chat has been reset."}), 200
    else:
        return jsonify({"error": "Failed to reset chat."}), 500

@app.route("/ask", methods=["POST"])
def ask_gemini():
    global chat

    if chat is None:
        if not initialize_chat():
            return jsonify({"error": "AI model is unavailable. Please try again later."}), 500

    try:
        data = request.get_json()
        if not data or "question" not in data:
            logging.warning("Invalid request: No 'question' in JSON")
            return jsonify({"error": "No question provided."}), 400
        
        user_text = data["question"].strip()
        if not user_text:
            logging.warning("Empty question received")
            return jsonify({"error": "Question cannot be empty."}), 400
        
        @retry(tries=3, delay=1, backoff=2, logger=logging)
        def send_message_with_retry(text):
            response = chat.send_message(text)
            return response
        
        response = send_message_with_retry(user_text)
        answer = response.text.strip() if response.text else "Sorry, I couldn't generate a response."
        logging.info(f"Successful response for question: {user_text[:50]}...")
        return jsonify({"answer": answer})
    
    except genai.types.StopCandidateException as e:
        logging.error(f"Gemini content policy violation: {e}")
        return jsonify({"error": "The question violated content policies. Please rephrase."}), 400
    except Exception as e:
        logging.error(f"Error processing request: {e}", exc_info=True)
        return jsonify({"error": "An internal error occurred. Please try again."}), 500

if __name__ == "__main__":
    initialize_chat() 
    app.run(debug=True, port=8000)