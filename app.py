import pickle
import faiss
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PyPDF2 import PdfReader
import os
import google.generativeai as genai
from dotenv import load_dotenv
import numpy as np
import torch
from waitress import serve

load_dotenv()
genai.configure(api_key=os.environ['GEMINI_API_KEY'])

app = Flask(__name__)
CORS(app)

def process_resume_and_query(query):
    chunked = pickle.load(open('docs/chunked_data.pkl', 'rb'))
    query_embed_result = genai.embed_content(model="models/text-embedding-004", content=query)
    query_embedding = torch.tensor(query_embed_result['embedding'], dtype=torch.float32)
    query_embedding_array = np.expand_dims(np.array(query_embedding, dtype='float32'), axis=0)
    index = faiss.read_index("docs/resume_index.faiss")
    distances, indices = index.search(query_embedding_array, 3)
    context = "\n".join(chunked)
    system_prompt = "You are a professional assistant. Answer all the user query on my behalf. Provide concise, accurate, and relevant answers to the user's queries. Keep your responses short and straight to the point, avoiding unnecessary details. Maintain a professional tone at all times."
    input_prompt = f"{system_prompt}\n\nContext: {context}\n\nQuestion: {query}\nAnswer:"
    llm = genai.GenerativeModel('gemini-1.5-flash')
    response = llm.generate_content(input_prompt)
    print(response.text)
    return response.text

@app.route('/', methods=['GET'])
def home():
    return "App is running!", 200


@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    if not user_message:
        return jsonify({"response": "Sorry, I didn't understand your message."})

    # Example logic for chatbot functionality
    if "hello" in user_message.lower():
        bot_response = "Hi there! How can I assist you today?"
    elif "bye" in user_message.lower():
        bot_response = "Goodbye! Have a great day!"
    else:
        bot_response = process_resume_and_query(user_message)
        print(bot_response)
        print(type(bot_response))

    return jsonify({"response": bot_response})


if __name__ == "__main__":
    # port = int(os.environ.get("PORT", 5000))
    # serve(app, host='0.0.0.0', port=port)
    app.run(port=8001, debug=True)
