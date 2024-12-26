from safetensors.torch import load_file
import torch
from flask import Flask, request, jsonify
from transformers import T5Tokenizer, T5ForConditionalGeneration

app = Flask(__name__)

# Paths to the model files
MODEL_PATH = "/Users/vishnuvardhan/development/WebDev/FYP/your_t5_model"

# Load the tokenizer and model
tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)

# Use safetensors to load the model weights
model_state_dict = load_file('/Users/vishnuvardhan/development/WebDev/FYP/your_t5_model/model.safetensors')  # Assuming model.safetensors is in the same directory
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH, state_dict=model_state_dict)

# Flask route to handle requests from the frontend
@app.route('/get', methods=['POST'])
def generate_response():
    user_input = request.form['msg']

    # Tokenize the input
    input_ids = tokenizer.encode(user_input, return_tensors='pt')

    # Generate a response from the model
    outputs = model.generate(input_ids, max_length=100, num_beams=5, early_stopping=True)

    # Decode the generated output to string
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)
