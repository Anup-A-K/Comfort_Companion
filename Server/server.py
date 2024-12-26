from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import uvicorn
import logging
import json
import os
import requests
from datetime import datetime
import random
import google.generativeai as genai
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from config import Config
from typing import Optional
import time
from metrics import MetricsTracker

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="T5 Conversational Model Server")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
MODEL_PATH = "modelmodel"
HISTORY_FILE = "conversation_history.json"
REWARD_THRESHOLD = 0.7
DEFAULT_CITY = "Kollam"

# Configure Gemini - move this inside a function or method
def initialize_ai_service():
    try:
        ai_token = Config.get_service_token('AI_SERVICE_TOKEN')
        genai.configure(api_key=ai_token)
        return genai.GenerativeModel('gemini-pro')
    except Exception as e:
        logger.error(f"Failed to initialize AI service: {e}")
        raise

class ConversationManager:
    def __init__(self, history_file):
        self.history_file = history_file
        self.conversation_pairs = self.load_history()
        self.recent_conversations = []
        self.learning_rate = 0.1
        self.gemini_model = initialize_ai_service()
        self.metrics = MetricsTracker()
        
    def load_history(self):
        """Load conversation history from file"""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r') as f:
                    data = json.load(f)
                    self.conversation_pairs = data.get("conversation_pairs", {})
                    self.recent_conversations = data.get("recent_conversations", [])
                    return self.conversation_pairs
            return {}
        except Exception as e:
            logger.error(f"Error loading history: {e}")
            return {}

    def get_g_response(self, text):
        try:
            prompt = f"Give a conversational response to: {text}"
            response = self.gemini_model.generate_content(prompt)
            if response.text:
                # Remove truncation entirely
                clean_response = response.text.strip()
                self.add_conversation(text, clean_response, 0.7)
                return clean_response, 0.7
            return None, 0
        except Exception as e:
            logger.error(f"API error: {e}")
            return None, 0

    def get_weather(self, city=DEFAULT_CITY):
        """Get weather information using Weather API"""
        try:
            weather_token = Config.get_service_token('WEATHER_SERVICE_TOKEN')
            url = f"http://api.weatherapi.com/v1/current.json?key={weather_token}&q={city}&aqi=no"
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                condition = data['current']['condition']['text']
                temp_c = data['current']['temp_c']
                humidity = data['current']['humidity']
                return f"The weather in {city} is {condition} with a temperature of {temp_c}°C and humidity of {humidity}%"
            return None
        except Exception as e:
            logger.error(f"Weather API error: {e}")
            return None

    def get_joke(self):
        """Get a joke from JokeAPI"""
        try:
            url = "https://v2.jokeapi.dev/joke/Any?blacklistFlags=nsfw,religious,political,racist,sexist,explicit&type=single"
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                if data.get("type") == "single":
                    return data.get("joke"), 0.9
            return None, 0
        except Exception as e:
            logger.error(f"Joke API error: {e}")
            return None, 0

    def get_quote(self):
        """Get a happiness quote from API Ninjas"""
        try:
            url = "https://api.api-ninjas.com/v1/quotes?category=happiness"
            api_key = Config.get_service_token('QUOTES_SERVICE_TOKEN')
            headers = {
                'X-Api-Key': api_key
            }
            logger.info(f"Requesting happiness quote from API Ninjas...")
            logger.info(f"Using URL: {url}")
            
            response = requests.get(url, headers=headers)
            logger.info(f"Response status code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"Quote API response: {data}")
                if data and len(data) > 0:
                    quote = data[0]
                    formatted_quote = f"\"{quote['quote']}\" - {quote['author']}"
                    return formatted_quote, 0.9
                logger.error("Quote API returned empty data")
            else:
                logger.error(f"Quote API error status code: {response.status_code}")
                logger.error(f"Response content: {response.text}")
                if response.status_code == 401:
                    logger.error("Authentication failed - check your API key")
                elif response.status_code == 429:
                    logger.error("Rate limit exceeded")
            return None, 0
        except Exception as e:
            logger.error(f"Quotes API error: {e}")
            logger.exception("Full traceback:")
            return None, 0

    def get_response(self, input_text):
        """Get response with metrics tracking"""
        start_time = time.time()
        input_text = input_text.lower()

        # 1. First check exact match in conversation pairs
        if input_text in self.conversation_pairs:
            entry = self.conversation_pairs[input_text]
            if isinstance(entry, dict):
                response = entry.get('response')
                confidence = entry.get('confidence', 0.8)
            else:
                response = entry
                confidence = 0.8
            logger.info(f"Found exact match in history: {input_text}")
            
            # Evaluate response quality if we have a reference
            if isinstance(entry, dict) and 'reference' in entry:
                self.metrics.evaluate_response(entry['reference'], response)
            
            self.metrics.track_latency(start_time)
            return response, 'history', confidence

        # 2. Check for similar questions in recent conversations
        for conv in self.recent_conversations:
            if conv['question'].lower() == input_text:
                logger.info(f"Found match in recent conversations: {input_text}")
                self.metrics.track_latency(start_time)
                return conv['answer'], 'history', conv.get('confidence', 0.7)

        # 3. Check for similar questions (partial matches) in history
        for question, entry in self.conversation_pairs.items():
            if question in input_text or input_text in question:
                if isinstance(entry, dict):
                    response = entry.get('response')
                    confidence = entry.get('confidence', 0.7)
                else:
                    response = entry
                    confidence = 0.7
                logger.info(f"Found partial match in history: {question}")
                self.metrics.track_latency(start_time)
                return response, 'history', confidence

        # 4. Check for quote requests
        if any(word in input_text for word in ['quote', 'inspire me', 'inspiration']):
            logger.info("No history match found, trying quotes API")
            quote, confidence = self.get_quote()
            if quote:
                self.add_conversation(input_text, quote, confidence)
                self.metrics.track_latency(start_time)
                return quote, 'quote', confidence
            return "Sorry, I couldn't get a quote right now.", 'error', 0.1

        # 4. Only if no history match is found, proceed with API calls
        
        # Check for joke requests
        if any(word in input_text for word in ['joke', 'tell me a joke', 'make me laugh']):
            logger.info("No history match found, trying joke API")
            joke, confidence = self.get_joke()
            if joke:
                self.add_conversation(input_text, joke, confidence)
                self.metrics.track_latency(start_time)
                return joke, 'joke', confidence
            return "Sorry, I couldn't get a joke right now.", 'error', 0.1

        # Check for weather queries
        elif any(word in input_text for word in ['weather', 'temperature']):
            logger.info("No history match found, trying weather API")
            weather_response = self.get_weather()
            if weather_response:
                self.add_conversation(input_text, weather_response, 0.8)
                self.metrics.track_latency(start_time)
                return weather_response, 'weather', 0.8
            return "Sorry, I couldn't get the weather information.", 'error', 0.1

        else:
            logger.info("No history match found, using API")
            g_response, confidence = self.get_g_response(input_text)
            if g_response:
                self.metrics.track_latency(start_time)
                return g_response, 'gemini', confidence
            return "I'll keep learning to give better responses.", 'default', 0.1

    def extract_city(self, text):
        cities = ["bangalore", "mumbai", "delhi", "chennai", "kolkata"]
        words = text.lower().split()
        for word in words:
            if word in cities:
                return word.capitalize()
        return DEFAULT_CITY

    def add_conversation(self, question, answer, confidence=None):
        """Add conversation to history with full details"""
        timestamp = datetime.now().isoformat()
        
        # Create conversation entry
        conversation_entry = {
            "question": question,
            "answer": answer,
            "timestamp": timestamp,
            "confidence": confidence,
            "source": "user_interaction"
        }
        
        # Add to recent conversations
        self.recent_conversations.append(conversation_entry)
        
        # Add to conversation pairs (for learning)
        if confidence and confidence > REWARD_THRESHOLD:
            self.conversation_pairs[question.lower()] = {
                'response': answer,
                'confidence': confidence,
                'timestamp': timestamp,
                'uses': self.conversation_pairs.get(question.lower(), {}).get('uses', 0) + 1
            }
        
        # Save both recent conversations and pairs
        self.save_history()
        logger.info(f"New conversation added to history with confidence {confidence}")

    def save_history(self):
        """Save both conversation pairs and recent history"""
        try:
            history_data = {
                "conversation_pairs": self.conversation_pairs,
                "recent_conversations": self.recent_conversations,
                "last_updated": datetime.now().isoformat()
            }
            
            with open(self.history_file, 'w') as f:
                json.dump(history_data, f, indent=2)
            logger.info("Conversation history saved successfully")
        except Exception as e:
            logger.error(f"Error saving history: {e}")

    def update_model_weights(self, input_text, response, reward):
        if reward > 0:
            current_confidence = self.conversation_pairs.get(input_text, {}).get('confidence', 0.5)
            new_confidence = current_confidence + self.learning_rate * (reward - current_confidence)
            self.conversation_pairs[input_text] = {
                'response': response,
                'confidence': new_confidence,
                'uses': self.conversation_pairs.get(input_text, {}).get('uses', 0) + 1
            }
            self.save_history()

# Initialize managers and model
gemini_model = initialize_ai_service()
conversation_manager = ConversationManager(HISTORY_FILE)

try:
    Config.validate_tokens()
except ValueError as e:
    logger.error(f"Configuration error: {e}")
    raise

try:
    logger.info("Loading tokenizer and model...")
    tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)
    
    if torch.cuda.is_available():
        logger.info("Using GPU")
        model = model.cuda()
    else:
        logger.info("Using CPU")
    
    model.eval()
    logger.info("Model loaded successfully!")
    
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise

class GenerationRequest(BaseModel):
    text: str
    max_length: int = 64
    temperature: float = 0.7
    feedback: float = None

@app.post("/generate")
async def generate_text(request: GenerationRequest):
    try:
        input_text = request.text.lower()
        
        # Try to get response from conversation manager
        response, source, confidence = conversation_manager.get_response(input_text)
        
        if response:
            return {
                "generated_texts": [response],
                "input_text": request.text,
                "source": source,
                "confidence": confidence
            }

        # Fall back to T5 model with increased max_length
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=128,  # Increased from 64
            padding=True
        )
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=256,  # Increased from request.max_length
                temperature=request.temperature,
                num_return_sequences=1
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        
        if generated_text:
            conversation_manager.add_conversation(input_text, generated_text, 0.6)
            return {
                "generated_texts": [generated_text],
                "input_text": request.text,
                "source": "model",
                "confidence": 0.6
            }
            
        return {
            "generated_texts": ["I'm not sure how to respond to that."],
            "input_text": request.text,
            "source": "default",
            "confidence": 0.1
        }

    except Exception as e:
        logger.error(f"Generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
async def provide_feedback(question: str, answer: str, rating: float):
    """Save feedback and update conversation history"""
    conversation_manager.update_model_weights(question, answer, rating)
    conversation_manager.metrics.add_user_rating(rating)
    
    # Add the interaction to history
    conversation_manager.add_conversation(question, answer, rating)
    
    return {"status": "success", "message": "Feedback recorded"}

@app.get("/history")
async def get_history():
    return {
        "conversation_pairs": conversation_manager.conversation_pairs,
        "recent_conversations": conversation_manager.recent_conversations
    }

@app.get("/metrics")
async def get_metrics():
    """Get current evaluation metrics"""
    return conversation_manager.metrics.get_metrics()

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
