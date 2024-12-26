import nltk

print("Downloading NLTK data...")
try:
    nltk.download('punkt')
    print("Setup complete!")
except Exception as e:
    print(f"Error downloading NLTK data: {e}") 