import os
import traceback
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
print(f"API Key found: {api_key[:10]}...")

try:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash-latest")
    response = model.generate_content("Generate a simple navbar HTML")
    print("Success!")
    print(response.text[:100])
except Exception as e:
    traceback.print_exc()
