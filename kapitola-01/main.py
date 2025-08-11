from openai import OpenAI
from dotenv import load_dotenv
import os

# Načteme API klíč ze souboru .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("API klíč není nastaven v .env souboru.")

# Inicializujeme OpenAI klienta
client = OpenAI(api_key=api_key)

# Zavoláme OpenAI model
response = client.responses.create(
    model="gpt-5-nano",
    input="Jak si vyřídit nemocenskou?"
)

# Vypíšeme odpověď
print("AI odpověď:")
print(response.output_text)