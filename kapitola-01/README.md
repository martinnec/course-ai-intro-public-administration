# Kurz AI pro veřejnou správu - Kapitola 1

Vítejte v kurzu o umělé inteligenci (AI) pro veřejnou správu.
V tomto kurzu se naučíte, jak vytvářet jednoduché aplikace využívající umělou inteligenci pomocí nástrojů OpenAI.  

V první kapitole si připravíte vývojové prostředí, vytvoříte si jednoduchý Python skript, ve kterém si zavoláte velký jazykový model (LLM) od OpenAI.

## Postup

### 1. Příprava projektu

Vytvořte složku pro projekt a přejděte do ní:

- **Bash (Linux/Mac):**
  ```bash
  mkdir ai-pro-verejnou-spravu
  cd ai-pro-verejnou-spravu
  ```
- **PowerShell (Windows):**
  ```powershell
  mkdir ai-pro-verejnou-spravu
  cd ai-pro-verejnou-spravu
  ```

### 2. Vytvoření virtuálního prostředí

- **Bash:**
  ```bash
  python -m venv .venv
  ```
- **PowerShell:**
  ```powershell
  python -m venv .venv
  ```

Aktivujte prostředí:

- **Bash:**
  ```bash
  source .venv/bin/activate
  ```
- **PowerShell:**
  ```powershell
  .venv\Scripts\Activate.ps1
  ```

Po aktivaci se vám v příkazové řádce zobrazí `(.venv)`, což značí, že jste ve správném prostředí.

### 3. Instalace knihoven

- **Bash:**
  ```bash
  pip install openai python-dotenv
  pip freeze > requirements.txt
  ```
- **PowerShell:**
  ```powershell
  pip install openai python-dotenv
  pip freeze > requirements.txt
  ```

### 4. Vytvoření souborů

- **Bash:**
  ```bash
  touch main.py .env .gitignore
  ```
- **PowerShell:**
  ```powershell
  New-Item main.py -ItemType File
  New-Item .env -ItemType File
  ```

### 5. Získání OpenAI API klíče

1. Přejděte na [https://platform.openai.com/account/api-keys](https://platform.openai.com/account/api-keys)
2. Přihlaste se nebo si založte účet
3. Vygenerujte nový klíč
4. Vložte jej do souboru `.env` ve tvaru:

```
OPENAI_API_KEY=sk-...váš-klíč...
```

> ⚠️ Klíč nikdy nesdílejte veřejně ani jej nezahrnujte do verzovacích systémů.

---

### 6. Vytvoření jednoduché AI aplikace

Do souboru `main.py` vložte tento kód:

```python
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
```

### Spuštění skriptu:

- **Bash:**
  ```bash
  python main.py
  ```
- **PowerShell:**
  ```powershell
  python main.py
  ```

---