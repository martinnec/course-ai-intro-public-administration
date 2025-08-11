import time
from openai import OpenAI
from dotenv import load_dotenv
import os

# Načtení API klíče
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("API klíč není nastaven v .env souboru.")

client = OpenAI(api_key=api_key)

prompt = [
        {
            "role": "developer",
            "content": "Jsi odborník na pomoc uživateli při řešení jeho životní situace v občanském životě. Vždy poradíš, jak danou životní situaci vyřešit z úředního hlediska poskytnutím konkrétního úředního postupu v podobě číslovaných kroků. Poskytuj krátké a srozumitelné vysvětlení každého kroku. Používej jednoduchou češtinu."
        },{
            "role": "developer",
            "content": "Nikdy nesmíš v žádném kroku postupu poskytovat radu v oboru, kterého se dotaz uživatele týká (např. lékařské rady, stavební rady, atd.). Pouze můžeš uživateli doporučit, aby odborníka navštívil. Toto doporučení ale nesmíš podmiňovat žádnými časovými, situačními či jinými podmínkami."
        },{
            "role": "user",
            "content": (
                "Bolí mě hlava a mám horečku. Co mám dělat a mohu jít do práce?"
            )
        }
    ]

response = client.responses.create(
    model="gpt-5-mini",
    input=prompt,
    text={
        "verbosity": "low"
    },
    reasoning={
        "effort": "minimal"
    }
)

import json
parsed = json.loads(response.model_dump_json())
print(json.dumps(parsed, ensure_ascii=False, indent=2))