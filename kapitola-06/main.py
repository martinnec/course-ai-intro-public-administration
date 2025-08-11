from openai import OpenAI
from dotenv import load_dotenv
import os
from pydantic import BaseModel, Field
import json

# Načteme API klíč ze souboru .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("API klíč není nastaven v .env souboru.")

# Inicializujeme OpenAI klienta
client = OpenAI(api_key=api_key)

class KrokPostupu(BaseModel):
    poradi: int = Field(description="Pořadí kroku v postupu")
    nazev: str = Field(description="Název kroku, který stručně popisuje, co je potřeba udělat")
    popis: str = Field(description="Podrobný popis kroku, který uživateli vysvětluje, co má dělat")
    sluzba_id: str = Field(description="Odkaz na ID služby, ze které tento krok vyplývá")

class Postup(BaseModel):
    uvod: str = Field(description="Úvodní text k návrhu řešení dané životní situace")
    kroky: list[KrokPostupu] = Field(description="Uspořádaný seznam kroků, které je potřeba provést")

# Načteme data služeb ze souboru
with open("data/sluzby_data.json", "r", encoding="utf-8") as f:
    sluzby = json.load(f)

sluzby_json = json.dumps(sluzby, ensure_ascii=False)

# Zavoláme OpenAI model
response = client.responses.parse(
    model="gpt-5-mini",
    input=[
        {
            "role": "developer",
            "content": "Jsi odborník na pomoc uživateli při řešení jeho různých životních situací v občanském životě. Vždy poradíš, jak danou životní situaci vyřešit z úředního hlediska poskytnutím konkrétního postupu v podobě číslovaných kroků. Uživatel potřebuje srozumitelné ale krátké vysvětlení každého kroku jednoduchou češtinou."
        },{
            "role": "developer",
            "content": "Odpovídej *VÝHRADNĚ* na základě přiloženého JSON se seznamem služeb (viz <JSON-SLUZBY> dále). Každá služba je uvedena ve struktuře id (jednoznačný identifikátor služby), nazev (krátký název služby), popis (delší popis služby), benefit (jaké výhody či přínosy služba má), jak-resit (jakým způsobem by měl uživatel při řešení služby postupovat), kdy-resit (kdy by měl nebo kdy může uživatel službu řešit), resit-pokud (uživatel by měl službu řešit za zde popsaných podmínek). Nikdy nepoužívej žádné znalosti mimo tento JSON. Vždy je ale tvým hlavním cílem postup v informacích o službách v přiloženém JSON zjistit a uživateli vysvětlit. Pokud však opravdu žádné informace v JSON nenajdeš, výslovně napiš, že nemáš potřebné informace a vrať prázdný seznam kroků."
        },{
            "role": "developer",
            "content": "Nikdy nesmíš v žádném kroku postupu poskytovat radu v oboru, kterého se dotaz uživatele týká, např. lékařské rady, stavební rady, atd. Uživateli pouze můžeš napsat, aby odborníka vyhledal a navštívil bez jakýchkoliv časových, situačních či jiných podmínek a doporučení."
        },{
            "role": "developer",
            "content": f"<JSON-SLUZBY>\n{sluzby_json}\n</JSON-SLUZBY>"
        },{
            "role": "user",
            "content": "Bolí mě hlava a mám asi horečku. Co si na to mám vzít? Co mám dělat? A mohu jít do práce?"
        }
    ],
    text={"verbosity": "medium"},     # volitelně: úroveň detailu odpovědi
    reasoning={"effort": "minimal"},      # volitelně: hloubka uvažování
    text_format=Postup
)

# Vypíšeme odpověď
print("AI odpověď:")
postup = response.output_parsed
print(f"\n{postup.uvod}\n")
for krok in postup.kroky:
    print(f"{krok.poradi}. {krok.nazev} ({krok.sluzba_id})\n   {krok.popis}\n")