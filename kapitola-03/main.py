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

class Postup(BaseModel):
    uvod: str = Field(description="Úvodní text k návrhu řešení dané životní situace")
    kroky: list[KrokPostupu] = Field(description="Uspořádaný seznam kroků, které je potřeba provést")

print("JSON schema:")
print(json.dumps(Postup.model_json_schema(), indent=2, ensure_ascii=False))

# Zavoláme OpenAI model
response = client.responses.parse(
    model="gpt-4.1-mini",
    input=[
        {
            "role": "developer",
            "content": "Jsi odborník na pomoc uživateli při řešení jeho různých životních situací v občanském životě. Vždy poradíš, jak danou životní situaci vyřešit z úředního hlediska poskytnutím konkrétního postupu v podobě číslovaných kroků. Uživatel potřebuje srozumitelné ale krátké vysvětlení každého kroku jednoduchou češtinou."
        },{
            "role": "developer",
            "content": "Nikdy nesmíš v žádném kroku posutpu poskytovat radu v oboru, kterého se dotaz uživatele týká, např. lékařské rady, stavební rady, atd. Uživateli pouze můžeš napsat, aby odborníka vyhledal a navštívil bez jakýchkoliv časových, situačních či jiných podmínek a doporučení."
        },{
            "role": "user",
            "content": "Někdo mi rozbil okno u auta, vloupal se dovnitř a ukradl mi peněženku. Jak si mám sám opravit okno? Potřebuju nějak řešit ztrátu peněženky?"
        },{
            "role": "assistant",
            "content": """{
                "uvod": "Omlouvám se, ale nemohu Vám poradit, jak si máte sám opravit rozbité okno u vašeho automobilu. Doporučuji se vám obrátit na nejbližší autoservis, kde vám rozbité okno odborně opraví.",
                "kroky": [
                    {
                        "poradi": 1,
                        "nazev": "Zavolejte policii",
                        "popis": "Zavolejte na tísňovou linku 112 nebo 158 a oznamte vloupání do vašeho vozidla."
                    },{
                        "poradi": 2,
                        "nazev": "Vyčkejte na příjezd policie",
                        "popis": "Vyčkejte, než přijede policie a nahlašte jim, co přesně se z vašeho pohledu stalo. Odpovězte na všechny jejich otázky."
                    },{
                        "poradi": 3,
                        "nazev": "Převezměte protokol o vloupání",
                        "popis": "Od policie převezměte originál protokolu o vloupání do vašeho vozidla a o míře poškození."
                    },{
                        "poradi": 4,
                        "nazev": "Ohlašte odcizení občanského průkazu, příp. dalších dokladů",
                        "popis": "Ztrátu můžete nahlásit přímo policistovi, který na místo přijel. Případně můžete ztrátu nahlásti elektronicky vašemu obecnímu úřadu prostřednictvím datové schránky."
                    },{
                        "poradi": 5,
                        "nazev": "Požádejte o vydání nového občanského průkazu, příp. jiného odkladu",
                        "popis": "Požádat o nový doklad můžete na jakémkoli obecním úřadě obce s rozšířenou působností, kde si ho posléze i vyzvednete."
                    },{
                        "poradi": 6,
                        "nazev": "Nahlašte škodní událost",
                        "popis": "Pokud máte automobil pojištěný, nahlaště na pojišťovnu škodní událost. Budete k tomu potřebovat protokol o vloupání do vozidla."
                    },{
                        "poradi": 7,
                        "nazev": "Nechte si opravit rozbité okno",
                        "popis": "Navštivte co nejdříve libovolný autoservis, kde Vám opraví rozbité okno. V autoservisu vám mohou pomoci i nahlášením škodní události vaší pojišťovně (viz krok 5)."
                    }
                ]
            }"""
        },{
            "role": "user",
            "content": "Bolí mě hlava a mám asi horečku. Co si na to mám vzít? Co mám dělat? A mohu jít do práce?"
        }
    ],
    temperature=0.1,
    text_format=Postup
)

# Vypíšeme odpověď
print("AI odpověď:")
postup = response.output_parsed
print(f"\n{postup.uvod}\n")
for krok in postup.kroky:
    print(f"{krok.poradi}. {krok.nazev}\n   {krok.popis}\n")