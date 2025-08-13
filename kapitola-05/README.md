# Kurz AI pro veřejnou správu - Kapitola 5: Strukturované výstupy modelu

V této kapitole se naučíte, jak přimět model vracet odpovědi **ve strukturované podobě**, se kterou lze dále bezpečně a spolehlivě pracovat v aplikaci. Projdeme teorii, návrh schématu definujícího požadovanou strukturu, volání API, validaci a zpracování výsledků.

## 1. Proč strukturovaný výstup

LLM můžete v aplikaci z pohledu uživatele použít dvěma způsoby:

- **Viditelně**, kdy se uživatel ptá v aplikaci volným textem a aplikace mu ukazuje odpovědi opět formou volného textu. Zde je vaše aplikace pouze tenkou nadstavbou nad modelem. Příkladem je např. aplikace ChatGPT.
- **„Na pozadí“**, kdy je model skryt uvnitř aplikace jako jedna z více aplikačních komponent a aplikace jej využívá pro konkrétní úlohy. Model generuje podklady pro rozhraní (kroky postupu, tagy, klasifikace…) a aplikace s nimi **algoritmicky** pracuje.

Pro druhý způsob použití potřebujeme, aby model nevracel volný text, ale aby odpověď byla v podobě strukturovaných dat, která můžeme dále zpracovávat v našem aplikačním kódu.

---

## 2. Návrh cílového schématu v Pydantic

V naší aplikaci chceme uživateli ukázat návrh řešení životní situace tak, že nejprve ukážeme **úvodní popis** a potom zobrazíme setříděny **seznam kroků**. Každý krok má pořadí, název a popis.

Na základě tohoto požadavku můžeme **definovat datovou strukturu**, se kterou budeme v našem aplikačním kódu pracovat. K tomu využijeme knihovnu **Pydantic**.

```python
from pydantic import BaseModel, Field

class KrokPostupu(BaseModel):
    poradi: int = Field(description="Pořadí kroku v postupu")
    nazev: str = Field(description="Název kroku, který stručně popisuje, co je potřeba udělat")
    popis: str = Field(description="Podrobný popis kroku, který uživateli vysvětluje, co má dělat.")

class Postup(BaseModel):
    uvod: str = Field(description="Úvodní text k návrhu řešení dané životní situace")
    kroky: list[KrokPostupu] = Field(description="Uspořádaný seznam kroků, které je potřeba provést")
```

---

## 3) Volání API s `parse` a `text_format`

Místo `responses.create` použijeme `responses.parse` a model požádáme, aby výsledek odevzdal **přímo v této struktuře**. `output_parsed` pak bude instance naší Pydantic třídy.

```python
from openai import OpenAI
from dotenv import load_dotenv
import os
from pydantic import BaseModel, Field

# Načteme API klíč ze souboru .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("API klíč není nastaven v .env souboru.")

# Inicializujeme OpenAI klienta
client = OpenAI(api_key=api_key)

# Definujeme datové třídy
class KrokPostupu(BaseModel):
    poradi: int = Field(description="Pořadí kroku v postupu")
    nazev: str = Field(description="Název kroku, který stručně popisuje, co je potřeba udělat")
    popis: str = Field(description="Podrobný popis kroku, který uživateli vysvětluje, co má dělat")

class Postup(BaseModel):
    uvod: str = Field(description="Úvodní text k návrhu řešení dané životní situace")
    kroky: list[KrokPostupu] = Field(description="Uspořádaný seznam kroků, které je potřeba provést")

# Zavoláme OpenAI model
response = client.responses.parse(
    model="gpt-5-mini",
    input=[
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
    ],
    text={"verbosity": "medium"},          # volitelně: úroveň detailu odpovědi
    reasoning={"effort": "low"},           # volitelně: hloubka uvažování
    text_format=Postup
)

# Vypíšeme odpověď
postup = response.output_parsed  # -> instance Postup
print(postup)
```

---

## 4) Prezentace a práce s daty

`output_parsed` je běžný Python objekt a můžeme s ním tak v našem aplikačním kódu dále algoritmicky pracovat, např. jednoduše vypsat.

```python
print(f"\n{postup.uvod}\n")
for krok in postup.kroky:
    print(f"{krok.poradi}. {krok.nazev}\n   {krok.popis}\n")
```

Nebo vypsat do JSONu.

```python
import json
parsed_postup = json.loads(response.model_dump_json())
print(json.dumps(parsed_postup, ensure_ascii=False, indent=2))
```

---

## 5) JSON Schema a „co přesně model vrací“

Každé Pydantic schéma má odpovídající **JSON Schema**. Model se snaží výstup tomuto schématu přizpůsobit.

```python
print(json.dumps(Postup.model_json_schema(), indent=2, ensure_ascii=False))
```

---

## 6) One‑shot learning pro strukturovaný výstup

Ukázkový *assistant* výstup vložíme přímo jako **JSON** ve tvaru schématu. Model se podle něj „naladí“.

```python
from openai import OpenAI
from dotenv import load_dotenv
import os
from pydantic import BaseModel, Field

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

# Zavoláme OpenAI model
response = client.responses.parse(
    model="gpt-5-mini",
    input=[
        {
            "role": "developer",
            "content": "Jsi odborník na pomoc uživateli při řešení jeho životní situace v občanském životě. Vždy poradíš, jak danou životní situaci vyřešit z úředního hlediska poskytnutím konkrétního úředního postupu v podobě číslovaných kroků. Poskytuj krátké a srozumitelné vysvětlení každého kroku. Používej jednoduchou češtinu."
        },{
            "role": "developer",
            "content": "Nikdy nesmíš v žádném kroku postupu poskytovat radu v oboru, kterého se dotaz uživatele týká (např. lékařské rady, stavební rady, atd.). Pouze můžeš uživateli doporučit, aby odborníka navštívil. Toto doporučení ale nesmíš podmiňovat žádnými časovými, situačními či jinými podmínkami."
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
    text={"verbosity": "medium"},     # volitelně: úroveň detailu odpovědi
    reasoning={"effort": "low"},          # volitelně: hloubka uvažování
    text_format=Postup
)

# Vypíšeme odpověď
print("AI odpověď:")
postup = response.output_parsed
print(f"\n{postup.uvod}\n")
for krok in postup.kroky:
    print(f"{krok.poradi}. {krok.nazev}\n   {krok.popis}\n")
```

---

## Shrnutí

- `responses.parse` + `text_format=PydanticTřída` = **přímý strukturovaný výstup**.
- Data odpovídají vašemu **JSON Schema** a lze je bezpečně validovat.
- *One‑shot* JSON příklad pomáhá modelu trefit přesný tvar a styl.