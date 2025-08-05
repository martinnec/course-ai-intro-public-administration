# Kurz AI pro veřejnou správu - Kapitola 3

Ve třetí kapitole se naučíte, jak získat odpověď modelu ve strukturované podobě, se kterou můžeme dále algoritmicky pracovat v rámci naší aplikace nebo služby.

## Postup

### 1. Definice struktury

Dnešní velké jazykové modely předpokládají, že budou integrovány jako komponenty v rámci různých webových či jiných softwarových aplikací a služeb.
Někdy je model pro uživatele viditelný formou např. chatu.
Jindy je skrytý na pozadí, kdy aplikace nabízí běžné uživatelské rozhraní, do kterého jsou různé výstupy modelu integrovány.
K tomu potřebujeme být schopni s výstupy modelu dále v našem programovém kódu pracovat a tedy potřebujeme, aby nám model vracel odpovědi ve strukturované podobě, která je dále zpracovatelná.

Tato struktura není předdefinovaná tvůrci modelu.
Programátor aplikace, který chce integrovat odpovědi jazykového modelu, si může tuto strukturu zadefinovat sám.
Pokud programujeme v Pythonu, je dnes běžnou praxí definovat strukturu jako *Pydantic* třídy (datové struktury).

V našem případě chceme, aby model vracel návrh postupu řešení životní situace strukturovaný do dvou částí:

- Úvodní text k návrhu řešení dané životní situace
- Uspořádaný seznam kroků, které je potřeba provést

Pro každý krok potom potřebujeme následující informace:

- Pořadí kroku v postupu
- Název kroku, který stručně popisuje, co je potřeba udělat
- Podrobný popis kroku, který uživateli vysvětluje, co má dělat

To můžeme přepsat do následující definice datové struktury

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

Tuto definici můžeme přímo vložit do našeho skriptu `main.py`.

V předchozích příkladech jsme model volali pomocí operace `create`.
Ta je určena pro situace, kdy chceme, aby model odpověděl nestrukturovaným textem.

Můžeme jej však instruovat, aby odpověď poskytl v podobě dat strukturovaných dle námi definované datové struktury.
K tomu slouží operace `parse`.
Její volání je podobné jako volání operace `create`.
Pouze přidáme parametr `text_format`, do kterého předáme definici naší cílové datové struktury.

Výsledek je potom uložen v odpovědi, jako hodnota `output_parsed`.
Touto hodnotou je objekt, který je instancí třídy definující naší datovou strukturu.

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
            "content": "Bolí mě hlava a mám asi horečku. Co si na to mám vzít? Co mám dělat? A mohu jít do práce?"
        }
    ],
    temperature=0.1,
    text_format=Postup
)

# Vypíšeme odpověď
print("AI odpověď:")
print(response.output_parsed)
```

### 2. Zpracování strukturovaného výstupu

Jak vidíme vy výsledku spuštění předchozího skriptu, hodnota uložená v `response.output_parsed` je opravdu instancí naší třídy `Postup`.
Můžeme s ní dále pracovat běžným způsobem v našem programovém kódu, např. ji vypsat v uživatelsky přívětivé podobě.
Výhodou tohoto přístupu je, že nejsme závislí na tom, v jaké podobě nám model odpověď formátuje.
Získáme strukturovaný výstup v podobě, kterou potřebujeme a kterou si v našem aplikačním kódu zpracujeme tak, jak potřebujeme.

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
```

### 3. Detailnější pohled na strukturovaný výstup

Definice datové struktury pomocí Pydantic tříd odpovídá definici pomocí JSON Schema.
Na toto JSON Schema se můžeme podívat, jak ukazuje následující skript.

```python
from pydantic import BaseModel, Field
import json

class KrokPostupu(BaseModel):
    poradi: int = Field(description="Pořadí kroku v postupu")
    nazev: str = Field(description="Název kroku, který stručně popisuje, co je potřeba udělat")
    popis: str = Field(description="Podrobný popis kroku, který uživateli vysvětluje, co má dělat")

class Postup(BaseModel):
    uvod: str = Field(description="Úvodní text k návrhu řešení dané životní situace")
    kroky: list[KrokPostupu] = Field(description="Uspořádaný seznam kroků, které je potřeba provést")

print("JSON schema:")
print(json.dumps(Postup.model_json_schema(), indent=2, ensure_ascii=False))
```

Model ve skutečnosti vrací strukturovaný výstup v podobě JSON dokumentu, který je strukturován tak, jak definuje toto JSON Schema.
Na tento JSON dokument se můžeme také podívat.

```python
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
            "content": "Bolí mě hlava a mám asi horečku. Co si na to mám vzít? Co mám dělat? A mohu jít do práce?"
        }
    ],
    temperature=0.1,
    text_format=Postup
)

# Vypíšeme odpověď
print("AI odpověď:")
postup = response.output_parsed
print(postup.model_dump_json(indent=2))
```

### 4. Příklady pro *one-shot learning* při strukturovaném výstupu

Výše uvedený náhled na JSON podobu strukturovaného výstupu využijeme pro zápis příkladu pro *one-shot learning* techniku, kterou jsme si vysvětlovali v předchozí kapitole.

Příklad, který vložíme do instrukcí, bude z věcného hlediska stejný.
Pouze ho formátujeme v očekávané struktuře a zapíšeme jako JSON dokument.

```python
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
```

---