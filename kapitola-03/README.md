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

---