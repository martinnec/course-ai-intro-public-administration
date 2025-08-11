# Kurz AI pro veřejnou správu - Kapitola 3: Parametry volání API

V této kapitole se naučíte, jak ovlivnit chování a styl odpovědí modelu nastavením vybraných parametrů při volání API:

1. Volba modelu
2. Teplota (temperature)
3. Upovídanost (verbosity)
4. Hloubka přemýšlení (reasoning\_effort)

---

## 1. Volba modelu

Při návrhu aplikace využívající velký jazykový model máme na výběr mezi různými modely. Liší se:

- výrobcem,
- zaměřením (text-to-text, text-to-image, speech-to-text…),
- výkonem, rychlostí a cenou,
- schopnostmi (např. podpora tzv. reasoning – „přemýšlení“).

V našem kurzu se zaměříme na **text-to-text** modely od OpenAI. Použijeme tři varianty modelu GPT-5:

- **gpt-5** – nejvýkonnější, ale také nejpomalejší a nejdražší.
- **gpt-5-mini** – kompromis mezi výkonem a rychlostí.
- **gpt-5-nano** – nejrychlejší a nejlevnější, ale s omezenou kvalitou.

📝 **Úkol:** Spusťte stejný dotaz na všechny tři varianty, změřte dobu odezvy a porovnejte kvalitu odpovědí.

```python
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

models = ["gpt-5", "gpt-5-mini", "gpt-5-nano"]

print("\n=== Výsledky ===")
print(f"{'Model':<12} {'Čas (s)':<10} Odpověď")
for model in models:
    start = time.time()
    response = client.responses.create(
        model=model,
        input=prompt)
    duration = time.time() - start
    print(f"{model:<12} {duration:<10.2f} {response.output_text.strip()}")
```


## 2. Teplota (temperature)

Teplota určuje míru náhodnosti při generování odpovědi.

- **Nízká hodnota (např. 0.2)** → odpovědi jsou konzistentnější a více deterministické.
- **Vyšší hodnota (např. 0.8)** → odpovědi jsou různorodější, ale může stoupat riziko „halucinací“.
- Rozsah hodnot: 0–2 (doporučeno 0–1).

📝 **Úkol:** Změňte hodnotu `temperature` a pozorujte rozdíly ve výstupech.

```python
response = client.responses.create(
    model="gpt-5-mini",
    input=prompt,
    temperature=0.2
)

print("AI odpověď:")
print(response.output_text)
```

---

## 3. Upovídanost (verbosity)

Parametr `verbosity` určuje množství detailů ve výstupu. Hodnoty:

- `low` – stručné odpovědi.
- `medium` – vyvážená míra detailu.
- `high` – velmi podrobné odpovědi.

📝 **Úkol:** Vyzkoušejte různé úrovně upovídanosti pro model `gpt-5-mini`.

```python
for verbosity in ["low", "medium", "high"]:
    print(f"\n--- Verbosity: {verbosity} ---")
    response = client.responses.create(
        model="gpt-5-mini",
        input=prompt,
        text={
            "verbosity": verbosity
        }
    )
    print(response.output_text)
```

📝 **Úkol:** Přidejte do výstupu informaci o počtu vstupních a výstupních tokenů.

```python
for verbosity in ["low", "medium", "high"]:
    print(f"\n--- Verbosity: {verbosity} ---")
    start = time.time()
    response = client.responses.create(
        model="gpt-5-mini",
        input=prompt,
        text={
            "verbosity": verbosity
        }
    )
    duration = time.time() - start
    print(f"--- Duration: {duration:.2f} ---")
    print(f"--- Input tokens: {response.usage.input_tokens} ---")
    print(f"--- Output tokens: {response.usage.output_tokens} ---")
    print(f"--- Total tokens: {response.usage.total_tokens} ---")
    print(response.output_text)
```

---

## 4. Hloubka přemýšlení (`reasoning_effort`)

Parametr `reasoning_effort` určuje, kolik interního „uvažování“ (reasoning tokenů) model věnuje přípravě odpovědi:

- `minimal` – nejrychlejší volba, model provádí jen základní interní zpracování, vhodné pro jednoduché úlohy (např. extrakce, formátování), kde je klíčová rychlost a nízká cena.
- `low` – model věnuje mírnou míru analýzy; doporučeno pro běžné, relativně jednoduché případy.
- `medium` – výchozí vyvážený přístup; model věnuje přiměřené úsilí analýze i koherence.
- `high` – nejhlouběji přemýšlející přístup; model investuje více prostředků, výsledkem je typicky nejkvalitnější (a nejpomalejší) odpověď.

Každá vyšší úroveň může zlepšit kvalitu odpovědí, ale také prodloužit dobu generování a zvýšit náklady.

📝 **Úkol:** Porovnejte odpovědi pro různé úrovně `reasoning_effort` u modelu `gpt-5-mini`.

```python
for effort in ["minimal", "low", "medium", "high"]:
    print(f"\n--- Reasoning effort: {effort} ---")
    start = time.time()
    response = client.responses.create(
        model="gpt-5-mini",
        input=prompt,
        reasoning={
            "effort": effort
        }
    )
    duration = time.time() - start
    print(f"--- Duration: {duration:.2f} ---")
    print(f"--- Input tokens: {response.usage.input_tokens} ---")
    print(f"--- Output tokens: {response.usage.output_tokens} ---")
    print(f"--- Total tokens: {response.usage.total_tokens} ---")
    print(response.output_text)
```

---

## 5. Defaultní nastavení

Vraťme se ještě k volání bez jakýchkoliv parametrů a prozkoumejme defaultní nastavení.

📝 **Úkol:** Prozkoumejte výsledek zpracování modelem jeho detailním prozkoumáním a nejděte hodnoty defaultního nastavení.

```python
response = client.responses.create(
    model="gpt-5-mini",
    input=prompt
)

import json
parsed = json.loads(response.model_dump_json())
print(json.dumps(parsed, ensure_ascii=False, indent=2))
```

## 6. Kombinace parametrů

V reálném kódu reálné aplikace či služby typicky voláme model s konkrétním nastavením parametrů.
Ty můžeme kombinovat dle potřeby.

📝 **Úkol:** Zavolejte model s konkrétním nastavením parametrů a prozkoumejte výsledek, abyste ověřili, že model s nastavením pracoval.

```python
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
```

---

## Shrnutí

- **Volba modelu** ovlivňuje kvalitu, rychlost i cenu.
- **Teplota** mění míru náhodnosti výstupu.
- **Upovídanost** řídí detailnost odpovědí.
- **Hloubka přemýšlení** určuje, kolik interní analýzy model provede a jaké množství reasoning tokenů využije.

Kombinací těchto parametrů můžete doladit chování modelu pro potřeby vaší aplikace.