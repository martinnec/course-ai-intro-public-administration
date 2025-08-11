# Kurz AI pro veřejnou správu - Kapitola 6: Práce s externími znalostmi

V této kapitole si krok za krokem ukážeme, jak dodat modelu **konkrétní znalostní kontext** z externích dat (např. otevřených dat státu) tak, aby odpovídal **výhradně z těchto zdrojů** a neimprovizoval z vlastních, potenciálně zastaralých znalostí. Naučíte se:

- kde vzít data a jak je připravit,
- jak je bezpečně předat do promptu,
- proč použít roli `developer` a jak přesně formulovat omezení,
- jak rozšířit návratovou strukturu o odkazy na použité služby

---

## 1. Proč dodávat modelu externí znalosti

V předchozích kapitolách jsme model učili *jak* odpovídat (styl, formát, priority), ale **odkud** čerpal fakta, určoval jeho vlastní trénink. To je problém, pokud potřebujete:

- **aktuálnost** (zákony, formuláře, postupy se mění),
- **lokálně specifické informace** (české veřejné služby),
- **konzistenci** s oficiálním zdrojem.

Řešením je tzv. *context injection*: předáte modelu **referenční data (znalosti)** přímo v promptu a omezíte ho, aby odpovídal jen podle nich.

---

## 2. Získání a příprava dat

1. Stáhněte JSON a vyfiltrujte služby relevantní k dané situaci.
2. Pro každou službu vyberte relevantní informace.
3. Udržujte konzistentní pole (např. `id`, `nazev`, `popis`, `benefit`, `jak-resit`, `kdy-resit`, `resit-pokud`).
4. Uložte jako `data/sluzby_data.json`.

**Příklad struktury:**

```json
[
  {
    "id": "sluzba-123",
    "nazev": "Vystavení potvrzení o pracovní neschopnosti",
    "popis": "Služba umožňuje ...",
    "benefit": "Zajištění nároku na nemocenskou ...",
    "jak-resit": "Navštivte svého lékaře ...",
    "kdy-resit": "Při vzniku pracovní neschopnosti ...",
    "resit-pokud": "Pokud jste nemocní a nemůžete do práce ..."
  }
]
```

> **Poznámka k velikosti:** Pokud je JSON dlouhý, zvažte výběr jen nejdůležitějších položek, případně před-sumarizaci. Důležité je zachovat **identifikátory služeb** (`id`), ke kterým budeme v odpovědi odkazovat.

## 3. Načtení dat v Pythonu

Nejprve tato data načteme do proměnné a převedeme je na JSON řetězec, který vložíme do promptu.

```python
import json

# Načteme data služeb ze souboru
with open("data/sluzby_data.json", "r", encoding="utf-8") as f:
    sluzby = json.load(f)

sluzby_json = json.dumps(sluzby, ensure_ascii=False)
```

---

## 4. Instruování modelu pro práci s externím JSON

V této části je klíčové použít roli `developer`. Proč?

- Zprávy v roli `developer` mají **vyšší prioritu** než `user` a pomáhají tak odolávat pokusům uživatele instrukce obejít.
- Můžeme je použít vícekrát a každou zprávu věnovat jiné zásadě: *kdo jsme a jak odpovídáme*, *přesná pravidla omezení znalostí*, *bezpečnostní zásady*, *vložená data*.

**Co dělají jednotlivé zprávy níže:**

1. **Definují roli a formát odpovědi** – stručné kroky, jednoduchá čeština.
2. **Zakazují čerpání mimo JSON** – výraz „*VÝHRADNĚ*“ je záměrně zvýrazněn, aby model bral omezení vážně; současně vysvětluje **strukturu JSON** (aby věděl, co v datech hledat).
3. **Bezpečnostní mantinely** – zákaz odborných rad mimo rámec veřejné správy.
4. **Předávají data** – JSON vkládáme „in-line“ mezi značky `<JSON-SLUZBY>…</JSON-SLUZBY>`; je to jasný ukotvující kontext, na který se můžeme v instrukcích odkazovat.
5. **Dotaz uživatele** – až poté přichází `user` zpráva s konkrétním problémem.

```python
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
```

---

## 5. Rozšíření výstupního schématu o odkaz na službu

Chceme, aby každý krok obsahoval odkaz na zdrojovou službu. Přidáme tedy `sluzba_id`.

```python
class KrokPostupu(BaseModel):
    poradi: int = Field(description="Pořadí kroku v postupu")
    nazev: str = Field(description="Název kroku")
    popis: str = Field(description="Podrobný popis kroku")
    sluzba_id: str = Field(description="Odkaz na ID služby")
```

> **Kontrola konzistence:** Po obdržení odpovědi si v aplikaci ověřte, že vrácená `sluzba_id` skutečně existují v `sluzby_data.json`. Pokud ne, krok vyřaďte nebo požádejte model o doplnění s přiloženým logem chyby.

---

## 6. Kompletní ukázka kódu

Následující ukázka spojuje načtení JSON, instrukce, parsování do Pydantic tříd a výpis výsledku.

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
```

---

## 7. Experimentujte a testujte odolnost

Vyzkoušejte různé modely (`gpt-5`, `gpt-5-mini`) a parametry (`text.verbosity`, `reasoning.effort`). Sledujte:

- zda model **nevybočuje** z JSON,
- jak podrobné odpovědi poskytuje,
- zda správně uvádí `sluzba_id` a zda se shodují s přiloženými službami.

> **Tip:** Při zjištění vybočení přidejte přísnější `developer` věty (např. „Každý krok MUSÍ vycházet z konkrétní položky v JSON a uvést `sluzba_id`. Pokud takovou službu nenalezneš, krok nevracej.“).

---

## Shrnutí

- Externí data vložená do promptu tvoří **zdroj pravdy**; pomocí role `` zajistíte, že mají **vyšší prioritu** než přání uživatele.
- Popis **struktury JSON** přímo v instrukcích zvyšuje šanci, že model data správně přečte a použije.
- Rozšířením výstupního schématu o `sluzba_id` lze každé doporučení **doložit** konkrétní službou (auditovatelnost).
- Pravidelně testujte „vybočení“ a dolaďujte instrukce; u větších JSONů zvažte výběr/sumarizaci dat, abyste zůstali pod limitům tokenů.