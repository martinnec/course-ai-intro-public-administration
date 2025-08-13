# Kurz AI pro veřejnou správu - Kapitola 8: Použití RAG s lokální cache a XML serializací

V této kapitole si ukážeme, jak využít vyhledávání služeb ve vytvořené vektorové databázi a získávání detailů o nalezených službách k vybudování plnohodnotného RAG řešení.

Předpokládáme, že už máte připravenou znalostní bázi — tedy stažené a uložené informace o službách veřejné správy, včetně jejich embeddingů v ChromaDB. To znamená, že při spuštění aplikace už nemusíme nic stahovat ze SPARQL endpointu ani počítat embeddingy. Vše se načte z **lokální cache**.

---

## 8.1 Vytvoření instance a načtení z lokální cache

Nejprve vytvoříme instanci `GovernmentServicesStore` a načteme data z cache. Jelikož embeddingy už máme spočítané, tento krok je rychlý a bez potřeby internetu.

```python
from government_services_store import GovernmentServicesStore

store = GovernmentServicesStore()
store.load_services()
```

---

## 8.2 Ověření, že máme data k dispozici

Po načtení si ověříme, kolik služeb máme k dispozici a kolik z nich má embeddingy.

```python
stats = store.get_services_embedding_statistics()
print(f"Načteno {stats['total_services']} služeb.")
print(f"Embeddings v ChromaDB: {stats['total_embeddings']} (coverage: {stats['coverage_percentage']}%)")
```

To je důležité — pokud by se coverage rovnala 0 %, znamenalo by to, že embeddingy chybí a musíme provést bootstrap podle kapitoly 7.

---

## 8.3 Sémantické vyhledávání služeb

Dotaz uživatele převedeme na embedding a v ChromaDB vyhledáme nejpodobnější služby. Tím získáme jen relevantní podmnožinu služeb.

```python
user_query = "Bolí mě hlava a mám asi horečku. Co si na to mám vzít? Co mám dělat? A mohu jít do práce?"
results = store.search_services(user_query, k=10)
for s in results:
    print(f"{s.id}: {s.name}")
```

---

## 8.4 Příprava na volání LLM

Vložíme kód pro incializaci OpenAI klienta a pro definici datových struktur popisujících očekávaný strukturovaný výstup.
Je to stejné, jako v předchozích kapitolách.

```python
from openai import OpenAI
from dotenv import load_dotenv
import os
from pydantic import BaseModel, Field

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("API klíč není nastaven v .env souboru.")

client = OpenAI(api_key=api_key)

class KrokPostupu(BaseModel):
    poradi: int = Field(description="Pořadí kroku v postupu")
    nazev: str = Field(description="Název kroku, který stručně popisuje, co je potřeba udělat")
    popis: str = Field(description="Podrobný popis kroku, který uživateli vysvětluje, co má dělat")
    sluzba_id: str = Field(description="Odkaz na ID služby, ze které tento krok vyplývá")

class Postup(BaseModel):
    uvod: str = Field(description="Úvodní text k návrhu řešení dané životní situace")
    kroky: list[KrokPostupu] = Field(description="Uspořádaný seznam kroků, které je potřeba provést")
```

---

## 8.5 Serializace výsledků do XML

Oproti kapitole 6, kde jsme služby serializovali do **JSON** formátu, zde použijeme **XML**. To je plně podporované a některé studie naznačují, že pro LLM může být XML dokonce lepší — zejména proto, že má jasně vymezené značky a je pro model snadno parsovatelné.

```python
if results:
    sluzby_xml = "<sluzby>\n"
    
    for service in results:
        sluzby_xml += f"  <sluzba>\n"
        sluzby_xml += f"    <id>{service.id}</id>\n"
        sluzby_xml += f"    <nazev>{service.name}</nazev>\n"
        sluzby_xml += f"    <popis>{service.description}</popis>\n"
        
        detail = store.get_service_detail_by_id(service.id)
        if detail:
            sluzby_xml += f"    <detail>{detail}</detail>\n"
        
        steps = store.get_service_steps_by_id(service.id)
        if steps:
            sluzby_xml += f"    <kroky>\n"
            for step in steps:
                sluzby_xml += f"      <krok>{step}</krok>\n"
            sluzby_xml += f"    </kroky>\n"
        
        sluzby_xml += f"  </sluzba>\n"
    
    sluzby_xml += "</sluzby>"
else:
    print("Žádné služby nenalezeny, asistent vám bohužel nemůže pomoci")
```

---

## 8.6 Použití XML kontextu v promptu

XML můžeme vložit přímo do systémové nebo uživatelské zprávy pro LLM. Díky značkám `<service>`, `<id>`, `<name>` a `<description>` má model jasné oddělení jednotlivých částí.

```python
if results:
    response = client.responses.parse(
        model="gpt-5-mini",
        input=[
            {
                "role": "developer",
                "content": "Jsi odborník na pomoc uživateli při řešení jeho různých životních situací v občanském životě. Vždy poradíš, jak danou životní situaci vyřešit z úředního hlediska poskytnutím konkrétního postupu v podobě číslovaných kroků. Uživatel potřebuje srozumitelné ale krátké vysvětlení každého kroku jednoduchou češtinou."
            },{
                "role": "developer",
                "content": "Odpovídej *VÝHRADNĚ* na základě přiloženého XML se seznamem služeb (viz <sluzby> dále). Každá služba je uvedena ve struktuře <id> (jednoznačný identifikátor služby), <nazev> (krátký název služby), <popis> (delší popis služby), <detail> (detailní popis služby) a <kroky> (úřední kroky, v rámci kterých je potřeba službu řešit). Kroky služeb nemusíš v popisu striktně dodržovat, ale zkombinuj je tak, aby dávaly v kontextu situace uživatele smysl."
            },{
                "role": "developer",
                "content": "*Nikdy nepoužívej žádné znalosti mimo ty uvedené v XML*. Vždy je ale tvým hlavním cílem postup v informacích o službách v přiloženém XML zjistit a uživateli vysvětlit. Může se ale stát, že XML neobsahuje informace o žádných relevantních službách. V takovém případě *NESMÍŠ* poskytnout žádné kroky, tj. seznam kroků bude prázdný, a v úvodu *MUSÍŠ* výslovně napsat, že nemáš potřebné informace a tedy neposkytuješ žádný návod ani postup."
            },{
                "role": "developer",
                "content": "*Nikdy nesmíš v žádném kroku postupu poskytovat radu v oboru, kterého se dotaz uživatele týká, např. lékařské rady, stavební rady, atd.* Uživateli pouze můžeš napsat, aby odborníka vyhledal a navštívil bez jakýchkoliv časových, situačních či jiných podmínek a doporučení."
            },{
                "role": "developer",
                "content": sluzby_xml
            },{
                "role": "user",
                "content": user_query
            }
        ],
        text={"verbosity": "medium"},
        reasoning={"effort": "minimal"},
        text_format=Postup
    )
```

## 8.7 Vypsání výsledku

Nakonec vypíšeme výsledek, což je opět stejné, jako dříve.

```python
if results:
    print("AI odpověď:")
    postup = response.output_parsed
    print(f"\n{postup.uvod}\n")
    for krok in postup.kroky:
        print(f"{krok.poradi}. {krok.nazev} ({krok.sluzba_id})\n   {krok.popis}\n")
```

---

### 8.8 Prázdný výsledek

Pro uživatelský dotaz *Bolí mě hlava a mám asi horečku. Co si na to mám vzít? Co mám dělat? A mohu jít do práce?* pravděpodobně model odpověděl, že nenašel žádné relevantní služby a neposkytl žádný postup. Sice se podařilo najít nějaké služby, ale model usoudil, že nejsou pro uživatelský dotaz relevantní. Je to způsobeno tím, že služby nejsou v oblasti nemocenské a neschopenek popsány moc dobře. Zkuste si sami, zda se vám podaří ručně nějaké relevantní služby najít. **Vidíme tak, jak silně jsme závislí na kvalitě podkladových dat!**

Můžete zkusit experimentovat s parametry ve skriptu:
- Zvyšte počet služeb, které ve vektorové databázi hledáte, např. na 20 nebo 25. Jak jsme si už říkali, kontextová okna jsou dost velká na to, aby se tam takový počet služeb se svými detailními popisy vešel. Čím více dat ale do promptu vložíme, tím více platí rizika spojená s vkládáním velkého znalostního kontextu do promptu (viz úvod ke kapitole 7).
- Použijte lepší model, např. `gpt-5` místo `gpt-5-mini`.
- Zvyšte parametry `verbosity` a `reasoning-effort`.

### 8.9 Příklady dalších uživatelských dotazů.

Experimentujte s dalšími dotazy, např. níže uvedenými. Nebo si vymyslete vlastní.

```python
user_query = "Bolí mě hlava a mám asi horečku. Co si na to mám vzít? Co mám dělat? A mohu jít do práce?"
#user_query = "Jsem OSVČ a jsem nemocný. Můžete mi pomoct?"
#user_query = "Začal jsem stavět garáž na mém pozemku, ale soused mě vynadal, že stavím bez povolení. Nic takového jsem nevyřizoval, nevím zda je to potřeba. Co mám dělat?"
#user_query = "Bojím se, že moje dítě není ještě připraveno na základní školu. Je nějaká možnost odkladu nebo přípravy?"
#user_query = "Starám se sama o dvě malé děti. Vyhodili mě z nájmu v bytu a už nemám peníze ani na jídlo."
```

---

## Shrnutí

- Oproti kapitole 7 nyní **neprovádíme bootstrap**, ale načítáme z lokální cache.
- Stejně jako v kapitole 6, vkládáme informace o službách přímo do promptu. Nyní jsme ale využili vektorovou databázi k vyhledání služeb relevantních pro uživatelský dotaz ve všech známých službách veřejné správy.
- Vyhledané služby **serializujeme do XML** namísto JSON.
- XML může být pro LLM přehlednější a usnadnit mu práci s daty.
- Pokud model vrátí pro daný uživatelský dotaz prázdný postup, můžeme zkusit volat znovu se silnějším nastavením parametrů (širší znalostní báze ~ více služeb, lepší model, vyšší upovídanost a úroveň přemýšlení).

