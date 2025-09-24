# Kapitola 9 – První AI agent

V předchozí kapitole jsme si ukázali, že dokážeme zlepšit odpověď uživateli, pokud modelu dodáme správný znalostní kontext – konkrétně seznam služeb veřejné správy, které se vztahují k uživatelově životní situaci. To ale pořád znamenalo, že jsme uživatelský dotaz přímo použili k vyhledávání.

Nyní uděláme krok dál. V této kapitole se seznámíme s pojmem **softwarový agent** a vyzkoušíme si postavit svého prvního jednoduchého agenta, který bude uživateli pomáhat komplexněji.

---

## Co je softwarový agent?

Softwarový agent je program, který **vykonává komplexnější úkol autonomně** – podobně, jako by to udělal člověk. Agent:
- dostane úkol (například „pomoz mi vyřešit mou životní situaci“),
- rozdělí si ho na dílčí kroky,
- využívá k tomu dostupné nástroje (např. vyhledávání služeb),
- a nakonec uživateli předá výsledek.

Agent tedy není jenom funkce, která vrátí výstup – je to koordinátor, který dokáže řídit jednotlivé části řešení a sám si rozhoduje, jaké akce provést.

---

## Co bude dělat náš agent?

Agent v této kapitole bude mít za úkol uživateli pomoci se životní situací, podobně jako v kapitole 8. Tentokrát ale postupuje chytřeji:

1. **Samostatně formuluje vhodné vyhledávací dotazy** – nevyhledáváme přímo textem od uživatele, ale agent si vygeneruje několik variant dotazů, které pokrývají různé aspekty situace.
2. **Hledá služby** pomocí těchto dotazů.
3. **Filtruje služby podle relevance** vůči životní situaci.
4. **Generuje finální doporučený postup**, jak má uživatel situaci řešit.

Pokud dokážeme tento postup popsat krok za krokem, můžeme ho naprogramovat. Každá část pak bude funkce, kterou agent zavolá.

---

## Jak bude vypadat program?

Celý agent bude poskládaný z několika kroků:

1. **Generování vyhledávacích dotazů** – modelu předáme uživatelův popis situace a on navrhne 5–7 krátkých dotazů.
2. **Vyhledání služeb** – pro každý dotaz se vyhledají odpovídající služby z našeho úložiště.
3. **Filtrování služeb** – model vyhodnotí, jestli je daná služba skutečně relevantní k situaci uživatele.
4. **Vytvoření finálního postupu** – model zkombinuje dostupné služby a jejich kroky do přehledného návodu.

---

## Postupná stavba agenta

Pojďme nyní jednotlivé části naprogramovat.

Nejprve krátká příprava, potřebujeme importovat knihovny, zavést naše úložiště služeb pro vyhledávání z předchozích kapitol, nahrát služby a také OpenAI API klíč a zkontruovat OpenAI API klienta.

```python
from typing import List
from government_services_store import GovernmentServicesStore
from openai import OpenAI
from dotenv import load_dotenv
import os
from pydantic import BaseModel, Field

store = GovernmentServicesStore()
store.load_services()

stats = store.get_services_embedding_statistics()
print(f"Načteno {stats['total_services']} služeb.")
print(f"Embeddings v ChromaDB: {stats['total_embeddings']} (coverage: {stats['coverage_percentage']}%)")

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("API klíč není nastaven v .env souboru.")

client = OpenAI(api_key=api_key)
```

Také znovu použijeme náš zásobník předpřipravených životních situací.

```python
#user_query = "Bolí mě hlava a mám asi horečku. Co mám dělat? A mohu jít do práce?"
user_query = "Jsem OSVČ a jsem nemocný. Můžete mi pomoct?"
#user_query = "Začal jsem stavět garáž na mém pozemku, ale soused mě vynadal, že stavím bez povolení. Nic takového jsem nevyřizoval, nevím zda je to potřeba. Co mám dělat?"
#user_query = "Bojím se, že moje dítě není ještě připraveno na základní školu. Je nějaká možnost odkladu nebo přípravy?"
#user_query = "Starám se sama o dvě malé děti. Vyhodili mě z nájmu v bytu a už nemám peníze ani na jídlo."
```

Dále budeme pracovat s datovými strukturami, které si potřebujeme zadefinovat. Vysvětlíme si je později.

```python
# Definice datových struktur pro parsování odpovědí
class KrokPostupu(BaseModel):
    poradi: int = Field(description="Pořadí kroku v postupu")
    nazev: str = Field(description="Název kroku, který stručně popisuje, co je potřeba udělat")
    popis: str = Field(description="Podrobný popis kroku, který uživateli vysvětluje, co má dělat")
    sluzba_id: str = Field(description="Odkaz na ID služby, ze které tento krok vyplývá")

class Postup(BaseModel):
    uvod: str = Field(description="Úvodní text k návrhu řešení dané životní situace")
    kroky: list[KrokPostupu] = Field(description="Uspořádaný seznam kroků, které je potřeba provést")

class NavrzenaVyhledavani(BaseModel):
    dotazy: list[str] = Field(description="Seznam fulltextových dotazů pro vyhledání relevantních služeb")
```

### 1. Generování vyhledávacích dotazů

Nejprve vytvoříme funkci, která z uživatelského dotazu připraví seznam dotazů pro vyhledávání služeb:

```python
def generuj_navrhy_vyhledavacich_dotazu(user_query: str) -> List[str]:

    response = client.responses.parse(
        model="gpt-5-mini",
        input=[
            {
                "role": "developer",
                "content": "Jsi odborník na vyhledávání služeb veřejné správy vhodných pro řešení životních situací občanů v jejich občanském životě. Uživatel popíše svojí životní situaci. Na základě popisu navrhni seznam full-textových dotazů, kterými můžeme vhodné služby najít. Dotazů zkonstruuj 5-7, aby si pokryl různé části životní situace. V životních situacích ignoruj žádosti o odborné rady (např. lékařské, stavební) a soustřeď se pouze na úřední pomoc. V dotazech preferuj zobecnění konkrétních doménově specifických pojmů do obecnějších úředních pojmů (např. 'Potřebuji opravit rozbitou brzdu v autě' -> 'porucha vozidla oprava', 'Ztratil jsem občanku a potřebuji si zařídit novou.' -> 'vydání nových dokladů totožnosti.', 'Potřebuji si nechat schválit vystavění zdi kolem mé zahrádky.' -> 'získání stavebního povolení'). Každý výsledný dotaz by měl být krátký, ideálně do 7 slov. Dotazy musí být v češtině."
            },{
                "role": "assistant",
                "content": "Jaká je vaše životní situace?"
            },{
                "role": "user",
                "content": user_query
            }
        ],
        text={"verbosity": "low"},
        reasoning={"effort": "medium"},
        text_format=NavrzenaVyhledavani
    )

    navrzena_vyhledavani = response.output_parsed
    return navrzena_vyhledavani.dotazy
```

Tato funkce vrátí seznam krátkých dotazů v češtině, které agent dále použije.

---

### 2. Vyhledání služeb

Každý z dotazů použijeme k vyhledání služeb v našem úložišti. Funkce vrátí slovník služeb:

```python
def vyhledej_sluzby(dotazy: List[str]) -> dict:  
    sluzby = {}
    for dotaz in dotazy:
        results = store.search_services(dotaz, k=3)
        for sluzba in results:
            sluzby[sluzba.id] = sluzba
    return sluzby
```

---

### 3. Filtrování služeb

Aby se do výsledku nedostaly nesouvisející služby, agent každou posoudí jazykovým modelem. Pokud model odpoví `TRUE`, služba je považována za relevantní:

```python
def filtruj_relevantni_sluzby(sluzby: dict, user_query: str) -> dict:
    
    filtrovany_seznam_sluzeb = {}
    if sluzby not in (None, {}):
        for sluzba in sluzby.values():
            sluzby_xml = f"<sluzba>\n"
            sluzby_xml += f"  <id>{sluzba.id}</id>\n"
            sluzby_xml += f"  <nazev>{sluzba.name}</nazev>\n"
            sluzby_xml += f"  <popis>{sluzba.description}</popis>\n"        
            sluzby_xml += f"</sluzba>\n"


            response = client.responses.create(
                model="gpt-5-mini",
                input=[
                    {
                        "role": "developer",
                        "content": "Jsi odborník na pomoc uživateli při řešení jeho různých životních situací v občanském životě. Uživatel popíše svojí životní situaci. Ty srovnáš jeho situaci s danou službou veřejné správy a posoudíš její relevanci. Služba je relevantní, pokud je splněna jedna z následujících podmínek: 1) služba přímo umožní řešit popsanou situaci nebo její část, 2) služba se nějak dotýká potřeb občana, které souvisejí se situací nebo alespoň s její částí, 3) služba by mohla být pro občana v rámci jeho životní situace užitečná, i když ji přímo nevyužije."
                    },{
                        "role": "developer",
                        "content": "Odpověz *VÝHRADNĚ* hodnotou TRUE/FALSE. TRUE znamená, že služba je relevantní, FALSE že není. Jiný výstup než TRUE nebo FALSE není přípustný."
                    },{
                        "role": "assistant",
                        "content": "Jaká je vaše životní situace?"
                    },{
                        "role": "user",
                        "content": user_query
                    },{
                        "role": "assistant",
                        "content": "Jakou službu mám posoudit?"
                    },{
                        "role": "user",
                        "content": f"- název služby: {sluzba.name}\n-popis služby: {sluzba.description}"
                    }
                ],
                text={"verbosity": "low"},
                reasoning={"effort": "low"}
            )

            posouzeni_relevance = response.output_text
            print(f"- {sluzba.name} {sluzba.id}: {posouzeni_relevance}")
            if posouzeni_relevance == "TRUE":
                filtrovany_seznam_sluzeb[sluzba.id] = sluzba

    return filtrovany_seznam_sluzeb
```

---

### 4. Vygenerování finálního postupu

Nakonec připravíme postup pro uživatele. Model dostane seznam relevantních služeb, jejich detaily a případné kroky a z nich sestaví přehledný návod:

```python
def vygeneruj_finalni_postup(sluzby: dict, user_query: str) -> Postup:

    sluzby_xml = "<sluzby>\n"

    for sluzba in sluzby.values():
        sluzby_xml += f"  <sluzba>\n"
        sluzby_xml += f"    <id>{sluzba.id}</id>\n"
        sluzby_xml += f"    <nazev>{sluzba.name}</nazev>\n"
        
        detail = store.get_service_detail_by_id(sluzba.id)
        if detail:
            sluzby_xml += f"    <detail>{detail}</detail>\n"
        
        steps = store.get_service_steps_by_id(sluzba.id)
        if steps:
            sluzby_xml += f"    <kroky>\n"
            for step in steps:
                sluzby_xml += f"      <krok>{step}</krok>\n"
            sluzby_xml += f"    </kroky>\n"
        
        sluzby_xml += f"  </sluzba>\n"
    
    sluzby_xml += "</sluzby>"

    response = client.responses.parse(
        model="gpt-5-mini",
        input=[
            {
                "role": "developer",
                "content": "Jsi odborník na pomoc uživateli při řešení jeho různých životních situací v občanském životě. Vždy poradíš, jak danou životní situaci vyřešit z úředního hlediska poskytnutím konkrétního postupu v podobě číslovaných kroků na základě popisů služeb veřejné správy, které lze pro řešení životní situace použít. Uživatel potřebuje srozumitelné ale krátké vysvětlení každého kroku jednoduchou češtinou."
            },{
                "role": "developer",
                "content": "Odpovídej *VÝHRADNĚ* na základě strukturovaného seznamemu služeb dodaného uživatelem. Každá služba je uvedena ve struktuře: <id> (jednoznačný identifikátor služby), <nazev> (krátký název služby), <detail> (detailní informace o službě) a <kroky> (úřední kroky, v rámci kterých je potřeba službu řešit). <detail> se skládá z následujícíh částí: <popis> (Základní vymezení služby a upřesnění názvu, pokud není dost jednoznačný.), <benefit> (Atribut popisuje, jaký přínos má pro klienta využití služby.), <jak-resit> (Jakým způsobem se služba řeší elektronicky včetně případného ID datové schránky, mailové adresy či jiných digitálních kanálů.), <kdy-resit> (Popisuje, v jakou chvíli může nebo musí být iniciováno čerpání služby.), <resit-pokud> (Vymezení toho, kdo může službu využívat a za jakých podmínek se ho týká.), <zpusob-vyrizeni> (Co potřebuje klient, aby mohl službu řešit elektronicky (typicky doklady, žádosti apod.)"
            },{
                "role": "developer",
                "content": "Kroky služeb nemusíš ve výsledném postupu striktně dodržovat, ale zkombinuj je tak, aby dávaly v kontextu situace uživatele smysl."
            },{
                "role": "developer",
                "content": "*Nikdy nepoužívej žádné znalosti mimo ty uvedené v seznamu*. Vždy je ale tvým hlavním cílem postup v uvedných informacích o službách zjistit a uživateli vysvětlit. Může se ale stát, že nejsou přiloženy informace o žádných relevantních službách. V takovém případě *NESMÍŠ* poskytnout žádné kroky, tj. seznam kroků bude prázdný, a v úvodu *MUSÍŠ* výslovně napsat, že nemáš potřebné informace a tedy neposkytuješ žádný návod ani postup."
            },{
                "role": "developer",
                "content": "*Nikdy nesmíš v žádném kroku výsledného postupu poskytovat radu v oboru, kterého se životní situace uživatele týká, např. lékařské rady, stavební rady, atd.* Uživateli pouze můžeš napsat, aby odborníka vyhledal a navštívil bez jakýchkoliv časových, situačních či jiných podmínek a doporučení."
            },{
                "role": "assistant",
                "content": "Jaká je vaše životní situace?"
            },{
                "role": "user",
                "content": user_query
            },{
                "role": "assistant",
                "content": "Jaké služby mám při přípravě postupu použít?"
            },{
                "role": "user",
                "content": sluzby_xml
            }
        ],
        text={"verbosity": "medium"},
        reasoning={"effort": "medium"},
        text_format=Postup
    )

    return response.output_parsed
```

Výstupem je struktura `Postup`, která obsahuje úvodní text a seznam kroků s odkazy na příslušné služby.

---

## Finální spuštění agenta

Všechny funkce spojíme do hlavního běhu programu:

```python
def main():
    navrh_dotazy = generuj_navrhy_vyhledavacich_dotazu()
    if navrh_dotazy is None:
        print("Nepodařilo se vygenerovat vyhledávací dotazy.")
        return

    print("Navržené dotazy pro vyhledání služeb:")
    for dotaz in navrh_dotazy:
        print(f"- {dotaz}")

    sluzby = vyhledej_sluzby(navrh_dotazy)
    if not sluzby:
        print("Nebyly nalezeny žádné služby.")
        return
    
    print(f"Nalezeno služeb: {len(sluzby)}")

    filtrovane_sluzby = filtruj_relevantni_sluzby(sluzby, user_query)
    if not filtrovane_sluzby:
        print("Nebyly nalezeny žádné relevantní služby.")
        return
    
    print(f"Nalezeno relevantních služeb: {len(filtrovane_sluzby)}")

    postup = vygeneruj_finalni_postup(filtrovane_sluzby, user_query)
    if not postup or not postup.uvod:
        print("Nepodařilo se vygenerovat postup.")
    print("AI odpověď:")
    print(f"{postup.uvod}\n")
    if postup.kroky and len(postup.kroky) > 0:
        for krok in postup.kroky:
            print(f"{krok.poradi}. {krok.nazev} ({krok.sluzba_id})\n   {krok.popis}\n")
```

Po spuštění program načte popis životní situace od uživatele, agent ji zpracuje a uživateli vrátí srozumitelný postup.

---

## Shrnutí

- Poznali jsme pojem **softwarový agent** - program, který autonomně řeší komplexní úkol.
- Náš první agent pomáhá uživateli se životní situací tak, že:
  1. generuje vyhledávací dotazy,
  2. hledá služby,
  3. filtruje je podle relevance,
  4. a sestaví přehledný postup.
- Každý krok jsme si naprogramovali jako samostatnou funkci a pak je spojili do jednoho celku.

Tímto jsme položili základ pro další rozvoj – agent může být časem rozšířen o více nástrojů a větší míru autonomie.

