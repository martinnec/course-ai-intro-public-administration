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

user_query = "Bolí mě hlava a mám asi horečku. Co si na to mám vzít? Co mám dělat? A mohu jít do práce?"
#user_query = "Jsem OSVČ a jsem nemocný. Můžete mi pomoct?"
#user_query = "Začal jsem stavět garáž na mém pozemku, ale soused mě vynadal, že stavím bez povolení. Nic takového jsem nevyřizoval, nevím zda je to potřeba. Co mám dělat?"
#user_query = "Bojím se, že moje dítě není ještě připraveno na základní školu. Je nějaká možnost odkladu nebo přípravy?"
#user_query = "Starám se sama o dvě malé děti. Vyhodili mě z nájmu v bytu a už nemám peníze ani na jídlo."
results = store.search_services(user_query, k=10)
for s in results:
    print(f"{s.id}: {s.name}")

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

if results:
    sluzby_xml = "<sluzby>\n"
    
    for service in results:
        sluzby_xml += f"  <sluzba>\n"
        sluzby_xml += f"    <id>{service.id}</id>\n"
        sluzby_xml += f"    <nazev>{service.name}</nazev>\n"
        
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

if results:
    response = client.responses.parse(
        model="gpt-5-mini",
        input=[
            {
                "role": "developer",
                "content": "Jsi odborník na pomoc uživateli při řešení jeho různých životních situací v občanském životě. Vždy poradíš, jak danou životní situaci vyřešit z úředního hlediska poskytnutím konkrétního postupu v podobě číslovaných kroků. Uživatel potřebuje srozumitelné ale krátké vysvětlení každého kroku jednoduchou češtinou."
            },{
                "role": "developer",
                "content": "Odpovídej *VÝHRADNĚ* na základě přiloženého XML se seznamem služeb (viz <sluzby> dále). Každá služba je uvedena ve struktuře: <id> (jednoznačný identifikátor služby), <nazev> (krátký název služby), <detail> (detailní informace o službě) a <kroky> (úřední kroky, v rámci kterých je potřeba službu řešit). <detail> se skládá z následujícíh částí: <popis> (Základní vymezení služby a upřesnění názvu, pokud není dost jednoznačný.), <benefit> (Atribut popisuje, jaký přínos má pro klienta využití služby.), <jak-resit> (Jakým způsobem se služba řeší elektronicky včetně případného ID datové schránky, mailové adresy či jiných digitálních kanálů.), <kdy-resit> (Popisuje, v jakou chvíli může nebo musí být iniciováno čerpání služby.), <resit-pokud> (Vymezení toho, kdo může službu využívat a za jakých podmínek se ho týká.), <zpusob-vyrizeni> (Co potřebuje klient, aby mohl službu řešit elektronicky (typicky doklady, žádosti apod.)"
            },{
                "role": "developer",
                "content": "Kroky služeb nemusíš v popisu striktně dodržovat, ale zkombinuj je tak, aby dávaly v kontextu situace uživatele smysl."
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

    print("AI odpověď:")
    postup = response.output_parsed
    print(f"\n{postup.uvod}\n")
    for krok in postup.kroky:
        print(f"{krok.poradi}. {krok.nazev} ({krok.sluzba_id})\n   {krok.popis}\n")