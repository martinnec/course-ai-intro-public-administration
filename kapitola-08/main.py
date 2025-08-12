from openai import OpenAI
from dotenv import load_dotenv
import os
from pydantic import BaseModel, Field
import json
from government_services_store import GovernmentServicesStore

store = GovernmentServicesStore()
store.load_services()

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

user_query = "Bolí mě hlava a mám asi horečku. Co si na to mám vzít? Co mám dělat? A mohu jít do práce?"
#user_query = "Jsem OSVČ a jsem nemocný. Jak získám neschopenku a nemocenskou?"
#user_query = "Začal jsem stavět garáž na mém pozemku, ale soused mě vynadal, že stavím bez povolení. Nic takového jsem nevyřizoval, nevím zda je to potřeba. Co mám dělat?"
#user_query = "Bojím se, že moje dítě není ještě připraveno na základní školu. Je nějaká možnost odkladu nebo přípravy?"

results = store.search_services(user_query, k=10)
if results:
    # Construct XML for all services
    sluzby_xml = "<sluzby>\n"
    
    for service in results:
        sluzby_xml += f"  <sluzba>\n"
        sluzby_xml += f"    <id>{service.id}</id>\n"
        sluzby_xml += f"    <nazev>{service.name}</nazev>\n"
        sluzby_xml += f"    <popis>{service.description}</popis>\n"
        
        # Get additional details for this service
        detail = store.get_service_detail_by_id(service.id)
        if detail:
            sluzby_xml += f"    <detail>{detail}</detail>\n"
        
        # Get steps for this service
        steps = store.get_service_steps_by_id(service.id)
        if steps:
            sluzby_xml += f"    <kroky>\n"
            for step in steps:
                sluzby_xml += f"      <krok>{step}</krok>\n"
            sluzby_xml += f"    </kroky>\n"
        
        sluzby_xml += f"  </sluzba>\n"
    
    sluzby_xml += "</sluzby>"

    # Zavoláme OpenAI model
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

else:
    print("Žádné služby nenalezeny.")

