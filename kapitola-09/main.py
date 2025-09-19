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

#user_query = "Bolí mě hlava a mám asi horečku. Co si na to mám vzít? Co mám dělat? A mohu jít do práce?"
#user_query = "Jsem OSVČ a jsem nemocný. Můžete mi pomoct?"
user_query = "Začal jsem stavět garáž na mém pozemku, ale soused mě vynadal, že stavím bez povolení. Nic takového jsem nevyřizoval, nevím zda je to potřeba. Co mám dělat?"
#user_query = "Bojím se, že moje dítě není ještě připraveno na základní školu. Je nějaká možnost odkladu nebo přípravy?"
#user_query = "Starám se sama o dvě malé děti. Vyhodili mě z nájmu v bytu a už nemám peníze ani na jídlo."

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("API klíč není nastaven v .env souboru.")

client = OpenAI(api_key=api_key)

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


def generuj_navrhy_vyhledavacich_dotazu() -> List[str]:

    response = client.responses.parse(
        model="gpt-5-mini",
        input=[
            {
                "role": "developer",
                "content": "Jsi odborník na vyhledávání služeb veřejné správy vhodných pro řešení životních situací občanů v jejich občanském životě. Uživatel popíše svojí životní situaci. Na základě popisu navrhni seznam full-textových dotazů, kterými můžeme vhodné služby najít. Dotazů zkonstruuj více, aby si pokryl různé části životní situace z různých úhlů pohledu (samotná životní situace, problémy občana v rámci životní situace, související úřední postupy, zobecnění od konkrétních pojmů volených uživatelem do obecnějších které se pravděpodobnějí používají v popisech služeb). Pro danou část a úhel pohledu zvol různé styly formulace (legislativní, úředně-procesní, laický). Každý výsledný dotaz by měl být krátký, ideálně do 7 slov. Dotazy musí být v češtině."
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

def vyhledej_sluzby(dotazy: List[str]) -> dict:  
    sluzby = {}
    for dotaz in dotazy:
        results = store.search_services(dotaz, k=3)
        for sluzba in results:
            sluzby[sluzba.id] = sluzba
    return sluzby

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
                        "content": "Jsi odborník na pomoc uživateli při řešení jeho různých životních situací v občanském životě. Uživatel popíše svojí životní situaci. Ty srovnáš jeho situaci s danou službou veřejné správy a posoudíš její relevanci. Služba je relevantní, pokud je splněna jedna z následujících podmínek: 1) služba přímo umožní řešit popsanou situaci nebo její část, 2) služba se nějak dotýká potřeb občana, které souvisejí se situací nebo alespoň s její částí, 3) povědomí o službě by mohlo být pro občana v rámci jeho životní situace užitečné, i když ji přímo nevyužije."
                    },{
                        "role": "developer",
                        "content": "Odpověz *VÝHRADNĚ* hodnotou TRUE/FALSE. TRUE znamená, že služba je relevantní, FALSE že není. Jiný výstup než TRUE nebo FALSE není přípustný."
                    },{
                        "role": "developer",
                        "content": f"Hodnoť pouze relevanci následující služby:\nnázev služby: {sluzba.name}\npopis služby: {sluzba.description}"
                    },{
                        "role": "user",
                        "content": user_query
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
        reasoning={"effort": "medium"},
        text_format=Postup
    )

    return response.output_parsed


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

if __name__ == "__main__":
    try:
        main()
    finally:
        # Clean up resources to prevent threading errors on exit
        store.close()