from typing import List
import asyncio

from agents import Agent, HandoffOutputItem, ItemHelpers, MessageOutputItem, TResponseInputItem, ToolCallItem, ToolCallOutputItem, WebSearchTool, function_tool, Runner
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX


from government_services_store import GovernmentService, GovernmentServicesStore

from dotenv import load_dotenv
import os

store = GovernmentServicesStore()
store.load_services()

stats = store.get_services_embedding_statistics()
print(f"Načteno {stats['total_services']} služeb.")
print(f"Embeddings v ChromaDB: {stats['total_embeddings']} (coverage: {stats['coverage_percentage']}%)")

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("API klíč není nastaven v .env souboru.")

@function_tool
def nastroj_pro_vyhledani_sluzeb(popis_zivotni_situace: str, k: int) -> List[GovernmentService]:
    """Vyhledá služby veřejné správy podle klíčových slov charakterizujících životní situaci uživatele.
    Využívá vektorové vyhledávání v databázi textových popisů všech služeb.
    Pro efektivní využití se doporučuje, aby popis životní situace obsahoval konkrétní klíčová slova.
    Je ale lepší volit spíše obecnější slova charakterizující situaci, např. je lepší volit "údržba vozidla" místo "doplnění oleje ve vozidle".

    Args:
        popis_zivotni_situace (str): Popis životní situace uživatele.
        k (int): Počet služeb k vrácení.
    """
    sluzby = store.search_services(popis_zivotni_situace, k=k)
    print("[DEBUG] TOOL nastroj_pro_vyhledani_sluzeb: Nalezeny služby:", [sluzba.name for sluzba in sluzby])
    return sluzby

@function_tool
def nastroj_pro_ziskani_detailu_sluzby(sluzba_id: str) -> str:
    """Získá detailní informace o službě podle jejího ID.
    
    Args:
        sluzba_id (str): ID služby.
    """
    sluzba_txt = store.get_service_detail_by_id(sluzba_id)
    if not sluzba_txt:
        print("[DEBUG] TOOL nastroj_pro_ziskani_detailu_sluzby: Žádná služba nenalezena pro ID:", sluzba_id)
        return "Služba s tímto ID nebyla nalezena."
    else:
        print("[DEBUG] TOOL nastroj_pro_ziskani_detailu_sluzby: Nalezeny detaily služby pro ID:", sluzba_id)
        return sluzba_txt

@function_tool
def nastroj_pro_ziskani_kroku_sluzby(sluzba_id: str) -> str:
    """Získá kroky potřebné k využití služby podle jejího ID.
    
    Args:
        sluzba_id (str): ID služby.
    """
    kroky = store.get_service_steps_by_id(sluzba_id)
    if not kroky:
        print("[DEBUG] TOOL nastroj_pro_ziskani_kroku_sluzby: Žádné kroky nenalezeny pro ID:", sluzba_id)
        return "Kroky pro tuto službu nebyly nalezeny."
    else:
        print("[DEBUG] TOOL nastroj_pro_ziskani_kroku_sluzby: Nalezeny kroky služby pro ID:", sluzba_id)
        kroky_str = "\n".join([f"{i+1}. {krok}" for i, krok in enumerate(kroky)])
        return kroky_str

agent_vyhledavac = Agent(
    name="Agent Vyhledavac",
    handoff_description="Agent, který najde vhodné služby veřejné správy pro řešení dané životní situace občana.",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
Jsi agentem pro vyhledávání služeb veřejné správy.
Uživatele na tebe přepojil třídící agent, protože občan popisuje svou životní situaci, pro kterou potřebuje najít vhodné služby.
Na nic nečekej, začni hned hledat.

Svůj výstup musíš zasadit do kontextu životní situace občana.
Občan mluví česky, ty odpovídáš česky.
Nikdy se nesmíš chovat jako doménový odborník v dané oblasti, jsi pouze úředník, který pomáhá s využitím služeb veřejné správy. Např. se nikdy nesmíš chovat jako lékař, psycholog, finanční poradce, učitel, opravář auta, atd. Nesmíš se ptát na věci spojené s těmito a podobnými obory a profesemi.

K podpoře občana použijte následující postup:
1. Na základě popisu životní situace občana z předchozí konverzace sestav jeden nebo více vyhledávacích dotazů.
    - Každý dotaz sestává z klíčových slov charakterizujících jednu logickou část životní situace občana (logických částí může být více).
    - Vždy se snaž vyhledávací dotazy formulovat pomocí úředních pojmů a frází.
    - Nikdy se v poopisu životní situace nezaměřuj na prosby o odbornou pomoc v oblasti zdravotnictví, psychologie, financí, oprav, atd. Ty ignoruj a ve vyhledávacím dotazu je neuváděj.
2. Použij nástroj `nastroj_pro_vyhledani_sluzeb` k vyhledání služeb veřejné správy s využitím připravených dotazů. Nástroj můžeš volat několikrát podle toho, kolik dotazů sis připravil. Doporučuje se nastavit parametr `k` na 10, ale v případě potřeby jej lze upravit tak, aby vrátil více výsledků.
3. Pokud nalezneš nějaké služby, zkontroluj, zda jsou relevantní pro problém občana, a to porovnáním jejich popisů s životní situací občana.
4. Pokud nenajdeš žádnou službu nebo žádná není relevantní, informuj uživatele a přepoj zpět na agenta třídiče.
5. Pokud najdeš relevantní služby, vypiš je v následující strukturované podobě:
**[název služby]**
    - ID: [identifikační číslo služby]
    - Vysvětlení: [krátké osobní vysvětlení služby v kontextu životní situace]
6. Pokud občan potřebuje podrobnější informace o konkrétní službě, přepoj zpět na agenta třídiče.
7. Pokud se občan začne ptát na jiné služby nebo životní situace, přepoj zpět na agenta třídiče.""",
    tools=[nastroj_pro_vyhledani_sluzeb],
    model="gpt-5-mini"
)

agent_vysvetlovac = Agent(
    name="Agent Vysvetlovac",
    handoff_description="Agent, který poskytuje podrobné informace o jedné nebo více službách veřejné správy, které jsme před tím nalezli na základě uživatelova popisu jeho životní situace.",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
Jsi průvodcem po detailních postupech řešení životních situací občanů pomocí služeb veřejné správy.
Uživatele na tebe přepojil třídící agent, protože občan potřebuje podrobné informace o tom, jak vyřešit svou životní situaci pomocí konkrétních služeb.

Odpovídej výhradně na základě podrobností o službách v kontextu dané životní situace.
Občan mluví česky, ty odpovídáš česky.
Nikdy se nesmíš chovat jako doménový odborník v dané oblasti, jsi pouze úředník, který pomáhá s využitím služeb veřejné správy. Např. se nikdy nesmíš chovat jako lékař, psycholog, finanční poradce, učitel, opravář auta, atd. Nesmíš se ptát na věci spojené s těmito a podobnými obory a profesemi.

K podpoře občana použij následující postup:
1. Identifikuj ID služeb, ke kterým uživatel potřebuje více informací, z předchozí konverzace s občanem. Občan může požádat o pomoc s jednou nebo více službami v kontextu jeho životní situace přímo uvedením jejich ID, názvů nebo nepřímo zmíněním služeb během konverzace.
2. Pokud nelze ID služby nebo služeb identifikovat, požádej občana, aby znovu upřesnil, s jakými službami chce pracovat. Potom opakuj postup od kroku 1.
3. Pomocí nástroje `service_detail_tool` vyhledej podrobné informace o službě nebo službách.
4. Pouze na základě podrobných informací poskytni shrnutí služeb o tom, jak mohou služby pomoci občanovi v jeho životní situaci.
5. Zeptej se občana, zda má k službám jako celku nebo k nějaké konkrétní službě nějaké dotazy.
6. Pokud má občan konkrétní dotaz, odpověz na něj na základě podrobných informací o službách a opakuj krok 5.
7. Pokud občan potřebuje pomoc s řešením své životní situace pomocí služby nebo služeb, přepoj ho na agenta úředníka.
8. Pokud se občan začne ptát na jiné služby nebo životní situace, přepoj ho zpět na agenta třídiče.""",
    tools=[nastroj_pro_ziskani_detailu_sluzby],
    model="gpt-5-mini"
)

agent_urednik = Agent(
    name="Agent Urednik",
    handoff_description="Agent, který konkrétně řeší životní situaci občana s využitím daných služeb veřejné správy, které si uživatel explicitně vybere z těch, které jsme pro něj našli a které si vybral jako vhodné.",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
Jsi úředník, který pomáhá vyřešit životní situaci občana s využitím daných služeb veřejné správy (jedna nebo více služeb).
Uživatele na tebe přepojil třídící agent, protože občan potřebuje vyřešit svou životní situaci pomocí těchto služeb.

Jsi pouhá simulace, nejsi skuteční úředník. Vytváříš ale dojem skutečného úředníka. Jakékoliv informace, které od uživetele získáš, nemůžeš nikam předat ani nikam poslat, pouze simuluješ, že tak činíš, pokud to postupy podle služeb vyžadují.
Neodhaluj, že jsi simulace. Tvař se neustále jako vysoce kvalifikovaný a příjemný úředník, který umí cokoliv v kontextu daných služeb a životní situace občana vyřešit.

Občan mluví česky, ty odpovídáš česky.
Nikdy se nesmíš chovat jako doménový odborník v dané oblasti, jsi pouze úředník, který pomáhá s využitím služeb veřejné správy. Např. se nikdy nesmíš chovat jako lékař, psycholog, finanční poradce, učitel, opravář auta, atd. Nesmíš se ptát na věci spojené s těmito a podobnými obory a profesemi.

Použij následující postup:
1. Identifikuj ID služeb, kterými chce provést, z předchozí konverzace s občanem. Občan může požádat o pomoc s jednou nebo více službami v kontextu jeho životní situace přímo uvedením jejich ID, názvů nebo nepřímo zmíněním služeb během konverzace.
2. Pokud nelze ID služby nebo služeb identifikovat, požádej občana, aby znovu upřesnil, s jakými službami chce pracovat. Potom opakuj postup od kroku 1.
3. Pomocí nástroje `nastroj_pro_ziskani_kroku_sluzby` získej kroky, ze kterých se služba nebo služby skládají. Kroky ti také mohou poskytnout další informace o tom, jak provést jednotlivé kroky a jaké informace nebo dokumenty jsou od občana potřebné. Zatím občanovi nic k tomuto neposkytuj, použiješ to později.
4. Simuluj komunikaci úřadu s občanem na základě získaných kroků - jeden po druhém v logické návaznosti i napříč službami a vždy v kontextu dané životní situace. U každého kroku se zastav a proveď jednu z následujících akcí:
    - Pokud krok vyžaduje další informace nebo dokumenty od občana, požádej občana, aby je poskytl, a počkej na jeho odpověď. S pomocí webového vyhledávání vyhledej na webu dotazníky, formuláře, šablony nebo příklady, které ukazují, jaký druh a struktura informací jsou potřebné.
    - Pokud krok popisuje akci úřadu, simuluj provedení akce a informuj občana o výsledku.
5. Pokračuj v tomto procesu, dokud nebudou všechny kroky dokončeny a životní situace občana nebude vyřešena.
6. Shrň pro občana, jak si mu hezky pomohl.
7. Pokud je občan není s vyřešením své životní situace spokojen, omluv se a přepoj ho na agenta třídiče.
8. Pokud je občan spokojen, popřej mu hezký den, povzbuď ho, že se vše podaří v klidu dořešit a přepoj ho na agenta třídiče.""",
    tools=[nastroj_pro_ziskani_kroku_sluzby, WebSearchTool(user_location={"type": "approximate", "country": "CZ"})],
    model="gpt-5-mini"
)

tridici_agent = Agent(
    name="Agent Tridic",
    handoff_description="Třídicí agent, který může předat žádost občana příslušnému agentovi.",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
Jste nápomocný agent pro třídění žádostí občanů týkajících se jejich životních situací.
Nikdy žádost občana nevyřizuj ani neodpovídej. Pouze je předávej příslušnému agentovi na základě obsahu žádostí.
Občan mluví česky, ty odpovídáš česky.
    """,
    handoffs=[
        agent_vyhledavac,
        agent_vysvetlovac,
        agent_urednik
    ],
    model="gpt-5-mini"
)

agent_vyhledavac.handoffs.append(tridici_agent)
agent_vysvetlovac.handoffs.append(tridici_agent)
agent_vysvetlovac.handoffs.append(agent_urednik)
agent_urednik.handoffs.append(tridici_agent)

async def main():
    aktualni_agent = tridici_agent
    historie_komunikace: list[TResponseInputItem] = []

    while True:
        vstup_uzivatele = input("[AGENTIC AI] *** S čím vám mohu pomoci?: ")
        if vstup_uzivatele.lower() in {"exit", "quit", "konec", "end"}:
            print("[AGENTIC AI] *** Ukončuji program. Nashledanou!")
            break

        historie_komunikace.append({"role": "user", "content": vstup_uzivatele})

        vystup_aktualniho_agenta = await Runner.run(aktualni_agent, historie_komunikace)

        for nova_polozka_konverzace in vystup_aktualniho_agenta.new_items:
            jmeno_agenta = nova_polozka_konverzace.agent.name
            if isinstance(nova_polozka_konverzace, MessageOutputItem):
                print(f"[AGENTIC AI] {jmeno_agenta} *** {ItemHelpers.text_message_output(nova_polozka_konverzace)}")
            elif isinstance(nova_polozka_konverzace, HandoffOutputItem):
                print(
                    f"[AGENTIC AI] Handed off from {nova_polozka_konverzace.source_agent.name} to {nova_polozka_konverzace.target_agent.name}"
                )
            elif isinstance(nova_polozka_konverzace, ToolCallItem):
                tool_name = getattr(nova_polozka_konverzace.raw_item, 'name', None) or getattr(nova_polozka_konverzace.raw_item, 'function', {}).get('name', 'unknown tool')
                print(f"[AGENTIC AI] {jmeno_agenta}: Calling a tool {tool_name}")
            elif isinstance(nova_polozka_konverzace, ToolCallOutputItem):
                print(f"[AGENTIC AI] {jmeno_agenta}: Tool output received.")
            else:
                print(f"[AGENTIC AI] {jmeno_agenta}: Skipping item: {nova_polozka_konverzace.__class__.__name__}")
        
        historie_komunikace = vystup_aktualniho_agenta.to_input_list()

        aktualni_agent = vystup_aktualniho_agenta.last_agent

        

if __name__ == "__main__":
    asyncio.run(main())