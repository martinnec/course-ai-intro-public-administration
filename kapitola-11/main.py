from typing import List
import asyncio

from agents import Agent, HandoffOutputItem, ItemHelpers, MessageOutputItem, ModelSettings, TResponseInputItem, ToolCallItem, ToolCallOutputItem, function_tool, Runner
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX
from openai.types.shared import Reasoning

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
def nastroj_pro_vyhledani_sluzeb(charakteristika_zivotni_situace: str, k: int) -> List[GovernmentService]:
    """Vyhledá služby veřejné správy podle klíčových slov charakterizujících životní situaci uživatele.
    Využívá vektorové vyhledávání v databázi textových popisů všech služeb.
    Pro efektivní využití se doporučuje, aby popis životní situace obsahoval konkrétní klíčová slova.
    Je vhodné volit spíše obecnější slova charakterizující situaci, např. je lepší volit "údržba vozidla" místo "doplnění oleje ve vozidle".
    V případě složitější situace zahrnující více aspektů je vhodné zavolat tento nástroj vícekrát s různými popisy situace.

    Args:
        charakteristika_zivotni_situace (str): Charakteristika životní situace pomocí klíčových slov.
        k (int): Počet služeb k vrácení.
    """
    sluzby = store.search_services(charakteristika_zivotni_situace, k=k)
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

agent = Agent(
    name="Agent Urednik",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
Jste nápomocný agent, který pomáhá uživatelům - občanům v jejich životních situacích.
Uživatel mluví česky, ty odpovídáš česky.
Umíš vyhledávat služby veřejné správy, poskytovat podrobné informace o službách a konkrétně řešit životní situace uživatelů s využitím nalezených služeb.
Nikdy se nesmíš chovat jako doménový odborník v dané oblasti, jsi pouze úředník, který pomáhá s využitím služeb veřejné správy. Např. se nikdy nesmíš chovat jako lékař, psycholog, finanční poradce, učitel, opravář auta, atd. Nesmíš se ptát na věci spojené s těmito a podobnými obory a profesemi.
Všechny tvoje odpovědi, reakce a akce musí být výhradně v kontextu životní situace uživatele a dostupných dat o službách veřejné správy.
Nikdy nesmíš odpovídat mimo tento kontext.
Data o službách veřejné správy můžeš získat pomocí svých nástrojů.
Pracuj s uživatelem ve dvou fázích:

*Fáze 1*
V první fázi konverzace se zaměř na pomoc uživateli naformulovat jeho životní situaci a vyhledávání vhodných služeb k řešení této situace.
Uživatel může být v různě těžkých životních situacích a z toho důvodu může být zmatený, vystresovaný nebo rozrušený.
Je tedy potřeba k němu přistupovat s empatií a trpělivostí.
To, zda je daná vyhledaná služba veřejné správy vhodná, posuzuj podle jejího popisu a porovnej ho s životní situací uživatele.
Je možné, že situaci uživatele lze vyřešit pouze kombinací více služeb.
Uživateli vysvětli, jaké služby si našel a proč by mu mohly pomoci.
Pro přehlednost a jednoznačnost používej i názvy a kódy služeb.
Nikdy nenabízej nic, co by přímo nevyplývalo z nalezených služeb.

*Fáze 2*
Pokud chce uživatel službu nebo služby využít, vypiš mu kroky, které musí podniknout, aby svoji situaci vyřešil.
K tomu můžeš potřebovat vhodně kombinovat kroky z více různých služeb - dbej na logické návaznosti mezi kroky.
U každého kroku uveď, z jaké služby pochází.
Nikdy nenabízej kroky, které by přímo nevyplývaly z nalezených služeb.""",
    tools=[nastroj_pro_vyhledani_sluzeb, nastroj_pro_ziskani_detailu_sluzby, nastroj_pro_ziskani_kroku_sluzby],
    model="gpt-5-mini",
    model_settings=ModelSettings(reasoning=Reasoning(effort="medium"), verbosity="medium")
)

async def main():
    historie_komunikace: list[TResponseInputItem] = []

    while True:
        vstup_uzivatele = input("[AGENTIC AI] *** S čím vám mohu pomoci?: ")
        if vstup_uzivatele.lower() in {"exit", "quit", "konec", "bye", "end"}:
            print("[AGENTIC AI] *** Ukončuji program. Nashledanou!")
            break

        historie_komunikace.append({"role": "user", "content": vstup_uzivatele})

        vystup_agenta = await Runner.run(agent, historie_komunikace)

        for nova_polozka_konverzace in vystup_agenta.new_items:
            if isinstance(nova_polozka_konverzace, MessageOutputItem):
                print(f"[AGENTIC AI]: *** {ItemHelpers.text_message_output(nova_polozka_konverzace)}")
            elif isinstance(nova_polozka_konverzace, ToolCallItem):
                tool_name = getattr(nova_polozka_konverzace.raw_item, 'name', None) or getattr(nova_polozka_konverzace.raw_item, 'function', {}).get('name', 'unknown tool')
                print(f"[AGENTIC AI]: Calling a tool {tool_name}")
            elif isinstance(nova_polozka_konverzace, ToolCallOutputItem):
                print(f"[AGENTIC AI]: Tool output received.")
            else:
                print(f"[AGENTIC AI]: Skipping item: {nova_polozka_konverzace.__class__.__name__}")

        historie_komunikace = vystup_agenta.to_input_list()

if __name__ == "__main__":
    asyncio.run(main())