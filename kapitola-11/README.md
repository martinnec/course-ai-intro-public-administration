# Kapitola 11 – Konverzační agent s OpenAI Agents SDK

V kapitolách 9 a 10 jsme pracovali s agentem, jehož chování jsme přesně naprogramovali do jednotlivých kroků (generování dotazů, vyhledávání služeb, filtrování, sestavení postupu). Tento přístup je vhodný, pokud máme úlohu, kterou lze algoritmicky rozdělit na dílčí části.

V praxi je ale tento přístup často **příliš svázaný**. Ne všechny interakce s uživatelem totiž dokážeme převést do striktní posloupnosti kroků. Potřebujeme, aby **AI řídila samotný tok konverzace** – kdy se má ptát, kdy využít nástroj a kdy předat výsledek. K tomu slouží tzv. **konverzační agenti**.

---

## Co je konverzační agent?

Konverzační agent je AI, která:
- vede s uživatelem dialog,
- sama rozhoduje o krocích konverzace,
- podle potřeby využívá dostupné nástroje (funkce),
- a průběžně kombinuje přirozenou komunikaci s akcemi.

Tento přístup je flexibilnější – AI nemusí slepě vykonávat předem definovaný algoritmus, ale sama se rozhoduje, jak postupovat.

---

## OpenAI Agents SDK

Pro tvorbu konverzačních agentů využijeme knihovnu [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/). Tento framework:
- umožňuje snadno definovat agenta,
- přidávat k němu **nástroje** (funkce, které AI může volat),
- a spouštět konverzace v podobě smyčky, která propojuje uživatele, AI a nástroje.

Nejprve je potřeba nainstalovat:

```bash
pip install openai-agents
```

---

## První agent – bez nástrojů

Začněme s úplně jednoduchým agentem, který pouze komunikuje s uživatelem:

```python
from agents import Agent

agent = Agent(
    name="Agent Urednik",
    instructions="Jste nápomocný agent, který odpovídá česky a pomáhá občanům v jejich životních situacích.",
    model="gpt-5-mini"
)
```

Tento agent dokáže vést dialog, ale zatím neumí nic víc než reagovat přirozeným jazykem.

### Spuštění agenta jednorázově

Abychom si mohli agenta vyzkoušet, spustíme ho v asynchronní funkci `main`:

```python
import asyncio
from agents import Runner

async def main():
    user_query = "Jsem OSVČ a jsem nemocný. Můžete mi pomoct?"
    vystup_agenta = await Runner.run(agent, user_query)
    print(vystup_agenta.final_output)

if __name__ == "__main__":
    asyncio.run(main())
```

Pokud tento skript spustíte, agent zpracuje dotaz a vypíše odpověď. Velmi pravděpodobně se uživatele bude chtít dále doptat.

### Smyčka pro interaktivní komunikaci

Proto si hned ukážeme, jak zavést jednoduchou konverzační smyčku. Ta bude jen v základní podobě:

```python
async def main():
    while True:
        user_query = input("Vy: ")
        if user_query.lower() in {"exit", "quit", "konec"}:
            print("Agent: Nashledanou!")
            break

        vystup_agenta = await Runner.run(agent, user_query)
        print("Agent:", vystup_agenta.final_output)
```

Tento jednoduchý kód uživatelům umožní si s agentem přímo „povídat“. Agent zatím neumí používat žádné nástroje, pouze reaguje na vstup textem.

### Přidání historie konverzace

Výše uvedená smyčka má jednu nevýhodu – agent si nepamatuje, co bylo řečeno dříve. Aby si uchovával kontext, musíme mu dodávat celou historii komunikace:

```python
from agents import TResponseInputItem

async def main():
    historie_komunikace: list[TResponseInputItem] = []

    while True:
        user_query = input("Vy: ")
        if user_query.lower() in {"exit", "quit", "konec"}:
            print("Agent: Nashledanou!")
            break

        historie_komunikace.append({"role": "user", "content": user_query})
        vystup_agenta = await Runner.run(agent, historie_komunikace)

        print("Agent:", vystup_agenta.final_output)

        # aktualizujeme historii pro další kolo
        historie_komunikace = vystup_agenta.to_input_list()
```

Díky tomu si agent pamatuje předchozí dialog a konverzace je souvislá.

---

## Přidání nástrojů

Aby agent mohl vyhledávat služby, musíme mu dát k dispozici nástroje. Nástroj je funkce, kterou si může agent kdykoliv dle svého uvážení zavolat a využít vrácené výsledky. V závislosti na frameworku nebo SDK, které používáte, dostáváte nějakou sadu předpřipravených nástrojů, viz např. [nástroje zahrnuté v rámci OpenAI Agents SDK](https://openai.github.io/openai-agents-python/tools/). Máme zde např. nástroj pro vyhledávání na webu, `WebSearchTool`.

Nástroj pro vyhledávání na webu můžete agentovi poskytnout následující úpravou kódu:

```python
from agents import WebSearchTool

agent = Agent(
    name="Agent Urednik",
    instructions="Jste nápomocný agent, který odpovídá česky a pomáhá občanům v jejich životních situacích.",
    model="gpt-5-mini",
    tools=[
        WebSearchTool(user_location={"type": "approximate", "country": "CZ"})
    ]
)
```

### Vlastní nástroje

V našem případě práce se službami potřebujeme ale agentovi dát schopnost vyhledávat přímo v datech o službách veřejné správy a získávat o nich detaily. K tomu můžeme pro agenta naprogramovat vlastní nástroje. Jedná se o běžné funkce, které opatříme anotací `@function_tool` a které dobře zdokumentujeme, aby agent rozumněl jejich použití.

Pro naše nástroje jsme již připraveni. Postavíme je s využitím předpřipraveného úložiště dat o službách, které máme naprogramováno v `government_services_store.py`. Úložiště si přidáme do našeho `main.py`:

```python
from government_services_store import GovernmentService, GovernmentServicesStore

store = GovernmentServicesStore()
store.load_services()

stats = store.get_services_embedding_statistics()
print(f"Načteno {stats['total_services']} služeb.")
print(f"Embeddings v ChromaDB: {stats['total_embeddings']} (coverage: {stats['coverage_percentage']}%)")
```

A potom přidáme nástroje:

```python
from typing import List
from agents import function_tool

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
```

---

## Definice agenta s vlastními nástroji

Agentovi nyní předáme připravené nástroje. Zároveň mu přidáme detailnější instrukce, aby se choval a využíval nástroje tak, jak chceme.

```python
from agents import ModelSettings
from openai.types.shared import Reasoning
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX

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
```

Agent má nyní možnost vyhledávat služby a získávat jejich detaily i kroky.

---

## Smyčka konverzace

Hlavní logika běží v nekonečné smyčce. Smyčku rozšiřme o detailnější analýzu výstupu agenta, abychom mohli pozorovat, jak agent pracuje.

```python
from agents import Runner, MessageOutputItem, ToolCallItem, ToolCallOutputItem, ItemHelpers

async def main():
    historie_komunikace = []

    while True:
        vstup_uzivatele = input("[AGENTIC AI] *** S čím vám mohu pomoci?: ")
        if vstup_uzivatele.lower() in {"exit", "quit", "konec"}:
            print("[AGENTIC AI] *** Ukončuji program. Nashledanou!")
            break

        historie_komunikace.append({"role": "user", "content": vstup_uzivatele})

        vystup_agenta = await Runner.run(agent, historie_komunikace)

        for polozka in vystup_agenta.new_items:
            if isinstance(polozka, MessageOutputItem):
                print("[AGENTIC AI]:", ItemHelpers.text_message_output(polozka))
            elif isinstance(polozka, ToolCallItem):
                print("[AGENTIC AI]: Volám nástroj.")
            elif isinstance(polozka, ToolCallOutputItem):
                print("[AGENTIC AI]: Výstup z nástroje.")

        historie_komunikace = vystup_agenta.to_input_list()
```

---

## Shrnutí

- Předchozí přístup byl přesně naprogramovaný – AI jen plnila předem definované funkce.
- Nyní jsme přešli ke **konverzačnímu agentovi**, který rozhoduje o krocích v dialogu.
- Nejprve jsme si ukázali agenta bez nástrojů:
  - jednorázové spuštění,
  - jednoduchou smyčku,
  - a pokročilejší variantu se zapamatováním historie.
- Poté jsme mu přidali nástroje a spustili ho v plné konverzační smyčce.

Tento přístup otevírá cestu k mnohem volnější a přirozenější interakci s uživateli.

