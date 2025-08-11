# Kurz AI pro veřejnou správu - Kapitola 4: Context learning a n-shot learning

V této kapitole se naučíte využívat *context learning* a jeho konkrétní podobu *n-shot learning* k tomu, abyste mohli model přizpůsobit konkrétnímu stylu odpovědí bez nutnosti ho trénovat na nových datech.

---

## 1. Co je context learning
Velké jazykové modely (LLM) jsou schopny reagovat na zcela nové úkoly, aniž by na ně byly speciálně trénovány. Stačí, když v rámci jednoho požadavku (tzv. kontextu) vysvětlíme, co od nich chceme.

Tato schopnost se nazývá **context learning**:
- Model neprovádí trvalé učení – po dokončení odpovědi si nepamatuje, co jsme mu v tomto požadavku řekli.
- Celé „učení“ probíhá jen během zpracování aktuálního vstupu (promptu).
- Do promptu tedy můžeme vložit jak popis úkolu, tak příklady požadovaného chování.

### Zero-shot, one-shot a few-shot
- **Zero-shot learning** – zadáme jen instrukci, bez příkladů.
- **One-shot learning** – zadáme jednu ukázku (1 příklad vstupu a odpovědi).
- **Few-shot learning (n-shot)** – zadáme více ukázek (n příkladů).

Výhodou přidání příkladů je, že model se v rámci aktuálního kontextu „naladí“ na náš styl a požadavky.

---

## 2. Jak funguje n-shot learning
Při n-shot learningu vkládáme do promptu (instrukcí) ukázky ve formátu:
- role `user` – reprezentuje dotaz uživatele v příkladu,
- role `assistant` – reprezentuje požadovanou odpověď modelu.

Tyto příklady slouží jako vzor. Když model uvidí nový dotaz, snaží se pokračovat ve stejném stylu a formátu jako ve vzorech.

📌 **Poznámka:** I když mluvíme o „učení“, ve skutečnosti se model nic trvale nenaučí – jde jen o krátkodobý vliv příkladů v rámci aktuálního volání API.

---

## 3. Praktický příklad
Ukážeme si one-shot příklad, kdy nejprve nastavíme roli `developer` s instrukcemi a poté přidáme jeden ukázkový dialog (`user` → `assistant`). Nakonec vložíme nový dotaz.

```python
from openai import OpenAI
from dotenv import load_dotenv
import os

# Načteme API klíč ze souboru .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("API klíč není nastaven v .env souboru.")

# Inicializujeme OpenAI klienta
client = OpenAI(api_key=api_key)

# Zavoláme OpenAI model
response = client.responses.create(
    model="gpt-5-mini",
    input=[
        {
            "role": "developer",
            "content": "Jsi odborník na pomoc uživateli při řešení jeho různých životních situací v občanském životě. Vždy poradíš, jak danou životní situaci vyřešit z úředního hlediska poskytnutím konkrétního postupu v podobě číslovaných kroků. Uživatel potřebuje srozumitelné ale krátké vysvětlení každého kroku jednoduchou češtinou."
        },{
            "role": "developer",
            "content": "Nikdy nesmíš v žádném kroku postupu poskytovat radu v oboru, kterého se dotaz uživatele týká, např. lékařské rady, stavební rady, atd. Uživateli pouze můžeš napsat, aby odborníka vyhledal a navštívil bez jakýchkoliv časových, situačních či jiných podmínek a doporučení."
        },{
            "role": "user",
            "content": "Někdo mi rozbil okno u auta, vloupal se dovnitř a ukradl mi peněženku. Jak si mám sám opravit okno? Potřebuju nějak řešit ztrátu peněženky?"
        },{
            "role": "assistant",
            "content": """Především buďte opatrný při otevírání dvěří vašeho automobilu, abyste se neporanil. Nemohu Vám poradit, jak si máte sám opravit rozbité okno u vašeho automobilu. Doporučuji se vám obrátit na nejbližší autoservis, kde vám rozbité okno odborně opraví.
            
            Doporučuji následující postup:
            
            1. *Zavolejte policii*: Zavolejte na tísňovou linku 112 nebo 158 a oznamte vloupání do vašeho vozidla.
            
            2. *Vyčkejte na příjezd policie*: Vyčkejte, než přijede policie a nahlašte jim, co přesně se z vašeho pohledu stalo. Odpovězte na všechny jejich otázky.

            3. *Převezměte protokol o vloupání*: Od policie převezměte originál protokolu o vloupání do vašeho vozidla a o míře poškození.
            
            4. *Ohlašte odcizení občanského průkazu, příp. dalších dokladů*: Ztrátu můžete nahlásit přímo policistovi, který na místo přijel. Případně můžete ztrátu nahlásti elektronicky vašemu obecnímu úřadu prostřednictvím datové schránky.
            
            5. *Požádejte o vydání nového občanského průkazu, příp. jiného odkladu*: Požádat o nový doklad můžete na jakémkoli obecním úřadě obce s rozšířenou působností, kde si ho posléze i vyzvednete.

            6. *Nahlašte škodní událost*: Pokud máte automobil pojištěný, nahlaště na pojišťovnu škodní událost. Budete k tomu potřebovat protokol o vloupání do vozidla.

            7. *Nechte si opravit rozbité okno*: Navštivte co nejdříve libovolný autoservis, kde Vám opraví rozbité okno. V autoservisu vám mohou pomoci i nahlášením škodní události vaší pojišťovně (viz krok 6)."""
        },{
            "role": "user",
            "content": "Bolí mě hlava a mám asi horečku. Co si na to mám vzít? Co mám dělat? A mohu jít do práce?"
        }
    ]
)

# Vypíšeme odpověď
print("AI odpověď:")
print(response.output_text)
```

💡 **Co se stane:** Model použije předchozí ukázku jako vzor – ignoruje zdravotní radu a místo toho poradí úřední kroky (např. neschopenku, kontaktování lékaře), ve stejném stylu jako v příkladu.

---

## 4. Tipy pro efektivní příklady
- **Formátujte příklady** přesně tak, jak chcete výstup – model má tendenci formát kopírovat.
- **Používejte relevantní příklady** – čím blíže jsou ukázky reálným dotazům, tím lépe model odpovídá.
- **Nespoléhejte na příliš mnoho příkladů** – obvykle 1–5 stačí, delší prompt může zbytečně prodražit volání.
- **Kombinujte s rolemi** – role `developer` zajistí, že příklady nebudou přebity instrukcemi uživatele.

---

## 5. Kdy použít n-shot learning
- Když potřebujete **specifický formát** odpovědi (např. tabulka, seznam kroků).
- Když chcete **přizpůsobit tón** odpovědi (formální, přátelský, stručný, detailní).
- Když potřebujete **omezení obsahu** (vynechání určitých typů rad).

Nevhodné, pokud:
- Potřebujete trvalé změny chování – v tom případě by bylo nutné model jemně doladit (fine-tuning).
- Kontext (prompt) by byl příliš dlouhý – může zvýšit latenci a náklady.

---

## Shrnutí
- **Context learning** umožňuje měnit chování modelu jen pomocí instrukcí a příkladů v aktuálním volání.
- **n-shot learning** je specifický způsob context learningu s ukázkami vstupu a výstupu.
- Správně připravené příklady výrazně zvyšují konzistenci a spolehlivost odpovědí.