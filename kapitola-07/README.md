# Kurz AI pro veřejnou správu - Kapitola 7: Od promptu k RAGu

V minulé kapitole jsme si ukázali, jak dodat modelu **externí znalost** tak, že ji vložíme přímo do promptu. Tento přístup funguje, ale má své limity. V této kapitole si vysvětíme, **proč to není ideální**, co je to **RAG** a jak ho v našem případě využijeme.

---

## 1. Proč nevkládat všechno do promptu

Naším cílem je vybudovat asistenta, který umí poradit občanovi v **mnoha různých životních situacích**. To znamená, že potřebujeme znát informace o **všech službách veřejné správy**.

Nejjednodušší možnost by byla vytvořit **jeden obrovský JSON soubor** a jeho obsah při každém dotazu vložit do promptu. Tento přístup má ale vážné nevýhody:

- **Velikost kontextu** – sice se kontextová okna modelů zvětšují, ale tisíce služeb s popisy by je snadno zaplnily.
- **Plýtvání tokeny** – při každém dotazu bychom posílali všechna data, i když velká část nebude relevantní.
- **Menší kontrola nad odpovědí** – necháváme na modelu, aby si sám prošel veškerou znalost a rozhodl, co použije.
- **Horší škálovatelnost** – čím víc dat, tím horší latence a vyšší náklady.

---

## 2. Co je RAG a proč ho chceme

**RAG (Retrieval-Augmented Generation)** kombinuje dva kroky:

1. **Retrieval** – najdeme pouze ty informace, které jsou pro dotaz relevantní.
2. **Generation** – předáme je modelu spolu s dotazem a necháme ho vygenerovat odpověď.

V našem případě to znamená:
- Uchováváme si znalost o veškerých službách ve **vektorové databázi**.
- Při dotazu vyhledáme jen ty služby, které nejlépe odpovídají uživatelské otázce.
- Do promptu vložíme pouze tuto **malou, relevantní podmnožinu znalostí**.

Tím:
- šetříme tokeny,
- zrychlujeme odezvu,
- a předcházíme tomu, aby model používal irelevantní nebo zastaralé informace.

---

## 3. Komponenty RAGu v našem projektu

- **Vektorová databáze** – slouží k rychlému vyhledávání podobných textů podle jejich embeddingů. My použijeme **ChromaDB** (snadno se používá, umí běžet v paměti a nepotřebuje server).
- **GovernmentServicesStore** – náš Python modul, který uchovává znalosti o službách ve vektorové databázi a umí v nich vyhledávat.
- **Otevřená data služeb** – zdroj aktuálních informací, který budeme používat při inicializaci znalostní báze.

Abychom mohli RAG implementovat, musíme si doinstalovat dva Python moduly:
- `chromadb` pro podporu vektorové databáze
- `rdflib` pro práci s externím SPARQL endpointem

- **Bash:**
  ```bash
  pip install chromadb rdflib
  pip freeze > requirements.txt
  ```
- **PowerShell:**
  ```powershell
  pip install chromadb rdflib
  pip freeze > requirements.txt
  ```

---

## 4. Jak funguje GovernmentServicesStore

Tento modul nebudeme implementovat.
Máte jej již k dispozici v materiálech v souboru `government_services_store.py`.
Je dobré vědět, co dělá:
1. **Ukládá služby** do vektorové DB.
2. **Vyhledává** nejrelevantnější služby k dotazu uživatele.
3. **Získává detaily služeb** přímo z otevřených dat.

Klíčové pro nás je, že při dotazu nedostane model všechno, ale jen malý balíček relevantních služeb.

### 4.1 Načítání dat a práce s cache

`load_services()` zajišťuje kompletní načtení a přípravu dat. Postupuje následovně:

1. **Lokální cache** – Nejprve zkusí načíst data z lokálního úložiště (`_load_services_from_local_cache()`), což je nejrychlejší varianta a nevyžaduje internetové připojení.
2. **Externí SPARQL endpoint** – Pokud lokální data nejsou k dispozici, stáhne aktuální seznam služeb ze vzdáleného SPARQL endpointu (`_load_from_external_store()`) Registru práv a povinností. Konkrétně pro každou službu získáme její název (vlastnost `má-název-služby`) a popis (vlastnost `má-popis-služby`) 
3. **Doplnění detailů** – Ihned po načtení doplní popisy a klíčová slova z doplňujícího JSON souboru z datové sady "Detailní popisy služeb veřejné správy" (`_load_services_with_details()`), aby byla data bohatší. Obsah této datové sady není dostupný ze SPARQL endpointu a tak musíme stáhnout příslušnou distribuci datové sady v JSONu, ten celý přečíst a pro každou službu z něj získat potřebné údaje. Zatím bereme `popis` (jedná se o jiný popis, než popis získaný ze SPARQL endpointu), `jaký-má-služba-benefit`  a `klíčová-slova`.
4. **Výpočet embeddingů** – Každou službu převede na číselný vektor (embedding) pomocí OpenAI modelu (`_compute_services_embeddings()`). Tyto vektory jsou uloženy do **ChromaDB** – in-memory vektorové databáze, kterou používáme pro rychlé sémantické vyhledávání. Na číselný vektor můžeme převést pouze textový řetězec a tak jej musíme nejprve pro službu vytvořit. Zatím jej vytváříme ze spojení názvu, popisu a klíčových slov služby, které jsme si připravili v předchozích krocích.
5. **Uložení cache** – Kompletní dataset se uloží zpět do lokálního souboru (`_store_services_to_local_cache()`), aby bylo načítání při příštím spuštění rychlejší.

ChromaDB zde hraje roli **indexu embeddingů**: při vyhledávání nám bude stačit porovnat embedding dotazu s embeddingy uloženými v ChromaDB a rychle získat nejpodobnější služby.

### 4.2 Sémantické vyhledávání služeb pomocí vektorové databáze

Embedding je číselná reprezentace textu. V našem případě se pro každou službu vytvoří embedding kombinující její název a popis (`_compute_services_embeddings()`). Tento embedding se uloží do ChromaDB společně s metadaty o každé službě

- `id` služby,
- názvem a popisem,
- samotným textem, který jsme embedovali (nyní opět název a popis spojené do jednoho textu).

Vyhledávání (`search_services()`) funguje tak, že se dotaz uživatele převede na embedding stejným modelem, ChromaDB vrátí `k` nejpodobnějších výsledků pomocí spočítání podobnosti vektorů a tyto výsledky se vrátí jako seznam služeb.

### 4.3 Získávání informací o vyhledaných službách

Pokud jsme našli službu, můžeme s pomocí jejího `id` získat detailní informace:
- Metoda `get_service_detail_by_id()` vrací detailní textové informace o službě uvedené v datové sadě "Detailní popisy služeb veřejné správy".
- Metoda `get_service_steps_by_id()` provede SPARQL dotaz na endpoint Registru práv a povinností a získá seznam úkonů služby, které lze řešit digitálně a jsou dostupné přes kanál **Datová schránka**. Výsledky jsou seřazeny podle pořadí kroků a formátovány do textové podoby. Tento postup filtruje pouze digitální kroky.

---

## 5. Hlavní skript – načtení detailů služeb, uložení do vektorové DB a ukázka vyhledávání

V této části si ukážeme, jak použít `GovernmentServicesStore` v praxi. Postupně projdeme jednotlivé kroky a k nim vždy přidáme funkční kód.

### 5.1 Vytvoření instance a načtení služeb

Nejprve vytvoříme instanci `GovernmentServicesStore` a zavoláme `load_services()`, aby se načetla data buď z cache, nebo ze SPARQL endpointu. Při prvním běhu se také vypočítají embeddingy a uloží do ChromaDB.

```python
from government_services_store import GovernmentServicesStore

store = GovernmentServicesStore()

store.load_services()
```

### 5.2 Ověření počtu služeb a základní statistika embeddingů

Po načtení si můžeme vypsat statistiky embeddingů pomocí `get_services_embedding_statistics()`.

```python
stats = store.get_services_embedding_statistics()

# Počet načtených služeb a rychlá statistika embeddingů (ChromaDB)
total_services = stats.get("total_services", 0)
total_embeddings = stats.get("total_embeddings", 0)
coverage = stats.get("coverage_percentage", 0.0)

print(f"Načteno {total_services} služeb.")
print(f"Embeddings v ChromaDB: {total_embeddings} (coverage: {coverage}%)")
```

### 5.3 Test sémantického vyhledávání

Vyzkoušíme sémantické vyhledávání pomocí `search_services(query, k)`, které vrací objektové reprezentace služeb.

```python
query = "Bolí mě hlava a mám asi horečku. Co si na to mám vzít? Co mám dělat? A mohu jít do práce?"
results = store.search_services(query, k=10)
for s in results:
    print(f"{s.id}: {s.name}")
```

### 5.4 Získání detailu a kroků služby

Získáme detail a digitální kroky první nalezené služby.

```python
if results:
    service_id = results[0].id
    detail = store.get_service_detail_by_id(service_id)
    steps = store.get_service_steps_by_id(service_id)
    print("Detail služby:", detail or "Detail není k dispozici.")
    print("Kroky služby:")
    if steps:
        for step in steps:
            print("-", step)
    else:
        print("Žádné digitální kroky nebyly pro službu nalezeny.")
```

---

## 6. Shrnutí

- Vkládání celé znalostní báze do promptu je neefektivní.
- RAG nám umožňuje vybrat a vložit jen relevantní část dat.
- V našem projektu to realizujeme pomocí `GovernmentServicesStore` a `ChromaDB`.
- Před prvním použitím musíme znalostní bázi inicializovat bootstrap skriptem.