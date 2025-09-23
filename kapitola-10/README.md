# Kapitola 10 – Interaktivní agent

V minulé kapitole jsme si ukázali agenta, který dokázal uživateli pomoci s řešením životní situace. Vše proběhlo plně automaticky – uživatel zadal svou situaci, agent vyhledal služby, přefiltroval je a vrátil finální doporučený postup.

V praxi ale často potřebujeme, aby agent **s uživatelem interaktivně komunikoval**. To znamená, že úkol si agent a uživatel postupně **zpřesňují, doplňují a dolaďují**. Agent tedy:
- začne z uživatelského vstupu,
- zkusí provést svou práci (v našm případě tedy najít služby),
- vysvětlí uživateli, co našel,
- a zeptá se, zda to odpovídá jeho potřebě,
- pokud ne, pomůže uživateli lépe formulovat dotaz a celý proces zopakuje.

Takto se oba „domlouvají“, dokud se nedostanou k výsledku, se kterým je uživatel spokojený. 

---

## Vylepšení našeho agenta o nové prvky

Náš agent nyní funguje na stejném základním principu jako v kapitole 9 (generování dotazů → vyhledání služeb → filtrování → sestavení postupu). Hlavní rozdíl je v tom, že celý proces je rozšířen o **interaktivní krokování s uživatelem**. Přibyly tři důležité změny:

1. **Interaktivní smyčka** – hlavní funkce (`main`) běží v nekonečném cyklu. Po každém vyhledání služeb dostává uživatel možnost vyjádřit se, zda jsou výsledky v pořádku.
2. **Vysvětlení nalezených služeb** – nová funkce `vysvetli_sluzby` shrne uživateli, jak nalezené služby souvisejí s jeho situací a jak mu mohou pomoci.
3. **Nápověda pro lepší formulaci dotazu** – pokud uživatel není spokojen, funkce `pomoz_zlepsit_dotaz` mu poradí, jak situaci popsat jinak. Uživatel pak může dotaz přeformulovat a celý proces běží znovu.

Finální postup (`vygeneruj_finalni_postup`) se volá až v okamžiku, kdy uživatel potvrdí, že je se službami spokojen.

---

## Funkce `vysvetli_sluzby`

Tato funkce převezme seznam nalezených relevantních služeb a vytvoří srozumitelné shrnutí pro uživatele:

```python
def vysvetli_sluzby(sluzby: dict, user_query: str) -> Postup:

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

    response = client.responses.create(
        model="gpt-5-mini",
        input=[
            {
                "role": "developer",
                "content": "Jsi odborník na pomoc uživateli při řešení jeho různých životních situací v občanském životě. Uživatel popsal svoji situaci. Našel jsi několik služeb, které by mu mohly pomoci s vyřešením jeho situace. Shrň uživateli, jak by mu mohly tyto služby pomoci."
            },{
                "role": "developer",
                "content": "Odpovídej *VÝHRADNĚ* na základě přiloženého XML se seznamem služeb (viz <sluzby> dále). Každá služba je uvedena ve struktuře: <id> (jednoznačný identifikátor služby), <nazev> (krátký název služby), <detail> (detailní informace o službě) a <kroky> (úřední kroky, v rámci kterých je potřeba službu řešit). <detail> se skládá z následujícíh částí: <popis> (Základní vymezení služby a upřesnění názvu, pokud není dost jednoznačný.), <benefit> (Atribut popisuje, jaký přínos má pro klienta využití služby.), <jak-resit> (Jakým způsobem se služba řeší elektronicky včetně případného ID datové schránky, mailové adresy či jiných digitálních kanálů.), <kdy-resit> (Popisuje, v jakou chvíli může nebo musí být iniciováno čerpání služby.), <resit-pokud> (Vymezení toho, kdo může službu využívat a za jakých podmínek se ho týká.), <zpusob-vyrizeni> (Co potřebuje klient, aby mohl službu řešit elektronicky (typicky doklady, žádosti apod.)"
            },{
                "role": "developer",
                "content": "*Nikdy nesmíš ve shrnutí poskytovat radu ani informace v oboru, kterého se životní situace uživatele týká, např. lékařské rady, stavební rady, atd.* Uživateli pouze můžeš napsat, aby odborníka vyhledal a navštívil bez jakýchkoliv časových, situačních či jiných podmínek a doporučení."
            },{
                "role": "assistant",
                "content": "Jaká je vaše životní situace?"
            },{
                "role": "user",
                "content": user_query
            },{
                "role": "assistant",
                "content": "Zde jsou služby, které by vám mohly pomoci:"
            },{
                "role": "assistant",
                "content": sluzby_xml
            },{
                "role": "user",
                "content": "Shrň mi, jak by mi tyto služby mohly pomoci vyřešit mou situaci."
            }
        ],
        text={"verbosity": "medium"},
        reasoning={"effort": "medium"}
    )

    return response.output_text
```

Díky tomu uživatel ví, proč byly služby vybrány a jak mu mohou pomoci.

---

## Funkce `pomoz_zlepsit_dotaz`

Pokud uživatel není s výsledkem spokojen, spustíme tuto funkci. Jejím cílem je doporučit uživateli, **jak situaci popsat lépe**:

```python
def pomoz_zlepsit_dotaz(sluzby: dict, user_query: str) -> Postup:

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

    response = client.responses.create(
        model="gpt-5-mini",
        input=[
            {
                "role": "developer",
                "content": "Jsi odborník na pomoc uživateli při řešení jeho různých životních situací v občanském životě. Uživatel popsal svoji situaci. Bohužel se ti nepodařilo najít služby, se kterými by uživatel souhlasil, že mu pomohou situaci vyřešit. Pomoz mu popsat jeho situaci lépe, aby byla větší šance, že najdeme pro něj relevantní služby."
            },{
                "role": "developer",
                "content": "Odpovídej *VÝHRADNĚ* na základě následujícího popisu životní situace od uživatele, na základě kterého jsme ale nenalezli vhodné služby a také na základě přiloženého XML se seznamem služeb, které jsme nalezli ale uživatel s nimi nebyl spokojený (viz <sluzby> dále, seznam může být i prázdný). Každá služba je uvedena ve struktuře: <id> (jednoznačný identifikátor služby), <nazev> (krátký název služby), <detail> (detailní informace o službě) a <kroky> (úřední kroky, v rámci kterých je potřeba službu řešit). <detail> se skládá z následujícíh částí: <popis> (Základní vymezení služby a upřesnění názvu, pokud není dost jednoznačný.), <benefit> (Atribut popisuje, jaký přínos má pro klienta využití služby.), <jak-resit> (Jakým způsobem se služba řeší elektronicky včetně případného ID datové schránky, mailové adresy či jiných digitálních kanálů.), <kdy-resit> (Popisuje, v jakou chvíli může nebo musí být iniciováno čerpání služby.), <resit-pokud> (Vymezení toho, kdo může službu využívat a za jakých podmínek se ho týká.), <zpusob-vyrizeni> (Co potřebuje klient, aby mohl službu řešit elektronicky (typicky doklady, žádosti apod.)"
            },{
                "role": "developer",
                "content": "*Nikdy nesmíš uživatele navádět k vylepšení dotazu směrem do oboru, kterého se životní situace uživatele týká, např. popis zdravotního stavu, odborného stavu stavby, atd.* Uživateli můžeš doporučit, aby lépe popsal svoji situaci v tomto kontextu, ale ne konkrétními odbornými ukazateli."
            },{
                "role": "assistant",
                "content": "Jaká je vaše životní situace?"
            },{
                "role": "user",
                "content": user_query
            },{
                "role": "assistant",
                "content": "Zde jsou služby, které by vám mohly pomoci:"
            },{
                "role": "assistant",
                "content": sluzby_xml
            },{
                "role": "user",
                "content": "Nejsem spokojen s navrženými službami. Pomoz mi lépe popsat mou situaci, abychom měli větší šanci najít relevantní služby."
            }
        ],
        text={"verbosity": "medium"},
        reasoning={"effort": "medium"}
    )

    return response.output_text
```

Agent tak uživateli nenásilně poradí, jak zadat přesnější vstup.

---

## Hlavní smyčka programu

Hlavní logika je nyní řízená nekonečným cyklem:

1. Uživatelský vstup → `generuj_navrhy_vyhledavacich_dotazu`
2. Vyhledání a filtrování služeb.
3. Pokud jsou služby nalezeny:
   - zobrazíme jejich shrnutí (`vysvetli_sluzby`),
   - zeptáme se uživatele, zda je spokojen,
   - pokud ano → vygenerujeme finální postup a cyklus končí.
4. Pokud služby nejsou vhodné → použijeme `pomoz_zlepsit_dotaz`, aby uživatel mohl situaci upřesnit, a cyklus běží dál.

Nekonečná smyčka a závěr v `main` funkci:

```python
while True:
    navrh_dotazy = generuj_navrhy_vyhledavacich_dotazu(user_query)
    if navrh_dotazy is None:
        print("Nepodařilo se vygenerovat vyhledávací dotazy.")
        return

    print("Navržené dotazy pro vyhledání služeb:")
    for dotaz in navrh_dotazy:
        print(f"- {dotaz}")

    sluzby = vyhledej_sluzby(navrh_dotazy)
    filtrovane_sluzby = {}
    
    if sluzby and len(sluzby) > 0:
        
        print(f"Nalezeno služeb: {len(sluzby)}")

        filtrovane_sluzby = filtruj_relevantni_sluzby(sluzby, user_query)
        
        if len(filtrovane_sluzby) > 0:
            
            print(f"Nalezeno relevantních služeb: {len(filtrovane_sluzby)}")

            vysvetleni = vysvetli_sluzby(filtrovane_sluzby, user_query)
            print(vysvetleni)

            user_decision = input("Chcete na základě těchto služeb vygenerovat konkrétní postup? Pokud ano, napište 'ano' a stiskněte Enter. Pokud ne, napište cokoliv jiného a stiskněte Enter.")
            if user_decision.strip().lower() == 'ano':
                print("Generuji postup...")
                break
    
    pomoc_pro_zlepseni = pomoz_zlepsit_dotaz(filtrovane_sluzby, user_query)
    print(pomoc_pro_zlepseni)

    user_query = input("Zkuste nyní popsat svoji situaci lépe:")

postup = vygeneruj_finalni_postup(filtrovane_sluzby, user_query)
if not postup or not postup.uvod:
    print("Nepodařilo se vygenerovat postup.")
print("AI odpověď:")
print(f"{postup.uvod}\n")
if postup.kroky and len(postup.kroky) > 0:
    for krok in postup.kroky:
        print(f"{krok.poradi}. {krok.nazev} ({krok.sluzba_id})\n   {krok.popis}\n")
```

---

## Shrnutí

- Náš agent funguje stále na stejném základu jako v kapitole 9.
- **Vylepšení spočívá v přidání interaktivního dialogu** s uživatelem.
- Agent nově:
  - vysvětluje, co našel,
  - dává uživateli možnost souhlasit nebo nesouhlasit,
  - nabízí pomoc s upřesněním dotazu.
- Díky smyčce se hledání a filtrování služeb opakuje, dokud uživatel nepotvrdí spokojenost.

Tento přístup je bližší reálným scénářům, kde agenti s uživateli **vedou konverzaci**, místo aby vše proběhlo jednorázově.

