"""
Government Services Store - jednoduché in-memory úložiště pro specifikace služeb veřejné správy.

Co se v souboru naučíte:
1) Jak načíst data služeb (SPARQL → volitelně uložit/číst z lokální cache).
2) Jak doplnit podrobnosti z lokálního JSON souboru (čištění HTML, klíčová slova).
3) Jak spočítat vektorové reprezentace (embeddings) pro sémantické vyhledávání.
4) Jak pracovat s vektorovým indexem (Chroma) a dotazem „najdi podobné služby“.
"""

from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import re
from urllib.parse import urlparse
from rdflib import Graph
import json
import os
from pathlib import Path
import openai
import chromadb

# Konfigurace důležitých voleb na jednom místě:
# - EMBEDDINGS_MODEL: název embedding modelu pro výpočet vektorů
# - CHROMA_PATH: složka s lokálním úložištěm Chroma (vektorová DB)
# - SERVICES_CACHE: JSON cache se seznamem služeb (rychlé načtení na lekci)
# - DETAILS_PATH: JSON s detailními informacemi o službách (popisy, klíčová slova)
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "text-embedding-3-small")
CHROMA_PATH = Path("data/chromadb")
SERVICES_CACHE = Path("data/government_services_data.json")
DETAILS_PATH = Path("data/detailni-popis-sluzby-vs.json")


@dataclass
class GovernmentService:
    """Reprezentuje jednu službu veřejné správy."""
    uri: str
    id: str
    name: str
    description: str
    keywords: List[str] = None

    def __post_init__(self):
        """Po vytvoření objektu:
        - normalizuje pole `keywords` na prázdný seznam,
        - pokud chybí `id`, odvodí jej z `uri` (fragment nebo poslední segment cesty).
        """
        if self.keywords is None:
            self.keywords = []

        if not self.id and self.uri:
            self.id = _extract_id_from_uri(self.uri)

        if not self.id:
            raise ValueError("Service ID could not be determined from URI")


# Utility functions for processing URIs and text
def _extract_id_from_uri(uri: str) -> str:
    """Vrátí ID extrahované z URI (nejdříve fragment, jinak poslední segment cesty).
    
    Funguje jak s absolutními URI, tak s relativními cestami jako 'detailní-popis-služby-vs/S6192'.
    """
    if not uri:
        return ""
    
    parsed = urlparse(uri)
    if parsed.fragment:
        return parsed.fragment
    
    # Pro absolutní URI použij parsed.path, pro relativní URI použij celý string
    path_to_parse = parsed.path if parsed.scheme or parsed.netloc else uri
    
    if path_to_parse:
        parts = [p for p in path_to_parse.split("/") if p]
        if parts:
            return parts[-1]
    return ""


def _strip_html(text: str) -> str:
    """Odstraní HTML značky z textu (detaily služeb bývají formátované HTML)."""
    return re.sub(r"<[^>]+>", "", text) if text else text


def _get_cs(obj) -> Optional[str]:
    """Bezpečně vrátí českou hodnotu z objektu ve tvaru {'cs': '...'}; jinak None."""
    return obj.get("cs") if isinstance(obj, dict) and obj.get("cs") else None


def _safe_get_cs_from_item(item: dict, key: str) -> str:
    """Bezpečně extrahuje českou hodnotu z položky a odstraní HTML značky."""
    val = _get_cs(item.get(key))
    return _strip_html(val) if val else "Není k dispozici"


class GovernmentServicesStore:
    """
    Veřejné API třídy:
      - load_services(): načte služby (z cache, případně ze SPARQL) a doplní detaily
      - search_services_by_keywords(): jednoduché fulltextové vyhledávání
      - search_services_semantically(): sémantické vyhledávání na embeddings + Chroma
      - get_service_by_id(), get_all_services(), get_services_count()
      - get_service_detail_by_id(), get_service_howto_by_id(): získání rozšířených informací
      - get_service_steps_by_id(): získání kroků (úkonů) služby z SPARQL
      - get_embedding_statistics(): metriky vektorového indexu

    Interní kroky:
      - _load_from_local(), _store_to_local(): práce s cache JSON
      - _load_from_external_store(): SPARQL dotaz na otevřená data ČR
      - _load_auxiliary_details(): sloučení detailů (popisy/klíčová slova)
      - _initialize_semantic_search(): příprava OpenAI klienta a Chroma
      - _compute_embeddings(): výpočet embeddingů pro všechny služby
    """

    def __init__(self):
        """Vytvoří prázdné úložiště a připraví stav pro semantické vyhledávání."""
        self._services: Dict[str, GovernmentService] = {}
        self._services_list: List[GovernmentService] = []  # synchronizovaná kopie hodnot pro jednoduché procházení

        # Komponenty pro sémantické vyhledávání (lazy inicializace)
        self._openai_client = None
        self._chroma_client = None
        self._collection = None
        self._embeddings_computed = False

    def add_service(self, service: GovernmentService) -> None:
        """Přidá jednu službu do úložiště a zaktualizuje interní seznam."""
        self._services[service.id] = service
        self._services_list = list(self._services.values())

    def clear_services(self) -> None:
        """Vyprázdní úložiště (slovník i seznam)."""
        self._services.clear()
        self._services_list.clear()

    def load_services(self) -> None:
        """Načítání služeb s „fallback“ strategií."""
        if len(self._services) > 0:
            self.clear_services()

        if SERVICES_CACHE.exists():
            try:
                self._load_services_from_local_cache()
            except Exception as local_error:
                print(f"Warning [load_services]: Failed to load from local file: {local_error}")
                self.clear_services()

        if len(self._services) == 0:
            try:
                self._load_services_from_external_store()
                self._load_services_with_details()
                try:
                    print("Debug [load_services]: Computing embeddings for semantic search...")
                    self._compute_services_embeddings()
                except Exception as embedding_error:
                    print(f"Warning [load_services]: Failed to compute embeddings: {embedding_error}")
                    print("Warning [load_services]: Semantic search will not be available until embeddings are computed manually.")
                self._store_services_to_local_cache()
            except Exception as e:
                raise RuntimeError(f"Warning [load_services]: Failed to load services from both local and external sources: {e}")

    def _load_services_from_local_cache(self) -> None:
        """Načte služby z lokální cache (JSON)."""
        try:
            with open(SERVICES_CACHE, "r", encoding="utf-8") as f:
                data = json.load(f)
            for item in data:
                service = GovernmentService(**item)
                self.add_service(service)
        except Exception as e:
            raise RuntimeError(f"Warning [_load_services_from_local_cache]: Failed to load services from local file: {e}")

    def _load_services_from_external_store(self) -> None:
        """Načte služby ze SPARQL endpointu a uloží je do paměti."""
        sparql_endpoint = "https://rpp-opendata.egon.gov.cz/odrpp/sparql/"
        g = Graph()
        sparql_str = f"""
        PREFIX rppl: <https://slovník.gov.cz/legislativní/sbírka/111/2009/pojem/>
        PREFIX rppa: <https://slovník.gov.cz/agendový/104/pojem/>
        SELECT ?uri ?name ?description
        WHERE {{
            SERVICE <{sparql_endpoint}> {{
            ?uri a rppl:služba-veřejné-správy ;
                rppa:má-název-služby ?name ;
                rppa:má-popis-služby ?description .
            }}
        }}
        """
        results = g.query(sparql_str)
        for row in results:
            uri = str(row.uri)
            name = str(row.name) if row.name else ""
            description = str(row.description) if row.description else ""
            service = GovernmentService(uri=uri, id="", name=name, description=description, keywords=[])
            self.add_service(service)

    def _load_services_with_details(self) -> None:
        """Rozšíří data o popisy a klíčová slova z lokálního JSON souboru."""
        if not DETAILS_PATH.exists():
            print(f"Warning [_load_services_with_details]: Auxiliary details file not found at {DETAILS_PATH}")
            return
        with open(DETAILS_PATH, "r", encoding="utf-8") as f:
            details_data = json.load(f)

        # Check if data has the expected structure with "položky" key
        items = details_data.get("položky", details_data) if isinstance(details_data, dict) else details_data
        
        for item in items:
            # Handle both "iri" and "id" fields, prefer "id" if available
            if isinstance(item, dict):
                # Extract service ID from either "id" or "iri" field using consistent logic
                raw_id = item.get("id", "") or item.get("iri", "")
                service_id = _extract_id_from_uri(raw_id) if raw_id else ""
            else:
                continue  # Skip non-dictionary items
            if service_id in self._services:
                service = self._services[service_id]
                if 'popis' in item:
                    clean_text = _safe_get_cs_from_item(item, 'popis')
                    if clean_text != "Není k dispozici":
                        service.description += " " + clean_text
                if 'jaký-má-služba-benefit' in item:
                    clean_text = _safe_get_cs_from_item(item, 'jaký-má-služba-benefit')
                    if clean_text != "Není k dispozici":
                        service.description += " " + clean_text
                if 'klíčová-slova' in item and item['klíčová-slova'] is not None:
                    # Extract Czech keywords from the list of keyword objects
                    for keyword_obj in item['klíčová-slova']:
                        if isinstance(keyword_obj, dict) and 'cs' in keyword_obj and keyword_obj['cs']:
                            service.keywords.append(keyword_obj['cs'])

    def _compute_services_embeddings(self) -> None:
        """Spočítá embeddingy pro všechny služby, které je ještě nemají uložené v Chroma."""
        if not self._services_list:
            print("Warning [_compute_services_embeddings]: No services to compute embeddings for.")
            return

        self._initialize_search()
        existing_ids = set()
        try:
            existing_data = self._collection.get()
            existing_ids = set(existing_data['ids']) if existing_data['ids'] else set()
        except Exception:
            pass

        new_services = [s for s in self._services_list if s.id not in existing_ids]
        batch_size = 500
        for i in range(0, len(new_services), batch_size):
            batch = new_services[i:i+batch_size]
            service_texts = []
            for s in batch:
                text = f"{s.name}. {s.description}"
                if s.keywords:
                    for kw in s.keywords:
                        text += f" {kw}"
                service_texts.append(text)
            embeddings_response = self._openai_client.embeddings.create(
                input=service_texts,
                model=EMBEDDINGS_MODEL
            )
            embeddings = [e.embedding for e in embeddings_response.data]
            self._collection.add(
                ids=[s.id for s in batch],
                embeddings=embeddings,
                metadatas=[{"name": s.name, "description": s.description} for s in batch],
                documents=service_texts
            )
            print(f"Debug [_compute_services_embeddings]: Computed embeddings for services {i} - {i + batch_size - 1}.")
        self._embeddings_computed = True

    def _initialize_search(self) -> None:
        """Připraví OpenAI klienta a ChromaDB pro sémantické vyhledávání."""
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise RuntimeError("Warning [_initialize_search]: OPENAI_API_KEY environment variable is not set")

        self._openai_client = openai
        self._openai_client.api_key = api_key

        self._chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        self._collection = self._chroma_client.get_or_create_collection("government_services")

    def close(self) -> None:
        """Uzavře připojení k ChromaDB a uklidí resources."""
        if hasattr(self, '_chroma_client') and self._chroma_client is not None:
            try:
                # ChromaDB PersistentClient doesn't have explicit close method,
                # but we can clear the references to help with cleanup
                self._collection = None
                self._chroma_client = None
            except Exception as e:
                print(f"Warning [close]: Error during cleanup: {e}")
        
        # Clear OpenAI client reference
        self._openai_client = None

    def _store_services_to_local_cache(self) -> None:
        """Uloží aktuální seznam služeb do lokální cache (JSON)."""
        try:
            output_dir = SERVICES_CACHE.parent
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(SERVICES_CACHE, "w", encoding="utf-8") as f:
                json.dump([s.__dict__ for s in self._services_list], f, ensure_ascii=False, indent=2)
        except Exception as e:
            raise RuntimeError(f"Warning [_store_services_to_local_cache]: Failed to store services to local file: {e}")

    def search_services(self, query: str, k: int = 10) -> List[GovernmentService]:
        """Najde služby sémanticky podobné dotazu."""
        print(f"Debug [search_services]: called with query='{query}', k={k}")
        if not query.strip():
            print("Warning [search_services]: Empty query provided. Returning empty list.")
            return []

        if not self._collection:
            self._initialize_search()

        query_embedding_response = self._openai_client.embeddings.create(
            input=[query],
            model=EMBEDDINGS_MODEL
        )
        query_embedding = query_embedding_response.data[0].embedding

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
        ids = results['ids'][0]
        return [self._services[i] for i in ids if i in self._services]

    def get_service_detail_by_id(self, service_id: str) -> Optional[str]:
        """Vrátí rozšířené textové detaily o službě (z details JSON)."""
        if not DETAILS_PATH.exists():
            print(f"Warning [get_service_detail_by_id {service_id}]: Details file not found at {DETAILS_PATH}")
            return None
        with open(DETAILS_PATH, "r", encoding="utf-8") as f:
            details_data = json.load(f)

        # Check if data has the expected structure with "položky" key
        items = details_data.get("položky", details_data) if isinstance(details_data, dict) else details_data
        
        for item in items:
            if isinstance(item, dict):
                # Extract service ID using consistent logic
                raw_id = item.get("id", "") or item.get("iri", "")
                item_id = _extract_id_from_uri(raw_id) if raw_id else ""
                if item_id == service_id:
                    return f"Popis: {_safe_get_cs_from_item(item, 'popis')}\nKde a jak službu řešit elektronicky: {_safe_get_cs_from_item(item, 'kde-a-jak-službu-řešit-el')}\nKdy službu řešit: {_safe_get_cs_from_item(item, 'kdy-službu-řešit')}\nTýká se uživatele pokud: {_safe_get_cs_from_item(item, 'týká-se-vás-to-pokud')}\nZpůsob vyřízení: {_safe_get_cs_from_item(item, 'způsob-vyřízení-el')}"

        return None

    def get_service_steps_by_id(self, service_id: str) -> List[str]:
        """
        Vrátí seznam „úkonů/kroků“ vybrané služby pomocí SPARQL dotazu.
        Každá položka je ve formátu: "název_úkonu: popis_úkonu".
        Filtrovány jsou digitální úkony realizované kanálem „DATOVA_SCHRANKA“.

        Args:
            service_id: ID služby, pro kterou chceme kroky získat.

        Returns:
            Seznam textových kroků ve formátu "název: popis". Pokud nic nenajdeme, vrací prázdný seznam.

        Raises:
            RuntimeError: Pokud selže dotaz na SPARQL endpoint.
        """
        if not service_id:
            return []

        sparql_endpoint = "https://rpp-opendata.egon.gov.cz/odrpp/sparql/"

        sparql_str = f"""
        PREFIX rppl: <https://slovník.gov.cz/legislativní/sbírka/111/2009/pojem/>
        PREFIX rppa: <https://slovník.gov.cz/agendový/104/pojem/>
        SELECT ?step ?name ?description
        WHERE {{
          SERVICE <{sparql_endpoint}> {{
            <https://rpp-opendata.egon.gov.cz/odrpp/zdroj/služba/{service_id}> rppa:skládá-se-z-úkonu ?step .
            ?step rppa:je-digitální true .
            ?step rppa:má-název-úkonu-služby ?name ;
                  rppa:má-popis-úkonu-služby ?description ;
                  rppa:je-realizován-kanálem/rppa:má-typ-obslužného-kanálu <https://rpp-opendata.egon.gov.cz/odrpp/zdroj/typ-obslužného-kanálu/DATOVA_SCHRANKA>
          }}
        }}
        ORDER BY ?step
        """

        try:
            g = Graph()
            results = g.query(sparql_str)

            steps: List[str] = []
            for row in results:
                try:
                    name = str(row.name) if row.name else ""
                    description = str(row.description) if row.description else ""

                    # Přeskočíme nekompletní záznamy bez názvu
                    if not name:
                        continue

                    # Výstupní formát "název: popis"
                    step_text = f"{name}: {description}" if description else name
                    steps.append(step_text)

                except Exception as step_error:
                    print(f"Warning [get_service_steps_by_id {service_id}]: Failed to process step from row {row}: {step_error}")
                    continue

            print(f"Debug [get_service_steps_by_id {service_id}]: Successfully retrieved {len(steps)} steps for service {service_id}")
            return steps

        except Exception as e:
            raise RuntimeError(f"Warning [get_service_steps_by_id {service_id}]: Failed to retrieve steps for service {service_id}: {e}")

    def get_services_embedding_statistics(self) -> Dict[str, Any]:
        """Vrátí základní statistiky o stavu embeddingů."""
        if not self._collection:
            try:
                self._initialize_search()
            except Exception:
                return {
                    "embeddings_computed": False,
                    "total_embeddings": 0,
                    "total_services": len(self._services_list),
                    "coverage_percentage": 0.0
                }

        total_embeddings = self._collection.count()
        total_services = len(self._services_list)
        coverage = (total_embeddings / total_services * 100) if total_services > 0 else 0

        return {
            "embeddings_computed": self._embeddings_computed,
            "total_embeddings": total_embeddings,
            "total_services": total_services,
            "coverage_percentage": round(coverage, 2)
        }