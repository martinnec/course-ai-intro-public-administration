from government_services_store import GovernmentServicesStore

store = GovernmentServicesStore()

store.load_services()

stats = store.get_services_embedding_statistics()

# Počet načtených služeb a rychlá statistika embeddingů (ChromaDB)
total_services = stats.get("total_services", 0)
total_embeddings = stats.get("total_embeddings", 0)
coverage = stats.get("coverage_percentage", 0.0)

print(f"Načteno {total_services} služeb.")
print(f"Embeddings v ChromaDB: {total_embeddings} (coverage: {coverage}%)")

query = "Bolí mě hlava a mám asi horečku. Co si na to mám vzít? Co mám dělat? A mohu jít do práce?"
results = store.search_services(query, k=10)
for s in results:
    print(f"{s.id}: {s.name}")

if results:
    service_id = results[0].id
    detail = store.get_service_detail_by_id(service_id)
    steps = store.get_service_steps_by_id(service_id)
    print("Detail služby:", detail or "Detail není k dispozici.")
    print(f"Kroky služby:")
    if steps:
        for step in steps:
            print("-", step)
    else:
        print("Žádné digitální kroky nebyly pro službu nalezeny.")