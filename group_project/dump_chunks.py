import json, weaviate
from weaviate.classes.init import Auth, AdditionalConfig, Timeout
from src.config import settings

client = weaviate.connect_to_weaviate_cloud(
    cluster_url=settings.WEAVIATE_URL,
    auth_credentials=Auth.api_key(settings.WEAVIATE_API_KEY),
    skip_init_checks=True,
    additional_config=AdditionalConfig(timeout=Timeout(init=30, query=60)),
)
try:
    col = client.collections.get(settings.WEAVIATE_COLLECTION)
    rows = []
    for obj in col.iterator():
        p = obj.properties
        rows.append({
            "uuid": str(obj.uuid),
            "source": p.get("source", ""),
            "doc_type": p.get("doc_type", ""),
            "chunk_index": p.get("chunk_index", 0),
            "content": p.get("content", ""),
        })
    rows.sort(key=lambda r: (r["doc_type"], r["source"], r["chunk_index"]))
    with open("data/chunks_export.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"Total chunks: {len(rows)}  | Collection: {settings.WEAVIATE_COLLECTION}")
    by_src = {}
    for r in rows:
        by_src[r["source"]] = by_src.get(r["source"], 0) + 1
    for s, n in sorted(by_src.items()):
        print(f"  {n:4d}  {s}")
    print("\nSaved -> data/chunks_export.json")
finally:
    client.close()
