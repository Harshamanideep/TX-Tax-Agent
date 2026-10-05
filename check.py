from src.rag import get_store

r = get_store().get(where_document={"$contains": "calendar year"})
for doc, meta in zip(r["documents"], r["metadatas"]):
    print("PAGE", meta["page"], "|", doc[:400].replace("\n", " "), "\n")
print(len(r["documents"]), "chunks found")