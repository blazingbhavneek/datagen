from chromadb import PersistentClient

client = PersistentClient(path="/run/media/blazingbhavneek/Common/Code/datagen/parser/tests/output/chroma_db")

cols = client.list_collections()
print("collections:", len(cols), [c.name for c in cols])

for c in cols:
    col = client.get_collection(c.name)

    # count vectors
    count = col.count()
    
    # try reading creation time from metadata
    # created = col.get()["metadata"].get("created_at", None)
    
    print(f"{c.name}: chunks={count}")
