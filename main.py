from pathlib import Path

import chromadb

client = chromadb.PersistentClient(path=str(Path("parser/tests/output/chroma_db")))

for c in client.list_collections():
    print(c.name)
