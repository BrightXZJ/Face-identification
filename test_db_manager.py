# test_db_manager.py
from core.db_manager import DatabaseManager

print("Initializing DatabaseManager...")
db = DatabaseManager()

print("\nAll registered persons:")
for name in db.get_all_names():
    emb = db.get_embedding_by_name(name)
    print(f"  - {name} : embedding shape {emb.shape}, norm {float(emb.dot(emb)):.4f}")

print("\n✅ Database manager test complete.")