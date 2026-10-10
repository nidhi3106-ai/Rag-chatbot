import chromadb

client = chromadb.Client()

print("ChromaDB connected!")

collection = client.get_or_create_collection(
    name="college_handbook"
)

print("Collection created!")

chunks = [
    "Students must maintain a minimum of 75% attendance in each subject to be eligible for the semester examination.",
    "Students must carry their college identity card during examinations.",
    "Students can borrow up to 3 books at a time."
]

collection.add(
    ids=["chunk1", "chunk2", "chunk3"],
    documents=chunks
)

print("Chunks stored!")

question = "What is the attendance requirement?"

results = collection.query(
    query_texts=[question],
    n_results=1
)

print("Relevant chunk:")
print(results["documents"][0][0])