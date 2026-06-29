import glob
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from dotenv import load_dotenv

load_dotenv()



# 1. Read every .txt file in the lore folder
documents = []
for filepath in glob.glob("lore/*.txt"):
    with open(filepath, "r", encoding="utf-8") as f:
        documents.append(f.read())

# 2. Set up the splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

# 3. Split the text into chunks
chunks = []
for doc in documents:
    chunks.extend(splitter.split_text(doc))

# 4. Report what happened
print(f"Loaded {len(documents)} file(s)")
print(f"Created {len(chunks)} chunks")
print("--- First chunk ---")
print(chunks[0])

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


Chroma.from_texts(
    texts=chunks,                    # your full chunk list
    embedding=embeddings,
    persist_directory="chroma_db"
)
print("Stored all chunks in Chroma")