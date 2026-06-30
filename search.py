from langchain_openai import OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from dotenv import load_dotenv
question = "What is Nen?"

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatAnthropic(model="claude-opus-4-8")  

# Load the EXISTING database (note: not from_texts this time)
db = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

def ask_question(question):
    results = db.similarity_search(question, k=3)

    # Combine the retrieved chunks into one block of text
    context = "\n\n".join(r.page_content for r in results)

    # Build the prompt
    prompt = f"""Answer the question using ONLY the context below.
    If the answer isn't in the context, say so.

    Context:
    {context}

    Question: {question}"""

    # Ask Claude
    answer = llm.invoke(prompt)
    return answer.content

def build_dossier(character):
    results = db.similarity_search(character, k=8)

    context = "\n\n".join(r.page_content for r in results)

    prompt = f"""Build a character dossier using ONLY the context below.
    Organize it into these sections:
    - Overview
    - Abilities / Nen
    - Affiliations
    - Key Events

    If a section has no information in the context, write "Not specified in available lore."

    Context:
    {context}

    Character: {character}"""

    answer = llm.invoke(prompt)
    return answer.content


result = build_dossier("Killua")
print(result)
