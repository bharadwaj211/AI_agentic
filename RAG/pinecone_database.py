import os
import uuid
from openai import OpenAI
from pinecone import Pinecone
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# Initialization of Clients
load_dotenv()  # loads .env file
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

index_name = "rag-documentation"
index = pc.Index(index_name)

reader = PdfReader("C:\\Users\\bhara\\Downloads\\rag_documentation.pdf")

full_text = ""
for page in reader.pages:
    full_text += page.extract_text() + "\n"

print(" Started PDF upload to the pinecone vector database ")

# Split the data into Chunks

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_text(full_text)

print(f" Total chunks created: {len(chunks)}")


# Embed + Store

for chunk in chunks:
    embedding = client.embeddings.create(
        model="text-embedding-3-large",
        input=chunk,
        dimensions=1024  # Crucial step
    ).data[0].embedding

    index.upsert([
        {
            "id": str(uuid.uuid4()),
            "values": embedding,
            "metadata": {"text": chunk}
        }
    ])

print(" PDF successfully uploaded to Pinecone ")