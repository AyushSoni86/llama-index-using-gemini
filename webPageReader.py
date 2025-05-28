import os
from dotenv import load_dotenv

from llama_index.core import VectorStoreIndex, Settings
from llama_index.readers.web import SimpleWebPageReader
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

load_dotenv()

def main(url: str) -> None:
    # Set Gemini API key
    api_key = os.environ["GEMINI_API_KEY"]

    # Use Gemini for LLM and Hugging Face for embeddings
    Settings.llm = GoogleGenAI(
        model="gemini-1.5-flash",  # or gemini-1.5-pro if needed
        api_key=api_key
    )

    Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Load document
    documents = SimpleWebPageReader(html_to_text=True).load_data(urls=[url])

    # Build index
    index = VectorStoreIndex.from_documents(documents=documents)
    # print(index)

    # Optional: Query
    query_engine = index.as_query_engine()
    response = query_engine.query("What is a LlamaIndex document?")
    print(response)


if __name__ == '__main__':
    main(url='https://docs.llamaindex.ai/en/stable/module_guides/loading/documents_and_nodes/')
