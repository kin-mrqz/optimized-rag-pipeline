from langchain.embeddings import HuggingFaceEmbeddings
from langchain_google_community import GoogleDriveLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma


def load_documents(docs_links, credentials_path, token_path, pdf_path):
    g_docs = GoogleDriveLoader(
        document_ids = [link.split("/")[5] for link in docs_links],
        credentials_path=credentials_path,
        token_path=token_path
    ).load()

    pdf_docs = PyPDFLoader(pdf_path).load()

    return g_docs + pdf_docs


def create_vectorstore(documents, embedding_model, persist_directory, chunk_size=300, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    splits = text_splitter.split_documents(documents)
    vectorestore = Chroma.from_documents(
        documents=splits,
        embedding=embedding_model,
        persist_directory=persist_directory
    )
    vectorestore.persist()

    return vectorestore

    
if __name__ == "__main__":
    docs_links = [
            "https://docs.google.com/document/d/1MvX9CTrVcoWg7WLAscq2MmnhTIrR0hZIGkpJMqhgflo/edit?tab=t.0",
            "https://docs.google.com/document/d/1NcV9_JGjMfA4WlihW3vNTduy24NBWdy1RWXrA2W0BIk/edit?usp=sharing",
            "https://docs.google.com/document/d/193rx2Rh6u-Ud40k-rgnqSQs-94SvHdeXPrPxOWK59X0/edit?usp=sharing",
            "https://docs.google.com/document/d/1vRDsn5o5mdymOEJ_O0tS4wcOjsAjt_2mLZqFfvgDUOs/edit?usp=sharing",
            "https://docs.google.com/document/d/1JceLBII727AZzSrDFfdGthJ1G4PhCDsA8sEm_dQMVr0/edit?usp=sharing",
            "https://docs.google.com/document/d/1yonU4qcysNkgd0BvbFmeIW9NF2ARErRJVW8QZynJyvM/edit?usp=sharing",
            "https://docs.google.com/document/d/1bq2AE1Jy6cQFt1xgjqtkof12Lw6F6fujqTlN1nZnh0A/edit?usp=sharing",
            "https://docs.google.com/document/d/1i-OcQeo7XOG83gS2ay2u0SLMWs4f8FG0JE_7l87qJkw/edit?usp=sharing",
            "https://docs.google.com/document/d/1PyZE8v3S3aUY66lFn97q0vaXDHc60lyso2oUFP0htjY/edit?usp=sharing",
            "https://docs.google.com/document/d/1Dudd7-6yl_UQrxfGa3MJZlKOfzHQdqWg0fb8Z9RzBec/edit?usp=sharing",
            "https://docs.google.com/document/d/1cjhrhCccuwiIh0ujj8QeamJ2JHhI6CjPmO84t1DSRZ0/edit?usp=sharing",
            "https://docs.google.com/document/d/1ESlfU6v8jseFlllZb3eaUeCJt69EIMsZiyMrDac-wX8/edit?usp=sharing",
            "https://docs.google.com/document/d/1xQhAkC3oP2cb262EjaHCV6CxEgeEGUsKrC8pH2p6RiY/edit?usp=sharing"  
        ]
    # Paths to credentials and documents
    credentials_path = "./src/credentials_google.json"
    token_path = "./src/token.json"
    pdf_path = "./src/data/resources/wine_food_pairing_knowledge.pdf"

    persist_directory = "src/chroma_db"
    documents = load_documents(docs_links, credentials_path, token_path, pdf_path)
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectorstore = create_vectorstore(
            documents=documents,
            embedding_model=embedding_model,
            persist_directory=persist_directory,
            chunk_size=300,
            chunk_overlap=50
        )