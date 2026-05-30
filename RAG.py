# Step 1. Load and extract text from all PDF files

import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

# Safe folder path
folder = Path(r"C:\Users\Asif\Desktop\Thesis_GM\Sbp_FMR_Policies")

documents = []

for file_path in folder.glob("*.pdf"):
    loader = PyPDFLoader(str(file_path))
    docs = loader.load()
    for d in docs:
        d.metadata["source"] = file_path.name
    documents.extend(docs)

print(f"Loaded {len(documents)} pages from {len(list(folder.glob('*.pdf')))} PDFs.")


# Create and save FAISS vector store locally (no Google Drive)

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings



from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

# Define your local folder path
save_dir = r"C:\Users\Asif\Desktop\Thesis_GM\vectorstore"
os.makedirs(save_dir, exist_ok=True)


# Split documents into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
texts = splitter.split_documents(documents)

# Create Hugging Face embeddings
hf_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create FAISS vector store
vectorstore_hf = FAISS.from_documents(texts, hf_embeddings)

# Save FAISS index locally
save_path = os.path.join(save_dir, "fund_policy_index_hf")
vectorstore_hf.save_local(save_path)

print(f"Vector store created and saved locally at:\n{save_path}")
