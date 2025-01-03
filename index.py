from typing import List, Dict, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import json
import os


class RAGSystem:
    def __init__(
        self, model_name: str = "all-MiniLM-L6-v2", index_path: str = "rag_index"
    ):
        """
        Initialize the RAG system with a sentence transformer model and storage paths.

        Args:
            model_name: The name of the sentence transformer model to use
            index_path: Directory to store the FAISS index and documents
        """
        self.encoder = SentenceTransformer(model_name)
        self.index_path = index_path
        self.dimension = self.encoder.get_sentence_embedding_dimension()

        # Initialize FAISS index
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product index

        # Storage for documents and their metadata
        self.documents = []
        self.doc_embeddings = []

        # Create storage directory if it doesn't exist
        os.makedirs(index_path, exist_ok=True)

        # Load existing index if available
        self._load_index()

    def add_document(self, text: str, metadata: Optional[Dict] = None) -> int:
        """
        Add a new document to the system.

        Args:
            text: The document text
            metadata: Optional metadata about the document (e.g., URL, timestamp)

        Returns:
            doc_id: The ID of the added document
        """
        # Create document object
        doc_id = len(self.documents)
        doc = {"id": doc_id, "text": text, "metadata": metadata or {}}

        # Compute embedding
        embedding = self.encoder.encode([text])[0]

        # Add to storage
        self.documents.append(doc)
        self.doc_embeddings.append(embedding)

        # Add to FAISS index
        self.index.add(np.array([embedding], dtype=np.float32))

        # Save updated index
        self._save_index()

        return doc_id

    def query(self, question: str, k: int = 5) -> List[Dict]:
        """
        Query the system with a question and retrieve relevant documents.

        Args:
            question: The query text
            k: Number of documents to retrieve

        Returns:
            List of relevant documents with their similarity scores
        """
        # Encode query
        query_embedding = self.encoder.encode([question])[0]

        # Search index
        scores, doc_indices = self.index.search(
            np.array([query_embedding], dtype=np.float32), k
        )

        # Prepare results
        results = []
        for score, doc_idx in zip(scores[0], doc_indices[0]):
            if doc_idx != -1:  # Valid index
                doc = self.documents[doc_idx].copy()
                doc["similarity_score"] = float(score)
                results.append(doc)

        return results

    def _save_index(self):
        """Save the current state of the system"""
        # Save FAISS index
        faiss.write_index(self.index, os.path.join(self.index_path, "index.faiss"))

        # Save documents and embeddings
        with open(os.path.join(self.index_path, "documents.json"), "w") as f:
            json.dump(self.documents, f)

        np.save(
            os.path.join(self.index_path, "embeddings.npy"),
            np.array(self.doc_embeddings),
        )

    def _load_index(self):
        """Load the saved state if it exists"""
        index_file = os.path.join(self.index_path, "index.faiss")
        docs_file = os.path.join(self.index_path, "documents.json")
        embeddings_file = os.path.join(self.index_path, "embeddings.npy")

        if all(os.path.exists(f) for f in [index_file, docs_file, embeddings_file]):
            self.index = faiss.read_index(index_file)

            with open(docs_file, "r") as f:
                self.documents = json.load(f)

            self.doc_embeddings = np.load(embeddings_file).tolist()

    def get_document_count(self) -> int:
        """Return the number of documents in the system"""
        return len(self.documents)


def check_if_information_found(
    rag: RAGSystem, query: str, threshold: float = 0.6
) -> bool:
    """
    Check if the RAG system has found the desired information.

    Args:
        rag: The RAG system instance
        query: The information we're looking for
        threshold: Similarity threshold to consider information as found

    Returns:
        bool: Whether the information has been found
    """
    results = rag.query(query, k=1)
    return bool(results and results[0]["similarity_score"] >= threshold)
