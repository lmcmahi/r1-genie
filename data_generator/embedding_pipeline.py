#!/usr/bin/env python3
"""
Embedding Pipeline for RAG System
Generates embeddings and stores them in Qdrant vector database
"""

import os
import uuid
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Import dependencies
try:
    from openai import OpenAI
except ImportError:
    print("❌ openai package not installed")
    OpenAI = None

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
except ImportError:
    print("❌ qdrant-client package not installed")
    QdrantClient = None

from document_parsers import DocumentChunk


class EmbeddingGenerator:
    """Generate embeddings using OpenAI API"""

    def __init__(
        self,
        model: str = None,
        api_key: str = None,
        batch_size: int = 100
    ):
        """
        Initialize embedding generator

        Args:
            model: OpenAI embedding model (default: from env)
            api_key: OpenAI API key (default: from env)
            batch_size: Number of texts to embed per batch
        """
        self.model = model or os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.batch_size = batch_size

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not set in environment")

        if OpenAI is None:
            raise ImportError("openai package required for embeddings")

        self.client = OpenAI(api_key=self.api_key)
        print(f"✅ Embedding generator initialized (model: {self.model})")

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        response = self.client.embeddings.create(
            input=text,
            model=self.model
        )
        return response.data[0].embedding

    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batch

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        response = self.client.embeddings.create(
            input=texts,
            model=self.model
        )
        return [item.embedding for item in response.data]

    def embed_chunks(self, chunks: List[DocumentChunk]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for document chunks with progress bar

        Args:
            chunks: List of DocumentChunk objects

        Returns:
            List of dicts with {text, embedding, metadata}
        """
        embedded_chunks = []

        # Process in batches
        for i in tqdm(range(0, len(chunks), self.batch_size), desc="Generating embeddings"):
            batch = chunks[i:i + self.batch_size]
            texts = [chunk.text for chunk in batch]

            try:
                embeddings = self.generate_embeddings_batch(texts)

                for chunk, embedding in zip(batch, embeddings):
                    embedded_chunks.append({
                        "text": chunk.text,
                        "embedding": embedding,
                        "metadata": chunk.metadata
                    })

            except Exception as e:
                print(f"⚠️  Error embedding batch {i//self.batch_size}: {e}")
                # Fallback to individual embeddings
                for chunk in batch:
                    try:
                        embedding = self.generate_embedding(chunk.text)
                        embedded_chunks.append({
                            "text": chunk.text,
                            "embedding": embedding,
                            "metadata": chunk.metadata
                        })
                    except Exception as e2:
                        print(f"❌ Error embedding chunk: {e2}")
                        continue

        return embedded_chunks


class QdrantVectorStore:
    """Qdrant vector database interface"""

    def __init__(
        self,
        host: str = None,
        port: int = None,
        collection_name: str = None
    ):
        """
        Initialize Qdrant client

        Args:
            host: Qdrant host (default: from env)
            port: Qdrant port (default: from env)
            collection_name: Collection name (default: from env)
        """
        self.host = host or os.getenv("QDRANT_HOST", "localhost")
        self.port = int(port or os.getenv("QDRANT_PORT", "6333"))
        self.collection_name = collection_name or os.getenv("QDRANT_COLLECTION", "vendor_apis")

        if QdrantClient is None:
            raise ImportError("qdrant-client package required")

        try:
            self.client = QdrantClient(host=self.host, port=self.port)
            print(f"✅ Connected to Qdrant at {self.host}:{self.port}")
        except Exception as e:
            print(f"❌ Failed to connect to Qdrant: {e}")
            print(f"   Make sure Qdrant is running: docker run -p 6333:6333 qdrant/qdrant")
            raise

    def create_collection(
        self,
        vector_size: int = None,
        distance: Distance = Distance.COSINE,
        force_recreate: bool = False
    ):
        """
        Create collection in Qdrant

        Args:
            vector_size: Embedding dimension (default: from env)
            distance: Distance metric
            force_recreate: Delete existing collection if exists
        """
        vector_size = vector_size or int(os.getenv("EMBEDDING_DIMENSION", "1536"))

        # Check if collection exists
        collections = self.client.get_collections().collections
        collection_exists = any(c.name == self.collection_name for c in collections)

        if collection_exists:
            if force_recreate:
                print(f"🗑️  Deleting existing collection: {self.collection_name}")
                self.client.delete_collection(self.collection_name)
            else:
                print(f"✅ Collection '{self.collection_name}' already exists")
                return

        # Create collection
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=vector_size, distance=distance)
        )
        print(f"✅ Created collection: {self.collection_name} (dimension: {vector_size})")

    def upsert_embeddings(
        self,
        embedded_chunks: List[Dict[str, Any]],
        batch_size: int = 100
    ) -> int:
        """
        Upload embeddings to Qdrant

        Args:
            embedded_chunks: List of {text, embedding, metadata} dicts
            batch_size: Upload batch size

        Returns:
            Number of points uploaded
        """
        points = []

        for chunk in embedded_chunks:
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=chunk["embedding"],
                payload={
                    "text": chunk["text"],
                    **chunk["metadata"]
                }
            )
            points.append(point)

        # Upload in batches
        total_uploaded = 0
        for i in tqdm(range(0, len(points), batch_size), desc="Uploading to Qdrant"):
            batch = points[i:i + batch_size]

            try:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )
                total_uploaded += len(batch)

            except Exception as e:
                print(f"⚠️  Error uploading batch {i//batch_size}: {e}")

        print(f"✅ Uploaded {total_uploaded} points to Qdrant")
        return total_uploaded

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        info = self.client.get_collection(self.collection_name)
        return {
            "name": self.collection_name,
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "status": info.status
        }

    def search(
        self,
        query_vector: List[float],
        limit: int = 5,
        filter_conditions: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search for similar vectors

        Args:
            query_vector: Query embedding
            limit: Number of results
            filter_conditions: Metadata filters

        Returns:
            List of search results
        """
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit,
            query_filter=filter_conditions
        )

        return [
            {
                "id": result.id,
                "score": result.score,
                "payload": result.payload
            }
            for result in results
        ]


class EmbeddingPipeline:
    """End-to-end embedding pipeline"""

    def __init__(self):
        """Initialize embedding pipeline"""
        self.embedding_generator = EmbeddingGenerator()
        self.vector_store = QdrantVectorStore()

    def process_document(
        self,
        chunks: List[DocumentChunk],
        create_collection: bool = False,
        force_recreate: bool = False
    ) -> Dict[str, Any]:
        """
        Process document chunks through complete pipeline

        Args:
            chunks: List of DocumentChunk objects
            create_collection: Create collection if doesn't exist
            force_recreate: Recreate collection even if exists

        Returns:
            Processing statistics
        """
        print(f"\n🚀 Starting embedding pipeline for {len(chunks)} chunks...")

        # Step 1: Create collection if needed
        if create_collection or force_recreate:
            self.vector_store.create_collection(force_recreate=force_recreate)

        # Step 2: Generate embeddings
        print("\n📊 Step 1/2: Generating embeddings...")
        embedded_chunks = self.embedding_generator.embed_chunks(chunks)

        # Step 3: Upload to Qdrant
        print("\n📤 Step 2/2: Uploading to Qdrant...")
        uploaded_count = self.vector_store.upsert_embeddings(embedded_chunks)

        # Get final stats
        stats = self.vector_store.get_collection_stats()

        result = {
            "chunks_processed": len(chunks),
            "embeddings_generated": len(embedded_chunks),
            "points_uploaded": uploaded_count,
            "collection_stats": stats
        }

        print(f"\n✅ Pipeline complete!")
        print(f"   Chunks processed: {result['chunks_processed']}")
        print(f"   Embeddings generated: {result['embeddings_generated']}")
        print(f"   Points uploaded: {result['points_uploaded']}")
        print(f"   Total collection size: {stats['vectors_count']}")

        return result


# Example usage
if __name__ == "__main__":
    import sys
    from document_parsers import DocumentParserFactory

    if len(sys.argv) < 3:
        print("Usage: python embedding_pipeline.py <file_path> <vendor>")
        print("Example: python embedding_pipeline.py viavi_api.yaml viavi")
        sys.exit(1)

    file_path = sys.argv[1]
    vendor = sys.argv[2]

    # Load canonical KPIs
    try:
        from r1_dataset_builder_v3 import CANONICAL_KPIS
    except:
        CANONICAL_KPIS = {}

    # Parse document
    print("📄 Parsing document...")
    parser_factory = DocumentParserFactory(CANONICAL_KPIS)
    chunks = parser_factory.parse(file_path, vendor)

    # Run embedding pipeline
    pipeline = EmbeddingPipeline()
    result = pipeline.process_document(
        chunks,
        create_collection=True,
        force_recreate=False
    )

    print(f"\n🎉 Success! Vendor documentation indexed for RAG retrieval.")
