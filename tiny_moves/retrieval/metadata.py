"""
Define metadata classes for managing indexing information for FAISS.

These metadata objects link vector embeddings to their corresponding
source documents, enabling semantic search, analysis, and traceability
across different data domains. When the FAISS indices are queried, the
metadata entries for the most relevant results are returned.

Classes:

    - Metadata:
        Base class for all metadata types associated with indexed content.
        Acts as a flexible parent class for domain-specific metadata used
        in retrieval pipelines.

"""

from pydantic import BaseModel

class Metadata(BaseModel):
    """Base class for index metadata."""

    pass



class PlainTextMetadata(Metadata):
    """Metadata for generic text faiss chunk."""

    filename: str
    faiss_string: str
    chunk_id: str




class GenericTextMetadata(Metadata):
    """Metadata for generic text faiss chunk."""

    filename: str
    faiss_string: str
    chunk_id: str
