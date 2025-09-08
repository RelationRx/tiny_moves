from tiny_moves.retrieval.faiss_constructor import FaissConstructor
from tiny_moves.retrieval.metadata import PlainTextMetadata
from tiny_moves.retrieval.utils import chunk_text_with_tokenizer, string_to_uuid


def transform(data: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Apply data transform function in the data loader."""
    result = []
    for filename, content in data:
        if ".txt" in filename:
            result.append((filename, content))
        else:
            print(f"{filename} not a txt file")

    return result


class PlainTextFaissConstructor(FaissConstructor[PlainTextMetadata]):
    """Generic Text FAISS index constructor."""

    def chunk_text(self, data: list[tuple[str, str]]) -> tuple[list[str], list[PlainTextMetadata], dict[str, str]]:
        """Generate chunks with metadata for each paragraph."""
        chunk_token_limit = self.embedding_model.max_seq_length
        chunks, metadata, chunk_context = [], [], {}
        for filename, content in data:
            for para in content.split("\n\n"):
                # chunk the text
                gene_chunks = chunk_text_with_tokenizer(
                    para, tokenizer=self.embedding_model.tokenizer, max_seq_length=chunk_token_limit, overlap=50
                )
                chunks.extend(gene_chunks)

                # create metadata for each chunk
                section_metadata = [
                    PlainTextMetadata(filename=filename, faiss_string=x, chunk_id=string_to_uuid(x))
                    for x in gene_chunks
                ]
                metadata.extend(section_metadata)

                # create a mapping of chunk_id to chunk text
                gene_chunk_context = {meta.chunk_id: para for meta in section_metadata}
                chunk_context.update(gene_chunk_context)

        return (chunks, metadata, chunk_context)
