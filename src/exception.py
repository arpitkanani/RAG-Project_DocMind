import sys
from src.logger import logging

def error_message_detail(error, error_detail: sys): # type: ignore
    _, _, exc_tb = sys.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename # pyright: ignore[reportOptionalMemberAccess]

    error_message = (
        "Error occurred in python script name [{0}] line number [{1}] error message [{2}]"
        .format(file_name, exc_tb.tb_lineno, str(error)) # pyright: ignore[reportOptionalMemberAccess]
    )
    return error_message


class CustomException(Exception):

    def __init__(self, error_message, error_detail: sys): # type: ignore
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message


class CollectionNotFoundError(Exception):
    """Raised when one or more requested Chroma/Qdrant collections do not exist."""

    def __init__(self, missing_collections: list[str]):
        self.missing_collections = missing_collections
        joined = ", ".join(missing_collections)
        super().__init__(f"Collection not found: {joined}")


class KnowledgeBaseEmptyError(Exception):
    """Raised when a query is attempted before any collection exists."""

    pass


class EmbeddingUnavailableError(Exception):
    """Raised when the active embedding service is unreachable or rate limited."""

    def __init__(self, message: str = "The embedding model is unavailable. You can switch the embedding model in Settings."):
        super().__init__(message)
        self.message = message


class CollectionEmbeddingMismatchError(Exception):
    """Raised when a collection's embedding model or dimension does not match active model."""

    def __init__(self, collection_name: str, expected_model: str, actual_model: str, expected_dim: int, actual_dim: int, doc_name: str = ""):
        self.collection_name = collection_name
        self.doc_name = doc_name or collection_name
        self.expected_model = expected_model
        self.actual_model = actual_model
        self.expected_dim = expected_dim
        self.actual_dim = actual_dim

        name_display = f"'{self.doc_name}'" if self.doc_name else f"collection '{collection_name}'"
        msg = (
            f"The uploaded document {name_display} was indexed using '{actual_model}' ({actual_dim}d), "
            f"but the currently selected model is '{expected_model}' ({expected_dim}d). "
            f"Please switch to '{actual_model}' from Settings or re-upload the document."
        )
        super().__init__(msg)
        self.message = msg


# Alias for the DocuVortex rebranding
DocuVortexException = CustomException
