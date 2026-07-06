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
    """Raised when one or more requested Chroma collections do not exist."""

    def __init__(self, missing_collections: list[str]):
        self.missing_collections = missing_collections
        joined = ", ".join(missing_collections)
        super().__init__(f"Collection not found: {joined}")


class KnowledgeBaseEmptyError(Exception):
    """Raised when a query is attempted before any collection exists."""

    pass
