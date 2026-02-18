from typing import TypedDict, List, Dict, Any, Annotated
from langgraph.graph.message import add_messages

class ClaimsGraphState(TypedDict):
    """
    Represents the state of the claim processing graph.
    """
    pdf_path: str
    pages: List[Dict[str, Any]]  # List of {page_number: int, text: str}
    classification: Dict[str, List[int]] # {doc_type: [page_numbers]}
    extracted_data: Dict[str, Any]
    messages: Annotated[List[Any], add_messages]
