from state import ClaimsGraphState

def aggregate_results(state: ClaimsGraphState):
    """
    Aggregates all extraction results into a final structured output.
    """
    
    classification = state.get('classification', {})
    extracted_data = state.get('extracted_data', {})
    
    # Create final structured output
    final_output = {
        "document_classification": classification,
        "extracted_information": {
            "identity": extracted_data.get('identity_data'),
            "discharge_summary": extracted_data.get('discharge_summary_data'),
            "itemized_bill": extracted_data.get('itemized_bill_data')
        },
        "processing_status": {
            "total_pages": len(state.get('pages', [])),
            "classified_pages": sum(len(pages) for pages in classification.values()),
            "agents_executed": [
                key.replace('_data', '') for key in extracted_data.keys() 
                if key.endswith('_data') and extracted_data[key] is not None
            ]
        }
    }
    
    
    # Store the final output in the state
    return {"extracted_data": {**extracted_data, "final_output": final_output}}
