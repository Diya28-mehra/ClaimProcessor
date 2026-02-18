import json
import re
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from state import ClaimsGraphState
from dotenv import load_dotenv

load_dotenv()

ITEMIZED_BILL_PROMPT = """
You are a medical claim itemized bill extractor.

Extract ALL billing items from these itemized bill pages:
- Item descriptions
- Individual costs
- Quantities (if available)
- Categories (consultation, pharmacy, procedures, etc.)

Calculate the total amount by summing all individual costs.

IMPORTANT: Respond with ONLY a valid JSON object. No explanation, no markdown, no extra text.

Format:
{{
  "items": [
    {{
      "description": "Item name",
      "quantity": 1,
      "unit_cost": 100.0,
      "total_cost": 100.0
    }}
  ],
  "total_amount": 100.0
}}

If a field is not found, use null or reasonable defaults.

Pages:
{pages_text}
"""

def extract_itemized_bill(state: ClaimsGraphState):
    """
    Extracts itemized bill information and calculates total.
    """
    
    classification = state.get('classification', {})
    bill_pages = classification.get('itemized_hospital_bill', [])
    
    if not bill_pages:
        return {"extracted_data": {**state.get('extracted_data', {}), "itemized_bill_data": None}}
    
    # Get only the pages assigned to this agent
    all_pages = state['pages']
    relevant_pages = [p for p in all_pages if p['page_number'] in bill_pages]
    
    # Format pages for the prompt
    pages_text = ""
    for page in relevant_pages:
        pages_text += f"Page {page['page_number']}:\n{page['text']}\n\n"
    
    prompt_template = PromptTemplate(
        input_variables=["pages_text"],
        template=ITEMIZED_BILL_PROMPT
    )
    
    prompt = prompt_template.format(pages_text=pages_text)
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    
    try:
        response = llm.invoke(prompt)
        content = response.content.strip()
        
        # Clean up markdown/code blocks
        content = re.sub(r"```(?:json)?", "", content).strip()
        
        # Extract JSON
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            content = json_match.group(0)
        
        bill_data = json.loads(content)
        
        # Update extracted_data in state
        current_extracted = state.get('extracted_data', {})
        current_extracted['itemized_bill_data'] = bill_data
        
        return {"extracted_data": current_extracted}
    except Exception as e:
        current_extracted = state.get('extracted_data', {})
        current_extracted['itemized_bill_data'] = None
        return {"extracted_data": current_extracted}