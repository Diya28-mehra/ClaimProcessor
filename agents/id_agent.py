import json
import re
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from state import ClaimsGraphState
from dotenv import load_dotenv

load_dotenv()

ID_AGENT_PROMPT = """
You are a medical claim identity information extractor.

Extract the following information from these identity document pages:
- Patient name (full name)
- Date of birth (DOB)
- ID numbers (policy number, patient ID, etc.)
- Policy details (policy number, insurer name)
- Contact information (if available)

IMPORTANT: Respond with ONLY a valid JSON object. No explanation, no markdown, no extra text.

Format:
{{
  "patient_name": "John Doe",
  "date_of_birth": "DD/MM/YYYY",
  "id_numbers": "ID-XXX-YYY-ZZZ",
  "policy_details": {{
    "insurer_name": "ABC Insurance",
    "policy_number": "POL123456"
  }},
  "contact": {{
    "phone": "++1-555-0123",
    "email": "example@email.com",
    "address": "Full address"
  }}
}}

If a field is not found, use null.

Pages:
{pages_text}
"""

def extract_identity(state: ClaimsGraphState):
    """
    Extracts identity information from identity document pages.
    """
    
    classification = state.get('classification', {})
    identity_pages = classification.get('identity_document', [])
    insurance_verification_pages = classification.get('insurance_verification_form', [])
    claim_pages = classification.get('claim_forms', [])

    # Combine identity and insurance verification pages (deduplicated)
    combined_pages = list(set(identity_pages + insurance_verification_pages + claim_pages))

    if not combined_pages:
        return {"extracted_data": {**state.get('extracted_data', {}), "identity_data": None}}
    
    # Get only the pages assigned to this agent
    all_pages = state['pages']
    relevant_pages = [p for p in all_pages if p['page_number'] in combined_pages]
    
    # Format pages for the prompt
    pages_text = ""
    for page in relevant_pages:
        pages_text += f"Page {page['page_number']}:\n{page['text']}\n\n"
    
    prompt_template = PromptTemplate(
        input_variables=["pages_text"],
        template=ID_AGENT_PROMPT
    )
    
    prompt = prompt_template.format(pages_text=pages_text)
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    
    try:
        response = llm.invoke(prompt)
        content = response.content.strip()
        
        # Clean up markdown/code blocks
        content = re.sub(r"```(?:json)?", "", content).strip()
        
        # Extract JSON
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            content = json_match.group(0)
        
        identity_data = json.loads(content)
        
        # Update extracted_data in state
        current_extracted = state.get('extracted_data', {})
        current_extracted['identity_data'] = identity_data
        
        return {"extracted_data": current_extracted}
        
    except Exception as e:
        current_extracted = state.get('extracted_data', {})
        current_extracted['identity_data'] = None
        return {"extracted_data": current_extracted}