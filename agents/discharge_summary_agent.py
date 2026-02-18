import json
import re
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from state import ClaimsGraphState
from dotenv import load_dotenv

load_dotenv()

DISCHARGE_SUMMARY_PROMPT = """
You are a medical claim discharge summary extractor.

Extract the following information from these discharge summary pages:
- Diagnosis (primary)
- Admit date
- Discharge date
- Physician details (name, specialization)
- Treatment summary
- Medications prescribed

IMPORTANT: Respond with ONLY a valid JSON object. No explanation, no markdown, no extra text.

Format:
{{
  "diagnosis": {{
    "primary": "Primary diagnosis"
  }},
  "admit_date": "DD/MM/YYYY",
  "discharge_date": "DD/MM/YYYY",
  "physician": {{
    "name": "Dr. Jane Smith",
    "specialization": "Cardiology",
  }},
  "treatment_summary": "Brief summary of treatment",
  "prescribed medicines": ["Med 1", "Med 2"]
}}

If a field is not found, use null.

Pages:
{pages_text}
"""

def extract_discharge_summary(state: ClaimsGraphState):
    """
    Extracts discharge summary information.
    """
    
    classification = state.get('classification', {})
    discharge_pages = classification.get('discharge_summary', [])
    prescription_pages = classification.get('prescription', [])

    # Combine discharge summary and prescription pages (deduplicated)
    combined_pages = list(set(discharge_pages + prescription_pages))
    
    if not combined_pages:
        return {"extracted_data": {**state.get('extracted_data', {}), "discharge_summary_data": None}}
    
    # Get only the pages assigned to this agent
    all_pages = state['pages']
    relevant_pages = [p for p in all_pages if p['page_number'] in combined_pages]
    
    # Format pages for the prompt
    pages_text = ""
    for page in relevant_pages:
        pages_text += f"Page {page['page_number']}:\n{page['text']}\n\n"
    
    prompt_template = PromptTemplate(
        input_variables=["pages_text"],
        template=DISCHARGE_SUMMARY_PROMPT
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
        
        discharge_data = json.loads(content)
        
        # Update extracted_data in state
        current_extracted = state.get('extracted_data', {})
        current_extracted['discharge_summary_data'] = discharge_data
        
        return {"extracted_data": current_extracted}
        
    except Exception as e:
        current_extracted = state.get('extracted_data', {})
        current_extracted['discharge_summary_data'] = None
        return {"extracted_data": current_extracted}
        