import os
import json
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from state import ClaimsGraphState
from dotenv import load_dotenv
import re 
load_dotenv()

SEGREGATOR_PROMPT = """
You are a medical claim document classifier.

Classify EACH page of the document into EXACTLY ONE of these types:
- claim_forms
- cheque_or_bank_details
- identity_document
- insurance_verification_form
- medical_history_questionnaire
- itemized_hospital_bill
- pharmacy_and_outpatient_bill
- discharge_summary
- prescription
- investigation_report
- cash_receipt
- other

Format your response as a single JSON object where keys are the types and values are lists of page numbers.
Example:
{{
  "identity_document": [1],
  "discharge_summary": [2, 3]
}}

IMPORTANT: 
1. Use ONLY the types listed above. 
2. Do NOT repeat any keys in the JSON.
3. Respond with ONLY the JSON object.

Pages to classify:
{pages}
"""
def segregate_pages(state: ClaimsGraphState):
    """
    Classifies pages of the PDF into specific document types using Gemini.
    """
    pages = state['pages']
    print(f"\n--- Segregator Debug ---")
    print(f"Number of pages to classify: {len(pages)}")
    
    # Format pages for the prompt - using shorter truncation for 8B model stability
    pages_text = ""
    for page in pages:
        pages_text += f"Page {page['page_number']}:\n{page['text'][:500]}\n\n"

    print(f"Pages text snippet (first 100 chars):\n{pages_text[:100]}...")

    prompt_template = PromptTemplate(
        input_variables=["pages"],
        template=SEGREGATOR_PROMPT
    )
    
    prompt = prompt_template.format(pages=pages_text)
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    
    try:
        response = llm.invoke(prompt)
        content = response.content
        print(f"Raw LLM Response Content: {content[:500]}...") # Print snippet to see if it loops

        content = content.strip()
        content = re.sub(r"```(?:json)?", "", content).strip()

        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            content = json_match.group(0)
        
        classification = json.loads(content)
        print(f"Parsed Classification: {classification}")
        
        return {"classification": classification}
    except Exception as e:
        print(f"!!! Segregator Error: {str(e)}")
        return {"classification": {}}
