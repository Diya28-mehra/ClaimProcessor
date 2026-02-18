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

Classify each page into one of these types:
- claim_forms
- cheque_or_bank_details
- identity_document
- itemized_bill
- discharge_summary
- prescription
- investigation_report
- cash_receipt
- other

IMPORTANT: Respond with ONLY a valid JSON object. No explanation, no markdown, no extra text.

Format:
{{
  "identity_document": [1, 3],
  "discharge_summary": [2],
  "itemized_bill": [4, 5]
}}

Only include document types that are actually present in the pages.

Pages:
{pages}
"""
def segregate_pages(state: ClaimsGraphState):
    """
    Classifies pages of the PDF into specific document types using Gemini.
    """
    pages = state['pages']
    
    # Format pages for the prompt
    pages_text = ""
    for page in pages:
        pages_text += f"Page {page['page_number']}:\n{page['text'][:1000]}\n\n" # Truncate for prompt efficiency if needed

    prompt_template = PromptTemplate(
        input_variables=["pages"],
        template=SEGREGATOR_PROMPT
    )
    
    prompt = prompt_template.format(pages=pages_text)
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    
    try:
        response = llm.invoke(prompt)
        content = response.content

        content = content.strip()
        content = re.sub(r"```(?:json)?", "", content).strip()

        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            content = json_match.group(0)
        
        classification = json.loads(content)
        
        return {"classification": classification}
    except Exception as e:
        return {"classification": {}}
