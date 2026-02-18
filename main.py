from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import tempfile
import os
from utils.pdf_parser import extract_pages
from graph import app as graph_app

app = FastAPI(title="Claim Processor API")

@app.post("/api/process")
async def process_claim(
    claim_id: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Process a medical claim PDF and extract structured information.
    
    Args:
        claim_id: Unique identifier for the claim
        file: PDF file to process
        
    Returns:
        JSON with extracted claim data
    """
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")
    
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        # Extract pages from PDF
        pages = extract_pages(temp_path)
        
        # Initialize state for the graph
        initial_state = {
            "pdf_path": temp_path,
            "pages": pages,
            "classification": {},
            "extracted_data": {},
            "messages": []
        }
        
        # Run the processing pipeline
        final_state = graph_app.invoke(initial_state)
        
        # Get the final aggregated output
        extracted_data = final_state.get('extracted_data', {})
        final_output = extracted_data.get('final_output', {})
        
        # Add claim_id to the response
        response = {
            "claim_id": claim_id,
            "status": "success",
            **final_output
        }
        
        # Clean up temp file
        os.unlink(temp_path)
        
        return JSONResponse(content=response)
        
    except Exception as e:
        # Clean up temp file if it exists
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.unlink(temp_path)
        
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Claim Processor API"}

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "ok"}
