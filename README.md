# Claim Processing Pipeline

FastAPI service that processes PDF claims using LangGraph to orchestrate document segregation and multi-agent extraction.

## Setup

1. **Create and Activate Virtual Environment:**
   ```bash
   # Create the environment
   python -m venv venv

   # Activate the environment
   source venv/bin/activate
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables:**
   Create a `.env` file in the root directory and add your Groq API key:
   ```env
   GROQ_API_KEY=your_api_key_here
   ```

4. **Run the Server:**
   ```bash
   uvicorn main:app --reload
   ```

## API

- `POST /api/process`: Process a claim PDF.
