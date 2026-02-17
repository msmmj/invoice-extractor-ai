import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
import fitz  # PyMuPDF
import json

class InvoiceExtractor:
    def __init__(self, groq_api_key):
        """Initialize with FREE Groq LLM"""
        self.llm = ChatGroq(
            temperature=0,
            model_name="llama-3.3-70b-versatile",
            groq_api_key=groq_api_key
        )
    
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF using PyMuPDF"""
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    
    def extract_invoice_data(self, invoice_text):
        """Use LLM to extract structured data from invoice"""
        
        prompt_text = f"""You are an expert at extracting data from invoices.

Extract the following information from this invoice text. If a field is not found, put "Not Found".

Invoice Text:
{invoice_text}

Extract and return ONLY a valid JSON object with these exact fields:
{{
    "vendor_name": "company that issued the invoice",
    "invoice_number": "invoice or reference number",
    "invoice_date": "date of invoice (format: YYYY-MM-DD if possible)",
    "due_date": "payment due date (format: YYYY-MM-DD if possible)",
    "total_amount": "total amount due (just the number, no currency symbol)",
    "currency": "currency (USD, AUD, EUR, etc)",
    "line_items": [
        {{
            "description": "item description",
            "quantity": "quantity",
            "unit_price": "price per unit",
            "total": "line total"
        }}
    ]
}}

Return ONLY the JSON, no other text.
"""
        
        # Run extraction directly with the LLM
        result = self.llm.invoke(prompt_text)
        
        # Parse JSON response
        try:
            content = result.content.strip()
            if content.startswith("```json"):
                content = content.split("```json")[1].split("```")[0].strip()
            elif content.startswith("```"):
                content = content.split("```")[1].split("```")[0].strip()
            
            data = json.loads(content)
            return data
        except json.JSONDecodeError as e:
            return {"error": f"Failed to parse JSON: {str(e)}", "raw_response": result.content}
    
    def process_invoice(self, pdf_path):
        """Full pipeline: PDF -> Text -> Structured Data"""
        text = self.extract_text_from_pdf(pdf_path)
        data = self.extract_invoice_data(text)
        
        return {
            "extracted_text": text,
            "structured_data": data
        }