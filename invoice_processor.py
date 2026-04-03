import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import fitz  # PyMuPDF
import json


class InvoiceExtractor:
    def __init__(self, groq_api_key):
        """
        Initialise with FREE Groq LLM and local HuggingFace embeddings.
        No OpenAI API key needed — embeddings run locally for free.
        """
        self.llm = ChatGroq(
            temperature=0,
            model_name="llama-3.3-70b-versatile",
            groq_api_key=groq_api_key
        )

        # Embeddings — runs locally on CPU, no API key, no cost
        # all-MiniLM-L6-v2 converts text into 384-dimensional vectors
        # so semantically similar text maps to nearby vectors
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )

        # Text splitter — breaks invoice text into overlapping chunks
        # chunk_size=500: each chunk is ~500 characters
        # chunk_overlap=50: adjacent chunks share 50 chars so no info lost at boundaries
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", " ", ""]
        )

        # Output parser — strips whitespace from LLM response
        self.output_parser = StrOutputParser()

    def extract_text_from_pdf(self, pdf_path):
        """Extract raw text from PDF using PyMuPDF"""
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text

    def build_vector_store(self, invoice_text):
        """
        RAG Step 1 and 2: Chunk the text and embed each chunk into FAISS.

        Why chunking matters:
        - Long invoices may have too much text to fit cleanly in one prompt
        - Chunking + retrieval lets us send only the MOST RELEVANT sections
          per question — e.g. only the totals section when asking about amounts
        - FAISS stores vectors in memory for fast cosine similarity search
        """
        chunks = self.text_splitter.split_text(invoice_text)
        vector_store = FAISS.from_texts(
            texts=chunks,
            embedding=self.embeddings
        )
        return vector_store

    def build_rag_chain(self, retriever):
        """
        RAG Step 3: Build the retrieval chain using LangChain Expression Language (LCEL).

        This is the modern LangChain 1.x way to build a RAG chain:
        - RunnablePassthrough passes the question through unchanged
        - retriever finds the top-k most relevant chunks for that question
        - PromptTemplate fills {context} with retrieved chunks and {question}
        - LLM generates the answer from that focused context
        - StrOutputParser cleans the response string
        """
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are an expert at extracting data from invoices.
Use ONLY the following invoice context to answer the question.
If the information is not in the context, return "Not Found".

Context from invoice:
{context}

Question: {question}

Return ONLY the requested value, nothing else. No explanation."""
        )

        def format_docs(docs):
            """Join retrieved document chunks into a single context string"""
            return "\n\n".join(doc.page_content for doc in docs)

        # LCEL chain: question -> retrieve chunks -> format -> fill prompt -> LLM -> parse
        rag_chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | prompt_template
            | self.llm
            | self.output_parser
        )

        return rag_chain

    def extract_invoice_data(self, invoice_text):
        """
        RAG Step 3: For each invoice field, retrieve the most relevant
        chunks from the vector store then pass those chunks to the LLM.

        Each question triggers a fresh similarity search — the LLM sees
        only focused relevant context per field, not the entire invoice.
        This is what makes it RAG rather than plain prompting.
        """
        vector_store = self.build_vector_store(invoice_text)

        # Retriever finds top-3 most semantically similar chunks per query
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )

        rag_chain = self.build_rag_chain(retriever)

        def ask(question):
            """Run a question through the RAG chain and return cleaned answer"""
            try:
                answer = rag_chain.invoke(question).strip()
                for prefix in ["Answer:", "Result:", "Value:"]:
                    if answer.startswith(prefix):
                        answer = answer[len(prefix):].strip()
                return answer if answer else "Not Found"
            except Exception:
                return "Not Found"

        # Each question triggers a different similarity search in the vector store
        vendor_name    = ask("What is the vendor or supplier company name that issued this invoice?")
        invoice_number = ask("What is the invoice number or reference number?")
        invoice_date   = ask("What is the invoice date? Return in YYYY-MM-DD format if possible.")
        due_date       = ask("What is the payment due date? Return in YYYY-MM-DD format if possible.")
        total_amount   = ask("What is the total amount due? Return only the number, no currency symbol.")
        currency       = ask("What is the currency used? Return the code only e.g. USD, AUD, EUR.")
        line_items_raw = ask(
            "List all line items. Format each as: description | quantity | unit_price | total. "
            "One line item per line."
        )

        line_items = self._parse_line_items(line_items_raw)

        return {
            "vendor_name":    vendor_name,
            "invoice_number": invoice_number,
            "invoice_date":   invoice_date,
            "due_date":       due_date,
            "total_amount":   total_amount,
            "currency":       currency,
            "line_items":     line_items
        }

    def _parse_line_items(self, raw_text):
        """Parse pipe-delimited line items into a list of dicts for DataFrame display"""
        if not raw_text or raw_text == "Not Found":
            return []
        line_items = []
        for line in raw_text.strip().split("\n"):
            line = line.strip()
            if not line or line == "Not Found":
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 4:
                line_items.append({
                    "description": parts[0],
                    "quantity":    parts[1],
                    "unit_price":  parts[2],
                    "total":       parts[3]
                })
            elif len(parts) >= 2:
                line_items.append({
                    "description": parts[0],
                    "quantity":    parts[1] if len(parts) > 1 else "N/A",
                    "unit_price":  parts[2] if len(parts) > 2 else "N/A",
                    "total":       parts[3] if len(parts) > 3 else "N/A"
                })
        return line_items

    def process_invoice(self, pdf_path):
        """
        Full RAG pipeline:
        PDF -> Raw Text -> Chunk + Embed -> FAISS Vector Store
        -> Per-field Retrieval -> LLM Extraction -> Structured JSON
        """
        text = self.extract_text_from_pdf(pdf_path)
        if not text.strip():
            return {
                "extracted_text": "",
                "structured_data": {
                    "error": "No text could be extracted. "
                             "This may be a scanned image-based PDF requiring OCR."
                }
            }
        data = self.extract_invoice_data(text)
        return {
            "extracted_text": text,
            "structured_data": data
        }