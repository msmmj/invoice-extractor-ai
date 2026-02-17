import streamlit as st
import os
from dotenv import load_dotenv
from invoice_processor import InvoiceExtractor
import json
import pandas as pd

load_dotenv()

st.set_page_config(
    page_title="AI Invoice Extractor",
    page_icon="🧾",
    layout="wide"
)

st.title("🧾 AI-Powered Invoice Data Extractor")
st.markdown("Upload invoice PDFs and automatically extract structured data using AI")

if 'extractor' not in st.session_state:
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        st.error("⚠️ Please set GROQ_API_KEY in .env file")
        st.info("Get a free API key from console.groq.com")
        st.stop()
    st.session_state.extractor = InvoiceExtractor(api_key)

with st.sidebar:
    st.header("🚀 Tech Stack")
    st.markdown("""
    **100% Free:**
    - 🦙 Groq API (Free LLM)
    - 🦜 LangChain (RAG)
    - 📄 PyMuPDF (PDF processing)
    - 🎈 Streamlit (Interface)
    
    Built by Maxson Stephen Mathew
    """)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload Invoice")
    uploaded_file = st.file_uploader(
        "Upload PDF invoice",
        type=['pdf']
    )
    
    if uploaded_file:
        temp_path = "./temp_invoice.pdf"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success("✅ Invoice uploaded!")
        
        if st.button("🤖 Extract Data", type="primary", use_container_width=True):
            with st.spinner("🔍 AI is processing... (10-15 seconds)"):
                try:
                    result = st.session_state.extractor.process_invoice(temp_path)
                    st.session_state.result = result
                    st.session_state.processed = True
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
            
            if os.path.exists(temp_path):
                os.remove(temp_path)

with col2:
    st.subheader("📊 Extracted Data")
    
    if 'processed' in st.session_state and st.session_state.processed:
        data = st.session_state.result['structured_data']
        
        if 'error' not in data:
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.metric("Vendor", data.get('vendor_name', 'N/A'))
                st.metric("Invoice Number", data.get('invoice_number', 'N/A'))
                st.metric("Invoice Date", data.get('invoice_date', 'N/A'))
            
            with col_b:
                st.metric("Due Date", data.get('due_date', 'N/A'))
                st.metric("Total Amount", f"{data.get('currency', '')} {data.get('total_amount', 'N/A')}")
            
            st.subheader("📋 Line Items")
            if data.get('line_items'):
                df = pd.DataFrame(data['line_items'])
                st.dataframe(df, use_container_width=True)
            
            st.subheader("💾 Export Data")
            json_str = json.dumps(data, indent=2)
            st.download_button(
                label="📥 Download JSON",
                data=json_str,
                file_name=f"invoice_data.json",
                mime="application/json"
            )
            
            with st.expander("🔍 View Raw JSON"):
                st.json(data)
        else:
            st.error("Failed to extract data")
            st.code(data.get('raw_response', 'Unknown error'))
    else:
        st.info("👈 Upload an invoice and click 'Extract Data'")

st.markdown("---")
st.markdown("💡 **Tip:** This uses AI - always verify important data manually")