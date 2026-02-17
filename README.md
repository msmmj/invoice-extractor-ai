# 🧾 AI Invoice Data Extractor



Automatically extract structured data from invoice PDFs using Retrieval-Augmented Generation (RAG) and Large Language Models.



## 🎯 Problem Solved



Manual invoice data entry is time-consuming, error-prone, and costs businesses thousands in labor hours. This tool automates the extraction of key invoice fields using AI.



## ✨ Features



\- 📤 Upload invoice PDFs of any format

\- 🤖 AI extracts: vendor, invoice number, dates, amounts, line items

\- 📊 Displays data in clean, structured format

\- 💾 Export to JSON for integration with accounting systems

\- 🆓 100% free to run (uses free Groq API)



## 🚀 Tech Stack



\- **LangChain**: RAG orchestration framework

\- **Groq API**: Free LLM inference (Llama 3.3 70B)

\- **PyMuPDF**: PDF text extraction

\- **Streamlit**: Web interface

\- **Python**: Core logic


## 🛠️ Installation

```bash

# Clone repository

git remote add origin git@github.com:msmmj/invoice-extractor-ai.git

cd invoice-extractor-ai



# Create virtual environment

python -m venv venv

source venv/bin/activate  # On Windows: venv\\Scripts\\activate


# Install dependencies

pip install -r requirements.txt


# Create .env file with your Groq API key

echo "GROQ\_API\_KEY=your\_key\_here" > .env



# Run the app

streamlit run app.py

```



## 🔑 Getting a Free Groq API Key



1\. Visit \[console.groq.com](https://console.groq.com)

2\. Sign up (no credit card required)

3\. Generate API key

4\. Add to `.env` file



## 🎓 Key Learnings

\- Building RAG pipelines for document processing

\- Prompt engineering for structured data extraction

\- Working with LLM APIs

\- Handling unstructured text → structured JSON

\- Deploying ML applications



## 💡 Use Cases



\- Accounts payable automation

\- Expense report processing

\- Financial document digitization

\- Invoice reconciliation

\- Small business bookkeeping



## 🔮 Future Enhancements

\-  Batch processing multiple invoices

\-  CSV export functionality

\-  Database integration for invoice storage

\-  OCR for scanned/image-based PDFs

\-  Duplicate invoice detection

\-  Multi-language support



## 👨‍💻 Author



**Maxson Stephen Mathew**  

Data Analyst | Melbourne, Australia  

Building AI solutions for real business problems


[LinkedIn](https://www.linkedin.com/in/maxson-stephen-mathew-98a000188/) | [Portfolio](https://msmmj.github.io/portfolio/)



## 🙏 Acknowledgments


Built as a portfolio project to demonstrate practical AI/RAG application development.

