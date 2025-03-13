# WSDM25 Confluence

## 🚀 Installation
From the root directory `wsdm25-confluence`, run:
```bash
pip install -e .
```

## 🔑 Prerequisites
Ensure you have an OpenAI API key ready:
- Store it as an environment variable:
  ```bash
  export OPENAI_API_KEY="your-api-key"
  ```
- Or enter it manually when prompted.

## 🛠️ Usage

### 1️⃣ Prepare the Dataset
Run the following command to prepare the dataset using the default configuration.
(You can customize settings in `src/config/multi-modal-config.yaml`.)

**Important note:** This only needs to be done **once** to set up the VectorDB.
```bash
python src/prepare.py --config src/config/multi-modal-config.yaml
```

### 2️⃣ Start Chatting!
Launch the RAG-based chatbot:
```bash
python src/start_rag.py
```
Disclaimer: The code here is not exactly RAGonite code that we use in our organization (which cannot be released due to policy), but rather a close proxy
that mimics RAGonite's basic functionalities.
