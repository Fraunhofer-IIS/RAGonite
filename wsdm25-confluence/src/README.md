Code will be released after internal OSS clearance, by 10.03.2025 (before the start of the conference).


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
```bash
python src/prepare.py --config src/config/multi-modal-config.yaml
```

### 2️⃣ Start Chatting!  
Launch the RAG-based chatbot:  
```bash
python src/start_rag.py
```

---

### 📌 Additional Enhancements:
- **Add a banner/logo** at the top if you have one.  
- **Use emojis** to make sections stand out.  
- **Consider adding a GIF or screenshot** of the chatbot in action.  