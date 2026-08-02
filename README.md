# 🤖 SupportAI – FAQ-Grounded Support Chatbot

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)
![Open Source](https://img.shields.io/badge/Open%20Source-Yes-orange?style=for-the-badge)

</p>

A lightweight FAQ-grounded AI chatbot that retrieves relevant answers from a knowledge base using **TF-IDF + Cosine Similarity** and optionally generates natural language responses using the **Anthropic Claude API**.

This project demonstrates the fundamentals of **Information Retrieval (IR)**, **Retrieval-Augmented Generation (RAG)**, and conversational AI using Python.

---

# ✨ Features

- 🔍 Keyword-based FAQ search
- 📄 TF-IDF + Cosine Similarity retrieval
- 🤖 Claude API integration (optional)
- 🔄 Automatic fallback when no API key is provided
- 💬 Interactive command-line chatbot
- 🧠 Sliding-window conversation memory
- 📦 Modular Python package structure
- ⚡ Easy to extend with additional FAQs or LLMs

---

# 📂 Project Structure

```text
FAQ-Knowledge-Base-Foundation/
│
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── faq_search.py             # Task 1: Keyword-based FAQ search
├── main.py                   # CLI chatbot entry point
│
└── supportai/
    ├── __init__.py           # Package initializer
    ├── chat.py               # Chat session & memory management
    ├── llm.py                # Claude API / LLM integration
    └── retriever.py          # TF-IDF + Cosine Similarity retrieval
```

---

# 🛠️ Technologies Used

- Python 3.10+
- TF-IDF
- Cosine Similarity
- Anthropic Claude API
- Command Line Interface (CLI)

---

# 🚀 Installation

## 1. Clone the Repository

```bash
git clone https://github.com/mundlapudimuneendra-ops/SupportAI-FAQ-Grounded-Support-Chatbot.git
cd SupportAI-FAQ-Grounded-Support-Chatbot
```

## 2. Create a Virtual Environment (Recommended)

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🔑 Configure Claude API (Optional)

For real AI-generated responses, set your Anthropic API key.

### Windows

```bash
set ANTHROPIC_API_KEY=your_api_key
```

### Linux / macOS

```bash
export ANTHROPIC_API_KEY=your_api_key
```

If no API key is configured, the chatbot automatically uses the built-in simulator.

---

# ▶️ Running the Project

## Task 1 – Keyword Search

```bash
python faq_search.py
```

---

## Task 2 – TF-IDF Retrieval

```bash
python -m supportai.retriever
```

---

## Task 3 – LLM Answer Generation

```bash
python -m supportai.llm
```

---

## Task 4 – Interactive Chatbot

```bash
python main.py
```

---

# 💬 Chat Commands

| Command | Description |
|---------|-------------|
| `history` | Display previous conversation |
| `reset` | Clear conversation history |
| `quit` | Exit chatbot |
| `exit` | Exit chatbot |

---

# ⚙️ How It Works

```text
User Question
      │
      ▼
Keyword Search
      │
      ▼
TF-IDF Vectorization
      │
      ▼
Cosine Similarity Ranking
      │
      ▼
Top Matching FAQ
      │
      ▼
Claude API (Optional)
      │
      ▼
Final Response
```

---

# 📸 Screenshots

## Chatbot

> Add a screenshot of your chatbot here.

Example:

```
screenshots/chatbot.png
```

---

## Retrieval Demo

> Add a screenshot of TF-IDF retrieval results here.

Example:

```
screenshots/retrieval.png
```

---

# 📈 Future Improvements

- 🌐 Streamlit web interface
- 🎤 Voice assistant support
- 🗄️ Vector database integration
- 🤖 OpenAI / Gemini support
- 🌍 Multi-language FAQ search
- 💾 Persistent chat history

---

# 🤝 Contributing

Contributions are welcome!

1. Fork the repository.
2. Create a new feature branch.

```bash
git checkout -b feature-name
```

3. Commit your changes.

```bash
git commit -m "Add new feature"
```

4. Push your branch.

```bash
git push origin feature-name
```

5. Open a Pull Request.

---

# 📄 License

This project is licensed under the **MIT License**.

---

# 👨‍💻 Author

**Mundlapudi Muneendra**

AI & ML Student

GitHub: https://github.com/mundlapudimuneendra-ops

---

# ⭐ Show Your Support

If you found this project helpful:

⭐ Star this repository

🍴 Fork it

💡 Share it with others

Happy Coding! 🚀
