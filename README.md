 # 🏥 MediAssist — Voice-Enabled Multilingual Medical Chatbot

A RAG-based AI chatbot that answers medical questions supporting voice input in 15+ languages.

## Features
- 🎙️ Voice input in Hindi, Bengali, Tamil, English & more
- 🌐 Auto language detection and translation
- 🔴 Severity classification (MILD/MODERATE/SEVERE/EMERGENCY)
- ⚡ Powered by Groq LLaMA 3.1 + FAISS vector store

## Tech Stack
LangChain | FAISS | HuggingFace | Groq LLaMA 3.1 | Gradio | SpeechRecognition

## Setup
1. `pip install -r requirements.txt`
2. Add Groq API key in `app.py`
3. Add medical PDF to project folder
4. Run `python build_db.py`
5. Run `python app.py`
