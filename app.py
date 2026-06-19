import os
from dotenv import load_dotenv
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
import gradio as gr
import speech_recognition as sr
from deep_translator import GoogleTranslator
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
DB_PATH = "vectorstore/db_faiss"

print("Loading FAISS database...")
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
db = FAISS.load_local(DB_PATH, embedding_model, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": 5})
print("FAISS DB loaded!")

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.5,
    max_tokens=512,
    api_key=GROQ_API_KEY
)

prompt = ChatPromptTemplate.from_template(
    "You are an experienced medical assistant with knowledge from the Gale Encyclopedia of Medicine.\n"
    "The user has provided symptoms. Based on the context, give a detailed and helpful medical response.\n"
    "Do NOT say I do not know. Always try to give useful medical information.\n\n"
    "Context:\n{context}\n\n"
    "Symptoms reported by user:\n{question}\n\n"
    "Please provide your response in this exact format:\n\n"
    "SEVERITY LEVEL: [Choose one: MILD / MODERATE / SEVERE / EMERGENCY]\n"
    "SEVERITY REASON: [One line explaining why this severity level was chosen]\n\n"
    "1. POSSIBLE CONDITIONS:\n[List possible medical conditions]\n\n"
    "2. LIKELY CAUSES:\n[List likely causes]\n\n"
    "3. SUGGESTED TREATMENTS:\n[List treatments and next steps]\n\n"
    "4. WHEN TO SEE A DOCTOR:\n[Mention specific warning signs that need immediate attention]\n\n"
    "Severity guide:\n"
    "EMERGENCY - chest pain, difficulty breathing, unconsciousness, severe bleeding\n"
    "SEVERE - high fever above 103F, severe pain, vomiting blood, confusion\n"
    "MODERATE - persistent fever, body pain, dizziness, nausea lasting more than 2 days\n"
    "MILD - mild cold, slight headache, minor itching, low grade fever"
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt | llm | StrOutputParser()
)

LANGUAGE_MAP = {
    "hi": "Hindi", "bn": "Bengali", "ta": "Tamil",
    "te": "Telugu", "mr": "Marathi", "gu": "Gujarati",
    "kn": "Kannada", "ml": "Malayalam", "pa": "Punjabi",
    "ur": "Urdu", "es": "Spanish", "fr": "French",
    "de": "German", "ar": "Arabic", "en": "English"
}

VOICE_LANG_MAP = {
    "Hindi": "hi-IN", "Bengali": "bn-IN", "Tamil": "ta-IN",
    "Telugu": "te-IN", "Marathi": "mr-IN", "Gujarati": "gu-IN",
    "Kannada": "kn-IN", "Malayalam": "ml-IN", "Punjabi": "pa-IN",
    "Urdu": "ur-PK", "English": "en-US", "Spanish": "es-ES",
    "French": "fr-FR", "German": "de-DE", "Arabic": "ar-SA",
}

def detect_and_translate_to_english(text):
    try:
        translated = GoogleTranslator(source="auto", target="en").translate(text)
        detected = GoogleTranslator(source="auto", target="en").detect(text)
        return translated, detected
    except:
        return text, "en"

def translate_answer_back(text, lang):
    try:
        if lang == "en":
            return text
        return GoogleTranslator(source="en", target=lang).translate(text)
    except:
        return text

def audio_to_text(audio_path, selected_lang="Hindi"):
    if audio_path is None:
        return None
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.record(source)
    lang_code = VOICE_LANG_MAP.get(selected_lang, "hi-IN")
    try:
        return recognizer.recognize_google(audio, language=lang_code)
    except sr.UnknownValueError:
        return "ERROR: Could not understand audio."
    except sr.RequestError:
        return "ERROR: Speech service unavailable."

def answer(text_input, audio_input, voice_lang, chat_history):
    if audio_input is not None:
        query = audio_to_text(audio_input, voice_lang)
        if query and query.startswith("ERROR"):
            chat_history.append(("Voice Input", query))
            return "", None, voice_lang, chat_history
        source_label = f"Voice ({voice_lang}): {query}"
    elif text_input and text_input.strip():
        query = text_input.strip()
        source_label = f"You: {query}"
    else:
        return "", None, voice_lang, chat_history

    try:
        english_query, user_lang = detect_and_translate_to_english(query)
        english_answer = rag_chain.invoke(english_query)
        final_answer = translate_answer_back(english_answer, user_lang)

        severity_badge = ""
        if "SEVERITY LEVEL: EMERGENCY" in english_answer:
            severity_badge = "🔴 EMERGENCY — Call ambulance immediately!\n\n"
        elif "SEVERITY LEVEL: SEVERE" in english_answer:
            severity_badge = "🟠 SEVERE — Visit hospital today!\n\n"
        elif "SEVERITY LEVEL: MODERATE" in english_answer:
            severity_badge = "🟡 MODERATE — See a doctor within 24 hours\n\n"
        elif "SEVERITY LEVEL: MILD" in english_answer:
            severity_badge = "🟢 MILD — Monitor at home, rest and hydrate\n\n"

        if user_lang != "en":
            lang_name = LANGUAGE_MAP.get(user_lang, user_lang.upper())
            full_answer = severity_badge + final_answer + f"\n\n[Detected: {lang_name} | Auto translated]"
        else:
            full_answer = severity_badge + final_answer
    except Exception as e:
        full_answer = f"Error: {str(e)}"

    chat_history.append((source_label, full_answer))
    return "", None, voice_lang, chat_history

with gr.Blocks(title="MediAssist", theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🏥 MediAssist - Medical Encyclopedia Chatbot")
    gr.Markdown("Type or speak in ANY language - Hindi, Bengali, Tamil, Spanish and more!")

    chatbot = gr.Chatbot(height=450, label="Conversation")
    chat_history = gr.State([])

    with gr.Row():
        text_input = gr.Textbox(
            placeholder="Type your symptoms here...",
            label="Type your question", scale=4
        )
        audio_input = gr.Audio(
            sources=["microphone"], type="filepath",
            label="Or record voice", scale=2
        )

    voice_lang = gr.Dropdown(
        choices=["Hindi","Bengali","Tamil","Telugu","Marathi",
                 "Gujarati","Kannada","Malayalam","Punjabi","Urdu",
                 "English","Spanish","French","German","Arabic"],
        value="Hindi",
        label="Select voice language before recording"
    )

    with gr.Row():
        submit_btn = gr.Button("Ask", variant="primary", scale=2)
        clear_btn = gr.Button("Clear Chat", scale=1)

    submit_btn.click(
        fn=answer,
        inputs=[text_input, audio_input, voice_lang, chat_history],
        outputs=[text_input, audio_input, voice_lang, chat_history]
    ).then(lambda h: h, inputs=chat_history, outputs=chatbot)

    text_input.submit(
        fn=answer,
        inputs=[text_input, audio_input, voice_lang, chat_history],
        outputs=[text_input, audio_input, voice_lang, chat_history]
    ).then(lambda h: h, inputs=chat_history, outputs=chatbot)

    clear_btn.click(lambda: ([], []), outputs=[chatbot, chat_history])

print("Launching MediAssist...")
app.launch()
