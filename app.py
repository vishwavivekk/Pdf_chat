
import os
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
import re
import random
# ---------------- Basic Conversation Handling ---------------- #

def is_greeting(text: str) -> bool:
    """
    Check if the message contains a greeting as a whole word or phrase.
    Uses word-boundary regex so 'which' won't match 'hi'.
    """
    greetings = [
        "hi", "hello", "hey", "good morning", "good afternoon",
        "good evening", "howdy", "what's up", "how are you",
        "greetings", "salutations", "hola", "namaste"
    ]
    text = text or ""
    # build regex that matches any greeting as a whole word/phrase
    pattern = r'\b(?:' + '|'.join(re.escape(g) for g in greetings) + r')\b'
    return bool(re.search(pattern, text, flags=re.IGNORECASE))

def is_goodbye(text: str) -> bool:
    """
    Check if the message contains a goodbye as a whole word or phrase.
    """
    goodbyes = [
        "bye", "goodbye", "see you", "farewell", "take care",
        "talk to you later", "ttyl", "catch you later", "adios"
    ]
    text = text or ""
    pattern = r'\b(?:' + '|'.join(re.escape(g) for g in goodbyes) + r')\b'
    return bool(re.search(pattern, text, flags=re.IGNORECASE))

def get_basic_response(text):
    """Generate basic conversational responses"""
    text_lower = text.lower()

    if is_greeting(text_lower):
        return "👋 Hello! I'm here to help you with questions about your uploaded documents. How can I assist you today?"

    elif is_goodbye(text_lower):
        return "👋 Goodbye! Feel free to come back anytime if you have more questions about your documents!"

    elif any(word in text_lower for word in ['thank you', 'thanks', 'appreciate']):
        return "😊 You're welcome! Is there anything else you'd like to know about your documents?"

    elif any(word in text_lower for word in ['how are you', 'how do you do']):
        return "I'm doing great, thanks for asking! 😊 I'm ready to help you explore your documents. What would you like to know?"

    return None

def generate_suggestions(query, vectorstore=None):
    """Generate helpful suggestions when no answer is found"""
    suggestions = [
        "Could you provide more specific details about what you're looking for?",
        "Try rephrasing your question with different keywords",
        "Consider asking about a specific topic or concept from your documents"
    ]

    # Add contextual suggestions based on query
    if 'what' in query.lower():
        suggestions.append("Try asking 'How does [topic] work?' or 'Why is [topic] important?'")
    elif 'how' in query.lower():
        suggestions.append("Try asking 'What is [topic]?' or 'When should I use [topic]?'")
    elif 'when' in query.lower():
        suggestions.append("Try asking 'How often should [topic]?' or 'What are the steps for [topic]?'")

    return suggestions

# ---------------- PDF Processing ---------------- #

def extract_pdf_text(pdf_files):
    """Extract text from a list of PDF files."""
    text = ""
    for pdf in pdf_files:
        try:
            reader = PdfReader(pdf)
            for page in reader.pages:
                content = page.extract_text()
                if content:
                    text += content
        except Exception as e:
            st.error(f"⚠️ Error reading {pdf.name}: {e}")
    return text

def split_text_into_chunks(text, chunk_size=1000, overlap=200):
    """Split long text into chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""]
    )
    return splitter.split_text(text)

def build_vectorstore(chunks):
    """Convert text chunks into a FAISS vectorstore."""
    embeddings = OpenAIEmbeddings()
    return FAISS.from_texts(chunks, embeddings)

# ---------------- Conversation Setup ---------------- #

def create_conversation_chains(vectorstore):
    """Create strict and fallback retriever chains with memory."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    # Strict retriever
    strict_retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.65}
    )

    # Fallback retriever
    fallback_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    strict_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=strict_retriever,
        memory=memory,
        return_source_documents=True,
        output_key="answer"
    )

    fallback_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=fallback_retriever,
        memory=memory,
        return_source_documents=True,
        output_key="answer"
    )

    return {"strict": strict_chain, "fallback": fallback_chain}

# ---------------- Chat Handling ---------------- #

def is_unhelpful_answer(answer: str) -> bool:
    """Detect unhelpful answers like 'I don't know'."""
    bad_phrases = [
        "i don't know", 
        "sorry", 
        "no information", 
        "not sure", 
        "i cannot find"
    ]
    return any(p in answer.lower() for p in bad_phrases)


def handle_user_query(query):
    """Process user query with basic conversation handling."""
    if not query.strip():
        return

    # Add user message to chat history first
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Create user message object
    user_msg = type("HumanMessage", (), {"type": "human", "content": query})()
    st.session_state.chat_history.append(user_msg)

    # Check for basic conversation
    basic_response = get_basic_response(query)
    if basic_response:
        bot_msg = type("AIMessage", (), {"type": "ai", "content": basic_response})()
        st.session_state.chat_history.append(bot_msg)
        return

    # Check if documents are processed
    if st.session_state.conversation is None:
        response = "📂 I'd love to help you! Please upload and process some documents first so I can answer your questions about them."
        bot_msg = type("AIMessage", (), {"type": "ai", "content": response})()
        st.session_state.chat_history.append(bot_msg)
        return

    # Try strict retriever
    try:
        strict_result = st.session_state.conversation["strict"](
            {"question": query, "chat_history": st.session_state.chat_history[:-1]}  # Exclude current user message
        )

        if strict_result.get("source_documents") and len(strict_result["source_documents"]) > 0:
            if not is_unhelpful_answer(strict_result["answer"]):
                bot_msg = type("AIMessage", (), {"type": "ai", "content": strict_result["answer"]})()
                st.session_state.chat_history.append(bot_msg)
                return
    except Exception as e:
        st.error(f"Error in strict search: {e}")

    # Try fallback retriever
    try:
        fallback_result = st.session_state.conversation["fallback"](
            {"question": query, "chat_history": st.session_state.chat_history[:-1]}
        )

        if fallback_result.get("source_documents") and len(fallback_result["source_documents"]) > 0:
            if not is_unhelpful_answer(fallback_result["answer"]):
                bot_msg = type("AIMessage", (), {"type": "ai", "content": fallback_result["answer"]})()
                st.session_state.chat_history.append(bot_msg)
                return
    except Exception as e:
        st.error(f"Error in fallback search: {e}")

    # No answer found - provide helpful suggestions + clarifying question
    clarifying_question = generate_clarifying_question(query)
    suggestions = generate_suggestions(query)

    response = f"""🤔 I couldn't find specific information about **{query}** in your documents.  

**Maybe you can help me by clarifying:**  
👉 {clarifying_question}  

**Here are some tips to refine your question:**  
• {suggestions[0]}  
• {suggestions[1]}  
• {suggestions[2]}  

**Example follow-ups you could try:**  
• "What topics are covered in the document?"  
• "Can you summarize the main points?"  
• "What does the document say about [specific keyword]?"  
"""

    bot_msg = type("AIMessage", (), {"type": "ai", "content": response})()
    st.session_state.chat_history.append(bot_msg)

def generate_clarifying_question(query):
    """Generate a clarifying question when no answer is found."""
    clarifications = [
        f"Could you clarify what aspect of '{query}' you’re most interested in?",
        f"Do you mean '{query}' in terms of definitions, examples, or applications?",
        f"Are you asking about the general concept of '{query}' or something specific in your documents?",
        f"Would you like me to summarize related topics to '{query}' from the documents?",
    ]
    return random.choice(clarifications)

def render_chat():
    """Render chat history in a container with limited height."""
    # Create a container for chat messages with fixed height
    chat_container = st.container()

    with chat_container:
        # Display messages in reverse order (newest at bottom)
        for i, msg in enumerate(st.session_state.chat_history):
            if msg.type == "human":
                st.write(user_template.replace("{{MSG}}", msg.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace("{{MSG}}", msg.content), unsafe_allow_html=True)

def clear_chat():
    """Clear chat history."""
    st.session_state.chat_history = []
    st.success("💬 Chat cleared!")

# ---------------- Main App ---------------- #

def main():
    load_dotenv()

    # API Key management
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except Exception:
        api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        st.error("❌ Please set your OPENAI_API_KEY in .env or Streamlit secrets.")
        st.stop()

    os.environ["OPENAI_API_KEY"] = api_key

    # Streamlit page setup
    st.set_page_config(
        page_title="AI Document Chat Assistant", 
        page_icon="🤖",
        layout="wide"
    )
    st.write(css, unsafe_allow_html=True)

    # Initialize session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Main layout
    col1, col2 = st.columns([3, 1])

    with col1:
        st.title("🤖 AI Document Chat Assistant")
        st.markdown("*Your intelligent companion for document Q&A*")

        # Chat interface
        st.markdown("---")

        # Display chat history in a scrollable container
        if st.session_state.chat_history:
            st.markdown("### 💬 Conversation")
            # Create a scrollable area for chat
            with st.container():
                render_chat()
        else:
            st.markdown("### 👋 Welcome!")
            st.info("Upload some documents and start chatting! Try saying 'Hello' to get started.")

        # Chat input at bottom
        st.markdown("---")

        # Use columns for input and buttons
        if query := st.chat_input("💭 Ask me anything about your documents..."):
            handle_user_query(query)
            st.rerun()

        # Optional clear button
        if st.button("🗑️ Clear Chat"):
            clear_chat()
            st.rerun()

    # Sidebar
    with col2:
        st.markdown("### 📁 Document Manager")

        # Document upload
        pdf_files = st.file_uploader(
            "Upload PDF Documents", 
            type="pdf", 
            accept_multiple_files=True,
            help="Upload one or more PDF files to start chatting!"
        )

        process_button = st.button("🔄 Process Documents", type="primary")

        if process_button:
            if not pdf_files:
                st.warning("⚠️ Please upload at least one PDF file.")
            else:
                with st.spinner("🔄 Processing your documents..."):
                    raw_text = extract_pdf_text(pdf_files)

                    if not raw_text.strip():
                        st.error("🚫 No text could be extracted from the PDFs.")
                    else:
                        chunks = split_text_into_chunks(raw_text)
                        vectorstore = build_vectorstore(chunks)
                        st.session_state.conversation = create_conversation_chains(vectorstore)

                        # Clear previous chat when new documents are processed
                        st.session_state.chat_history = []

                        st.success("✅ Documents processed successfully!")
                        st.balloons()

        # Document info
        if pdf_files:
            st.markdown("---")
            st.markdown("### 📊 Uploaded Files")
            for pdf in pdf_files:
                st.write(f"📄 {pdf.name}")

        # Instructions
        st.markdown("---")
        st.markdown("### 💡 Tips")
        st.markdown("""
        • **Start simple**: Try "Hello" or "Hi"
        • **Be specific**: Ask about particular topics
        • **Use keywords**: Include relevant terms from your documents
        • **Ask follow-ups**: I remember our conversation context
        """)

if __name__ == "__main__":
    main()
