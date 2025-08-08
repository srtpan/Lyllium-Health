import streamlit as st
import os
import time
from langchain_anthropic import ChatAnthropic
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --bg-dark: #0D1117;
    --bg-secondary: #161B22;
    --bg-tertiary: #21262D;
    --text-primary: #F0F6FC;
    --text-secondary: #8B949E;
    --accent-orange: #F97316;
    --accent-purple: #8B5CF6;
    --border-color: #30363D;
    --input-bg: #21262D;
    --hover-bg: #262C36;
}

.stApp {
    background-color: var(--bg-dark);
    color: var(--text-primary);
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Hide Streamlit elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.stDeployButton {display: none;}

/* Main container */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 48rem;
    background-color: var(--bg-dark) !important;
}

/* Force ALL backgrounds to be dark */
* {
    background-color: var(--bg-dark) !important;
}

.stApp, .stApp > div, .main, section[data-testid="stMain"] {
    background-color: var(--bg-dark) !important;
}

/* Containers */
.stVerticalBlock, .stHorizontalBlock, .block-container {
    background-color: var(--bg-dark) !important;
}

/* Modern header */
.claude-header {
    text-align: center;
    padding: 2rem 0 3rem 0;
    border-bottom: 1px solid var(--border-color);
    margin-bottom: 2rem;
}

.claude-header h1 {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(135deg, var(--accent-orange), var(--accent-purple));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.5rem;
}

.claude-header p {
    color: var(--text-secondary);
    font-size: 1rem;
    font-weight: 400;
}

/* Chat messages */
.stChatMessage {
    background: transparent !important;
    border: none !important;
    padding: 1.5rem 0 !important;
    margin: 0 !important;
}

.stChatMessage[data-testid="chat-message-user"] {
    background: transparent !important;
}

.stChatMessage[data-testid="chat-message-assistant"] {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 1rem !important;
    margin: 1rem 0 !important;
    padding: 1.5rem !important;
}

/* Message content styling */
.stChatMessage [data-testid="stMarkdown"] {
    color: var(--text-primary);
    line-height: 1.6;
    font-size: 0.95rem;
}

.stChatMessage[data-testid="chat-message-user"] [data-testid="stMarkdown"] {
    background: linear-gradient(135deg, var(--accent-orange), #F59E0B);
    color: white;
    padding: 1rem 1.25rem;
    border-radius: 1rem;
    margin-left: 4rem;
    border: 1px solid rgba(249, 115, 22, 0.3);
    box-shadow: 0 4px 12px rgba(249, 115, 22, 0.2);
}

.stChatMessage[data-testid="chat-message-assistant"] [data-testid="stMarkdown"] {
    padding: 0;
    background: transparent;
    color: var(--text-primary);
}

/* Hide avatars */
.stChatMessage > div:first-child {
    display: none !important;
}

.stChatMessage [data-testid="chat-avatar"] {
    display: none !important;
}

.stChatMessage img {
    display: none !important;
}

/* Input styling */
.stChatInput {
    background: var(--input-bg) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 1rem !important;
    color: var(--text-primary) !important;
}

.stChatInput input {
    background: var(--input-bg) !important;
    color: var(--text-primary) !important;
    border: none !important;
    font-size: 0.95rem !important;
    padding: 0.75rem 1rem !important;
}

.stChatInput input::placeholder {
    color: var(--text-secondary) !important;
}

/* Force input text color */
.stChatInput textarea {
    background: var(--input-bg) !important;
    color: var(--text-primary) !important;
    border: none !important;
}

.stChatInput button {
    background: var(--accent-orange) !important;
    border: none !important;
    border-radius: 0.5rem !important;
    color: white !important;
    transition: all 0.2s ease !important;
}

.stChatInput button:hover {
    background: #EA580C !important;
    transform: translateY(-1px);
}

/* Loading and success messages */
.stSuccess {
    background: rgba(34, 197, 94, 0.1) !important;
    border: 1px solid #22C55E !important;
    color: #4ADE80 !important;
    border-radius: 0.5rem !important;
}

.stError {
    background: rgba(239, 68, 68, 0.1) !important;
    border: 1px solid #EF4444 !important;
    color: #F87171 !important;
    border-radius: 0.5rem !important;
}

.stSpinner {
    color: var(--accent-orange) !important;
}

/* Status indicator */
.status-indicator {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.75rem 1rem;
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: 0.5rem;
    margin: 1rem 0;
    font-size: 0.875rem;
    color: var(--text-secondary);
}

.status-dot {
    width: 8px;
    height: 8px;
    background: #22C55E;
    border-radius: 50%;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

/* Markdown styling for bot responses */
.stChatMessage[data-testid="chat-message-assistant"] h1,
.stChatMessage[data-testid="chat-message-assistant"] h2,
.stChatMessage[data-testid="chat-message-assistant"] h3 {
    color: var(--text-primary);
    margin-top: 1.5rem;
    margin-bottom: 0.5rem;
}

.stChatMessage[data-testid="chat-message-assistant"] ul,
.stChatMessage[data-testid="chat-message-assistant"] ol {
    margin: 1rem 0;
    padding-left: 1.5rem;
}

.stChatMessage[data-testid="chat-message-assistant"] li {
    margin: 0.5rem 0;
    color: var(--text-primary);
}

.stChatMessage[data-testid="chat-message-assistant"] strong {
    color: var(--accent-orange);
    font-weight: 600;
}

.stChatMessage[data-testid="chat-message-assistant"] p {
    margin: 1rem 0;
    color: var(--text-primary);
}

/* Welcome message styling */
.welcome-message {
    text-align: center;
    padding: 2rem;
    color: var(--text-secondary);
    border: 1px solid var(--border-color);
    border-radius: 1rem;
    background: var(--bg-secondary);
    margin: 2rem 0;
}

.welcome-message h3 {
    color: var(--text-primary);
    margin-bottom: 0.5rem;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# Modern header
st.markdown("""
<div class="claude-header">
    <h1>Lyllium Health</h1>
    <p>Ask me anything about Women's Health</p>
</div>
""", unsafe_allow_html=True)


# Get API key
# api_key = os.getenv("ANTHROPIC_API_KEY") 
api_key = st.secrets['API_KEY']

if not api_key:
    st.error("Please set ANTHROPIC_API_KEY")
    st.stop()

# Load knowledge base
@st.cache_resource
def load_knowledge_base():
    with st.spinner("Loading knowledge base..."):
        try:
            if not os.path.exists("./knowledge"):
                st.error("Create a './knowledge' folder and add some .txt files!")
                st.stop()
            
            loader = DirectoryLoader("./knowledge", glob="*.txt")
            documents = loader.load()
            
            if not documents:
                st.error("No .txt files found in ./knowledge folder!")
                st.stop()
            
            # st.write(f"📄 Found {len(documents)} documents")
            
            splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            texts = splitter.split_documents(documents)
            # st.write(f"📝 Created {len(texts)} text chunks")
            
            # st.write("🔄 Creating embeddings...")
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            # st.write("💾 Building vector database...")
            vectorstore = FAISS.from_documents(texts, embeddings)
            
            return vectorstore
            
        except Exception as e:
            st.error(f"Error loading knowledge base: {e}")
            st.stop()

# Initialize
if os.path.exists("./knowledge"):
    vectorstore = load_knowledge_base()
    llm = ChatAnthropic(anthropic_api_key=api_key, model="claude-3-5-sonnet-20241022")
    # st.success("✅ Knowledge base ready!")
else:
    st.error("Please create a './knowledge' folder and add some .txt files!")
    st.stop()

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display welcome message if no messages
if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.write("👋 Hi, I am Lyllium, what would you like to chat about today?")

# Chat input
if prompt := st.chat_input("Type your question..."):
    # Show user message
    st.chat_message("user").write(prompt)
    
    # Get relevant docs from knowledge base
    docs = vectorstore.similarity_search(prompt, k=2)
    context = "\n".join([doc.page_content for doc in docs])
    
    # Create prompt with context - allowing external knowledge
    full_prompt = f"""You are Lyllium, a women's health assistant. You MUST follow these rules strictly:

    **RULE 1: SCOPE CHECK**
    First, determine if the question is about women's health (hormones, fertility, PCOS, menopause, periods, pregnancy, postpartum, breast health, reproductive health).

    **RULE 2: IF NOT WOMEN'S HEALTH**
    Respond ONLY with: "I specialize specifically in women's health topics like hormones, fertility, PCOS, menopause, and reproductive wellness. For other health questions, I'd recommend consulting with a healthcare provider or specialist. Is there anything about women's health I can help you with instead?"

    **RULE 3: IF WOMEN'S HEALTH BUT NO CONTEXT**
    If it's women's health but you don't have relevant information in the knowledge base, say a variation of "That's a great question! I don't have enough specific information about that in my current knowledge base. Please consult with a healthcare provider for accurate guidance on this topic."

    **RULE 4: IF WOMEN'S HEALTH WITH CONTEXT**
    Only then provide a helpful response using the knowledge base and general women's health knowledge.

    **Knowledge base context:** {context}

    **User's question:** {prompt}

    Follow the rules above step by step but do not mention the rules in your response. Only answer facts available in your knowledge base, politely decline otherwise."""

# Answer: (Use the knowledge base context when relevant, but use Menopause Society's guidelines when necessary and cite them)"""
    
    # Get response from Claude
    with st.chat_message("assistant"):
        # Show custom loading message
        loading_placeholder = st.empty()
        loading_placeholder.markdown("*Preparing the best response for you...*")
        
        response = llm.invoke(full_prompt)
        
        # Clear loading message and stream response
        loading_placeholder.empty()
        
        # Use st.markdown directly for better formatting
        full_response = ""
        words = response.content.split()
        st.markdown(response.content, unsafe_allow_html=True)

        # st.write(words)
        
        # # for word in words:
        # #     full_response += word + " "
        # #     # Use st.markdown with container for proper formatting
        # with st.container():
        #     st.markdown(full_response, unsafe_allow_html=True)
        #     time.sleep(0.03)

        # Auto-scroll to bottom after response
