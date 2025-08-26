import streamlit as st
import os
import time
from langchain_anthropic import ChatAnthropic
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
import nltk
from deepeval.metrics import (
  AnswerRelevancyMetric, 
  FaithfulnessMetric, 
  ContextualRelevancyMetric, 
  ContextualRecallMetric, 
  ContextualPrecisionMetric
)
from deepeval.test_case import LLMTestCase

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

#for better comprehension, moved the CSS to a separate file
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css('static/style.css')


st.markdown("""
<div class="claude-header">
    <h1>Lyllium Health</h1>
    <p>Ask me anything about Women's Health</p>
</div>
""", unsafe_allow_html=True)


# Get API key
# anthropic_api_key = os.getenv("ANTHROPIC_API_KEY") 
anthropic_api_key = st.secrets['API_KEY']

if not anthropic_api_key:
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
            
            splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=100)
            texts = splitter.split_documents(documents)
            
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            vectorstore = FAISS.from_documents(texts, embeddings)
            
            return vectorstore
            
        except Exception as e:
            st.error(f"Error loading knowledge base: {e}")
            st.stop()

# Initialize
if os.path.exists("./knowledge"):
    vectorstore = load_knowledge_base()
    llm = ChatAnthropic(anthropic_api_key=anthropic_api_key, model="claude-3-7-sonnet-latest")
else:
    st.error("Please create a './knowledge' folder and add some .txt files!")
    st.stop()

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display welcome message if no messages
if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.write("👋 Hi, I am Lyllium, my current knoweldge base includes postmenopausal osteoposris, but I am contantly learning. \n What would you like to chat about today?")

# Chat input
if prompt := st.chat_input("Type your question..."):
    # Show user message
    st.chat_message("user").write(prompt)
    
    # Get relevant docs from knowledge base
    docs = vectorstore.similarity_search(prompt, k=2)
    context = "\n".join([doc.page_content for doc in docs])
    
    # Create prompt with context - allowing external knowledge
    full_prompt = f"""  You are Lyllium, a warm and knowledgeable women's health assistant. You specialize in helping with questions about hormones, fertility, PCOS, menopause, periods, pregnancy, postpartum care, breast health, reproductive wellness, and how conditions like osteoporosis and heart health specifically affect women.
                        How to respond:
                        Use the information in the knowledge base context to answer questions thoroughly and helpfully. If the context contains relevant information, draw from it confidently to provide a complete answer. You can explain concepts naturally and connect related ideas to give users a comprehensive understanding.
                        For questions outside your specialty area, kindly redirect: "I focus specifically on women's health topics. For that question, I'd suggest checking with a healthcare provider who specializes in that area. Is there anything about women's health I can help you with?"
                        Only mention lacking information if the knowledge base truly has no relevant content for a women's health question. If there's related information available, use it to provide a helpful response.
                        Always be conversational and supportive. Include the brief and friendly reminder that healthcare providers give personalized guidance, but don't let this overshadow providing the helpful information you do have.
                        Keep responses natural and friendly - no formal headings or robotic language.

                        **Knowledge base context:** {context}

                        **User's question:** {prompt}

                        """
    
    # Get response from Claude
    with st.chat_message("assistant"):
        # Show custom loading message
        loading_placeholder = st.empty()
        loading_placeholder.markdown("*Preparing the best response for you...*")
        
        response = llm.invoke(full_prompt)
        
        # Clear loading message and stream response
        loading_placeholder.empty()
        
        # Use st.markdown directly for better formatting
        words = response.content.split()
        st.markdown(response.content, unsafe_allow_html=True)

        #adding metrics from deepeval 
        #1. To detect hallucinations - FaithfulnessMetric

        #use an OPENAI model -o3-mini 
        # openai_api_key = os.getenv("OPENAI_API_KEY") 
        openai_api_key = st.secrets['API_KEY']
        test_case = LLMTestCase(
                    input=input,
                    context = [context],
                    actual_output=response.content,
                    retrieval_context=[context],
                    )
    
        metric = FaithfulnessMetric(model="gpt-4o-mini")
        metric.measure(test_case)
        st.write(f"Faithfulness Score: {metric.score}")
        st.write(f"Explanation: {metric.reason}")
        # st.write(words)
        
        # # for word in words:
        # #     full_response += word + " "
        # #     # Use st.markdown with container for proper formatting
        # with st.container():
        #     st.markdown(full_response, unsafe_allow_html=True)
        #     time.sleep(0.03)

        # Auto-scroll to bottom after response
