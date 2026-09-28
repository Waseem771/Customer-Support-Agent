import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from crewai import Agent, Task, Crew, Process
from tools import KnowledgeBaseTool, OrderDatabaseTool, HumanEscalationTool
import os

# --- 1. Streamlit Page Configuration ---
st.set_page_config(page_title="Customer Support AI", page_icon="🤖", layout="wide")
st.title("🤖 AI Customer Support Agent")

# --- 2. Sidebar for Escalations Dashboard ---
with st.sidebar:
    st.header("📋 Pending Escalations")
    if "escalations" in st.session_state and st.session_state.escalations:
        for idx, esc in enumerate(st.session_state.escalations):
            st.warning(f"**Escalation #{idx+1}**\n\n**Status:** {esc['status']}\n\n**Summary:** {esc['summary']}")
    else:
        st.info("No pending escalations.")

# --- 3. API Key Management via Streamlit Secrets ---
# In Streamlit Cloud, these are set in the project settings.
if "GOOGLE_API_KEY" not in st.secrets:
    st.error("Missing GOOGLE_API_KEY. Please add it to your Streamlit secrets.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# --- 4. Initialize LLM and Agent (Cached) ---
@st.cache_resource
def get_llm():
    # Using the requested 2026 Gemini model
    return ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite", 
        temperature=0.3
    )

@st.cache_resource
def get_agent():
    return Agent(
        role="Customer Support Specialist",
        goal="Assist customers with queries, check order details, provide company information, and escalate issues when necessary.",
        backstory=(
            "You are a friendly, patient, and highly capable AI support agent for a company in Pakistan. "
            "You always strive to resolve the user's problem using your Knowledge Base or Order Database tools. "
            "If the user is frustrated, explicitly asks for a human, or you cannot resolve their issue, "
            "you immediately use the 'Escalate to Human Agent' tool to report it."
        ),
        verbose=True,
        allow_delegation=False,
        llm=get_llm(),
        tools=[KnowledgeBaseTool(), OrderDatabaseTool(), HumanEscalationTool()]
    )

support_agent = get_agent()

# --- 5. Chat Interface and Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm your virtual support assistant. How can I help you today?"}
    ]

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# --- 6. Handle User Input ---
if prompt := st.chat_input("Type your message here..."):
    # Append user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Compile chat history to maintain context
    chat_history = ""
    for m in st.session_state.messages[:-1]: # Exclude the current prompt
        role = "User" if m["role"] == "user" else "Assistant"
        chat_history += f"{role}: {m['content']}\n"

    # Define the CrewAI Task with dynamic context
    task_description = f"""
    You are currently chatting with a user. Read the chat history below to understand the ongoing context.
    
    === Chat History ===
    {chat_history}
    ====================
    
    Latest User Query: {prompt}
    
    Instructions:
    1. Respond to the user's latest query directly and naturally.
    2. If they ask about an order, use the 'Search Order Database' tool.
    3. If they ask a general company question, use the 'Search Company Knowledge Base' tool.
    4. If they request human help or their issue is beyond your capabilities, use the 'Escalate to Human Agent' tool.
    """

    task = Task(
        description=task_description,
        expected_output="A helpful, polite, and direct response to the user. Do not include internal thoughts in the final response.",
        agent=support_agent
    )

    crew = Crew(
        agents=[support_agent],
        tasks=[task],
        verbose=True,
        process=Process.sequential
    )

    # Run the agent
    with st.spinner("Agent is thinking..."):
        try:
            result = crew.kickoff()
            # Handle CrewAI output format gracefully
            response_text = getattr(result, 'raw', str(result))
        except Exception as e:
            response_text = f"I'm sorry, I encountered an internal error: {str(e)}"

    # Append and display agent response
    st.session_state.messages.append({"role": "assistant", "content": response_text})
    st.chat_message("assistant").write(response_text)
    
    # Rerun to update the sidebar if any escalations occurred
    st.rerun()
