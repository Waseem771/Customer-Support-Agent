import streamlit as st
from crewai import Agent, Task, Crew, Process, LLM
from tools import knowledge_base_tool, order_database_tool, human_escalation_tool
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

# --- 3. API Key Management — checks st.secrets then os.environ ---
def _get_groq_key() -> str:
    # 1. Streamlit secrets (local .streamlit/secrets.toml or Cloud dashboard)
    try:
        key = st.secrets.get("GROQ_API_KEY", "")
        if key:
            return key
    except Exception:
        pass
    # 2. Environment variable fallback (useful for Docker / CI)
    key = os.environ.get("GROQ_API_KEY", "")
    if key:
        return key
    return ""

GROQ_API_KEY_VALUE = _get_groq_key()

if not GROQ_API_KEY_VALUE:
    st.error(
        "❌ **Missing GROQ_API_KEY.**\n\n"
        "Add it to `.streamlit/secrets.toml`:\n"
        "```toml\n"
        'GROQ_API_KEY = "gsk_...your_key_here..."\n'
        "```\n"
        "Or set it as an environment variable before launching:\n"
        "```powershell\n"
        "$env:GROQ_API_KEY = 'gsk_...your_key_here...'\n"
        "streamlit run app.py\n"
        "```"
    )
    st.stop()

os.environ["GROQ_API_KEY"] = GROQ_API_KEY_VALUE

# -----------------------------------------------------------------------
# Model — groq/llama-3.3-70b-versatile (free tier, fast, reliable)
# The "groq/" prefix tells LiteLLM to route to Groq automatically.
# -----------------------------------------------------------------------
MODEL_NAME = "groq/llama-3.3-70b-versatile"

# --- 4. Initialize LLM and Agent (Cached) ---
@st.cache_resource
def get_agent():
    llm = LLM(
        model=MODEL_NAME,
        api_key=GROQ_API_KEY_VALUE,
        temperature=0.3
    )

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
        llm=llm,
        tools=[knowledge_base_tool, order_database_tool, human_escalation_tool]
    )

support_agent = get_agent()

# --- 5. Helper: Run Crew ---
def run_crew_with_fallback(task: Task) -> str:
    """
    Runs the crew using Groq's openai/gpt-oss-120b model.
    Returns the agent's response or a friendly error message.
    """
    try:
        crew = Crew(
            agents=[support_agent],
            tasks=[task],
            verbose=True,
            process=Process.sequential
        )
        result = crew.kickoff()
        return getattr(result, "raw", str(result))

    except Exception as e:
        err_str = str(e)
        if any(kw in err_str for kw in ("429", "rate", "quota", "overload")):
            return (
                "I'm sorry, the AI service is currently experiencing high demand. "
                "Please wait a moment and try again.\n\n"
                f"_Error: {err_str}_"
            )
        return f"I'm sorry, I encountered an error: {err_str}"

# --- 6. Chat Interface and Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm your virtual support assistant. How can I help you today?"}
    ]

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# --- 7. Handle User Input ---
if prompt := st.chat_input("Type your message here..."):
    # Append user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Compile chat history to maintain context
    chat_history = ""
    for m in st.session_state.messages[:-1]:  # Exclude the current prompt
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

    # Run the agent with automatic model fallback on 503 errors
    with st.spinner("Agent is thinking..."):
        response_text = run_crew_with_fallback(task)

    # Append and display agent response
    st.session_state.messages.append({"role": "assistant", "content": response_text})
    st.chat_message("assistant").write(response_text)

    # Rerun to update the sidebar if any escalations occurred
    st.rerun()
