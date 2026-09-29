import json
import numpy as np
import pandas as pd
import faiss
from fastembed import TextEmbedding
from crewai.tools import tool
import streamlit as st

# Cache the embedding model and FAISS index — loads only once per session
@st.cache_resource
def load_knowledge_base():
    try:
        # fastembed uses ONNX runtime (no torch/torchvision needed)
        # It auto-downloads the model on first boot
        model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
        index = faiss.read_index("faiss.index")
        with open("chunks.json", "r", encoding="utf-8") as f:
            chunks = json.load(f)
        return model, index, chunks
    except Exception as e:
        print(f"Warning: Knowledge base files not found. Error: {e}")
        return None, None, None

@tool("Search Company Knowledge Base")
def knowledge_base_tool(query: str) -> str:
    """Use this tool to find information about company policies, FAQs, product knowledge, and general information."""
    model, index, chunks = load_knowledge_base()
    if model is None:
        return "Knowledge base is currently unavailable. Please check if faiss.index and chunks.json are uploaded."

    try:
        # fastembed returns a generator — convert to numpy array
        embeddings = list(model.embed([query]))
        query_vector = np.array(embeddings).astype("float32")

        k = 3
        distances, indices = index.search(query_vector, k)

        results = []
        for i in indices[0]:
            if i != -1 and i < len(chunks):
                chunk_data = chunks[i]
                text = chunk_data.get('text', chunk_data.get('content', str(chunk_data)))
                source = chunk_data.get('source_filename', 'Unknown Source')
                results.append(f"Source: {source}\nContent: {text}")

        if results:
            return "\n\n---\n\n".join(results)
        return "No relevant information found in the knowledge base."
    except Exception as e:
        return f"Error accessing knowledge base: {str(e)}"

@tool("Search Order Database")
def order_database_tool(query: str) -> str:
    """Use this tool to look up customer order details such as status, shipping address, and products."""
    try:
        df = pd.read_excel("Orders_Tracking_Sheet.xlsx")
        match = df[(df['Order ID'].astype(str).str.contains(query, case=False, na=False)) |
                   (df['Customer Name'].astype(str).str.contains(query, case=False, na=False))]

        if not match.empty:
            return match.to_string(index=False)
        return f"No order found matching '{query}'."
    except Exception as e:
        return f"Error accessing order database: {str(e)}. Make sure 'orders.xlsx' is uploaded."

@tool("Escalate to Human Agent")
def human_escalation_tool(summary: str) -> str:
    """Use this tool ONLY when you cannot solve the problem yourself, or if the user explicitly requests to speak with a human."""
    if "escalations" not in st.session_state:
        st.session_state.escalations = []

    st.session_state.escalations.append({
        "summary": summary,
        "status": "Pending"
    })

    return "SUCCESS: The issue has been recorded in the escalation system. Tell the user that their request has been successfully escalated to a human agent who will reach out to them."
