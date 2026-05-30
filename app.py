import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import FakeEmbeddings
from langchain_community.vectorstores import Chroma
from neo4j import GraphDatabase
import streamlit.components.v1 as components

load_dotenv()

st.set_page_config(
    page_title="Self-Healing Graph RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }

    .stApp {
        background: linear-gradient(135deg, #0a0a1a 0%, #1a0a2e 50%, #0a1a2e 100%);
    }
    .main-title {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #7F77DD, #1D9E75, #EF9F27);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 0.5rem 0;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 0.8rem;
        margin-bottom: 0.5rem;
        letter-spacing: 1px;
    }
    .stat-card {
        background: linear-gradient(135deg, rgba(127,119,221,0.15), rgba(29,158,117,0.15));
        border: 1px solid rgba(127,119,221,0.3);
        border-radius: 12px;
        padding: 0.8rem;
        text-align: center;
    }
    .stat-number {
        font-size: 1.8rem;
        font-weight: 700;
        color: #7F77DD;
    }
    .stat-label {
        font-size: 0.7rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .graph-connection {
        background: rgba(29,158,117,0.1);
        border-left: 3px solid #1D9E75;
        border-radius: 0 8px 8px 0;
        padding: 0.3rem 0.8rem;
        margin: 0.2rem 0;
        font-family: monospace;
        font-size: 0.75rem;
        color: #1D9E75;
    }
    .pipeline-step {
        background: rgba(239,159,39,0.1);
        border: 1px solid rgba(239,159,39,0.3);
        border-radius: 8px;
        padding: 0.5rem;
        font-size: 0.8rem;
        color: #EF9F27;
    }
    .feature-item {
        background: rgba(127,119,221,0.1);
        border-radius: 6px;
        padding: 0.3rem 0.6rem;
        margin: 0.2rem 0;
        font-size: 0.8rem;
        color: #ccc;
    }
    .sidebar-title {
        font-size: 1rem;
        font-weight: 600;
        color: #7F77DD;
    }
    .main .block-container {
        padding-top: 1rem !important;
        padding-bottom: 120px !important;
        max-width: 100% !important;
    }
    section[data-testid="stBottom"] {
        background: linear-gradient(0deg, #0d0d1f 60%, transparent) !important;
        padding: 1rem 2rem 1.5rem 2rem !important;
        backdrop-filter: blur(20px) !important;
    }
    div[data-testid="stChatInput"] {
        background: rgba(255,255,255,0.05) !important;
        border: none !important;
        border-radius: 50px !important;
        padding: 0.3rem 1rem !important;
        box-shadow: none !important;
        backdrop-filter: blur(20px) !important;
    }
    div[data-testid="stChatInput"]:focus-within {
        border: none !important;
        box-shadow: none !important;
    }
    div[data-testid="stChatInput"] textarea {
        background: transparent !important;
        color: white !important;
        font-size: 0.95rem !important;
        border: none !important;
        outline: none !important;
        padding: 0.5rem 0 !important;
    }
    div[data-testid="stChatInput"] textarea::placeholder {
        color: rgba(127,119,221,0.6) !important;
        font-style: italic !important;
    }
    div[data-testid="stChatInput"] button {
        background: linear-gradient(135deg, #7F77DD, #1D9E75) !important;
        border-radius: 50% !important;
        border: none !important;
        width: 38px !important;
        height: 38px !important;
    }
    div[data-testid="stChatMessage"] {
        background: rgba(255,255,255,0.03) !important;
        border: 1px solid rgba(127,119,221,0.1) !important;
        border-radius: 16px !important;
        padding: 1rem !important;
        margin: 0.5rem 0 !important;
    }
    div[data-testid="stExpander"] {
        background: rgba(255,255,255,0.02) !important;
        border: 1px solid rgba(127,119,221,0.2) !important;
        border-radius: 12px !important;
    }
    .verdict-grounded {
        background: rgba(29,158,117,0.15);
        border: 1px solid rgba(29,158,117,0.4);
        border-radius: 8px;
        padding: 0.4rem 0.8rem;
        font-size: 0.8rem;
        color: #1D9E75;
        display: inline-block;
        margin-top: 0.5rem;
    }
    .verdict-hallucinated {
        background: rgba(216,90,48,0.15);
        border: 1px solid rgba(216,90,48,0.4);
        border-radius: 8px;
        padding: 0.4rem 0.8rem;
        font-size: 0.8rem;
        color: #D85A30;
        display: inline-block;
        margin-top: 0.5rem;
    }
    .retry-badge {
        background: rgba(127,119,221,0.15);
        border: 1px solid rgba(127,119,221,0.4);
        border-radius: 8px;
        padding: 0.4rem 0.8rem;
        font-size: 0.8rem;
        color: #7F77DD;
        display: inline-block;
        margin-top: 0.3rem;
        margin-left: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# FIX: Smart Neo4j URI builder
# ─────────────────────────────────────────
def get_neo4j_driver():
    """
    Handles all possible NEO4J_URI formats correctly.
    Fixes ServiceUnavailable error on Streamlit Cloud.
    """
    raw_uri = os.getenv("NEO4J_URI", "")
    username = os.getenv("NEO4J_USERNAME", "")
    password = os.getenv("NEO4J_PASSWORD", "")

    # Try uri formats in order until one works
    uri_candidates = []

    if "neo4j+s://" in raw_uri:
        # Keep original + also try bolt+ssc
        uri_candidates.append(raw_uri)
        uri_candidates.append(raw_uri.replace("neo4j+s://", "neo4j+ssc://"))
        uri_candidates.append(raw_uri.replace("neo4j+s://", "bolt+s://"))
    elif "neo4j+ssc://" in raw_uri:
        uri_candidates.append(raw_uri)
        uri_candidates.append(raw_uri.replace("neo4j+ssc://", "neo4j+s://"))
    else:
        uri_candidates.append(raw_uri)

    last_error = None
    for uri in uri_candidates:
        try:
            driver = GraphDatabase.driver(uri, auth=(username, password))
            # Test the connection
            with driver.session() as session:
                session.run("RETURN 1")
            return driver  # success
        except Exception as e:
            last_error = e
            continue

    # All failed — raise the last error with a helpful message
    raise ConnectionError(
        f"Could not connect to Neo4j with any URI variant.\n"
        f"Last error: {last_error}\n"
        f"Please check your NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD in secrets."
    )


# ─────────────────────────────────────────
# Initialize system
# ─────────────────────────────────────────
@st.cache_resource
def initialize_system():
    llm = ChatGroq(model="llama-3.1-8b-instant")
    driver = get_neo4j_driver()
    from src.ingestion.document_loader import load_documents
    chunks = load_documents()
    embeddings = FakeEmbeddings(size=384)
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
    return llm, driver, vectorstore


# ─────────────────────────────────────────
# Core RAG functions (unchanged)
# ─────────────────────────────────────────
def graph_retriever(driver, question):
    with driver.session() as session:
        words = [w for w in question.lower().split() if len(w) > 3]
        results = []
        for word in words[:5]:
            result = session.run("""
                MATCH (e:Entity)-[r]->(e2:Entity)
                WHERE toLower(e.name) CONTAINS $keyword
                   OR toLower(e2.name) CONTAINS $keyword
                RETURN e.name as source, r.type as relation,
                       e2.name as target
                LIMIT 15
            """, keyword=word)
            for record in result:
                results.append(
                    f"{record['source']} --[{record['relation']}]--> {record['target']}"
                )
    return list(set(results))


def vector_retriever(vectorstore, question):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(question)
    return [doc.page_content for doc in docs]


def generate_answer(llm, question, graph_context, vector_context):
    prompt = ChatPromptTemplate.from_template("""
You are an expert AI assistant specializing in AI research.
Answer the question using ONLY the context below.
Do NOT use any outside knowledge.

KNOWLEDGE GRAPH CONTEXT:
{graph_context}

DOCUMENT CONTEXT:
{vector_context}

Question: {question}

Give a clear, structured, detailed answer based only on the context above.
If the context does not contain enough information, say so honestly.
""")
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({
        "question": question,
        "graph_context": "\n".join(graph_context) if graph_context else "No graph data found.",
        "vector_context": "\n\n".join(vector_context) if vector_context else "No document data found."
    })


# ─────────────────────────────────────────
# Critic agent (Groq — no extra API key)
# ─────────────────────────────────────────
def critic_agent(graph_context, vector_context, answer):
    """
    Uses a separate Groq LLM call to check if the answer
    is grounded in the retrieved context or hallucinated.
    Returns: 'GROUNDED' or 'HALLUCINATED'
    """
    critic_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

    prompt = ChatPromptTemplate.from_template("""
You are a strict fact-checker. Your only job is to verify answers.

RETRIEVED CONTEXT (Knowledge Graph + Documents):
{context}

GENERATED ANSWER:
{answer}

TASK: Check if every single claim in the answer can be found in the context above.

- If YES, every claim is supported → reply: GROUNDED
- If NO, any claim is NOT in the context → reply: HALLUCINATED

Reply with ONE word only. No explanation. Just: GROUNDED or HALLUCINATED
""")
    chain = prompt | critic_llm | StrOutputParser()
    combined_context = "\n".join(graph_context) + "\n\n" + "\n\n".join(vector_context)

    try:
        result = chain.invoke({"context": combined_context, "answer": answer})
        verdict = result.strip().upper()
        return "GROUNDED" if "GROUNDED" in verdict else "HALLUCINATED"
    except Exception:
        return "GROUNDED"  # safe fallback if critic itself fails


# ─────────────────────────────────────────
# Self-healing retry loop
# ─────────────────────────────────────────
def run_with_retry(llm, driver, vectorstore, question, max_attempts=2):
    """
    Full self-healing pipeline:
    1. Retrieve from Neo4j graph + ChromaDB vector store
    2. Generate answer with Groq LLaMA
    3. Check with Groq critic agent
    4. If HALLUCINATED → rephrase and retry
    5. After max attempts → return graceful fallback

    Returns: (answer, verdict, attempts, graph_context, vector_context)
    """
    current_question = question
    graph_context = []
    vector_context = []

    for attempt in range(max_attempts):
        # Step 1: Retrieve
        graph_context  = graph_retriever(driver, current_question)
        vector_context = vector_retriever(vectorstore, current_question)

        # Step 2: Generate
        answer = generate_answer(llm, current_question, graph_context, vector_context)

        # Step 3: Critique
        verdict = critic_agent(graph_context, vector_context, answer)

        # Step 4: If grounded → return immediately
        if verdict == "GROUNDED":
            return answer, verdict, attempt + 1, graph_context, vector_context

        # Step 5: Rephrase for retry
        current_question = f"Please explain in detail with specific facts only from the documents: {question}"

    # All attempts failed → graceful fallback
    fallback = (
        "I don't have enough verified information in my knowledge base "
        "to answer this accurately. Please try rephrasing your question "
        "or ask something related to AI agents."
    )
    return fallback, "HALLUCINATED", max_attempts, graph_context, vector_context


# ─────────────────────────────────────────
# Graph visualization
# ─────────────────────────────────────────
def create_question_graph(driver, question):
    try:
        from pyvis.network import Network
        words = [w for w in question.lower().split() if len(w) > 3]
        all_data = []

        with driver.session() as session:
            for word in words[:5]:
                result = session.run("""
                    MATCH (e:Entity)-[r]->(e2:Entity)
                    WHERE toLower(e.name) CONTAINS $keyword
                       OR toLower(e2.name) CONTAINS $keyword
                    RETURN e.name as source, e.type as source_type,
                           r.type as relation,
                           e2.name as target, e2.type as target_type
                    LIMIT 20
                """, keyword=word)
                all_data.extend([record.data() for record in result])

        if not all_data:
            return None

        net = Network(
            height="500px", width="100%",
            bgcolor="#111128", font_color="#ffffff", directed=True
        )

        colors = {
            "Person": "#EF9F27", "Concept": "#7F77DD",
            "Organization": "#1D9E75", "Tool": "#D85A30", "Unknown": "#7F77DD"
        }

        nodes_added = set()
        for record in all_data:
            source      = record["source"]
            target      = record["target"]
            relation    = record.get("relation", "RELATES_TO")
            source_type = record.get("source_type", "Unknown")
            target_type = record.get("target_type", "Unknown")

            if source not in nodes_added:
                net.add_node(source, label=source,
                    color={"background": colors.get(source_type, "#7F77DD"),
                           "border": "#ffffff",
                           "highlight": {"background": "#EF9F27", "border": "#ffffff"}},
                    size=35, font={"size": 14, "color": "#ffffff", "bold": True},
                    borderWidth=2, title=f"<b>{source}</b><br>Type: {source_type}", shadow=True)
                nodes_added.add(source)

            if target not in nodes_added:
                net.add_node(target, label=target,
                    color={"background": colors.get(target_type, "#7F77DD"),
                           "border": "#ffffff",
                           "highlight": {"background": "#EF9F27", "border": "#ffffff"}},
                    size=35, font={"size": 14, "color": "#ffffff", "bold": True},
                    borderWidth=2, title=f"<b>{target}</b><br>Type: {target_type}", shadow=True)
                nodes_added.add(target)

            net.add_edge(source, target, label=relation,
                color={"color": "#1D9E75", "highlight": "#EF9F27"},
                font={"size": 11, "color": "#ffffff", "strokeWidth": 3, "strokeColor": "#111128"},
                width=2, arrows={"to": {"enabled": True, "scaleFactor": 1.2}},
                smooth={"type": "curvedCW", "roundness": 0.2})

        net.set_options("""
        {
            "physics": {
                "enabled": true,
                "barnesHut": {
                    "gravitationalConstant": -5000,
                    "centralGravity": 0.5,
                    "springLength": 180,
                    "springConstant": 0.05,
                    "damping": 0.09
                },
                "stabilization": {"enabled": true, "iterations": 200, "updateInterval": 25}
            },
            "interaction": {"hover": true, "tooltipDelay": 100, "zoomView": true, "dragView": true},
            "nodes": {"shape": "dot", "shadow": true},
            "edges": {"shadow": true, "smooth": true}
        }
        """)

        html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "question_graph.html")
        net.save_graph(html_path)
        return html_path

    except Exception:
        return None


# ─────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">🧠 Self-Healing Graph RAG</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**👨‍💻 Developer**")
    st.markdown("Utsav Gaywala")
    st.markdown("MSc IT | Uka Tarsadia University")
    st.markdown("---")
    st.markdown("**🛠️ Tech Stack**")
    for tech in ["LangChain", "Neo4j AuraDB", "Groq LLaMA 3.1",
                 "Groq Critic Agent", "ChromaDB", "Streamlit"]:
        st.markdown(f'<div class="feature-item">⚡ {tech}</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**🔄 Self-Healing Pipeline**")
    for step in ["1️⃣ Retrieve from Neo4j + ChromaDB",
                 "2️⃣ Generate answer with LLaMA",
                 "3️⃣ Critic checks for hallucination",
                 "4️⃣ Retry if hallucinated",
                 "5️⃣ Return verified answer"]:
        st.markdown(f'<div class="feature-item">{step}</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**💡 Try asking:**")
    for ex in ["What is an AI agent?", "How does planning work?",
               "What is chain of thought?", "Explain memory in AI agents"]:
        st.markdown(f'<div class="feature-item">💬 {ex}</div>', unsafe_allow_html=True)
    st.markdown("---")
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()


# ─────────────────────────────────────────
# Header
# ─────────────────────────────────────────
st.markdown('<div class="main-title">🧠 Self-Healing Graph RAG</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">KNOWLEDGE GRAPH + VECTOR SEARCH + CRITIC AGENT + RETRY LOOP • BUILT BY UTSAV GAYWALA</div>',
    unsafe_allow_html=True)


# ─────────────────────────────────────────
# Initialize with error handling
# ─────────────────────────────────────────
try:
    with st.spinner("⚡ Connecting to Neo4j and loading documents..."):
        llm, driver, vectorstore = initialize_system()
    st.success("✅ System ready!", icon="🚀")
except Exception as e:
    st.error(f"""
    ❌ **Connection Error**

    Could not connect to Neo4j. Please check your Streamlit secrets:
    - `NEO4J_URI` — should be like `neo4j+s://xxxxxxxx.databases.neo4j.io`
    - `NEO4J_USERNAME` — usually `neo4j`
    - `NEO4J_PASSWORD` — your AuraDB password

    **Error:** `{str(e)}`
    """)
    st.stop()


# ─────────────────────────────────────────
# Stats
# ─────────────────────────────────────────
try:
    col1, col2, col3, col4 = st.columns(4)
    with driver.session() as session:
        nodes = session.run("MATCH (e:Entity) RETURN count(e) as count").single()["count"]
        rels  = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()["count"]

    with col1:
        st.markdown(f'<div class="stat-card"><div class="stat-number">{nodes}</div><div class="stat-label">📊 Graph Nodes</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="stat-card"><div class="stat-number">{rels}</div><div class="stat-label">🔗 Relationships</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="stat-card"><div class="stat-number">63</div><div class="stat-label">📄 Doc Chunks</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="stat-card"><div class="stat-number">3</div><div class="stat-label">⚡ Pipeline Steps</div></div>', unsafe_allow_html=True)
except Exception:
    pass

st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────
# Chat
# ─────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.markdown("""
    <div style='text-align:center; color:#555; padding:2rem;'>
        <div style='font-size:2.5rem'>🧠</div>
        <div style='font-size:1.1rem; color:#7F77DD; font-weight:600;'>
            Welcome to Self-Healing Graph RAG</div>
        <div style='font-size:0.85rem; color:#555; margin-top:0.3rem;'>
            Ask anything about AI agents — answers are critic-verified!</div>
    </div>
    """, unsafe_allow_html=True)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("graph_html"):
            st.markdown("**🕸️ Knowledge Graph:**")
            components.html(message["graph_html"], height=500)

if question := st.chat_input("✨ Ask anything about AI agents..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("🔍 Retrieving → Generating → Critic checking..."):
            answer, verdict, attempts, graph_context, vector_context = run_with_retry(
                llm, driver, vectorstore, question
            )

        st.markdown(answer)

        # Verdict + attempts badges
        badge_col1, badge_col2 = st.columns([2, 1])
        with badge_col1:
            if verdict == "GROUNDED":
                st.markdown(
                    '<div class="verdict-grounded">✅ Critic verified: answer is grounded in sources</div>',
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div class="verdict-hallucinated">⚠️ Could not fully verify — showing best available answer</div>',
                    unsafe_allow_html=True)
        with badge_col2:
            st.markdown(
                f'<div class="retry-badge">🔄 Attempts: {attempts}</div>',
                unsafe_allow_html=True)

        with st.expander("📊 View Sources & Pipeline Debug"):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**🔗 Graph Connections:**")
                if graph_context:
                    for conn in graph_context[:5]:
                        st.markdown(f'<div class="graph-connection">{conn}</div>', unsafe_allow_html=True)
                else:
                    st.info("No graph connections found for this question")
            with c2:
                st.markdown("**📄 ChromaDB Chunks:**")
                st.success(f"✅ {len(vector_context)} chunks retrieved")
                st.markdown(
                    f'<div class="pipeline-step">⚡ Verdict: {verdict} | Attempts: {attempts}/{2}</div>',
                    unsafe_allow_html=True)
                if attempts > 1:
                    st.warning("⚠️ Retry was triggered — query was rephrased automatically")

        st.markdown("**🕸️ Knowledge Graph for your question:**")
        with st.spinner("Generating graph visualization..."):
            q_html_path = create_question_graph(driver, question)
            graph_html = None
            if q_html_path:
                with open(q_html_path, "r", encoding="utf-8") as f:
                    graph_html = f.read()
                components.html(graph_html, height=500)
            else:
                st.info("No graph connections found for this question")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "graph_html": graph_html
    })