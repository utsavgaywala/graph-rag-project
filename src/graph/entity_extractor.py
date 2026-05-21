import os
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from neo4j import GraphDatabase

load_dotenv()

# Initialize Groq LLM
llm = ChatGroq(model="llama-3.1-8b-instant")

# Initialize Neo4j
URI = os.getenv("NEO4J_URI").replace("neo4j+s://", "neo4j+ssc://")
USERNAME = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")
driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))

# Prompt to extract entities and relationships
extraction_prompt = ChatPromptTemplate.from_template("""
You are a knowledge graph expert. Extract entities and relationships from the text below.

Return ONLY a valid JSON object in this exact format, nothing else:
{{
  "entities": [
    {{"name": "entity name", "type": "Person/Concept/Organization/Tool"}},
    {{"name": "entity name", "type": "Person/Concept/Organization/Tool"}}
  ],
  "relationships": [
    {{"source": "entity1", "relation": "RELATION_TYPE", "target": "entity2"}},
    {{"source": "entity1", "relation": "RELATION_TYPE", "target": "entity2"}}
  ]
}}

Text: {text}

Return ONLY the JSON, no explanation, no markdown, no backticks.
""")

def extract_entities(text):
    """Extract entities and relationships from text using LLM"""
    try:
        chain = extraction_prompt | llm
        response = chain.invoke({"text": text})

        # Clean response
        content = response.content.strip()
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]

        data = json.loads(content)
        return data
    except Exception as e:
        print(f"Extraction error: {e}")
        return {"entities": [], "relationships": []}

def store_in_neo4j(data, chunk_text):
    """Store entities and relationships in Neo4j"""
    with driver.session() as session:
        # Create entities as nodes
        for entity in data.get("entities", []):
            if not entity.get("name"):
                continue
            session.run("""
                MERGE (e:Entity {name: $name})
                SET e.type = $type
            """, name=entity["name"], type=entity.get("type", "Unknown"))

        # Create relationships with safety check
        for rel in data.get("relationships", []):
            # Skip if any field is missing
            if not all(k in rel for k in ["source", "target", "relation"]):
                continue
            if not rel["source"] or not rel["target"]:
                continue
            session.run("""
                MERGE (a:Entity {name: $source})
                MERGE (b:Entity {name: $target})
                MERGE (a)-[r:RELATES_TO {type: $relation}]->(b)
            """, source=rel["source"],
                target=rel["target"],
                relation=rel["relation"])

def build_knowledge_graph(chunks):
    """Process chunks and build knowledge graph"""
    print(f"\nBuilding knowledge graph from {len(chunks)} chunks...")
    print("Processing first 5 chunks only for testing...\n")

    for i, chunk in enumerate(chunks[:5]):
        print(f"Processing chunk {i+1}/5...")

        # Extract entities
        data = extract_entities(chunk.page_content)

        entities_found = len(data.get("entities", []))
        relations_found = len(data.get("relationships", []))
        print(f"  Found {entities_found} entities, {relations_found} relationships")

        # Store in Neo4j
        store_in_neo4j(data, chunk.page_content)
        print(f"  ✅ Stored in Neo4j!")

    print("\n🎉 Knowledge graph built successfully!")

    # Show summary
    with driver.session() as session:
        result = session.run("MATCH (e:Entity) RETURN count(e) as count")
        count = result.single()["count"]
        print(f"📊 Total nodes in your graph: {count}")

        result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
        count = result.single()["count"]
        print(f"🔗 Total relationships in your graph: {count}")

    driver.close()

if __name__ == "__main__":
    import sys
    sys.path.append("E:\\graph-rag-project")
    from src.ingestion.document_loader import load_documents

    # Load documents
    chunks = load_documents()

    # Build knowledge graph
    build_knowledge_graph(chunks)