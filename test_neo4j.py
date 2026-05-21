import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

URI = os.getenv("NEO4J_URI")
USERNAME = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")

# Use bolt+ssc for custom SSL
URI_BOLT = URI.replace("neo4j+s://", "neo4j+ssc://")

driver = GraphDatabase.driver(URI_BOLT, auth=(USERNAME, PASSWORD))

driver.verify_connectivity()
print("✅ Connected to Neo4j AuraDB successfully!")

with driver.session() as session:
    session.run("CREATE (n:TestNode {name: 'Utsav', project: 'Graph RAG'})")
    result = session.run("MATCH (n:TestNode) RETURN n.name, n.project")
    for record in result:
        print(f"✅ Node created: {record['n.name']} - {record['n.project']}")

driver.close()
print("🎉 Neo4j AuraDB is working perfectly!")