import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
from pyvis.network import Network
import streamlit.components.v1 as components

load_dotenv()

def get_graph_data(driver, limit=50):
    """Fetch graph data from Neo4j"""
    with driver.session() as session:
        result = session.run("""
            MATCH (e:Entity)-[r]->(e2:Entity)
            RETURN e.name as source, e.type as source_type,
                   r.type as relation,
                   e2.name as target, e2.type as target_type
            LIMIT $limit
        """, limit=limit)
        return [record.data() for record in result]

def create_graph_visualization(driver, limit=50):
    """Create interactive graph visualization"""
    data = get_graph_data(driver, limit)

    if not data:
        return None

    # Create network
    net = Network(
        height="600px",
        width="100%",
        bgcolor="#0a0a1a",
        font_color="white",
        directed=True
    )

    # Color map for entity types
    colors = {
        "Person": "#EF9F27",
        "Concept": "#7F77DD",
        "Organization": "#1D9E75",
        "Tool": "#D85A30",
        "Unknown": "#888888"
    }

    # Add nodes and edges
    nodes_added = set()

    for record in data:
        source = record["source"]
        target = record["target"]
        relation = record.get("relation", "RELATES_TO")
        source_type = record.get("source_type", "Unknown")
        target_type = record.get("target_type", "Unknown")

        # Add source node
        if source not in nodes_added:
            net.add_node(
                source,
                label=source,
                color=colors.get(source_type, "#7F77DD"),
                size=25,
                font={"size": 12, "color": "white"},
                title=f"Type: {source_type}"
            )
            nodes_added.add(source)

        # Add target node
        if target not in nodes_added:
            net.add_node(
                target,
                label=target,
                color=colors.get(target_type, "#7F77DD"),
                size=25,
                font={"size": 12, "color": "white"},
                title=f"Type: {target_type}"
            )
            nodes_added.add(target)

        # Add edge
        net.add_edge(
            source,
            target,
            label=relation,
            color="#555555",
            font={"size": 9, "color": "#aaaaaa"},
            arrows="to"
        )

    # Physics settings for nice layout
    net.set_options("""
    {
        "physics": {
            "enabled": true,
            "barnesHut": {
                "gravitationalConstant": -8000,
                "centralGravity": 0.3,
                "springLength": 150,
                "springConstant": 0.04
            },
            "stabilization": {
                "iterations": 100
            }
        },
        "interaction": {
            "hover": true,
            "tooltipDelay": 200
        }
    }
    """)

    # Save to HTML
    html_path = "E:\\graph-rag-project\\graph_viz.html"
    net.save_graph(html_path)

    return html_path