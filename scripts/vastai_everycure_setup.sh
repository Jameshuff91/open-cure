#!/bin/bash
# Setup script for Every Cure MATRIX pipeline on Vast.ai
# Usage: NEO4J_PASSWORD=yourpassword ./vastai_everycure_setup.sh <PORT> <HOST>
#
# Environment variables:
#   NEO4J_PASSWORD - Neo4j database password (required for security)

set -e

PORT=${1:-22}
HOST=${2:-"ssh.vast.ai"}

echo "=== Every Cure MATRIX Setup ==="
echo "Target: root@${HOST}:${PORT}"

# Create setup script to run on remote
cat << 'REMOTE_SCRIPT' > /tmp/ec_setup.sh
#!/bin/bash
set -e

echo "=== Phase 1: System Setup ==="
apt-get update
apt-get install -y \
    git curl wget unzip \
    python3.11 python3.11-venv python3.11-dev \
    openjdk-11-jdk \
    cmake build-essential

# Install uv (fast Python package manager used by Every Cure)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env

echo "=== Phase 2: Neo4j Installation ==="
# Neo4j 5.x with Graph Data Science plugin
wget -O - https://debian.neo4j.com/neotechnology.gpg.key | apt-key add -
echo 'deb https://debian.neo4j.com stable 5' > /etc/apt/sources.list.d/neo4j.list
apt-get update
apt-get install -y neo4j

# Download and install GDS plugin (for GraphSAGE/Node2Vec)
NEO4J_HOME=/var/lib/neo4j
GDS_VERSION="2.6.0"
wget -q "https://graphdatascience.ninja/neo4j-graph-data-science-${GDS_VERSION}.jar" \
    -O ${NEO4J_HOME}/plugins/neo4j-graph-data-science-${GDS_VERSION}.jar

# Configure Neo4j
cat >> /etc/neo4j/neo4j.conf << 'EOF'
# Allow GDS
dbms.security.procedures.unrestricted=gds.*
dbms.security.procedures.allowlist=gds.*

# Memory settings
server.memory.heap.initial_size=4g
server.memory.heap.max_size=8g
server.memory.pagecache.size=4g

# Network
server.default_listen_address=0.0.0.0
EOF

# Set Neo4j password
neo4j-admin dbms set-initial-password ${NEO4J_PASSWORD:-changeme}

# Start Neo4j
systemctl enable neo4j
systemctl start neo4j

echo "=== Phase 3: Clone Repositories ==="
cd /root
git clone https://github.com/everycure-org/matrix.git everycure-matrix
git clone https://github.com/jimhuff/open-cure.git open-cure || true

echo "=== Phase 4: Setup Python Environment ==="
cd /root/everycure-matrix
uv venv --python 3.11
source .venv/bin/activate

# Install matrix pipeline dependencies
cd pipelines/matrix
uv pip install -e .

# Also install our project for data access
cd /root/open-cure
uv pip install -e . 2>/dev/null || pip install -r requirements.txt 2>/dev/null || true

echo "=== Phase 5: Download Knowledge Graph Data ==="
mkdir -p /root/data/kg

# Option 1: Use DRKG (smaller, what we have)
echo "Downloading DRKG..."
cd /root/data/kg
wget -q https://dgl-data.s3-us-west-2.amazonaws.com/dataset/DRKG/drkg.tar.gz
tar -xzf drkg.tar.gz

# Create import script for Neo4j
cat > /root/import_drkg.py << 'PYEOF'
#!/usr/bin/env python3
"""Import DRKG into Neo4j for GraphSAGE/Node2Vec training."""
import pandas as pd
from neo4j import GraphDatabase
from tqdm import tqdm

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "${NEO4J_PASSWORD:-changeme}"))

def import_drkg():
    # Load DRKG triples
    df = pd.read_csv("/root/data/kg/drkg/drkg.tsv", sep="\t", header=None,
                     names=["head", "relation", "tail"])
    print(f"Loaded {len(df)} triples")

    # Get unique entities
    entities = set(df["head"].unique()) | set(df["tail"].unique())
    print(f"Unique entities: {len(entities)}")

    with driver.session() as session:
        # Create entities
        print("Creating entity nodes...")
        batch_size = 10000
        entity_list = list(entities)
        for i in tqdm(range(0, len(entity_list), batch_size)):
            batch = entity_list[i:i+batch_size]
            session.run("""
                UNWIND $entities AS entity
                MERGE (n:Entity {id: entity})
                SET n.type = split(entity, '::')[1]
            """, entities=batch)

        # Create relationships
        print("Creating relationships...")
        for i in tqdm(range(0, len(df), batch_size)):
            batch = df.iloc[i:i+batch_size].to_dict('records')
            session.run("""
                UNWIND $triples AS t
                MATCH (h:Entity {id: t.head})
                MATCH (t2:Entity {id: t.tail})
                MERGE (h)-[r:RELATES {type: t.relation}]->(t2)
            """, triples=batch)

    print("DRKG import complete!")

if __name__ == "__main__":
    import_drkg()
PYEOF

echo "=== Phase 6: Create Training Scripts ==="
cat > /root/train_embeddings.py << 'PYEOF'
#!/usr/bin/env python3
"""Train GraphSAGE and Node2Vec embeddings using Neo4j GDS."""
from graphdatascience import GraphDataScience
import numpy as np
import json

# Connect to Neo4j
gds = GraphDataScience("bolt://localhost:7687", auth=("neo4j", "${NEO4J_PASSWORD:-changeme}"))
print(f"GDS version: {gds.version()}")

# Create in-memory graph projection
print("Creating graph projection...")
G, result = gds.graph.project(
    "drkg",
    "Entity",
    {
        "RELATES": {"orientation": "UNDIRECTED"}
    }
)
print(f"Graph: {G.node_count()} nodes, {G.relationship_count()} relationships")

# Train Node2Vec embeddings (faster, good baseline)
print("\n=== Training Node2Vec (512-dim) ===")
node2vec_result = gds.node2vec.mutate(
    G,
    embeddingDimension=512,
    walkLength=80,
    walksPerNode=10,
    windowSize=10,
    iterations=10,
    mutateProperty="node2vec_embedding"
)
print(f"Node2Vec trained in {node2vec_result['computeMillis']/1000:.1f}s")

# Export embeddings
print("Exporting embeddings...")
embeddings = gds.graph.nodeProperties.stream(G, ["node2vec_embedding"], separate_property_columns=True)
embeddings.to_parquet("/root/data/embeddings/node2vec_512.parquet")

# Train GraphSAGE (requires node features, so we'll use Node2Vec as features)
print("\n=== Training GraphSAGE (512-dim) ===")
try:
    model, train_result = gds.beta.graphSage.train(
        G,
        modelName="graphsage_drkg",
        featureProperties=["node2vec_embedding"],
        embeddingDimension=512,
        sampleSizes=[25, 10],
        epochs=5,
        learningRate=0.01,
        aggregator="mean"
    )
    print(f"GraphSAGE trained: {train_result}")

    # Generate GraphSAGE embeddings
    gds.beta.graphSage.mutate(
        G,
        modelName="graphsage_drkg",
        mutateProperty="graphsage_embedding"
    )

    # Export
    gs_embeddings = gds.graph.nodeProperties.stream(G, ["graphsage_embedding"], separate_property_columns=True)
    gs_embeddings.to_parquet("/root/data/embeddings/graphsage_512.parquet")
except Exception as e:
    print(f"GraphSAGE failed (may need node features): {e}")

print("\nEmbedding training complete!")
print("Files saved to /root/data/embeddings/")
PYEOF

mkdir -p /root/data/embeddings

echo "=== Setup Complete ==="
echo ""
echo "Neo4j running at: bolt://localhost:7687"
echo "  Username: neo4j"
echo "  Password: ${NEO4J_PASSWORD:-changeme}"
echo ""
echo "Next steps:"
echo "  1. Import DRKG: python /root/import_drkg.py"
echo "  2. Train embeddings: python /root/train_embeddings.py"
echo "  3. Run Every Cure pipeline: cd /root/everycure-matrix/pipelines/matrix && uv run kedro run"
echo ""
REMOTE_SCRIPT

# Copy and run on remote
echo "Uploading setup script..."
scp -P ${PORT} -o StrictHostKeyChecking=no /tmp/ec_setup.sh root@${HOST}:/tmp/ec_setup.sh

echo "Running setup (this will take 10-15 minutes)..."
ssh -p ${PORT} -o StrictHostKeyChecking=no root@${HOST} "chmod +x /tmp/ec_setup.sh && /tmp/ec_setup.sh"

echo ""
echo "=== Setup Complete ==="
echo "SSH: ssh -p ${PORT} root@${HOST}"
