Hey Ryland, thanks for reaching out, I'm glad the post resonated. Your background is exactly what this project needs more of. Let me give you the quick picture of where things stand and where you could have outsized impact.

The core of the project is an autonomous research agent that runs continuously -- it generates hypotheses about drug repurposing, tests them computationally, records what works and what doesn't, and proposes new directions, all without human intervention. It uses Claude Code in a loop, maintains a research roadmap, and commits findings to git as it goes. Think of it as a tireless research assistant that runs experiments 24/7.

The agent sits on top of a drug repurposing model built entirely from public data:

- DRKG (Drug Repurposing Knowledge Graph) -- 97K nodes, 5.8M edges linking drugs, diseases, genes, pathways, side effects
- Every Cure's ground truth -- 58K known drug-disease treatment pairs
- Graph embeddings (Node2Vec) that encode biological relationships into vectors
- A kNN collaborative filtering approach that leverages "similar diseases share treatments"

The whole thing runs on a laptop. No GPU required. All open source.

What's working: for any given disease, about 37% of known treatments appear in the model's top 30 predictions out of ~8,000 candidate drugs (using honest disease-holdout evaluation. The model has never seen these diseases during training). The research agent recently discovered that simple kNN collaborative filtering outperforms all the ML models I'd built by 10+ percentage points. The insight seems simple, find the most similar diseases in embedding space, see what drugs treat them, and rank by frequency. 

Beyond the Dantrolene finding, the model independently surfaced Rituximab for MS (now on the WHO Essential Medicines list), Lovastatin for multiple myeloma (RCT showed improved OS/PFS), and Empagliflozin for Parkinson's (HR 0.80 in a Korean cohort study). There's automated confounding detection that flags predictions that are actually artifact. For example, statins get "predicted" for diabetes, but statins actually increase diabetes risk. A validation pipeline automatically checks predictions against ClinicalTrials.gov and PubMed.

The research agent has now identified what I'm calling the "DRKG ceiling". '37% recall is approximately the maximum achievable using only the knowledge graph. The theoretical oracle ceiling (if we could perfectly identify which diseases share treatments) is around 60%. That 23 percentage point gap requires external data sources: phenotype ontologies, protein-protein interaction networks, clinical trial databases, or other biological priors not captured in DRKG.

This is actually where your expertise becomes most valuable. The computational infrastructure works. The agent can run experiments autonomously. What's missing is domain knowledge about which external data sources would help most. A geneticist's intuition about what biological features actually predict treatment transferability across diseases could break through this ceiling in ways that pure computational exploration can't.

You could have significant impact in a few areas:

The agent is domain-agnostic right now. It could be pointed at genetic skin diseases and inflammation specifically. It would generate hypotheses about drug candidates for specific skin conditions, test them against the knowledge graph, validate against clinical literature, and surface candidates you might not have considered. You'd provide the domain expertise to evaluate outputs and steer direction. The agent does the computational grunt work.

More strategically: which biological databases or ontologies do you use in your research that capture disease similarity in ways the knowledge graph might miss? Gene expression signatures? Phenotype hierarchies? Pathway databases? The model currently represents diseases as opaque embedding vectors. Grounding those in interpretable biological features you trust could be the key to both better generalization and predictions you can actually reason about mechanistically.

The whole repo is at https://github.com/jimhuff/open-cure. The simplest way to start would be a call where I walk you through the setup and we figure out which diseases in your research would be most interesting to target first. The research agent can be running experiments on your specific questions within a day of setup.

Happy to jump on a call anytime you'd like if this sounds interesting.
https://calendly.com/jamesdanielhuff/discussion-with-jim?month=2026-01

Jim
