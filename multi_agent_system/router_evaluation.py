from .router_agent import run_router_agent
import pandas as pd
import time as time
from .multi_agent import AgentState
from langchain_core.messages import HumanMessage
sentences = pd.read_csv(".\dataset\dataset_for_router_embedding_similarity.csv")
sentences = sentences["query"][:2000].tolist()
print(len(sentences))
total_time = 0
for i, sentence in enumerate(sentences,1):
    state = {"messages": [HumanMessage(content=sentence)]}
    start_time = time.time()
    route = run_router_agent(state)
    end_time = time.time()
    elapsed_time = end_time - start_time
    total_time += elapsed_time
    print(f"Query {i}: {sentence}")
    print(f"Decided route: {route}")

print(f"Average time per query: {total_time / len(sentences):.4f} seconds")
