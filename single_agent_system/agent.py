from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda, RunnableBranch
from index import retrieve_laws_reciprocal_ranking, retrieve_sentences_documents
from llm import format_prompt,llm, format_prompt_qa, router, router_classifier_based, router_matching_based, router_embedding_similarity
import time
retriever = RunnableParallel(
    {
        "context": RunnableLambda(lambda x: retrieve_laws_reciprocal_ranking(x)), "query": lambda x: x["query"]
    }
) 
agent_law = (
    retriever
    | RunnableLambda(format_prompt)
    | llm
)
retriever_sentences = RunnableParallel(
    {
        "context": RunnableLambda(lambda x: retrieve_sentences_documents(x)), "query": lambda x: x["query"]
    }
)
agent_qa = (
    {
        "uploaded_files": lambda x: x["uploaded_files"],
        "query": lambda x: x["query"],
    }
    | retriever_sentences
    | RunnableLambda(format_prompt_qa)
    | llm
)


agent = (
    {
        "query": lambda x: x["query"],
        "uploaded_files": lambda x: x.get("uploaded_files",[])
    }
    | RunnableLambda(router_embedding_similarity)
    | RunnableBranch(
        (
            lambda x: x["route"] == "documents", 
            agent_qa
        ),
        (
            lambda x: x["route"] == "extract_laws", 
            RunnableLambda(lambda x: {"query": x["query"]}) | agent_law
        ),
        RunnableLambda(lambda x: {"query": x["query"]}) | agent_law
    )
)
