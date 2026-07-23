"""Privacy-aware LangSmith input/output processors for custom spans."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _file_name(value: Any) -> str:
    return Path(getattr(value, "name", None) or str(value)).name


def trace_router_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    state = inputs.get("state") or {}
    messages = state.get("messages") or []
    query = getattr(messages[-1], "content", "") if messages else ""
    return {
        "query": query,
        "has_uploaded_files": bool(state.get("uploaded_files")),
        "uploaded_file_count": len(state.get("uploaded_files") or []),
        "session_id": state.get("session_id"),
    }


def trace_router_outputs(output: str) -> dict[str, str]:
    return {"route": output}


def trace_agent_state_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    state = inputs.get("state") or {}
    messages = state.get("messages") or []
    query = ""
    for message in reversed(messages):
        if message.__class__.__name__ == "HumanMessage":
            query = getattr(message, "content", "")
            break
    context = (
        state.get("docs_context", "")
        if state.get("route") == "documents"
        else state.get("laws_context", "")
    )
    return {
        "query": query,
        "route": state.get("route"),
        "session_id": state.get("session_id"),
        "message_count": len(messages),
        "context_length": len(context or ""),
    }


def trace_subquery_outputs(output: Any) -> dict[str, Any]:
    subqueries = list(output or [])
    return {"subquery_count": len(subqueries), "subqueries": subqueries}


def trace_verifier_outputs(output: Any) -> dict[str, Any]:
    final_answer, explanation = output
    return {
        "final_answer": final_answer,
        "explanation": explanation,
        "final_answer_length": len(final_answer),
    }


def trace_ingestion_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    files = inputs.get("uploaded_files") or []
    return {
        "session_id": inputs.get("session_id"),
        "file_count": len(files),
        "file_names": [_file_name(value) for value in files],
    }


def trace_ingestion_outputs(output: Any) -> dict[str, Any]:
    document_ids = list(output or [])
    return {
        "document_count": len(document_ids),
        "document_ids": document_ids,
    }


def trace_document_search_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    document_ids = list(inputs.get("document_ids") or [])
    return {
        "query": inputs.get("query"),
        "session_id": inputs.get("session_id"),
        "document_count": len(document_ids),
        "document_ids": document_ids,
    }


def _safe_results(results: Any) -> list[dict[str, Any]]:
    safe: list[dict[str, Any]] = []
    for result in results or []:
        payload = result.get("payload") or {}
        text = str(payload.get("text") or "")
        safe.append(
            {
                "id": result.get("id"),
                "score": result.get("score"),
                "rrf_score": result.get("rrf_score"),
                "matched_queries": result.get("matched_queries"),
                "file_name": payload.get("file_name"),
                "page": payload.get("page"),
                "legacy_id": payload.get("legacy_id"),
                "source": payload.get("source"),
                "text_length": len(text),
            }
        )
    return safe


def trace_search_outputs(output: Any) -> dict[str, Any]:
    safe = _safe_results(output)
    return {"result_count": len(safe), "results": safe}


def trace_legal_search_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    queries = list(inputs.get("queries") or [])
    return {"query_count": len(queries), "queries": queries}


def trace_legal_search_outputs(output: Any) -> dict[str, Any]:
    results_by_query = output or {}
    return {
        "query_count": len(results_by_query),
        "results": {
            query: _safe_results(results)
            for query, results in results_by_query.items()
        },
    }


def trace_rrf_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    results_by_query = inputs.get("results_by_query") or {}
    return {
        "query_count": len(results_by_query),
        "candidate_count": sum(len(items) for items in results_by_query.values()),
        "rrf_k": inputs.get("rrf_k"),
        "top_k": inputs.get("top_k"),
    }


def trace_cleanup_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    return {"session_id": inputs.get("session_id")}


def trace_cleanup_outputs(output: Any) -> dict[str, Any]:
    return {"deleted_point_count": int(output or 0)}
