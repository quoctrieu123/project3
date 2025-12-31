# Project: Legal Multi-Agent System

This project aims to build a robust multi-agent system for legal question answering and document analysis. It leverages a graph-based architecture to route queries, retrieve relevant legal information, and process user-uploaded documents.

## 1. General Architecture

The system is built using **LangGraph** to orchestrate a workflow of specialized agents. The architecture follows a state-machine approach where a shared state (`AgentState`) is passed between nodes (agents).

### Workflow Overview:
1.  **Router Agent**: The entry point. It uses a trained PyTorch model (`PathClassifier`) to classify the user's query into one of two paths:
    *   `extract_laws`: For general legal questions requiring external legal knowledge.
    *   `documents`: For questions related to specific files uploaded by the user.
2.  **Sub-Agents**:
    *   **Generate Subqueries Agent**: Decomposes complex legal queries into smaller, retrievable sub-queries.
    *   **Retrieve Laws Agent**: Uses the sub-queries to fetch relevant legal contexts from a vector database.
    *   **Laws Agent**: Synthesizes the retrieved legal information to answer the user's question.
    *   **Documents Agent**: Processes and answers questions based on the context of uploaded documents (e.g., PDFs).
    *   **Verifier Agent**: (Optional/In-progress) Validates the generated answers for accuracy.
    *   **Reasoning Agent**: (Optional/In-progress) Provides logical reasoning steps for complex deductions.

The system uses **Google Gemini** (via `ChatGoogleGenerativeAI`) as the underlying LLM for generation and reasoning.

## 2. File Explanations

### `multi_agent_system/`
This folder contains the core logic for the advanced multi-agent implementation.
*   **`multi_agent.py`**: The main entry point that defines the LangGraph workflow, nodes, and edges. It initializes the `AgentState` and orchestrates the execution flow.
*   **`router_agent.py`**: Contains the logic for the Router. It loads the trained `path_classifier_model.pth` to decide the execution path.
*   **`generate_subqueries_agent.py`**: An agent responsible for breaking down a user query into multiple search queries to improve retrieval coverage.
*   **`retrieve_laws_agent.py`**: Handles the retrieval of legal documents/articles based on the generated sub-queries.
*   **`laws_agent.py`**: The specialist agent that formulates answers using the retrieved legal context.
*   **`documens_agent.py`**: The agent dedicated to answering questions based on user-uploaded files (RAG on specific docs).
*   **`verifier_agent.py`**: An agent designed to critique and verify the correctness of the generated response.
*   **`reasoning_agent.py`**: An agent focused on providing step-by-step reasoning for the answer.
*   **`extract_docs_agent.py`**: Helper agent/module for extracting content from documents.
*   **`visualize_system.ipynb`**: A notebook to visualize the LangGraph structure.
*   **`agent_evaluation.ipynb` & `router_evaluation.py`**: Tools for evaluating the performance of the agents and the router.

### `single_agent_system/`
This folder contains the initial prototype of the project.
*   **`app.py`**: A Streamlit web application serving as the user interface for the single-agent version.
*   **`agent.py`**: The implementation of the single monolithic agent.
*   **`ChatbotLuatPhap.ipynb`**: A notebook for testing and prototyping the chatbot logic.
*   **`index.py`**: Logic for indexing documents for retrieval.
*   **`extractfile.py`**: Utilities for parsing PDF/text files.
*   **`config.py`**: Configuration settings for the single-agent system.

### Root Directory
*   **`classifier_based_path_classifier_training.py`**: The training script for the Router's classification model.
*   **`create_laws_100k_index.ipynb`**: Notebook for creating the vector index of legal documents (FAISS/Chroma).
*   **`models/`**: Stores trained models, such as `path_classifier_model.pth`.
*   **`dataset/`**: Contains the dataset used for training the classifier.

## 3. Development Strategy

The project followed an iterative development strategy:

1.  **Single-Agent System (Phase 1)**:
    *   Initially, I deployed a **Single-Agent System** (located in `single_agent_system/`).
    *   This version served as a Proof of Concept (PoC) to validate the basic RAG (Retrieval-Augmented Generation) pipeline and user interface using Streamlit.
    *   It handled all tasks (retrieval, generation, context management) within a single agentic flow.

2.  **Multi-Agent System (Phase 2)**:
    *   To improve accuracy, scalability, and handling of complex queries, I transitioned the project to a **Multi-Agent System** (located in `multi_agent_system/`).
    *   This architecture decomposes the problem into specialized sub-tasks (Routing, Retrieval, Reasoning, Verification).
    *   This transition allows for better control over the logic flow, more precise retrieval strategies (via sub-queries), and the ability to handle different types of requests (General Law vs. Specific Documents) more effectively.
