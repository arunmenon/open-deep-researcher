# Deep Research Flow: Architecture & Process

This document provides a detailed explanation of the Open Deep Research architecture, components, and workflow.

## System Architecture

```mermaid
graph TD
    U[User Query] --> |Input| M[Main Application]
    M --> |1. Analyze| BD[Breadth & Depth Analysis]
    M --> |2. Clarify| FQ[Follow-up Questions]
    M --> |3. Generate| SQ[Search Queries]
    
    subgraph "Deep Research Process"
        SQ --> |Concurrent Processing| S[Search Service]
        S --> PR[Process Results]
        PR --> |Extract| L[Learnings]
        PR --> |Generate| NQ[New Queries]
        NQ --> |Recursive Research| S
    end
    
    L --> |Aggregate| FR[Final Report]
    FR --> |Output| RP[Research Report]
    
    subgraph "Model Provider Layer"
        MP[Model Provider Interface] --> RP1[RunPod Provider]
        MP --> RP2[Future Providers...]
    end
    
    S --> |API Calls| MP
    PR --> |API Calls| MP
    BD --> |API Calls| MP
    FQ --> |API Calls| MP
    FR --> |API Calls| MP
    
    subgraph "Progress Tracking"
        RT[Research Tree]
        RT --> |Update| PS[Progress State]
        PS --> |Visualize| PV[Progress Visualization]
    end
    
    S --> |Update| RT
    PR --> |Update| RT
```

## Component Breakdown

### 1. User Interface (Main Application)

The main application (`main.py`) serves as the entry point and provides:
- Command-line interface for accepting user queries
- Parameter configuration (mode, number of queries, etc.)
- Environment variable management
- Research orchestration

### 2. DeepSearch Engine

The `DeepSearch` class (`src/deep_research.py`) is the core engine that:
- Determines research scope (breadth and depth)
- Generates follow-up questions for clarification
- Creates search queries
- Manages the research process
- Generates the final report

### 3. Model Provider Layer

The Model Provider system (`src/models/`) provides a modular abstraction for different LLM backends:

```mermaid
classDiagram
    class ModelProvider {
        <<interface>>
        +completion(model, messages, temperature, max_tokens)
        +format_messages_to_prompt(messages)
    }
    
    class RunpodProvider {
        -api_key
        -endpoint_id
        -api_base
        -status_base
        +completion(model, messages, temperature, max_tokens)
        +format_messages_to_prompt(messages)
        -_poll_for_results(job_id)
    }
    
    class FutureProviders {
        +completion(model, messages, temperature, max_tokens)
        +format_messages_to_prompt(messages)
    }
    
    ModelProvider <|-- RunpodProvider
    ModelProvider <|-- FutureProviders
```

#### RunPod Provider
The RunPod provider (`src/models/runpod_provider.py`) implements:
- REST API calls to RunPod
- Message formatting for Mistral
- Asynchronous job polling with exponential backoff
- Response parsing and standardization

### 4. Research Progress Tracking

The progress tracking system (`src/utils/research_progress.py`) maintains:

```mermaid
classDiagram
    class ResearchProgress {
        -total_depth
        -total_breadth
        -current_depth
        -current_breadth
        -queries_by_depth
        -query_order
        -query_parents
        -total_queries
        -completed_queries
        -query_ids
        -root_query
        +start_query(query, depth, parent_query)
        +add_learning(query, depth, learning)
        +complete_query(query, depth)
        +get_learnings_by_query()
        -_update_parent_status(parent_query)
        -_report_progress(action)
        -_build_research_tree()
    }
```

The research tree structure records:
- Parent-child relationships between queries
- Query status (completed/in-progress)
- Learnings associated with each query
- Research depth and breadth parameters
- Unique identifiers for each query

### 5. Response Models

Pydantic models (`src/utils/response_models.py`) ensure consistent data structures:

```mermaid
classDiagram
    class BaseModel {
        <<pydantic>>
    }
    
    class BreadthDepthResponse {
        +breadth: int
        +depth: int
        +explanation: str
    }
    
    class FollowUpQueriesResponse {
        +follow_up_queries: List[str]
    }
    
    class QueriesResponse {
        +queries: List[str]
    }
    
    class QuerySimilarityResponse {
        +are_similar: bool
    }
    
    class ProcessResultResponse {
        +learnings: List[str]
        +follow_up_questions: List[str]
    }
    
    BaseModel <|-- BreadthDepthResponse
    BaseModel <|-- FollowUpQueriesResponse
    BaseModel <|-- QueriesResponse
    BaseModel <|-- QuerySimilarityResponse
    BaseModel <|-- ProcessResultResponse
```

## Research Workflow

### 1. Query Analysis Phase

```mermaid
sequenceDiagram
    participant U as User
    participant M as Main App
    participant DS as DeepSearch
    participant MP as Model Provider
    
    U->>M: Initial Query
    M->>DS: Initialize(mode)
    M->>DS: determine_research_breadth_and_depth(query)
    DS->>MP: completion(breadth_depth_prompt)
    MP-->>DS: BreadthDepthResponse
    DS->>M: breadth, depth, explanation
    
    M->>DS: generate_follow_up_questions(query)
    DS->>MP: completion(follow_up_prompt)
    MP-->>DS: follow_up_questions
    DS->>M: questions
    
    U->>M: Question Answers
    M->>DS: Combined Query
```

### 2. Research Execution Phase

```mermaid
sequenceDiagram
    participant M as Main App
    participant DS as DeepSearch
    participant RP as ResearchProgress
    participant MP as Model Provider
    
    M->>DS: deep_research(query, breadth, depth)
    DS->>RP: Initialize(depth, breadth)
    
    DS->>RP: start_query(query, depth, null)
    
    DS->>MP: completion(generate_queries_prompt)
    MP-->>DS: queries list
    
    loop For each query
        par Concurrent Processing
            DS->>RP: start_query(query, depth, parent)
            DS->>MP: completion(search_prompt)
            MP-->>DS: search results
            DS->>MP: completion(process_results_prompt)
            MP-->>DS: learnings, follow_up_questions
            DS->>RP: add_learning(query, depth, learning)
            
            alt Comprehensive Mode & depth > 1
                DS->>MP: Generate follow-up query
                MP-->>DS: next_query
                DS->>DS: process_query(next_query, depth-1, query)
            end
            
            DS->>RP: complete_query(query, depth)
        end
    end
    
    DS->>M: all_learnings, visited_urls
```

### 3. Report Generation Phase

```mermaid
sequenceDiagram
    participant M as Main App
    participant DS as DeepSearch
    participant MP as Model Provider
    
    M->>DS: generate_final_report(query, learnings, visited_urls)
    DS->>MP: completion(report_prompt)
    MP-->>DS: formatted_report
    DS->>M: final_report
    M->>U: final_report.md
```

## Research Modes

The system supports three research modes that affect how deeply and broadly research is conducted:

```mermaid
graph TD
    subgraph Fast
        F1[Max 3 queries]
        F2[No recursive diving]
        F3[2-3 follow-ups per query]
        F4[~1-3 minute runtime]
    end
    
    subgraph Balanced
        B1[Max 7 queries]
        B2[No recursive diving]
        B3[3-5 follow-ups per query]
        B4[~3-6 minute runtime]
    end
    
    subgraph Comprehensive
        C1[Max 5 initial queries]
        C2[Recursive deep diving]
        C3[5-7 follow-ups with recursion]
        C4[~5-12 minute runtime]
    end
```

### Mode Selection Impact

Each mode configures:
1. Maximum number of concurrent queries
2. Whether recursive deep diving is enabled
3. Number of follow-up questions generated
4. Runtime expectations

The comprehensive mode is particularly distinctive as it implements a tree-based research strategy where each query can spawn sub-queries that explore topics in greater depth.

## RunPod Integration

```mermaid
sequenceDiagram
    participant DS as DeepSearch
    participant RP as RunpodProvider
    participant RAPI as RunPod API
    
    DS->>RP: completion(model, messages, temp, max_tokens)
    RP->>RP: format_messages_to_prompt(messages)
    RP->>RAPI: POST /run (formatted_prompt)
    RAPI-->>RP: job_id
    
    loop Polling with exponential backoff
        RP->>RAPI: GET /status/job_id
        RAPI-->>RP: status
        
        alt status = COMPLETED
            RP->>RP: Parse and format response
            RP-->>DS: standardized_response
        else status = FAILED
            RP-->>DS: Exception
        else status = RUNNING
            RP->>RP: Wait with backoff
        end
    end
```

The RunPod integration handles:
1. Formatting messages for the Mistral model
2. Submitting jobs to the RunPod API
3. Polling for job completion with exponential backoff
4. Parsing and standardizing the response

## Research Tree Structure

The research progress is tracked using a tree structure:

```mermaid
graph TD
    R[Root Query] --> Q1[Query 1]
    R --> Q2[Query 2]
    R --> Q3[Query 3]
    
    Q1 --> Q1_1[Sub-Query 1.1]
    Q1 --> Q1_2[Sub-Query 1.2]
    
    Q2 --> Q2_1[Sub-Query 2.1]
    
    subgraph "Node Structure"
        NS[Node] --- ID[UUID]
        NS --- Status[Status: completed/in_progress]
        NS --- Depth[Research Depth]
        NS --- L[Learnings Array]
        NS --- SQ[Sub-Queries Array]
        NS --- PQ[Parent Query]
    end
```

Each node in the tree represents a research query with:
- Unique identifier
- Completion status
- Current depth level
- List of learnings
- References to parent and child queries

## Conclusion

The Open Deep Research system provides a modular, extensible framework for performing multi-layered research using LLMs. Its key strengths are:

1. **Modularity**: Easy to add new model providers
2. **Adaptability**: Configurable research modes for different needs
3. **Depth**: Recursive research capabilities for comprehensive analysis
4. **Tracking**: Detailed progress visualization and research tree
5. **Concurrency**: Efficient parallel processing of queries

This architecture allows the system to perform increasingly sophisticated research operations while maintaining a clear separation of concerns between components.