# Deep Research with Mistral

A powerful open-source research assistant that performs deep, multi-layered research on any topic, powered by Mistral AI via custom RunPod deployment.

## Features

- Automated deep research with adjustable breadth and depth
- Follow-up question generation for better context
- Concurrent processing of multiple research queries
- Comprehensive final report generation with citations
- Three research modes: fast, balanced, and comprehensive
- Progress tracking and detailed logging
- Source tracking and citation management

## Prerequisites

- Python 3.9+
- RunPod API key (for using Mistral model via RunPod)
- RunPod Endpoint ID (for your vLLM Mistral deployment)
- Docker (if using dev container)
- VS Code with Dev Containers extension (if using dev container)

## Installation

You can set up this project in one of two ways:

### Option 1: Using Dev Container (Recommended)

1. Open the project in VS Code
2. When prompted, click "Reopen in Container" or run the "Dev Containers: Reopen in Container" command
3. Create a `.env` file in the root directory and add your Gemini API key:
   ```
   GEMINI_KEY=your_api_key_here
   ```

### Option 2: Local Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd open-gemini-deep-research
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the root directory and add your API keys:
   ```
   RUNPOD_API_KEY=your_runpod_api_key_here
   RUNPOD_ENDPOINT_ID=your_runpod_endpoint_id_here
   ```

## Usage

Run the main script with your research query:
```bash
python main.py "your research query here"
```

### Optional Arguments

- `--mode`: Research mode (choices: fast, balanced, comprehensive) [default: balanced]
- `--num-queries`: Number of queries to generate [default: 3]
- `--learnings`: List of previous learnings [optional]
- `--runpod-api-key`: RunPod API key (overrides RUNPOD_API_KEY environment variable)

Examples:
```bash
# Basic usage
python main.py "Impact of artificial intelligence on healthcare" --mode comprehensive --num-queries 5

# Specify RunPod API key directly
python main.py "Climate change solutions" --runpod-api-key your_api_key_here
```


## Output

The script will:
1. Analyze your query for optimal research parameters
2. Ask follow-up questions for clarification
3. Conduct multi-layered research
4. Generate a comprehensive report saved as `final_report.md`
5. Show progress updates throughout the process

## Project Structure

```
open-gemini-deep-research/
├── .devcontainer/
│   └── devcontainer.json
├── src/
│   ├── __init__.py
│   └── deep_research.py
├── .env
├── .gitignore
├── dockerfile
├── main.py
├── README.md
└── requirements.txt
```

## How It Works

### Research Modes

The application offers three research modes that affect how deeply and broadly the research is conducted:

1. **Fast Mode**
   - Performs quick, surface-level research
   - Maximum of 3 concurrent queries
   - No recursive deep diving
   - Typically generates 2-3 follow-up questions per query
   - Best for time-sensitive queries or initial exploration
   - Processing time: ~1-3 minutes

2. **Balanced Mode** (Default)
   - Provides moderate depth and breadth
   - Maximum of 7 concurrent queries
   - No recursive deep diving
   - Generates 3-5 follow-up questions per query
   - Explores main concepts and their immediate relationships
   - Processing time: ~3-6 minutes
   - Recommended for most research needs

3. **Comprehensive Mode**
   - Conducts exhaustive, in-depth research
   - Maximum of 5 initial queries, but includes recursive deep diving
   - Each query can spawn sub-queries that go deeper into the topic
   - Generates 5-7 follow-up questions with recursive exploration
   - Explores primary, secondary, and tertiary relationships
   - Includes counter-arguments and alternative viewpoints
   - Processing time: ~5-12 minutes
   - Best for academic or detailed analysis

### Research Process

1. **Query Analysis**
   - Analyzes initial query to determine optimal research parameters
   - Assigns breadth (1-10 scale) and depth (1-5 scale) values
   - Adjusts parameters based on query complexity and chosen mode

2. **Query Generation**
   - Creates unique, non-overlapping search queries
   - Uses semantic similarity checking to avoid redundant queries
   - Maintains query history to prevent duplicates
   - Adapts number of queries based on mode settings

3. **Research Tree Building**
   - Implements a tree structure to track research progress
   - Each query gets a unique UUID for tracking
   - Maintains parent-child relationships between queries
   - Tracks query order and completion status
   - Provides detailed progress visualization through JSON tree structure

4. **Deep Research** (Comprehensive Mode)
   - Implements recursive research strategy
   - Each query can generate one follow-up query
   - Reduces breadth at deeper levels (breadth/2)
   - Maintains visited URLs to avoid duplicates
   - Combines learnings from all levels

5. **Report Generation**
   - Synthesizes findings into a coherent narrative
   - Minimum 3000-word detailed report
   - Includes inline citations and source tracking
   - Organizes information by relevance and relationship
   - Adds creative elements like scenarios and analogies
   - Maintains factual accuracy while being engaging

### Technical Implementation

- Uses LiteLLM exclusively with Mistral 24B Instruct model
- Custom RunPod adapter for seamless integration with self-hosted vLLM endpoints
- Model operations include:
  - Query analysis and generation
  - Content processing and synthesis
  - Semantic similarity checking
  - Report generation
- Implements concurrent processing for queries
- Uses progress tracking system with tree visualization
- Maintains research tree structure for relationship mapping

#### Research Tree Implementation

The research tree is implemented through the `ResearchProgress` class that tracks:
- Query relationships (parent-child)
- Query completion status
- Learnings per query
- Query order
- Unique IDs for each query

The complete research tree structure is automatically saved to `research_tree.json` when generating the final report, allowing for later analysis or visualization of the research process.

Example tree structure:
```json
{
  "query": "root query",
  "id": "uuid-1",
  "status": "completed",
  "depth": 2,
  "learnings": ["learning 1", "learning 2"],
  "sub_queries": [
    {
      "query": "sub-query 1",
      "id": "uuid-2",
      "status": "completed",
      "depth": 1,
      "learnings": ["learning 3"],
      "sub_queries": [],
      "parent_query": "root query"
    }
  ],
  "parent_query": null
}
```
