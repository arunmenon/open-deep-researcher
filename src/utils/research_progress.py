import json
import uuid

class ResearchProgress:
    def __init__(self, depth: int, breadth: int):
        self.total_depth = depth
        self.total_breadth = breadth
        self.current_depth = depth
        self.current_breadth = 0
        self.queries_by_depth = {}
        self.query_order = []  # Track order of queries
        self.query_parents = {}  # Track parent-child relationships
        self.total_queries = 0  # Total number of queries including sub-queries
        self.completed_queries = 0
        self.query_ids = {}  # Store persistent IDs for queries
        self.root_query = None  # Store the root query

    def start_query(self, query: str, depth: int, parent_query: str = None):
        """Record the start of a new query"""
        if depth not in self.queries_by_depth:
            self.queries_by_depth[depth] = {}

        if query not in self.queries_by_depth[depth]:
            # Generate ID only once per query
            if query not in self.query_ids:
                self.query_ids[query] = str(uuid.uuid4())
                
            self.queries_by_depth[depth][query] = {
                "completed": False,
                "learnings": [],
                "id": self.query_ids[query]  # Use persistent ID
            }
            self.query_order.append(query)
            if parent_query:
                self.query_parents[query] = parent_query
            else:
                self.root_query = query  # Set as root if no parent
            self.total_queries += 1

        self.current_depth = depth
        self.current_breadth = len(self.queries_by_depth[depth])
        self._report_progress(f"Starting query: {query}")

    def add_learning(self, query: str, depth: int, learning: str):
        """Record a learning for a specific query"""
        if depth in self.queries_by_depth and query in self.queries_by_depth[depth]:
            if learning not in self.queries_by_depth[depth][query]["learnings"]:
                self.queries_by_depth[depth][query]["learnings"].append(learning)
                self._report_progress(f"Added learning for query: {query}")

    def complete_query(self, query: str, depth: int):
        """Mark a query as completed"""
        if depth in self.queries_by_depth and query in self.queries_by_depth[depth]:
            if not self.queries_by_depth[depth][query]["completed"]:
                self.queries_by_depth[depth][query]["completed"] = True
                self.completed_queries += 1
                self._report_progress(f"Completed query: {query}")

                # Check if parent query exists and update its status if all children are complete
                parent_query = self.query_parents.get(query)
                if parent_query:
                    self._update_parent_status(parent_query)

    def _update_parent_status(self, parent_query: str):
        """Update parent query status based on children completion"""
        # Find all children of this parent
        children = [q for q, p in self.query_parents.items() if p == parent_query]
        
        # Check if all children are complete
        parent_depth = next((d for d, queries in self.queries_by_depth.items() 
                           if parent_query in queries), None)
        
        if parent_depth is not None:
            all_children_complete = all(
                self.queries_by_depth[d][q]["completed"]
                for q in children
                for d in self.queries_by_depth
                if q in self.queries_by_depth[d]
            )
            
            if all_children_complete:
                # Complete the parent query
                self.complete_query(parent_query, parent_depth)

    def _report_progress(self, action: str):
        """Report current progress"""
        print(f"\nResearch Progress Update:")
        print(f"Action: {action}")

        # Build and print the tree starting from the root query
        if self.root_query:
            tree = self._build_research_tree()
            print("\nQuery Tree Structure:")
            print(json.dumps(tree, indent=2))

        print(f"\nOverall Progress: {self.completed_queries}/{self.total_queries} queries completed")
        print("")

    def _build_research_tree(self):
        """Build the full research tree structure"""
        def build_node(query):
            """Recursively build the tree node"""
            # Find the depth for this query
            depth = next((d for d, queries in self.queries_by_depth.items() 
                         if query in queries), 0)
            
            data = self.queries_by_depth[depth][query]
            
            # Find all children of this query
            children = [q for q, p in self.query_parents.items() if p == query]
            
            return {
                "query": query,
                "id": self.query_ids[query],
                "status": "completed" if data["completed"] else "in_progress",
                "depth": depth,
                "learnings": data["learnings"],
                "sub_queries": [build_node(child) for child in children],
                "parent_query": self.query_parents.get(query)
            }

        # Start building from the root query
        if self.root_query:
            return build_node(self.root_query)
        return {}

    def get_learnings_by_query(self):
        """Get all learnings organized by query"""
        learnings = {}
        for depth, queries in self.queries_by_depth.items():
            for query, data in queries.items():
                if data["learnings"]:
                    learnings[query] = data["learnings"]
        return learnings