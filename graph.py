from langgraph.graph import StateGraph, END
from state import ClaimsGraphState
from agents.segregator import segregate_pages
from agents.id_agent import extract_identity
from agents.discharge_summary_agent import extract_discharge_summary
from agents.itemized_bill_agent import extract_itemized_bill
from agents.aggregator import aggregate_results

# Initialize the graph
workflow = StateGraph(ClaimsGraphState)

# Add nodes
workflow.add_node("segregator", segregate_pages)
workflow.add_node("id_agent", extract_identity)
workflow.add_node("discharge_summary_agent", extract_discharge_summary)
workflow.add_node("itemized_bill_agent", extract_itemized_bill)
workflow.add_node("aggregator", aggregate_results)

# Set entry point
workflow.set_entry_point("segregator")

# Create a routing function that routes to next agent in sequence
def route_to_next_agent(state: ClaimsGraphState):
    """
    Routes to the next extraction agent based on classification.
    This creates a sequential chain: segregator -> id -> discharge -> bill -> END
    Each agent checks if it has pages to process.
    """
    classification = state.get('classification', {})
    
    # Check which node we're coming from based on extracted_data
    extracted = state.get('extracted_data', {})
    
    # Route from segregator to first available agent
    if 'identity_data' not in extracted:
        if classification.get('identity_document'):
            return "id_agent"
        # Skip to next if no identity pages
    
    # Route from id_agent (or segregator if skipped) to discharge_summary_agent
    if 'discharge_summary_data' not in extracted:
        if classification.get('discharge_summary'):
            return "discharge_summary_agent"
        # Skip to next if no discharge pages
    
    # Route from discharge_summary_agent (or previous if skipped) to itemized_bill_agent
    if 'itemized_bill_data' not in extracted:
        if classification.get('itemized_bill'):
            return "itemized_bill_agent"
    
    # All agents processed, go to aggregator
    return "aggregator"

# Add conditional edges from segregator
workflow.add_conditional_edges(
    "segregator",
    route_to_next_agent,
    {
        "id_agent": "id_agent",
        "discharge_summary_agent": "discharge_summary_agent",
        "itemized_bill_agent": "itemized_bill_agent",
        "aggregator": "aggregator",
        END: END
    }
)

# Add conditional edges from each agent to the next
workflow.add_conditional_edges(
    "id_agent",
    route_to_next_agent,
    {
        "discharge_summary_agent": "discharge_summary_agent",
        "itemized_bill_agent": "itemized_bill_agent",
        "aggregator": "aggregator",
        END: END
    }
)

workflow.add_conditional_edges(
    "discharge_summary_agent",
    route_to_next_agent,
    {
        "itemized_bill_agent": "itemized_bill_agent",
        "aggregator": "aggregator",
        END: END
    }
)

# Bill agent routes to aggregator
workflow.add_conditional_edges(
    "itemized_bill_agent",
    route_to_next_agent,
    {
        "aggregator": "aggregator",
        END: END
    }
)

# Aggregator is the final node, always goes to END
workflow.add_edge("aggregator", END)

# Compile the graph
app = workflow.compile()
