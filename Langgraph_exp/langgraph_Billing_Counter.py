import os
from typing import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph,END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# Load the environment variables from the .env file 
load_dotenv()

# Implementing the check here to see whether it is picking up the env file or
# raise a exception
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError ("If the OPENAI_API_KEY is not found the .env file")

# Define LLM Configuration
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_completion_tokens=300
)

# Define shared state for the agents
class RouterState(TypedDict):
    query: str
    category: str
    response: str
    agent_used: str

# Classifier agent
def classifier_agent(state:RouterState):
    print("\n Classifier analyzing query...\n")
    prompt = f"""
    Classify the following customer query in to one of these categories:
    - technical
    - billing
    - general

    Return only one word: technical, billing or general.

    Query:
    {state['query']}
"""
    result = llm.invoke([HumanMessage(content=prompt)]).content.strip().lower()
    state["category"] = result
    return state

# Technical agent
def technical_agent(state: RouterState):
    print("\n Technical agent handling query...\n")
    prompt = f"""
You are a technical support specialist.

Provide a clear and step-by-step solution
to the following technical issue:

{state['query']}
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    state["response"] = response.content
    state["agent_used"] = "Technical agent"
    return state

# Billing agent
def billing_agent(state:RouterState):
    print("\n Billing agent handling query...\n")
    prompt = f"""
You are a billing and payments specialist.

Respond professionally and clearly to this billing related question:

{state['query']}
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    state["response"] = response.content
    state["agent_used"] = "Billing Agent"
    return state

# General agent
def general_agent(state: RouterState):
    print("\n General Agent handling query.....\n")
    prompt = f"""
You are a customer service representative.
to the following general inquiry:

{state['query']}
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    state["response"] = response.content
    state["agent_used"] = "General Agent"
    return state

# Decision roadmap for the process to flow
def route_decision(state:RouterState):
    if state["category"] == "technical":
        return "technical"
    elif state["category"] == "billing":
        return "billing"
    else:
        return "general"
    
# Langraph flow
workflow = StateGraph(RouterState)

workflow.add_node("classifier", classifier_agent)
workflow.add_node("technical", technical_agent)
workflow.add_node("billing", billing_agent)
workflow.add_node("general", general_agent)

workflow.set_entry_point("classifier")

workflow.add_conditional_edges(
    "classifier",
    route_decision,
    {
        "technical": "technical",
        "billing": "billing",
        "general": "general"
    }
)

workflow.add_edge("technical", END)
workflow.add_edge("billing", END)
workflow.add_edge("general", END)

app = workflow.compile()

# Execution of the program for the agents mentioned in the script
if __name__ == "__main__":
    print("\n-------------------------------------")
    print("   Customer query/enquiry system ")    
    print("-------------------------------------\n")

    user_query = input("Enter your customer query: ")
    initial_state = {
        "query": user_query,
        "category": "",
        "response": "",
        "agent_used": ""
    }

    print("\n Routing query...\n")
    result = app.invoke(initial_state)
    
    print("\n---------------------------------------")
    print("           Final Response                ")
    print("----------------------------------------\n")

    print("Category Detected:", result["category"])
    print("Agent Used:", result["agent_used"])
    print("\nResponse:\n")
    print(result["response"])

