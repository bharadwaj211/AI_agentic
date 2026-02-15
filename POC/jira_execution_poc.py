import os
import requests
from collections import Counter
from typing import TypedDict, Dict, Any
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

# -------------------------
# Load environment variables
# -------------------------
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=dotenv_path)

JIRA_BASE_URL = os.getenv("JIRA_BASE_URL")
JIRA_EMAIL = os.getenv("JIRA_EMAIL")
JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN")
OPEN_API_KEY = os.getenv("OPEN_API_KEY")

# Check for missing variables
missing_vars = [
    var_name
    for var_name, value in [
        ("JIRA_BASE_URL", JIRA_BASE_URL),
        ("JIRA_EMAIL", JIRA_EMAIL),
        ("JIRA_API_TOKEN", JIRA_API_TOKEN),
        ("OPEN_API_KEY", OPEN_API_KEY),
    ]
    if not value
]

if missing_vars:
    raise ValueError(
        f"The following environment variables are missing or not loaded from .env: {', '.join(missing_vars)}"
    )

print("All environment variables loaded successfully!")

# -------------------------
# Setup the LLM
# -------------------------
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=OPEN_API_KEY
)

# -------------------------
# Define Agent State
# -------------------------
class AgentState(TypedDict):
    user_input: str
    jql: str
    issues: Dict[str, Any]
    counts: Dict[str, int]

# -------------------------
# Node 1: Generate JQL from user input
# -------------------------
def generate_jql(state: AgentState) -> AgentState:
    prompt = f"""
You are a Jira expert.
Convert the following natural language request into a **valid JQL query**.
Use correct operators like =, !=, >=, <=, IN, NOT IN, IS, IS NOT.
Only return the JQL query, no explanations.

Request:
{state['user_input']}
"""
    response = llm.invoke(prompt)
    jql_query = response.content.strip()
    print(f"Generated JQL: {jql_query}")  # debug output
    return {**state, "jql": jql_query}


# -------------------------
# Node 2: Fetch issues from Jira
# -------------------------
def fetch_issues(state: AgentState) -> AgentState:
    url = f"{JIRA_BASE_URL}/rest/api/3/search/jql"

    response = requests.get(
        url,
        headers={"Accept": "application/json"},
        params={
            "jql": state["jql"],
            "maxResults": 1000,
            "fields": "issuetype"
        },
        auth=(JIRA_EMAIL, JIRA_API_TOKEN),
    )

    if response.status_code != 200:
        raise Exception(f"Jira API Error: {response.text}")

    data = response.json()
    print(f"\nTotal Issues Returned: {data.get('total', 0)}")

    return {**state, "issues": data}

# -------------------------
# Node 3: Count issue types
# -------------------------
def count_issue_types(state: AgentState) -> AgentState:
    issues = state["issues"].get("issues", [])

    counter = Counter(
        issue["fields"]["issuetype"]["name"]
        for issue in issues
    )

    print("\nIssue Type Breakdown:")
    for k, v in counter.items():
        print(f"   {k}: {v}")

    return {**state, "counts": dict(counter)}

# -------------------------
# Define the LangGraph
# -------------------------
builder = StateGraph(AgentState)
builder.add_node("generate_jql", generate_jql)
builder.add_node("fetch_issues", fetch_issues)
builder.add_node("count_issue_types", count_issue_types)
builder.set_entry_point("generate_jql")
builder.add_edge("generate_jql", "fetch_issues")
builder.add_edge("fetch_issues", "count_issue_types")
builder.add_edge("count_issue_types", END)
graph = builder.compile()

# -------------------------
# Main execution
# -------------------------
def main():
    print("Jira LangGraph Agent")
    print("Type your request (e.g., 'How many bugs in project ABC this month?')\n")

    user_input = input(">> ")

    result = graph.invoke({
        "user_input": user_input,
        "jql": "",
        "issues": {},
        "counts": {}
    })

    print("\nDone")

if __name__ == "__main__":
    main()
