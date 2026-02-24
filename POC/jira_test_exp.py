import os
import json
import requests
import subprocess
import xml.etree.ElementTree as ET
from typing import Annotated, List, Union
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END, add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from dotenv import load_dotenv

# =========================
# Load environment variables
# =========================
load_dotenv()

JIRA_BASE_URL = os.getenv("JIRA_BASE_URL")
JIRA_EMAIL = os.getenv("JIRA_EMAIL")
JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN")
OPEN_API_KEY = os.getenv("OPEN_API_KEY")
ROBOT_TEST_PATH = os.getenv("ROBOT_TEST_PATH", "tests/")
ROBOT_OUTPUT_DIR = os.getenv("ROBOT_OUTPUT_DIR", "results")

# =================================
# Jira Helper (Updated for v3)
# ================================
def jira_request(method, endpoint, payload=None, params=None):
    """Handles authentication and routing to the Jira Cloud API."""
    endpoint = endpoint.lstrip('/')
    url = f"{JIRA_BASE_URL}/rest/api/3/{endpoint}"
    
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json"
    }
    auth = (JIRA_EMAIL, JIRA_API_TOKEN)
    
    if method.upper() == "GET":
        return requests.get(url, headers=headers, params=params, auth=auth)
    else:
        return requests.post(url, headers=headers, auth=auth, data=json.dumps(payload))

# ==============================
# Define Tools
# =============================

@tool
def search_jira_issues(jql: str):
    """
    Search for Jira issues using a JQL string. 
    Example JQL: 'project = "PROJ" AND issuetype = Bug'
    Returns summaries and status.
    """
    # Using the new /search/jql endpoint required by Atlassian
    params = {
        "jql": jql,
        "fields": "summary,status,issuetype",
        "maxResults": 50
    }
    
    response = jira_request("GET", "search/jql", params=params)
    
    if response.status_code != 200:
        return f"Jira API Error ({response.status_code}): {response.text}"
    
    data = response.json()
    issues = data.get("issues", [])
    
    if not issues:
        return f"No issues found for JQL: {jql}"
    
    results = []
    for i in issues:
        key = i['key']
        itype = i['fields']['issuetype']['name']
        summary = i['fields']['summary']
        status = i['fields']['status']['name']
        results.append(f"{key}: [{itype}] {summary} | Status: {status}")
        
    return "\n".join(results)

@tool
def run_robot_automation():
    """
    Triggers the Robot Framework test suite. 
    If tests fail, it automatically creates a Jira bug and reports the issue key.
    """
    os.makedirs(ROBOT_OUTPUT_DIR, exist_ok=True)
    # Note: Ensure 'robot' is in your PATH
    subprocess.run(["robot", "--outputdir", ROBOT_OUTPUT_DIR, ROBOT_TEST_PATH])
    
    output_file = os.path.join(ROBOT_OUTPUT_DIR, "output.xml")
    if not os.path.exists(output_file):
        return "Robot execution failed: No output.xml found in results directory."

    tree = ET.parse(output_file)
    root = tree.getroot()
    total_stat = root.find(".//total/stat")
    passed = int(total_stat.attrib.get("pass", 0))
    failed = int(total_stat.attrib.get("fail", 0))
    
    report = f"Tests Complete. Passed: {passed}, Failed: {failed}."
    
    if failed > 0:
        # Auto-create bug for failed tests
        payload = {
            "fields": {
                "project": {"key": "SCRUM"}, # Replace with your default project key
                "summary": f"Automation Failure: {failed} tests failed",
                "description": {
                    "type": "doc",
                    "version": 1,
                    "content": [{"type": "paragraph", "content": [{"type": "text", "text": report}]}]
                },
                "issuetype": {"name": "Bug"}
            }
        }
        res = jira_request("POST", "issue", payload)
        if res.status_code == 201:
            report += f" Created Jira issue: {res.json().get('key')}"
            
    return report

# Tool configuration for the graph
tools = [search_jira_issues, run_robot_automation]
tool_node = ToolNode(tools)

# =================================
# Agent/Graph Configuration
# ================================
class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

# Initialize LLM with tool binding
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPEN_API_KEY)
llm_with_tools = llm.bind_tools(tools)

def call_model(state: State):
    """The node that decides which tool to call."""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# Define the graph
builder = StateGraph(State)
builder.add_node("agent", call_model)
builder.add_node("tools", tool_node)

builder.set_entry_point("agent")
builder.add_conditional_edges("agent", tools_condition) # Routes to tools or END
builder.add_edge("tools", "agent") # Returns back to agent after tool execution

graph = builder.compile()

# ===================================
# Execution Loop
# ===================================
def main():
    print("SmartQA Agent → “TestBuddy AI” (Tool-Enabled)\n")
    while True:
        user_input = input("User: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        
        # Stream events from the graph
        inputs = {"messages": [HumanMessage(content=user_input)]}
        for event in graph.stream(inputs, stream_mode="values"):
            if "messages" in event:
                last_msg = event["messages"][-1]
                # Only print if it's an AI message with actual text (not tool calls)
                if last_msg.type == "ai" and last_msg.content:
                    print(f"Agent: {last_msg.content}")

if __name__ == "__main__":
    main()