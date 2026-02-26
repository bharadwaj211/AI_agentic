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
ROBOT_OUTPUT_DIR = os.getenv("ROBOT_OUTPUT_DIR")

# =================================
# Jira Helper
# ================================
def jira_request(method, endpoint, payload=None, params=None):
    endpoint = endpoint.lstrip('/')
    url = f"{JIRA_BASE_URL}/rest/api/3/{endpoint}"
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    auth = (JIRA_EMAIL, JIRA_API_TOKEN)
    
    if method.upper() == "GET":
        return requests.get(url, headers=headers, params=params, auth=auth)
    else:
        return requests.post(url, headers=headers, auth=auth, data=json.dumps(payload))

# ==============================
# Define Tools
# =============================

@tool
def create_jira_ticket(project_key: str, summary: str, description: str, issue_type: str):
    """Creates a Jira ticket (Story, Epic, Bug, or Task) with a description."""
    payload = {
        "fields": {
            "project": {"key": project_key},
            "summary": summary,
            "description": {
                "type": "doc", "version": 1,
                "content": [{"type": "paragraph", "content": [{"type": "text", "text": description}]}]
            },
            "issuetype": {"name": issue_type}
        }
    }
    res = jira_request("POST", "issue", payload)
    return f"Success: {issue_type} created ({res.json().get('key')})" if res.status_code == 201 else f"Error: {res.text}"

@tool
def close_jira_ticket(issue_key: str):
    """
    Closes a Jira ticket by transitioning it to 'Done', 'Closed', or 'Resolved'.
    Works for Bugs, Stories, and Epics.
    """
    res = jira_request("GET", f"issue/{issue_key}/transitions")
    if res.status_code != 200:
        return f"Error fetching transitions for {issue_key}: {res.text}"
    
    transitions = res.json().get("transitions", [])
    target_id = None
    target_name = ""
    close_keywords = ["done", "close", "resolve", "complete"]
    
    for t in transitions:
        if any(key in t["name"].lower() for key in close_keywords):
            target_id = t["id"]
            target_name = t["name"]
            break
    
    if not target_id:
        avail = [t["name"] for t in transitions]
        return f"Could not find a 'Close' transition for {issue_key}. Available: {avail}"

    move_res = jira_request("POST", f"issue/{issue_key}/transitions", {"transition": {"id": target_id}})
    
    if move_res.status_code == 204:
        return f"Successfully closed {issue_key} (Status: {target_name})."
    else:
        return f"Failed to close ticket: {move_res.text}"

@tool
def run_robot_automation():
    """Runs Robot tests. If they fail, creates a bug and moves it to 'In Progress'."""
    os.makedirs(ROBOT_OUTPUT_DIR, exist_ok=True)
    subprocess.run(["robot", "--outputdir", ROBOT_OUTPUT_DIR, ROBOT_TEST_PATH])
    output_file = os.path.join(ROBOT_OUTPUT_DIR, "output.xml")
    if not os.path.exists(output_file): return "Robot execution failed."
    
    tree = ET.parse(output_file)
    failed = int(tree.getroot().find(".//total/stat").attrib.get("fail", 0))
    report = f"Tests finished. Failures: {failed}."
    
    if failed > 0:
        create_res = jira_request("POST", "issue", {
            "fields": {"project": {"key": "SCRUM"}, "summary": f"Auto-Bug: {failed} fails", "issuetype": {"name": "Bug"},
                       "description": {"type": "doc", "version": 1, "content": [{"type": "paragraph", "content": [{"type": "text", "text": report}]}]}}
        })
        if create_res.status_code == 201:
            key = create_res.json().get('key')
            report += f" Created {key}."
            trans_res = jira_request("GET", f"issue/{key}/transitions")
            tid = next((t["id"] for t in trans_res.json().get("transitions", []) if "progress" in t["name"].lower()), None)
            if tid: jira_request("POST", f"issue/{key}/transitions", {"transition": {"id": tid}})
    return report

# =================================
# Graph Configuration
# ================================
tools = [run_robot_automation, create_jira_ticket, close_jira_ticket]
tool_node = ToolNode(tools)

class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPEN_API_KEY)
llm_with_tools = llm.bind_tools(tools)

def call_model(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

builder = StateGraph(State)
builder.add_node("agent", call_model)
builder.add_node("tools", tool_node)
builder.set_entry_point("agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")
graph = builder.compile()

def main():
    print("TestBuddy AI: Create, Run, or Close Jira tickets.")
    print("[Status: Connected]\n")
    
    while True:
        user_input = input("End-user: ").strip()
        
        if user_input.lower() in ["exit", "quit"]:
            print("\n Thank you for using AI assistant and have a productive day.")
            break
            
        for event in graph.stream({"messages": [HumanMessage(content=user_input)]}, stream_mode="values"):
            if "messages" in event:
                last_msg = event["messages"][-1]
                if last_msg.type == "ai" and last_msg.content:
                    print(f"AI-agent: {last_msg.content}")

if __name__ == "__main__":
    main()