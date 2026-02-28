import os
import json
import requests
import subprocess
import xml.etree.ElementTree as ET
import time
from typing import Annotated, List
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from dotenv import load_dotenv

# =========================
# Configuration & Env
# =========================
load_dotenv()
JIRA_BASE_URL = os.getenv("JIRA_BASE_URL").rstrip('/')
JIRA_EMAIL = os.getenv("JIRA_EMAIL")
JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN")
OPEN_API_KEY = os.getenv("OPEN_API_KEY")
ROBOT_TEST_PATH = os.getenv("ROBOT_TEST_PATH", "tests/")
ROBOT_OUTPUT_DIR = os.getenv("ROBOT_OUTPUT_DIR", "results")

CACHE_FILE = "jira_automation_cache.json"

# =================================
# Local Cache Helpers
# =================================
def get_cached_ticket():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                return json.load(f).get("issue_key")
        except: return None
    return None

def save_cached_ticket(issue_key):
    with open(CACHE_FILE, "w") as f:
        json.dump({"issue_key": issue_key}, f)

# =================================
# Jira REST Helpers
# =================================
def jira_request(method, endpoint, payload=None, params=None):
    url = f"{JIRA_BASE_URL}/rest/api/3/{endpoint.lstrip('/')}"
    auth = (JIRA_EMAIL, JIRA_API_TOKEN)
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    return requests.request(method, url, headers=headers, auth=auth, params=params, json=payload, timeout=20)

def transition_issue(issue_key, target_name):
    res = jira_request("GET", f"issue/{issue_key}/transitions")
    if res.status_code != 200: return False
    transitions = res.json().get("transitions", [])
    target = next((t for t in transitions if target_name.lower() in t["name"].lower()), None)
    if target:
        jira_request("POST", f"issue/{issue_key}/transitions", {"transition": {"id": target["id"]}})
        return True
    return False

# ==============================
# The Automation Tool
# ==============================

@tool
def run_robot_automation(project_key: str = "SCRUM"):
    """
    Executes Robot tests and manages Jira tickets via Local Cache.
    Bypasses Jira Search Indexing completely to prevent duplicates.
    """
    os.makedirs(ROBOT_OUTPUT_DIR, exist_ok=True)
    
    print(f"\n[System]: Running Robot Framework tests...")
    subprocess.run(["robot", "--outputdir", ROBOT_OUTPUT_DIR, ROBOT_TEST_PATH], capture_output=True)

    output_file = os.path.join(ROBOT_OUTPUT_DIR, "output.xml")
    if not os.path.exists(output_file): return "Error: output.xml not found."

    root = ET.parse(output_file).getroot()
    stat = root.find(".//total/stat")
    passed, failed = int(stat.attrib.get("pass", 0)), int(stat.attrib.get("fail", 0))

    # Parse errors
    failed_details = []
    for test in root.findall(".//test"):
        status_tag = test.find("status")
        if status_tag is not None and status_tag.attrib.get("status") == "FAIL":
            failed_details.append(f"* Test Case:* {test.attrib.get('name')}\n*Reason:* {status_tag.text}")
    
    report_body = f"Passed: {passed}, Failed: {failed}\n\n*Failed Details:*\n" + "\n".join(failed_details)

    # --- THE CACHE CHECK (No JQL Search) ---
    print("[System]: Checking local cache for existing ticket...")
    existing_key = get_cached_ticket()
    is_active_in_jira = False

    if existing_key:
        # Ask Jira specifically about THIS ticket key (Direct DB hit)
        res = jira_request("GET", f"issue/{existing_key}")
        if res.status_code == 200:
            status_cat = res.json()["fields"]["status"]["statusCategory"]["name"]
            # Only consider it "active" if it isn't in a 'Done' category
            if status_cat != "Done":
                is_active_in_jira = True
        else:
            existing_key = None # Ticket likely deleted

    if failed > 0:
        if is_active_in_jira:
            # Add a comment to the existing ticket
            payload = {"body": {"type": "doc", "version": 1, "content": [{"type": "paragraph", "content": [{"type": "text", "text": f"Automation Re-run Results:\n{report_body}"}]}]}}
            jira_request("POST", f"issue/{existing_key}/comment", payload)
            return f"Tests Failed ({failed}). Updated existing ticket {existing_key}. WORKFLOW COMPLETE."
        else:
            # Create a brand new ticket
            payload = {
                "fields": {
                    "project": {"key": project_key},
                    "summary": f"Automation Failure in {project_key}",
                    "description": {"type": "doc", "version": 1, "content": [{"type": "paragraph", "content": [{"type": "text", "text": report_body}]}]},
                    "issuetype": {"name": "Bug"},
                    "labels": ["automation_failure"]
                }
            }
            res = jira_request("POST", "issue", payload)
            if res.status_code == 201:
                new_key = res.json().get('key')
                save_cached_ticket(new_key) # Update cache with new key
                transition_issue(new_key, "progress")
                return f"Tests Failed ({failed}). Created {new_key} and set to IN PROGRESS. WORKFLOW COMPLETE."
            return f"Error creating ticket: {res.text}"
    else:
        # All passed
        if is_active_in_jira:
            transition_issue(existing_key, "done")
            return f"All tests passed! Closed existing ticket {existing_key}. WORKFLOW COMPLETE."
        return "All tests passed. No active tickets found. WORKFLOW COMPLETE."

# =================================
# Agent Setup
# =================================

class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

tools = [run_robot_automation]
tool_node = ToolNode(tools)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPEN_API_KEY).bind_tools(tools)

SYSTEM_PROMPT = SystemMessage(content=(
    "You are a QA Agent. Run tests with 'run_robot_automation'. "
    "The tool handles all Jira logic. Once 'WORKFLOW COMPLETE' is returned, "
    "summarize and stop. Do not repeat the run."
))

def call_model(state: State):
    return {"messages": [llm.invoke([SYSTEM_PROMPT] + state["messages"])]}

builder = StateGraph(State)
builder.add_node("agent", call_model)
builder.add_node("tools", tool_node)
builder.set_entry_point("agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")
graph = builder.compile()

def main():
    print("SMART-QA agent Online (Zero-Index Cache Mode).\n")
    while True:
        u = input("End-user: ").strip()
        if u.lower() in ["exit", "quit"]: break
        for event in graph.stream({"messages": [HumanMessage(content=u)]}, stream_mode="values"):
            if "messages" in event:
                m = event["messages"][-1]
                if m.type == "ai" and m.content: print(f"AI-Agent: {m.content}")
                elif m.type == "tool": print(f"\n[Tool Result]: {m.content}\n")

if __name__ == "__main__":
    main()