import os
import json
import requests
import subprocess
import xml.etree.ElementTree as ET
import uuid
from datetime import datetime
from typing import Annotated, List, Optional
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

# =========================
# Configuration & Env
# =========================
load_dotenv()

JIRA_BASE_URL = os.getenv("JIRA_BASE_URL", "").rstrip('/')
JIRA_EMAIL = os.getenv("JIRA_EMAIL")
JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN")
OPEN_API_KEY = os.getenv("OPEN_API_KEY")
ROBOT_TEST_PATH = os.getenv("ROBOT_TEST_PATH")
ROBOT_OUTPUT_DIR = os.getenv("ROBOT_OUTPUT_DIR", "results")

CACHE_FILE = "jira_automation_cache.json"

# =================================
# Jira & Analysis Helpers
# =================================

def jira_request(method, endpoint, payload=None, params=None):
    url = f"{JIRA_BASE_URL}/rest/api/3/{endpoint.lstrip('/')}"
    auth = (JIRA_EMAIL, JIRA_API_TOKEN)
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    try:
        res = requests.request(method, url, headers=headers, auth=auth, json=payload, params=params, timeout=20)
        return res
    except Exception as e:
        print(f"[DEBUG] Jira Connection Error: {e}")
        return None

def is_issue_closed(issue_key):
    """Checks if the issue is already in a terminal status like 'Done' or 'Closed'."""
    res = jira_request("GET", f"issue/{issue_key}?fields=status")
    if res and res.status_code == 200:
        status_name = res.json().get("fields", {}).get("status", {}).get("name", "").lower()
        # If the status is any of these, we consider it 'Closed'
        if status_name in ["done", "closed", "resolved"]:
            return True
    return False

def transition_to_done(issue_key):
    res = jira_request("GET", f"issue/{issue_key}/transitions")
    if not res or res.status_code != 200: return False
    transitions = res.json().get("transitions", [])
    target = next((t for t in transitions if "done" in t["name"].lower()), None)
    if target:
        payload = {"transition": {"id": target["id"]}, "fields": {"resolution": {"name": "Done"}}}
        jira_request("POST", f"issue/{issue_key}/transitions", payload)
        return True
    return False

def analyze_failures_with_ai(failure_text):
    analysis_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPEN_API_KEY)
    prompt = f"Analyze these Robot Framework failures and explain the root cause concisely:\n\n{failure_text}"
    response = analysis_llm.invoke([HumanMessage(content=prompt)])
    return response.content

# ==============================
# Tools
# ==============================

@tool
def run_robot_automation(project_key: str = "SCRUM"):
    """Runs Robot tests, analyzes failures, and creates/updates Jira bugs smartly."""
    os.makedirs(ROBOT_OUTPUT_DIR, exist_ok=True)
    subprocess.run(["robot", "--outputdir", ROBOT_OUTPUT_DIR, ROBOT_TEST_PATH], capture_output=True)

    output_file = os.path.join(ROBOT_OUTPUT_DIR, "output.xml")
    if not os.path.exists(output_file): return "Error: output.xml not found."

    tree = ET.parse(output_file)
    root = tree.getroot()
    stat = root.find(".//total/stat")
    passed, failed = int(stat.attrib.get("pass", 0)), int(stat.attrib.get("fail", 0))

    failures = []
    for test in root.findall(".//test"):
        status = test.find("status")
        if status is not None and status.attrib.get("status") == "FAIL":
            failures.append(f"Test: {test.attrib.get('name')} | Error: {status.text}")
    
    raw_failure_text = "\n".join(failures)
    ai_explanation = analyze_failures_with_ai(raw_failure_text) if failed > 0 else "All passed."
    
    # Check cache
    existing_key = None
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f: 
            existing_key = json.load(f).get("issue_key")

    # SMART CHECK: If ticket exists but is ALREADY DONE in Jira, treat it as None
    if existing_key and is_issue_closed(existing_key):
        print(f"[INFO] {existing_key} is already Done in Jira. Creating a new ticket.")
        existing_key = None
        if os.path.exists(CACHE_FILE): os.remove(CACHE_FILE)

    jira_info = ""
    if failed > 0:
        description = f"AI ANALYSIS:\n{ai_explanation}\n\nRAW ERRORS:\n{raw_failure_text}"
        if existing_key:
            # Update existing
            payload = {"body": {"type": "doc", "version": 1, "content": [{"type": "paragraph", "content": [{"type": "text", "text": description}]}]}}
            jira_request("POST", f"issue/{existing_key}/comment", payload)
            jira_info = f"Updated existing Bug: {existing_key}"
        else:
            # Create new
            payload = {
                "fields": {
                    "project": {"key": project_key},
                    "summary": f"Automation Failure - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    "description": {"type": "doc", "version": 1, "content": [{"type": "paragraph", "content": [{"type": "text", "text": description}]}]},
                    "issuetype": {"name": "Bug"}
                }
            }
            res = jira_request("POST", "issue", payload)
            if res and res.status_code == 201:
                new_key = res.json().get("key")
                with open(CACHE_FILE, "w") as f: json.dump({"issue_key": new_key}, f)
                jira_info = f"Created new Bug: {new_key}"
    else:
        # Success logic
        if existing_key:
            transition_to_done(existing_key)
            if os.path.exists(CACHE_FILE): os.remove(CACHE_FILE)
            jira_info = f"Tests Passed! Closed ticket {existing_key}."
        else:
            jira_info = "Tests Passed. No open bugs to close."

    return f"RESULTS: {passed} Passed, {failed} Failed.\nJIRA: {jira_info}\nAI ANALYSIS: {ai_explanation}"

@tool
def bulk_close_all_bugs(project_key: str = "SCRUM"):
    """Search for all open Bugs in the project and move them to Done."""
    params = {"jql": f'project = "{project_key}" AND issuetype = Bug AND status != "Done"'}
    res = jira_request("GET", "search/jql", params=params)
    if not res or res.status_code != 200: return "Error retrieving bugs."
    issues = res.json().get("issues", [])
    closed_keys = []
    for issue in issues:
        key = issue.get("key")
        if key and transition_to_done(key): closed_keys.append(key)
    if os.path.exists(CACHE_FILE): os.remove(CACHE_FILE)
    return f"Bulk Action: Closed {len(closed_keys)} bugs."

# =================================
# Agent Setup (Main execution logic)
# =================================

class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

tools = [run_robot_automation, bulk_close_all_bugs]
tool_node = ToolNode(tools)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPEN_API_KEY).bind_tools(tools)

def call_model(state: State):
    prompt = SystemMessage(content="Senior QA Expert. You create new Jira bugs if the old ones are closed. You check status before updating.")
    return {"messages": [llm.invoke([prompt] + state["messages"])]}

builder = StateGraph(State)
builder.add_node("agent", call_model); builder.add_node("tools", tool_node)
builder.set_entry_point("agent"); builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")
graph = builder.compile(checkpointer=MemorySaver())

def main():
    session_id = str(uuid.uuid4())
    print("--- SMART-QA AGENT READY ---")
    while True:
        user_input = input("USER: ")
        if user_input.lower() in ["exit", "quit"]: break
        config = {"configurable": {"thread_id": session_id}}
        for event in graph.stream({"messages": [HumanMessage(content=user_input)]}, config, stream_mode="values"):
            if "messages" in event:
                msg = event["messages"][-1]
                if msg.type == "ai" and msg.content: print(f"\n[AI]: {msg.content}\n")

if __name__ == "__main__":
    main()