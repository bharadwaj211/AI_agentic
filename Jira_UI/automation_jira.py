import os
import json
import requests
import subprocess
import xml.etree.ElementTree as ET
import uuid
from datetime import datetime
from typing import Annotated, List
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
HISTORY_LOG = "automation_history.log"

# =================================
# Helper Functions
# =================================

def log_history(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(HISTORY_LOG, "a") as f:
        f.write(f"[{timestamp}] {message}\n")

def get_cached_ticket():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            try: return json.load(f).get("issue_key")
            except: return None
    return None

def save_cached_ticket(issue_key):
    with open(CACHE_FILE, "w") as f:
        json.dump({"issue_key": issue_key}, f)

def jira_request(method, endpoint, payload=None):
    url = f"{JIRA_BASE_URL}/rest/api/3/{endpoint.lstrip('/')}"
    auth = (JIRA_EMAIL, JIRA_API_TOKEN)
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    return requests.request(method, url, headers=headers, auth=auth, json=payload, timeout=20)

def transition_issue(issue_key, target_name):
    res = jira_request("GET", f"issue/{issue_key}/transitions")
    if res.status_code != 200: return False
    transitions = res.json().get("transitions", [])
    target = next((t for t in transitions if target_name.lower() in t["name"].lower()), None)
    if target:
        jira_request("POST", f"issue/{issue_key}/transitions", {"transition": {"id": target["id"]}})
        return True
    return False

def find_all_failures(element, failures_list):
    if element is None: return
    for test in element.findall("test"):
        status = test.find("status")
        if status is not None and status.attrib.get("status") == "FAIL":
            failures_list.append({"name": test.attrib.get("name"), "message": status.text})
    for suite in element.findall("suite"):
        find_all_failures(suite, failures_list)

def analyze_failures_with_ai(failure_text):
    analysis_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPEN_API_KEY)
    prompt = f"Analyze these Robot Framework failures and explain the root cause concisely:\n\n{failure_text}"
    response = analysis_llm.invoke([HumanMessage(content=prompt)])
    return response.content

# ==============================
# Tools
# ==============================

@tool
def clear_session_command():
    """Resets the AI's internal state so it can run tests again."""
    if os.path.exists(HISTORY_LOG):
        os.remove(HISTORY_LOG)
    return "Memory reset. I am ready for a fresh command."

@tool
def run_robot_automation(project_key: str = "SCRUM"):
    """Runs Robot Framework tests, updates Jira tickets, and performs initial failure analysis."""
    os.makedirs(ROBOT_OUTPUT_DIR, exist_ok=True)
    log_history("AI Agent running: Initiating automation...")

    subprocess.run(["robot", "--outputdir", ROBOT_OUTPUT_DIR, ROBOT_TEST_PATH], capture_output=True)

    output_file = os.path.join(ROBOT_OUTPUT_DIR, "output.xml")
    if not os.path.exists(output_file):
        return "Error: output.xml not found."

    tree = ET.parse(output_file)
    root = tree.getroot()
    stat = root.find(".//total/stat")
    passed, failed = int(stat.attrib.get("pass", 0)), int(stat.attrib.get("fail", 0))

    failures = []
    find_all_failures(root.find("suite"), failures)
    raw_failure_text = "\n".join([f"Test: {f['name']} | Error: {f['message']}" for f in failures])
    
    ai_explanation = analyze_failures_with_ai(raw_failure_text) if failed > 0 else "All tests passed."
    existing_key = get_cached_ticket()
    jira_status = "No update needed."

    if failed > 0:
        comment_body = f"AI ANALYSIS: {ai_explanation}\n\nRAW FAILURES:\n{raw_failure_text}"
        if existing_key:
            jira_request("POST", f"issue/{existing_key}/comment", {"body": {"type": "doc", "version": 1, "content": [{"type": "paragraph", "content": [{"type": "text", "text": comment_body}]}]}})
            jira_status = f"Updated existing Jira ticket: {existing_key}"
        else:
            payload = {"fields": {"project": {"key": project_key}, "summary": f"Automation Failure: {project_key}", "description": {"type": "doc", "version": 1, "content": [{"type": "paragraph", "content": [{"type": "text", "text": comment_body}]}]}, "issuetype": {"name": "Bug"}, "labels": ["automation_failure"]}}
            res = jira_request("POST", "issue", payload)
            if res.status_code == 201:
                existing_key = res.json().get("key")
                save_cached_ticket(existing_key)
                transition_issue(existing_key, "progress")
                jira_status = f"Created new Jira Bug: {existing_key}"
    else:
        if existing_key:
            transition_issue(existing_key, "done")
            jira_status = f"Tests Passed. Closed ticket: {existing_key}"
            if os.path.exists(CACHE_FILE): os.remove(CACHE_FILE)

    return f"RESULTS: {passed} Passed, {failed} Failed.\nJIRA: {jira_status}\nRAW FAILURE DATA: {raw_failure_text}\nAI SUMMARY: {ai_explanation}"

# =================================
# Agent Setup
# =================================

class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

tools = [run_robot_automation, clear_session_command]
tool_node = ToolNode(tools)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPEN_API_KEY).bind_tools(tools)

# Updated System Prompt for Conversational Troubleshooting
SYSTEM_PROMPT = SystemMessage(content=(
    "You are a Senior QA Automation Expert. "
    "1. Use 'run_robot_automation' ONLY when the user explicitly asks to run tests. "
    "2. If tests have already been run in this session, use the 'RAW FAILURE DATA' from the chat history "
    "to answer follow-up questions like 'why did it fail?' or 'how do I fix this?'. "
    "3. DO NOT re-run the automation if the failure data is already visible in the history. "
    "4. Provide detailed technical solutions, code suggestions, and troubleshooting steps. "
    "5. Always report the Jira ticket status clearly."
))

def call_model(state: State):
    return {"messages": [llm.invoke([SYSTEM_PROMPT] + state["messages"])]}

memory = MemorySaver()
builder = StateGraph(State)
builder.add_node("agent", call_model)
builder.add_node("tools", tool_node)
builder.set_entry_point("agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")
graph = builder.compile(checkpointer=memory)

# =================================
# Main Execution Loop
# =================================

def main():
    session_id = str(uuid.uuid4())
    print(f"SMART-QA agent active. [Session ID: {session_id[:8]}]\n")
    
    while True:
        user_input = input("USER: ")
        
        if user_input.lower() in ["exit", "quit"]:
            break

        if user_input.lower() == "clear session":
            session_id = str(uuid.uuid4())
            print(f"\n[AI]: Session cleared. (New Session: {session_id[:8]})\n")
            continue

        config = {"configurable": {"thread_id": session_id}}
        
        # We stream the values so the AI can see its own tool outputs in the message history
        events = graph.stream({"messages": [HumanMessage(content=user_input)]}, config, stream_mode="values")
        
        for event in events:
            if "messages" in event:
                msg = event["messages"][-1]
                # Only print the final AI response to the user for a clean conversation
                if msg.type == "ai" and msg.content:
                    print(f"\n[AI]: {msg.content}\n")
                elif msg.type == "tool":
                    print(f"\n[SYSTEM]: Processing tool results...\n")

if __name__ == "__main__":
    main()