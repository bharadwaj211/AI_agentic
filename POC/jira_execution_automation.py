import os
import json
import requests
import subprocess
import xml.etree.ElementTree as ET
from typing import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
import re

# -------------------------
# Load environment variables
# -------------------------
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=dotenv_path)

JIRA_BASE_URL = os.getenv("JIRA_BASE_URL")
JIRA_EMAIL = os.getenv("JIRA_EMAIL")
JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN")
OPEN_API_KEY = os.getenv("OPEN_API_KEY")
ROBOT_TEST_PATH = os.getenv("ROBOT_TEST_PATH", "tests/")
ROBOT_OUTPUT_DIR = os.getenv("ROBOT_OUTPUT_DIR", "results")

for var, val in [
    ("JIRA_BASE_URL", JIRA_BASE_URL),
    ("JIRA_EMAIL", JIRA_EMAIL),
    ("JIRA_API_TOKEN", JIRA_API_TOKEN),
    ("OPEN_API_KEY", OPEN_API_KEY)
]:
    if not val:
        raise ValueError(f"{var} not set in .env")

print("Environment variables loaded successfully!")

# -------------------------
# LLM Configuration
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
    intent: str
    jql: str
    issue_key: str
    response_message: str

# -------------------------
# Jira Utility
# -------------------------
def jira_request(method, endpoint, payload=None, params=None):
    url = f"{JIRA_BASE_URL}/rest/api/3/{endpoint}"
    headers = {"Accept": "application/json"}
    if method.upper() == "GET":
        return requests.get(url, headers=headers, params=params, auth=(JIRA_EMAIL, JIRA_API_TOKEN))
    elif method.upper() == "POST":
        headers["Content-Type"] = "application/json"
        return requests.post(url, headers=headers, auth=(JIRA_EMAIL, JIRA_API_TOKEN), data=json.dumps(payload))
    else:
        raise ValueError("Unsupported Jira method")

# -------------------------
# Jira: Move issue to In Progress
# -------------------------
def move_issue_to_in_progress(issue_key: str):
    response = jira_request("GET", f"issue/{issue_key}/transitions")
    if response.status_code != 200:
        print(f"[Move Issue] Failed to fetch transitions for {issue_key}: {response.text}")
        return False

    transitions = response.json().get("transitions", [])
    in_progress_transition = next(
        (t for t in transitions if t["to"]["name"].lower() == "in progress"), None
    )
    if not in_progress_transition:
        print(f"[Move Issue] No 'In Progress' transition found for {issue_key}")
        return False

    payload = {"transition": {"id": in_progress_transition["id"]}}
    post_resp = jira_request("POST", f"issue/{issue_key}/transitions", payload)
    if post_resp.status_code in [204, 200]:
        print(f"[Move Issue] {issue_key} moved to In Progress")
        return True
    else:
        print(f"[Move Issue] Failed to move {issue_key}: {post_resp.text}")
        return False

# -------------------------
# Agent: Intent Detection
# -------------------------
def detect_intent(state: AgentState) -> AgentState:
    prompt = f"""
Classify the intent of this request:

{state['user_input']}

Return ONLY one word from:
SEARCH
CREATE
RUN_AUTOMATION
"""
    intent = llm.invoke(prompt).content.strip()
    return {**state, "intent": intent}

# -------------------------
# Agent: Extract Jira Issue Key
# -------------------------
def extract_issue_key(state: AgentState) -> AgentState:
    prompt = f"""
Extract Jira issue key from:
{state['user_input']}

Return only key like ABC-123.
If none found return NONE.
"""
    key = llm.invoke(prompt).content.strip()
    if key == "NONE":
        return state
    return {**state, "issue_key": key}

# -------------------------
# Agent: Generate JQL
# -------------------------
def generate_jql(state: AgentState) -> AgentState:
    if state["intent"] != "SEARCH":
        return state

    project_match = re.search(r'project\s+(\w+)', state['user_input'], re.IGNORECASE)
    project = project_match.group(1) if project_match else "SCRUM"

    issue_types = []
    if re.search(r'\bbugs?\b', state['user_input'], re.IGNORECASE):
        issue_types.append("Bug")
    if re.search(r'\bstories?\b', state['user_input'], re.IGNORECASE):
        issue_types.append("Story")
    if re.search(r'\bepics?\b', state['user_input'], re.IGNORECASE):
        issue_types.append("Epic")

    if not issue_types:
        issue_types = ["Bug", "Story", "Epic"]

    issue_types_quoted = [f'"{it}"' for it in issue_types]

    jql = f'project = "{project}" AND issuetype IN ({", ".join(issue_types_quoted)})'
    if re.search(r'this month', state['user_input'], re.IGNORECASE):
        jql += " AND created >= startOfMonth() AND created <= endOfMonth()"

    print(f"[JQL Agent] Generated JQL: {jql}")
    return {**state, "jql": jql}

# -------------------------
# Agent: Search Jira Issues
# -------------------------
def search_issues(state: AgentState) -> AgentState:
    if state["intent"] != "SEARCH":
        return state

    response = jira_request("GET", "search/jql", params={"jql": state["jql"], "maxResults": 100, "fields": "summary,status,issuetype"})
    if response.status_code != 200:
        raise Exception(f"Jira API Error: {response.status_code} - {response.text}")
    
    data = response.json()
    issues = data.get("issues", [])
    bug_count = story_count = epic_count = 0
    details = []

    for issue in issues:
        key = issue.get("key", "Unknown")
        fields = issue.get("fields", {})
        summary = fields.get("summary", "No Summary")
        status = fields.get("status", {}).get("name", "Unknown")
        itype = fields.get("issuetype", {}).get("name", "Unknown")
        if itype.lower() == "bug":
            bug_count += 1
        elif itype.lower() == "story":
            story_count += 1
        elif itype.lower() == "epic":
            epic_count += 1
        details.append(f"{key} | {itype} | {status} | {summary}")

    summary_message = f"""
Jira Search Summary
-------------------
Total Issues: {len(issues)}
Bugs: {bug_count}
Stories: {story_count}
Epics: {epic_count}

Issue Details:
--------------
"""
    summary_message += "\n".join(details) if details else "No issues found."
    return {**state, "response_message": summary_message}

# -------------------------
# Agent: Create Jira Issue
# -------------------------
created_issues_log = set()
def create_issue(state: AgentState, summary: str, description: str) -> str:
    if summary in created_issues_log:
        return None
    payload = {
        "fields": {
            "project": {"key": "SCRUM"},
            "summary": summary,
            "description": {"type": "doc","version": 1,"content":[{"type":"paragraph","content":[{"type":"text","text": description}]}]},
            "issuetype": {"name": "Bug"}
        }
    }
    response = jira_request("POST", "issue", payload)
    if response.status_code != 201:
        print(f"[Create Issue Agent] Failed: {response.text}")
        return None
    key = response.json().get("key")
    created_issues_log.add(summary)
    return key

# -------------------------
# Agent: Run Robot Framework Tests
# -------------------------
def run_robot(state: AgentState) -> AgentState:
    if state["intent"] != "RUN_AUTOMATION":
        return state
    os.makedirs(ROBOT_OUTPUT_DIR, exist_ok=True)
    result = subprocess.run(["robot", "--outputdir", ROBOT_OUTPUT_DIR, ROBOT_TEST_PATH])
    output_file = os.path.join(ROBOT_OUTPUT_DIR, "output.xml")
    if not os.path.exists(output_file):
        return {**state, "response_message": "No Robot output found."}

    tree = ET.parse(output_file)
    root = tree.getroot()
    total_stat = root.find(".//total/stat")
    passed = int(total_stat.attrib.get("pass", 0))
    failed = int(total_stat.attrib.get("fail", 0))
    summary = f"Automation Result:\nTotal: {passed+failed}\nPassed: {passed}\nFailed: {failed}"

    if failed > 0:
        issue_key = create_issue(state, f"{failed} Robot Test Failures", summary)
        if issue_key:
            summary += f"\nCreated Jira Ticket: {issue_key}"
            moved = move_issue_to_in_progress(issue_key)
            if moved:
                summary += " (Moved to In Progress)"

    return {**state, "response_message": summary}

# -------------------------
# Build Agent Graph
# -------------------------
builder = StateGraph(AgentState)
builder.add_node("detect_intent", detect_intent)
builder.add_node("extract_issue_key", extract_issue_key)
builder.add_node("generate_jql", generate_jql)
builder.add_node("search_issues", search_issues)
builder.add_node("run_robot", run_robot)
builder.set_entry_point("detect_intent")
builder.add_edge("detect_intent", "extract_issue_key")
builder.add_edge("extract_issue_key", "generate_jql")
builder.add_edge("generate_jql", "search_issues")
builder.add_edge("search_issues", "run_robot")
builder.add_edge("run_robot", END)
graph = builder.compile()

# -------------------------
# Tool/Agent Runner (User-based)
# -------------------------
def run_user_task(user_input: str):
    state = graph.invoke({
        "user_input": user_input,
        "intent": "",
        "jql": "",
        "issue_key": "",
        "response_message": ""
    })
    print(state.get("response_message", ""))

# -------------------------
# Main: User Interaction
# -------------------------
def main():
    print("AI Jira + Robot Agent (User-driven)\n")
    while True:
        user_input = input("Enter your task (or 'exit'): ").strip()
        if user_input.lower() in ["exit", "quit"]:
            break
        run_user_task(user_input)

if __name__ == "__main__":
    main()
