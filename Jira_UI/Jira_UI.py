import os
import json
import requests
import subprocess
import xml.etree.ElementTree as ET
from typing import Annotated, List
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from dotenv import load_dotenv

# =========================
# Load environment variables
# =========================
load_dotenv()

JIRA_BASE_URL = os.getenv("JIRA_BASE_URL").rstrip('/')
JIRA_EMAIL = os.getenv("JIRA_EMAIL")
JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN")
OPEN_API_KEY = os.getenv("OPEN_API_KEY")
ROBOT_TEST_PATH = os.getenv("ROBOT_TEST_PATH", "tests/")
ROBOT_OUTPUT_DIR = os.getenv("ROBOT_OUTPUT_DIR", "results")

# =================================
# Jira Helper
# =================================
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
# Define Tools (FIXED WITH DOCSTRINGS)
# ==============================

@tool
def search_jira_issues(jql: str):
    """
    Search for Jira issues using JQL.
    JQL MUST include a project restriction (example: project = 'SCRUM').
    Returns issue key, type, summary, and status.
    """
    params = {"jql": jql, "fields": "summary,status,issuetype", "maxResults": 20}
    response = jira_request("GET", "search/jql", params=params)

    if response.status_code != 200:
        return f"Jira API Error: {response.status_code} - {response.text}"

    issues = response.json().get("issues", [])
    if not issues:
        return f"No issues found for: {jql}"

    output = ["Results:"]
    for i in issues:
        output.append(
            f"- {i['key']}: "
            f"[{i['fields']['issuetype']['name']}] "
            f"{i['fields']['summary']} "
            f"({i['fields']['status']['name']})"
        )

    return "\n".join(output)


@tool
def create_jira_ticket(project_key: str, summary: str, description: str, issue_type: str = "Bug"):
    """
    Create a new Jira ticket.
    Issue type can be Bug, Story, or Epic.
    Returns confirmation with created ticket key.
    """
    payload = {
        "fields": {
            "project": {"key": project_key},
            "summary": summary,
            "description": {
                "type": "doc",
                "version": 1,
                "content": [{
                    "type": "paragraph",
                    "content": [{"type": "text", "text": description}]
                }]
            },
            "issuetype": {"name": issue_type}
        }
    }

    res = jira_request("POST", "issue", payload)

    if res.status_code == 201:
        return f"Success: Created {res.json().get('key')}"
    else:
        return f"Error: {res.text}"


@tool
def close_jira_ticket(issue_key: str):
    """
    Close a Jira ticket by automatically detecting a Done/Closed transition.
    """
    res = jira_request("GET", f"issue/{issue_key}/transitions")

    if res.status_code != 200:
        return f"Error fetching transitions: {res.text}"

    transitions = res.json().get("transitions", [])

    target = next(
        (t for t in transitions if any(
            k in t["name"].lower() for k in ["done", "close", "complete"]
        )),
        None
    )

    if not target:
        return "No valid 'Close' transition found for this ticket."

    move_res = jira_request(
        "POST",
        f"issue/{issue_key}/transitions",
        {"transition": {"id": target["id"]}}
    )

    if move_res.status_code == 204:
        return f"Ticket {issue_key} successfully closed."
    else:
        return f"Failed to close: {move_res.text}"


@tool
def run_robot_automation(project_key: str = "SCRUM"):
    """
    Execute Robot Framework tests.
    If tests fail, automatically create a Jira Bug.
    Returns test execution summary.
    """
    os.makedirs(ROBOT_OUTPUT_DIR, exist_ok=True)

    subprocess.run(
        ["robot", "--outputdir", ROBOT_OUTPUT_DIR, ROBOT_TEST_PATH],
        capture_output=True
    )

    output_file = os.path.join(ROBOT_OUTPUT_DIR, "output.xml")

    if not os.path.exists(output_file):
        return "Error: output.xml not found."

    tree = ET.parse(output_file)
    root = tree.getroot()
    stat = root.find(".//total/stat")

    passed = int(stat.attrib.get("pass", 0))
    failed = int(stat.attrib.get("fail", 0))

    report = f"Robot Results - Passed: {passed}, Failed: {failed}."

    if failed > 0:
        payload = {
            "fields": {
                "project": {"key": project_key},
                "summary": f"Automation Failure: {failed} tests failed",
                "description": {
                    "type": "doc",
                    "version": 1,
                    "content": [{
                        "type": "paragraph",
                        "content": [{"type": "text", "text": report}]
                    }]
                },
                "issuetype": {"name": "Bug"}
            }
        }

        create_res = jira_request("POST", "issue", payload)
        if create_res.status_code == 201:
            key = create_res.json().get('key')
            report += f" Created Bug: {key}."

    return report


# =================================
# Agent Configuration
# =================================

tools = [search_jira_issues, create_jira_ticket, close_jira_ticket, run_robot_automation]
tool_node = ToolNode(tools)

SYSTEM_PROMPT = SystemMessage(content=(
    "You are a QA Automation Agent. You possess tools for Jira and Robot Framework. "
    "1. For 'list/show' requests, use 'search_jira_issues' and ALWAYS include 'project = ...' in JQL. "
    "2. For 'run tests', use 'run_robot_automation'. "
    "3. For 'close', use 'close_jira_ticket'. "
    "Execute the tool directly."
))

class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPEN_API_KEY)
llm_with_tools = llm.bind_tools(tools)

def call_model(state: State):
    return {"messages": [llm_with_tools.invoke([SYSTEM_PROMPT] + state["messages"])]}

builder = StateGraph(State)
builder.add_node("agent", call_model)
builder.add_node("tools", tool_node)
builder.set_entry_point("agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")

graph = builder.compile()


# =================================
# UI-READY FUNCTION
# =================================

def run_agent(user_input: str):
    responses = []

    for event in graph.stream(
        {"messages": [HumanMessage(content=user_input)]},
        stream_mode="values"
    ):
        if "messages" in event:
            last_msg = event["messages"][-1]
            if last_msg.type == "ai" and last_msg.content:
                responses.append(("ai", last_msg.content))
            elif last_msg.type == "tool":
                responses.append(("tool", last_msg.content))

    return responses


# =================================
# Optional CLI
# =================================

def main():
    print("SMART-QA agent Online. Type 'exit' to quit.\n")
    while True:
        user_input = input("End-user: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            break

        responses = run_agent(user_input)

        for role, message in responses:
            if role == "ai":
                print(f"AI-Agent: {message}")
            elif role == "tool":
                print(f"\n[Tool Result]: {message}\n")


if __name__ == "__main__":
    main()