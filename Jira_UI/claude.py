import os
import subprocess
import xml.etree.ElementTree as ET
import requests
from datetime import datetime
from typing import Optional
from typing_extensions import TypedDict
from dotenv import load_dotenv
import json

from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

# ===============================
# ENV CONFIG
# ===============================
load_dotenv()

CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
anthropic_client = Anthropic(api_key=CLAUDE_API_KEY)

ROBOT_TEST_PATH = os.getenv("ROBOT_TEST_PATH")
ROBOT_OUTPUT_DIR = os.getenv("ROBOT_OUTPUT_DIR", "results")

JIRA_BASE_URL = os.getenv("JIRA_BASE_URL", "").rstrip("/")
JIRA_EMAIL = os.getenv("JIRA_EMAIL")
JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN")

CACHE_FILE = "jira_automation_cache.json"

# ===============================
# STRUCTURED STATE
# ===============================
class QAState(TypedDict):
    test_results: Optional[dict]
    ai_analysis: Optional[str]
    jira_status: Optional[str]

# ===============================
# JIRA HELPERS
# ===============================
def jira_request(method, endpoint, payload=None, params=None):
    url = f"{JIRA_BASE_URL}/rest/api/3/{endpoint}"
    auth = (JIRA_EMAIL, JIRA_API_TOKEN)
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json"
    }
    try:
        return requests.request(method, url, headers=headers, auth=auth, json=payload, params=params, timeout=20)
    except Exception as e:
        print(f"[Jira Error] {e}")
        return None

def is_issue_closed(issue_key):
    res = jira_request("GET", f"issue/{issue_key}?fields=status")
    if res and res.status_code == 200:
        status = res.json()["fields"]["status"]["name"].lower()
        return status in ["done", "closed", "resolved"]
    return False

def transition_to_done(issue_key):
    res = jira_request("GET", f"issue/{issue_key}/transitions")
    if res and res.status_code == 200:
        transitions = res.json().get("transitions", [])
        target = next((t for t in transitions if "done" in t["name"].lower()), None)
        if target:
            jira_request("POST", f"issue/{issue_key}/transitions", {"transition": {"id": target["id"]}})
            return True
    return False

# ===============================
# TestAgent: Runs Robot tests
# ===============================
def TestAgent(state: QAState) -> QAState:
    os.makedirs(ROBOT_OUTPUT_DIR, exist_ok=True)
    subprocess.run(["robot", "--outputdir", ROBOT_OUTPUT_DIR, ROBOT_TEST_PATH], capture_output=True)
    
    output_file = os.path.join(ROBOT_OUTPUT_DIR, "output.xml")
    if not os.path.exists(output_file):
        state["test_results"] = {"passed": 0, "failed": 0, "failures": []}
        return state

    tree = ET.parse(output_file)
    root = tree.getroot()
    stat = root.find(".//total/stat")
    passed = int(stat.attrib.get("pass", 0))
    failed = int(stat.attrib.get("fail", 0))
    failures = []
    for test in root.findall(".//test"):
        status = test.find("status")
        if status is not None and status.attrib.get("status") == "FAIL":
            failures.append(f"{test.attrib.get('name')} :: {status.text}")
    
    state["test_results"] = {"passed": passed, "failed": failed, "failures": failures}
    return state

# ===============================
# AnalysisAgent: AI analysis using Claude 4.x
# ===============================
def AnalysisAgent(state: QAState) -> QAState:
    if state["test_results"] and state["test_results"]["failed"] > 0 and not state.get("ai_analysis"):
        print("→ AnalysisAgent: Analyzing failed tests...")
        failure_text = "\n".join(state["test_results"]["failures"])
        prompt = f"Analyze the root cause of these failed Robot Framework tests:\n{failure_text}"
        resp = anthropic_client.completions.create(
            model="claude-4.1",  # Change to claude-4.3 or claude-4.1-100k if needed
            prompt=f"{HUMAN_PROMPT}{prompt}{AI_PROMPT}",
            max_tokens_to_sample=1000,
            temperature=0
        )
        state["ai_analysis"] = resp["completion"].strip()
    return state

# ===============================
# JiraAgent: handles Jira tickets
# ===============================
def JiraAgent(state: QAState) -> QAState:
    if not state["test_results"] or state["test_results"]["failed"] == 0:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, "r") as f:
                key = json.load(f).get("issue_key")
            if key:
                transition_to_done(key)
                os.remove(CACHE_FILE)
                state["jira_status"] = f"Closed ticket {key}"
                print(f"→ JiraAgent: Closed {key}")
        return state

    failed_tests_text = "\n".join([f"{idx+1}) {t}" for idx, t in enumerate(state["test_results"]["failures"])])

    if state.get("jira_status"):
        return state

    summary = f"Automation Failure - {datetime.now()}"
    existing_key = None
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            existing_key = json.load(f).get("issue_key")
    if existing_key and is_issue_closed(existing_key):
        existing_key = None
        os.remove(CACHE_FILE)

    if existing_key:
        comment_payload = {
            "body": {
                "type": "doc",
                "version": 1,
                "content": [{"type": "paragraph", "content": [{"type": "text", "text": f"AI Analysis:\n{state['ai_analysis']}"}]}]
            }
        }
        res = jira_request("POST", f"issue/{existing_key}/comment", comment_payload)
        if res and res.status_code == 201:
            state["jira_status"] = f"Updated existing ticket {existing_key} with AI analysis comment"
            print(f"→ JiraAgent: Updated {existing_key} with AI analysis comment")
        else:
            state["jira_status"] = f"Failed to update existing ticket {existing_key}"
    else:
        description_payload = {
            "type": "doc",
            "version": 1,
            "content": [{"type": "paragraph", "content": [{"type": "text", "text": failed_tests_text}]}]
        }
        payload = {
            "fields": {
                "project": {"key": "SCRUM"},
                "summary": summary,
                "description": description_payload,
                "issuetype": {"name": "Bug"}
            }
        }
        res = jira_request("POST", "issue", payload)
        if res and res.status_code == 201:
            new_key = res.json()["key"]
            with open(CACHE_FILE, "w") as f:
                json.dump({"issue_key": new_key}, f)
            comment_payload = {
                "body": {
                    "type": "doc",
                    "version": 1,
                    "content": [{"type": "paragraph", "content": [{"type": "text", "text": f"AI Analysis:\n{state['ai_analysis']}"}]}]
                }
            }
            jira_request("POST", f"issue/{new_key}/comment", comment_payload)
            state["jira_status"] = f"Created new ticket {new_key} with AI analysis comment"
            print(f"→ JiraAgent: Created {new_key} with AI analysis comment")
        else:
            state["jira_status"] = "Failed to create Jira ticket"
    return state

# ===============================
# Supervisor/Main Loop
# ===============================
def main():
    print("=== MULTI-AGENT QA ORCHESTRATOR READY ===")
    state: QAState = {"test_results": None, "ai_analysis": None, "jira_status": None}

    while True:
        user_input = input("USER: ").strip().lower()
        if user_input in ["exit", "quit"]:
            break

        if "run tests" in user_input:
            state = TestAgent(state)
            state = AnalysisAgent(state)
            state = JiraAgent(state)
        elif "failed test cases" in user_input:
            if state.get("test_results") and state["test_results"]["failed"] > 0:
                print("\n→ Failed Test Cases:")
                for idx, t in enumerate(state["test_results"]["failures"]):
                    print(f"{idx+1}) {t}")
                print()
            else:
                print("\n→ No failed test cases.\n")
            continue
        elif "why" in user_input:
            if state.get("ai_analysis"):
                print(f"\n→ AnalysisAgent: {state['ai_analysis']}\n")
            else:
                print("\n→ AnalysisAgent: No failures or analysis yet.\n")
            continue

        print("\n=== FINAL SUMMARY ===")
        if state.get("test_results"):
            print(f"Passed: {state['test_results']['passed']}")
            print(f"Failed: {state['test_results']['failed']}")
        print(f"Jira Status: {state.get('jira_status')}")
        print("===================================")

if __name__ == "__main__":
    main()