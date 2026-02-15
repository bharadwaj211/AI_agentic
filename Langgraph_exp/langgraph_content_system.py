import os
from typing import TypedDict, List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage

# Load the variables from the .env file
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in .env file")

# Define LLM configuration 
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_completion_tokens=100
)

# Defining the shared states
class ReviewState(TypedDict):
    topic: str
    content: str
    feedback: str
    review_status: str
    review_count: int
    content_versions: List[str]

# Node 1: Writer agent
def writer_agent(state: ReviewState):
    print("\n Writer Agent generating content...\n")

    prompt = f"""
    Write a well-structured article about:
    {state['topic']}

    Ensure clarity, proper grammar, and completeness.
    """

    response = llm.invoke([HumanMessage(content=prompt)])

    state["content"] = response.content
    state["content_versions"].append(response.content)
    state["review_count"] = 0

    return state

# Node 2: Reviewer agent
def reviewer_agent(state: ReviewState):
    print(f"\n🔍 Reviewer Agent reviewing... (Attempt {state['review_count'] + 1})\n")

    prompt = f"""
    Evaluate the following content for:
    - Grammar
    - Clarity
    - Completeness
    - Structure

    If content is good, respond EXACTLY with:
    PASS

    If content needs improvement, respond in this format:
    FAIL: <clear feedback>

    Content:
    {state['content']}
    """

    response = llm.invoke([HumanMessage(content=prompt)])
    review_result = response.content.strip()

    state["review_count"] += 1

    if review_result.startswith("PASS"):
        state["review_status"] = "pass"
        state["feedback"] = ""
        print(" Review Passed")
    else:
        state["review_status"] = "fail"
        state["feedback"] = review_result
        print(" Review Failed - Sending to Editor")

    return state

# Node 3: Editor agent 
def editor_agent(state: ReviewState):
    print("\n Editor Agent improving content...\n")

    prompt = f"""
    Improve the following content based on reviewer feedback.

    Feedback:
    {state['feedback']}

    Content:
    {state['content']}

    Return improved version only.
    """

    response = llm.invoke([HumanMessage(content=prompt)])

    state["content"] = response.content
    state["content_versions"].append(response.content)

    return state

# Conditional Logic decision 
def review_decision(state: ReviewState):

    # If review passed → End
    if state["review_status"] == "pass":
        return "end"

    # If max 3 attempts reached → End
    if state["review_count"] >= 3:
        print("\n Maximum review attempts reached.")
        return "end"

    # Otherwise → Go to Editor
    return "edit"

# Building the langgraph flow
workflow = StateGraph(ReviewState)

workflow.add_node("writer", writer_agent)
workflow.add_node("reviewer", reviewer_agent)
workflow.add_node("editor", editor_agent)

workflow.set_entry_point("writer")

workflow.add_edge("writer", "reviewer")

workflow.add_conditional_edges(
    "reviewer",
    review_decision,
    {
        "edit": "editor",
        "end": END
    }
)

workflow.add_edge("editor", "reviewer")

app = workflow.compile()

# Execution of the main logic
if __name__ == "__main__":

    print("===================================")
    print("   Content Review System Started   ")
    print("===================================\n")

    # Input AFTER execution begins
    user_topic = input(" Enter a topic for content generation: ")

    initial_state = {
        "topic": user_topic,
        "content": "",
        "feedback": "",
        "review_status": "",
        "review_count": 0,
        "content_versions": []
    }

    print("\n Starting workflow...\n")

    result = app.invoke(initial_state)

    print("\n===================================")
    print("           FINAL OUTPUT            ")
    print("===================================\n")

    print(result["content"])

    print("\nReview Count:", result["review_count"])
    print("Total Versions Generated:", len(result["content_versions"]))



