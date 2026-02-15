import os
from typing import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# Load the environment variables from the env file 
load_dotenv()

# Implementing the check here to see whether it is picking up the env file or
# raise a exception
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in .env file")

# Define LLM configuration 
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_completion_tokens=100
)

# Define shared state for the agents
class DocumentState(TypedDict):
    input_text:str
    extracted_info:str
    summary:str
    formatted_output:str

# Extractor agent
def extractor_agent(state: DocumentState):
    prompt = f"""
    You are an information extraction expert.

    Extract the key points, main ideas, and important facts 
    from the following document:

    {state['input_text']}
    """

    response = llm.invoke([HumanMessage(content=prompt)])
    state["extracted_info"] = response.content
    return state

# Summarizer agent
def summarizer_agent(state :DocumentState):
    prompt = f"""
    You are an professional summarizer.

    Create a concise and clear summary from the
    following extracted information:

    {state['extracted_info']}
    """

    response = llm.invoke([HumanMessage(content=prompt)])
    state["summary"] = response.content
    return state

# Formatter agent
def formatter_agent(state:DocumentState):
    prompt = f"""
    Format the following summary into structured JSON format.

    Summary:
    {state['summary']}

    Return ONLY valid JSON in this format:

    {{
        "title": "Generated Title",
        "summary": "Summary text",
        "key_takeaways": ["Point 1", "Point 2", "Point 3"]
    }}
    """

    response = llm.invoke([HumanMessage(content=prompt)])
    state["formatted_output"] = response.content
    return state

# Building the Langgraph flow for the above agents mentioned
workflow = StateGraph(DocumentState)

workflow.add_node("extractor", extractor_agent)
workflow.add_node("summarizer", summarizer_agent)
workflow.add_node("formatter", formatter_agent)

workflow.set_entry_point("extractor")
workflow.add_edge("extractor", "summarizer")
workflow.add_edge("summarizer", "formatter")
workflow.add_edge("formatter", END)

app = workflow.compile()

# Execution of the logic for the agents to be behaving in sequential flow logic
if __name__ == "__main__":

    print("Enter your document (press Enter twice to finish):\n")

    lines = []
    while True:
        line = input()
        if line == "":
            break
        lines.append(line)

    input_document = "\n".join(lines)

    initial_state = {
        "input_text": input_document,
        "extracted_info": "",
        "summary": "",
        "formatted_output": ""
    }

    result = app.invoke(initial_state)

    print("\nFinal Structured Output:\n")
    print(result["formatted_output"])



