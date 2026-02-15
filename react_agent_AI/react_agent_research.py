from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_classic.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

# OpenAI API key
load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")


# Initialize LLM by taking the model of OpenAI
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3,
    max_tokens=600
)

# Research Tool
@tool
def research_tool(topic: str) -> str:
    """Generates well-structured research content on a given topic using an LLM."""
    
    research_prompt = f"""
You are a research assistant.

Generate well-researched and structured content on the topic below.

Topic: {topic}

Include:
- Overview
- Key concepts
- Everyday use of examples
- Real-world applications with examples and how it impacts
- Conclusion
"""
    response = llm.invoke(research_prompt)
    return response.content

# Tool list
# Invoking the research tool by mentioning here
tools = [research_tool]


# Agent Prompt for showing us the data
react_prompt = PromptTemplate.from_template(
    """
You are a Research Assistant Agent.

You must always use the research_tool to generate content.

Question: {input}
{agent_scratchpad}
"""
)

# Create Agent
agent = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    prompt=react_prompt
)

# Agent Executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)


# User Input for the research topic
topic = input("Enter research topic: ")

query = f"Conduct detailed research on the topic: {topic}"


# Execution of the logic for the Agent
result = agent_executor.invoke({"input": query})

print("\n========== STRUCTURED RESEARCH SUMMARY ==========\n")
print(result["output"])
