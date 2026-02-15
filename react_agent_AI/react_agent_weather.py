from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_classic.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os


# OpenAI API key
load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")

# Hardcoded weather data for the cities that show up on the execution
Weather_database = {
    "Delhi": {"temperature": 15, "condition": "cloudy", "humidity": 70, "Wind_Speed": 18, "AQI": "Severe"},
    "Mumbai": {"temperature": 35, "condition": "Sunny", "humidity": 50, "Wind_Speed": 20, "AQI": "Satisfactory"},
    "Jaipur": {"temperature": 25, "condition": "Dry", "humidity": 60, "Wind_Speed": 35, "AQI": "Good"},
    "Ahmedabad": {"temperature": 18, "condition": "clear", "humidity": 30, "Wind_Speed": 14, "AQI": "Poor"},
    "Hyderabad": {"temperature": 28, "condition": "Hot", "humidity": 20, "Wind_Speed": 32, "AQI": "Moderate"},
    "Nagpur": {"temperature": 20, "condition": "breezy", "humidity": 70, "Wind_Speed": 40, "AQI": "Moderate"},
    "Kochi": {"temperature": 30, "condition": "rainy", "humidity": 45, "Wind_Speed": 70, "AQI": "Satisfactory"},
    "Goa": {"temperature": 32, "condition": "clear", "humidity": 30, "Wind_Speed": 45, "AQI": "Good"},
    "Lucknow": {"temperature": 40, "condition": "Sunny", "humidity": 40, "Wind_Speed": 15, "AQI": "Moderate"},
}

# Custom tool to fetch weather
@tool
def get_weather(city: str) -> str:
    """Returns the current weather information for a given city from the database."""
    city_key = city.strip().title()
    if city_key not in Weather_database:
        return f"Apologies, the weather data for the city {city} could not be retrieved from the database."
    
    data = Weather_database[city_key]
    return (
        f"- Weather in {city_key}:\n"
        f"- Temperature: {data['temperature']}°C\n"
        f"- Condition: {data['condition']}\n"
        f"- Humidity: {data['humidity']}%\n"
        f"- Wind Speed: {data['Wind_Speed']} km/h\n"
        f"- AQI: {data['AQI']}"
    )


# Details of the LLM used 
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=200)

# Prompt template
prompt = PromptTemplate.from_template(
    "You are a helpful weather assistant.\nQuestion: {input}\n{agent_scratchpad}"
)

# Tool list
tools = [get_weather]

# Create agent
agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)

# AgentExecutor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Interactive city input
if __name__ == "__main__":
    while True:
        city_name = input("\nEnter the city name (or type 'exit' to quit): ")
        if city_name.lower() == "exit":
            print("Exiting Weather Assistant.")
            break
        response = agent_executor.invoke({"input": f"What is the weather like in {city_name}?"})
        print("\nFinal Answer:\n", response["output"])
