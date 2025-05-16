from dotenv import load_dotenv
from fastmcp import FastMCP
from langchain_groq import ChatGroq
from crewai import Agent, Task, Crew, Process
from crewai.memory import EntityMemory
from crewai.memory.storage.rag_storage import RAGStorage
from crewai_tools import MCPServerAdapter
from mcp import StdioServerParameters
import os

load_dotenv()
mcp = FastMCP("multi-agent-server")


# Function for per-user memory
def get_user_memory(user_id: str):
    return EntityMemory(
        storage=RAGStorage(
            embedder_config={
                "provider": "openai",
                "config": {"model": "text-embedding-3-small"},
            },
            type="short_term",
            path=f"./memory_store/{user_id}/",
        )
    )


@mcp.tool(name="multi_analyst")
async def multi_analyst_tool(question: str, user_id: str) -> str:
    """Handle financial and DB questions using unified tool access."""
    yfinance_params = StdioServerParameters(command="uvx", args=["yfmcp@latest"])
    supabase_params = StdioServerParameters(
        command="npx",
        args=["-y", "@supabase/mcp-server-supabase@latest"],
        env={"SUPABASE_ACCESS_TOKEN": os.getenv("SUPABASE_ACCESS_TOKEN"), **os.environ},
    )

    mcp_adapters = []
    try:
        yfinance_adapter = MCPServerAdapter(yfinance_params)
        supabase_adapter = MCPServerAdapter(supabase_params)
        mcp_adapters = [yfinance_adapter, supabase_adapter]

        tools = yfinance_adapter.tools + supabase_adapter.tools
        llm = ChatGroq(
            model="mixtral-8x7b-32768",  # Ou llama3-70b, etc., dependendo do que quer usar
            temperature=0.7,
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
        memory = get_user_memory(user_id)

        multi_analyst = Agent(
            role="Professional Data & Finance Analyst",
            goal="Answer any financial or database question using YFinance and Supabase tools.",
            backstory="Expert in SQL, stocks, KPIs, and databases. Decides the best tool for each query.",
            tools=tools,
            verbose=True,
            llm=llm,
            allow_delegation=False,
            memory=memory,
        )

        task = Task(
            description=f"Handle this user question: {question}",
            expected_output="Useful response using the most suitable tool.",
            tools=tools,
            agent=multi_analyst,
            memory=memory,
        )

        crew = Crew(
            agents=[multi_analyst],
            tasks=[task],
            process=Process.sequential,
            memory=True,
            entity_memory=memory,
            verbose=True,
        )

        result = await crew.kickoff_async()
        return result
    finally:
        for adapter in mcp_adapters:
            try:
                adapter.stop()
            except Exception:
                pass


if __name__ == "__main__":
    mcp.run(transport="sse", host="127.0.0.1", port=8005)
