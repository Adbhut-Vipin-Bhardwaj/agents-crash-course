import asyncio
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from search import SearchEngine

SYSTEM_PROMPT = """
You are a helpful assistant for a Github repo. \
You can search the markdown files of the repo.

Always search for relevant information before answering. 
If the first search doesn't give you enough information, try different search terms.

Make multiple searches if needed to provide comprehensive answers.
"""

class RepoAgent:
    def __init__(
        self, 
        search_engine: SearchEngine, 
        model_name: str = 'gpt-4o-mini',
        api_key_path: str = "./openai_api_key.txt"
    ):
        self.search_engine = search_engine
        
        with open(api_key_path, "r") as f:
            api_key = f.read().strip()

        self.model = OpenAIChatModel(
            model_name,
            provider=OpenAIProvider(api_key=api_key)
        )
        self.system_prompt = SYSTEM_PROMPT
        
        self.agent = Agent(
            name="repo_agent",
            instructions=self.system_prompt,
            tools=[self.search_engine.text_search],
            model=self.model
        )

    async def run(self, question: str):
        """Run the agent on a user question."""
        result = await self.agent.run(user_prompt=question)
        return result

if __name__ == "__main__":
    # Initialize search engine
    search_engine = SearchEngine(
        owner="microsoft",
        repo="vscode-copilot-chat",
    )
    search_engine.initialize()

    # Initialize agent
    repo_agent = RepoAgent(search_engine=search_engine)

    # Run question
    question = "What is the repo all about?"
    result = asyncio.run(repo_agent.run(question))
    
    print("\nAgent Response:")
    print(result.output)
