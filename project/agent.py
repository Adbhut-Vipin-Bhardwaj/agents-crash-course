import asyncio
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from search import text_search

system_prompt = """
You are a helpful assistant for a VSCode extension.

Always search for relevant information before answering. 
If the first search doesn't give you enough information, try different search terms.

Make multiple searches if needed to provide comprehensive answers.
"""

with open("./openai_api_key.txt", "r") as f:
    api_key = f.read().strip()

model = OpenAIChatModel(
    'gpt-4o-mini',
    provider=OpenAIProvider(api_key=api_key)
)
agent = Agent(
    name="faq_agent",
    instructions=system_prompt,
    tools=[text_search],
    model=model
)

question = "List the features pls"

result = asyncio.run(agent.run(user_prompt=question))
