"""Example: tool-calling agent variant of the assistant.

This module is intentionally not imported by the service. It shows how the
same retrieval layer plugs into a LangChain tool-calling agent, which is the
kind of construct AIBOM scanners look for when identifying agents and tools.
"""

from __future__ import annotations

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from policy_assistant.agent import PolicyAgent
from policy_assistant.config import Settings
from policy_assistant.prompts import SYSTEM_PROMPT_FILE, load_prompt

assistant = PolicyAgent(settings=Settings.from_env())


@tool
def search_policy_kb(query: str) -> str:
    """Search the HR, travel and security policy knowledge base."""
    return "\n\n".join(assistant.search_policy_kb(query))


@tool
def create_escalation(summary: str) -> str:
    """Open a ticket with the human policy review queue."""
    return assistant.create_escalation(summary)


def build_agent() -> AgentExecutor:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", load_prompt(SYSTEM_PROMPT_FILE)),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    agent = create_tool_calling_agent(llm, [search_policy_kb, create_escalation], prompt)
    return AgentExecutor(agent=agent, tools=[search_policy_kb, create_escalation])


if __name__ == "__main__":
    executor = build_agent()
    print(executor.invoke({"input": "Can I expense a hotel at 300 USD in London?"}))
