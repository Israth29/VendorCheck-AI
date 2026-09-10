from langsmith import traceable
from tools.web_search import web_search


@traceable(name="research_agent")
def run_research_agent(query, max_results=3):
    """Web Research Agent.
    Responsibility: ground the verification in live, external information via
    web search (this project's Google-Search-grounding equivalent).
    """
    return web_search(query, max_results=max_results)