from langsmith import traceable
from config import tavily


@traceable(name="web_search")
def web_search(query, max_results=3):
    try:
        response = tavily.search(query=query, max_results=max_results)
        results = []
        for r in response.get("results", []):
            results.append({
                "title": r.get("title"),
                "content": r.get("content"),
                "url": r.get("url")
            })
        return results
    except Exception:
        return []