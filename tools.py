from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import tool
from datetime import datetime
import os

@tool
def save_text_to_file(data: str, filename: str = "research_output.txt") -> str:
    """Saves the final research results into a local text file for storage."""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_text = f"--- Research Output ---\nTimestamp: {timestamp}\n\n{data}\n\n"
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else ".", exist_ok=True)
        with open(filename, "a", encoding="utf-8") as f:
            f.write(formatted_text)
        return f"Successfully saved to {filename}"
    except Exception as e:
        return f"Error saving file: {str(e)}"

@tool
def google_search(query: str) -> str:
    """Search the web for current events, real-time news, and general info."""
    return DuckDuckGoSearchRun().run(query)

@tool
def wikipedia(query: str) -> str:
    """Search Wikipedia for technical definitions and historical context."""
    api_wrapper = WikipediaAPIWrapper(top_k_results=3, doc_content_chars_max=4000)
    return WikipediaQueryRun(api_wrapper=api_wrapper).run(query)

tools = [google_search, wikipedia, save_text_to_file]