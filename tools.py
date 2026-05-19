# tools.py
# Defines all LangChain tools available to the agent:
#   1. get_weather       — real-time weather + forecast via WeatherAPI
#   2. web_search        — real-time web search via DuckDuckGo (no API key needed)
#   3. create_kb_tool()  — searches the session RAG knowledge base (built dynamically)
#
# get_base_tools()  → [weather, web_search]          before any docs are loaded
# get_all_tools()   → [search_documents, weather, web_search]  after docs are loaded

import requests
from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

from config import WEATHER_API_KEY


# ══════════════════════════════════════════════════════════════
#  TOOL 1 — WEATHER
# ══════════════════════════════════════════════════════════════

@tool
def get_weather(location: str) -> str:
    """
    Get real-time weather conditions and a 3-day forecast for any city or location.
    Use this tool for ANY weather-related question such as current temperature,
    humidity, wind, rain chance, UV index, or air quality.
    Input should be a city name or location string e.g. 'Delhi', 'New York', 'London UK'.
    """
    try:
        url = "http://api.weatherapi.com/v1/forecast.json"
        params = {
            "key":    WEATHER_API_KEY,
            "q":      location,
            "days":   3,
            "aqi":    "yes",
            "alerts": "yes",
        }
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        loc      = data["location"]
        cur      = data["current"]
        forecast = data["forecast"]["forecastday"]
        alerts   = data.get("alerts", {}).get("alert", [])

        output = f"""
📍 Location  : {loc['name']}, {loc['region']}, {loc['country']}
🕐 Local Time: {loc['localtime']}

🌡️ CURRENT CONDITIONS
  Temperature : {cur['temp_c']}°C  /  {cur['temp_f']}°F
  Feels Like  : {cur['feelslike_c']}°C  /  {cur['feelslike_f']}°F
  Condition   : {cur['condition']['text']}
  Humidity    : {cur['humidity']}%
  Wind        : {cur['wind_kph']} km/h {cur['wind_dir']}
  Visibility  : {cur['vis_km']} km
  UV Index    : {cur['uv']}
  AQI (US EPA): {cur.get('air_quality', {}).get('us-epa-index', 'N/A')}

📅 3-DAY FORECAST"""

        for day in forecast:
            d = day["day"]
            output += f"""
  {day['date']}
    High / Low   : {d['maxtemp_c']}°C  /  {d['mintemp_c']}°C
    Condition    : {d['condition']['text']}
    Rain Chance  : {d['daily_chance_of_rain']}%
    Avg Humidity : {d['avghumidity']}%"""

        if alerts:
            output += "\n\n⚠️ WEATHER ALERTS"
            for alert in alerts:
                output += f"\n  • {alert['headline']}"

        return output.strip()

    except requests.exceptions.HTTPError as e:
        resp = getattr(e, 'response', None)
        if resp is not None:
            return f"WeatherAPI HTTP error: {getattr(resp, 'status_code', 'N/A')} — {getattr(resp, 'text', '')}"
        return f"WeatherAPI HTTP error: {str(e)}"
    except requests.exceptions.RequestException as e:
        return f"WeatherAPI connection error: {str(e)}"
    except KeyError as e:
        return f"Unexpected WeatherAPI response — missing key: {str(e)}"


# ══════════════════════════════════════════════════════════════
#  TOOL 2 — DUCKDUCKGO WEB SEARCH
# ══════════════════════════════════════════════════════════════

_ddg = DuckDuckGoSearchRun()

@tool
def web_search(query: str) -> str:
    """
    Search the web for real-time information, news, current events, prices,
    or anything requiring up-to-date data beyond the uploaded documents.
    Use this for recent events, live data, or topics not in the knowledge base.
    Input should be a clear search query string.
    """
    try:
        result = _ddg.run(query)
        if not result:
            return f"No results found for: '{query}'"
        return f"🔍 Web search results for '{query}':\n\n{result}"
    except Exception as e:
        return f"DuckDuckGo search error: {str(e)}"


# ══════════════════════════════════════════════════════════════
#  TOOL 3 — KNOWLEDGE BASE SEARCH (dynamic — built per session)
# ══════════════════════════════════════════════════════════════

def create_kb_tool(retriever):
    """
    Factory — creates a search_documents tool bound to the current
    session's retriever. Called each time new documents are added
    so the tool always points to the latest retriever.
    """

    @tool
    def search_documents(query: str) -> str:
        """
        Search the uploaded knowledge base for relevant information from
        documents, PDFs, text files, web pages, or JSON files the user has added.
        Use this when the question is about content from uploaded files.
        Input should be a descriptive search query.
        """
        try:
            docs = retriever.invoke(query)
            if not docs:
                return "No relevant passages found in the uploaded documents."

            seen   = set()
            output = "📄 Relevant passages from the knowledge base:\n\n"
            shown  = 0

            for doc in docs:
                snippet = doc.page_content[:500]
                if snippet in seen:
                    continue
                seen.add(snippet)
                shown  += 1
                source  = doc.metadata.get("source", "Unknown")
                output += f"[Source {shown}: {source}]\n{snippet}\n\n{'─'*40}\n\n"

            return output.strip()

        except Exception as e:
            return f"Error searching knowledge base: {str(e)}"

    return search_documents


# ══════════════════════════════════════════════════════════════
#  TOOL FACTORIES
# ══════════════════════════════════════════════════════════════

def get_base_tools() -> list:
    """Weather + web search — always available, no docs needed."""
    return [get_weather, web_search]


def get_all_tools(retriever) -> list:
    """Full tool set once documents are loaded. KB search listed first."""
    return [create_kb_tool(retriever), get_weather, web_search]