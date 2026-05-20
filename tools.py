

import requests
from langchain.tools import tool

from datetime import datetime
from tavily import TavilyClient

from config import WEATHER_API_KEY
from datetime import datetime, timezone
from config import WEATHER_API_KEY, TAVILY_API_KEY
import os
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY or ""



# this is a work in progress and will be expanded with more tools and better formatting, but the idea is:
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
#  TOOL 4 — CURRENT DATE & TIME
# ══════════════════════════════════════════════════════════════

@tool
def get_current_datetime(query: str) -> str:
    """
    Returns the current real date and time.
    Use this tool whenever the user asks what today's date is,
    what time it is, what day of the week it is, or anything
    related to the current date or time.
    Input can be anything — it is ignored.
    """
    now = datetime.now()
    return f"""
📅 Current Date & Time
  Date      : {now.strftime("%A, %B %d, %Y")}
  Time      : {now.strftime("%I:%M %p")}
  Day       : {now.strftime("%A")}
  Week No.  : {now.strftime("%W")}
""".strip()




_tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

@tool
def web_search(query: str) -> str:
    """
    Search the web for real-time information, news, current events, prices,
    or anything requiring up-to-date data beyond the uploaded documents.
    Use this for recent events, live data, or topics not in the knowledge base.
    Input should be a clear search query string.
    """
    try:
        today = datetime.now().strftime("%Y-%m-%d")

        response = _tavily_client.search(
            query=f"{query} {today}",   # appends today's date to force fresh results
            search_depth="advanced",    # deeper search than basic
            max_results=6,
            include_answer=True,        # Tavily generates a direct answer too
            include_published_date=True # forces date filtering
        )

        results  = response.get("results", [])
        answer   = response.get("answer", "")

        if not results:
            return f"No results found for: '{query}'"

        output = f"🔍 Search results for '{query}':\n\n"

        if answer:
            output += f"📌 Quick answer: {answer}\n\n"

        for i, r in enumerate(results, 1):
            pub_date = r.get("published_date", "Date unknown")
            output += f"{i}. **{r.get('title', 'No title')}**\n"
            output += f"   Source : {r.get('url', '')}\n"
            output += f"   Date   : {pub_date}\n"
            output += f"   Summary: {r.get('content', 'No content')[:300]}\n\n"

        return output.strip()

    except Exception as e:
        return f"Search error: {str(e)}"




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




def get_base_tools() -> list:
    return [get_current_datetime, get_weather, web_search]

def get_all_tools(retriever) -> list:
    return [create_kb_tool(retriever), get_current_datetime, get_weather, web_search]