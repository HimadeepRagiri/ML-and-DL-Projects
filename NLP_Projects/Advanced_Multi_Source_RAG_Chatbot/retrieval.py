import wikipedia
import yt_dlp
from model_config import search_tool

# Wikipedia Retrieval
def search_wikipedia(query):
    try:
        return wikipedia.summary(query, sentences=2)
    except:
        return "No relevant Wikipedia information found."

# YouTube Transcription Retrieval
def get_youtube_transcript(url):
    if url:  # Only process if URL is provided
        ydl_opts = {
            "quiet": True,
            "format": "bestaudio/best",
            "cookiefile": "cookies.txt"  # Use the exported cookies
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                transcript = info.get("description", "")
            return transcript if transcript else "No transcript available."
        except Exception as e:
            return f"Error retrieving transcript: {str(e)}"
    return ""

# Retrieve Top Sources
def retrieve_sources(query, vector_db=None, youtube_url=None):
    sources = ""

    # Document search if PDF is loaded
    if vector_db is not None:
        retrieved_docs = vector_db.similarity_search(query, k=10)
        ranked_docs = [doc.page_content for doc in retrieved_docs]
        sources += f"Documents: {ranked_docs}\n"

    # Web search
    web_search = search_tool.run(query)
    sources += f"Web: {web_search}\n"

    # Wikipedia
    wiki_summary = search_wikipedia(query)
    sources += f"Wikipedia: {wiki_summary}\n"

    # YouTube transcript if URL provided
    if youtube_url:
        transcript = get_youtube_transcript(youtube_url)
        sources += f"YouTube Transcript: {transcript}"

    return sources
