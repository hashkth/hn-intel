import asyncio
import aiohttp
from bs4 import BeautifulSoup
from google import genai
from datetime import datetime
import re

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

BASE_HN_ITEM_URL = "https://hacker-news.firebaseio.com/v0/item"
HN_SEARCH_URL = "https://hn.algolia.com/api/v1/search"

BASE_MODEL = "gemini-2.5-flash"

MAX_COMMENTS = 100
NUM_STORIES = 3
CONCURRENCY = 50

# Context management constants
MAX_CHAT_TURNS = 6       # Max full turns before summarizing older messages
SUMMARY_KEEP_TURNS = 2   # Recent turns always kept verbatim

console = Console()
client = None

def initialize_client():
    global client

    api_key = input("Enter Gemini API Key: ").strip()

    try:
        temp_client = genai.Client(api_key=api_key)
        temp_client.models.generate_content(
            model=f"models/{BASE_MODEL}",
            contents="Test"
        )
        client = temp_client
        console.print("[green]Gemini API Key validated successfully[/green]")

    except Exception as e:
        console.print(f"[bold red]Invalid Gemini API Key[/bold red]\n{e}")
        exit(1)

def get_text_models():
    models = []

    try:
        for m in client.models.list():
            name = getattr(m, "name", "").lower()

            if not name.startswith("models/gemini"):
                continue

            excluded = [
                "preview", "tts", "audio", "image",
                "embedding", "robotics", "live", "computer-use"
            ]
            if any(k in name for k in excluded):
                continue

            if not any(k in name for k in ["pro", "flash", "lite"]):
                continue

            models.append(name)

    except Exception as e:
        console.print(f"[red]Model fetch failed:[/red] {e}")
        return [f"models/{BASE_MODEL}"]

    def score_model(name):
        score = 0
        if "latest" in name:
            score += 1000
            if "pro" in name:
                score += 30
            elif "flash" in name:
                score += 20
            elif "lite" in name:
                score += 10

        match = re.search(r"(\d+(\.\d+)?)", name)
        version = float(match.group(1)) if match else 0
        score += version * 10

        if "pro" in name:
            score += 3
        elif "flash" in name:
            score += 2
        elif "lite" in name:
            score += 1

        return score

    return sorted(models, key=score_model, reverse=True)

def generate_with_fallback(prompt):
    models = get_text_models()

    for model in models:
        try:
            console.print(f"[yellow]Using model:[/yellow] {model}")
            res = client.models.generate_content(model=model, contents=prompt)
            if res and getattr(res, "text", None):
                return res.text
        except Exception:
            console.print(f"[red]Failed:[/red] {model}")

    return "All models failed."

async def fetch_json(session, url):
    try:
        async with session.get(url, timeout=5) as res:
            return await res.json()
    except:
        return None

async def search_stories(session, query):
    try:
        params = {"query": query, "tags": "story", "hitsPerPage": NUM_STORIES}
        async with session.get(HN_SEARCH_URL, params=params, timeout=5) as res:
            data = await res.json()
            return data.get("hits", [])
    except:
        return []

def clean_html(text):
    return BeautifulSoup(text or "", "html.parser").get_text()

async def fetch_comments(session, story_id):
    story = await fetch_json(session, f"{BASE_HN_ITEM_URL}/{story_id}.json")
    if not story or "kids" not in story:
        return []

    ids = story["kids"][:MAX_COMMENTS]
    sem = asyncio.Semaphore(CONCURRENCY)

    async def fetch_one(cid):
        async with sem:
            item = await fetch_json(session, f"{BASE_HN_ITEM_URL}/{cid}.json")
            if item and item.get("text"):
                return {
                    "author": item.get("by"),
                    "depth": 0,
                    "text": clean_html(item.get("text")),
                    "time": item.get("time"),
                    "score": item.get("score", 0)
                }
        return None

    tasks = [fetch_one(cid) for cid in ids]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return [r for r in results if isinstance(r, dict)]

def filter_comments(comments):
    filtered = []
    discarded = 0

    for c in comments:
        if not c["text"] or len(c["text"]) < 40:
            discarded += 1
            continue
        filtered.append(c)

    return filtered, discarded

def format_comment(c):
    time_str = datetime.fromtimestamp(c["time"]).strftime("%Y-%m-%d") if c["time"] else "unknown"
    return f"- ({c['author']} | score:{c['score']} | {time_str}) {c['text']}"

def rank_comments(comments):
    def score(c):
        s = 0
        s += c.get("score", 0) * 2
        s += min(len(c["text"]) / 50, 20)
        s += max(0, 5 - c.get("depth", 0))
        if any(k in c["text"].lower() for k in ["performance", "latency", "design", "system", "memory"]):
            s += 5
        return s

    return sorted(comments, key=score, reverse=True)

def chunk_comments(comments, chunk_size=25):
    chunks = []
    current = []

    for c in comments:
        current.append(format_comment(c))
        if len(current) >= chunk_size:
            chunks.append("\n".join(current))
            current = []

    if current:
        chunks.append("\n".join(current))

    return chunks

def generate_audit(total, filtered, discarded):
    avg_len = filtered and sum(len(c["text"]) for c in filtered) // len(filtered) or 0

    return f"""
            ## Data Audit

            - Total comments fetched: {total}
            - Comments used (after filtering): {len(filtered)}
            - Comments discarded: {discarded} ({round(discarded / total * 100) if total else 0}% discard rate)
            - Average comment length: {avg_len} chars
            - Max possible raw comments: {MAX_COMMENTS} per story × {NUM_STORIES} stories = {MAX_COMMENTS * NUM_STORIES}

            ### Filtering Decisions
            - Removed comments with no text or fewer than 40 characters (noise, one-liners, jokes)
            - Preserved metadata per comment: author, upvote score, timestamp, thread depth
            - Ranked by upvotes, length, depth, and technical keyword presence
            - Only top {MAX_COMMENTS} ranked comments passed to the LLM
            """

# =========================
# CONTEXT MANAGEMENT (Stage 4)
# =========================
def summarize_chat_history(history):
    """
    Compress old chat turns into a summary to avoid overflowing the context window.

    Strategy: sliding window with summarization.
    - Keep the last SUMMARY_KEEP_TURNS turns verbatim (most recent context).
    - Summarize all earlier turns into a compact bullet-point block.
    - The summary block replaces the older turns in the history list.

    This ensures the model always has the most recent exchange in full,
    while older topics are retained as a compressed memory rather than dropped.
    """
    if len(history) <= SUMMARY_KEEP_TURNS * 2:
        return history  # Not enough history to warrant summarization

    older = history[: -(SUMMARY_KEEP_TURNS * 2)]
    recent = history[-(SUMMARY_KEEP_TURNS * 2):]

    transcript = "\n".join(
        f"{'User' if i % 2 == 0 else 'Assistant'}: {msg}"
        for i, msg in enumerate(older)
    )

    summary_prompt = f"""
            Summarize this Q&A exchange in 3-5 bullet points.
            Capture key facts discussed, the user's focus areas, and any conclusions reached.
            Be concise — this will serve as background memory for future questions.

            {transcript}
            """
    summary = generate_with_fallback(summary_prompt)
    summary_block = f"[Summary of earlier conversation]\n{summary}"
    return [summary_block] + recent


def build_chat_prompt(chunks, history, question):
    """Build the full prompt for a chat turn with grounded context and history."""
    context = "\n\n".join(chunks)

    history_text = ""
    if history:
        turns = [
            f"{'User' if i % 2 == 0 else 'Assistant'}: {msg}"
            for i, msg in enumerate(history)
        ]
        history_text = "\n".join(turns)

    return f"""
            You are a research assistant strictly grounded in Hacker News discussion data.

            Rules:
            1. Answer ONLY using the discussion data provided below.
            2. If information is absent, say exactly: "Not discussed in the threads."
            3. If there are contradictory opinions in the data, present BOTH sides fairly — do not pick one.
            4. Do NOT agree with false premises. If a user claims "everyone agreed X", verify it against the data.
            5. If a question references earlier conversation, use the conversation history to answer it.

            --- DISCUSSION DATA ---
            {context}

            --- CONVERSATION HISTORY ---
            {history_text if history_text else "(No prior conversation)"}

            --- CURRENT QUESTION ---
            {question}
            """

def generate_digest(chunks):
    prompt = f"""
            Analyze these Hacker News discussions and produce a structured digest.
            Be specific — extract real arguments, named tools, and concrete trade-offs.
            Do not produce generic summaries like "opinions are mixed."

            ## Key Themes
            ## Pros
            ## Cons
            ## Notable Insights
            ## Tools & Alternatives Mentioned
            ## Overall Take

            Use ONLY the data below:

            {chr(10).join(chunks)}
            """
    return generate_with_fallback(prompt)

def grounded_chat(chunks, history, question):
    prompt = build_chat_prompt(chunks, history, question)
    return generate_with_fallback(prompt)

async def async_main():
    query = input("Enter topic: ")

    async with aiohttp.ClientSession() as session:
        console.print("[cyan]Searching HN...[/cyan]")
        stories = await search_stories(session, query)

        if not stories:
            console.print("[red]No stories found[/red]")
            return

        console.print(f"[cyan]Fetching comments from {len(stories)} stories...[/cyan]")
        tasks = [fetch_comments(session, s["objectID"]) for s in stories]
        results = await asyncio.gather(*tasks)
        all_comments = [c for sub in results for c in sub]

    if not all_comments:
        console.print("[red]No comments fetched[/red]")
        return

    filtered, discarded = filter_comments(all_comments)
    ranked = rank_comments(filtered)
    top_comments = ranked[:MAX_COMMENTS]

    console.print(Panel(Markdown(generate_audit(len(all_comments), filtered, discarded)), title="Data Audit"))

    chunks = chunk_comments(top_comments)

    console.print("[cyan]Generating digest...[/cyan]")
    digest = generate_digest(chunks)
    console.print(Panel(Markdown(digest), title="Digest"))

    chat_history = []  # flat list: [user_q1, assistant_ans1, user_q2, ...]

    while True:
        q = input("\nAsk a follow-up question (or type 'exit'): ").strip()
        if q.lower() == "exit":
            break
        if not q:
            continue

        # Summarize older turns when history grows too long
        if len(chat_history) >= MAX_CHAT_TURNS * 2:
            console.print("[dim]Compressing earlier conversation to manage context window...[/dim]")
            chat_history = summarize_chat_history(chat_history)

        ans = grounded_chat(chunks, chat_history, q)
        console.print(Markdown(ans))

        chat_history.append(q)
        chat_history.append(ans)

if __name__ == "__main__":
    initialize_client()
    asyncio.run(async_main())