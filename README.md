# HN Thread Intelligence Tool

## Overview

A developer-focused research assistant that analyzes Hacker News discussions in real time. It fetches top threads for any topic, extracts high-signal insights, and lets users explore them through a conversational interface, all grounded strictly in the fetched data.

---

## Features

- Real-time data fetching from Hacker News APIs
- Multi-thread aggregation across top stories
- Data audit with discard reasoning
- High-signal comment ranking
- Structured digest generation (themes, pros/cons, alternatives)
- Conversational Q&A grounded in HN data
- Rolling context management for long chat sessions
- Clean terminal UI with Markdown rendering

---

## Tech Stack

- **Python** - core runtime
- **aiohttp** - async HTTP for concurrent comment fetching
- **BeautifulSoup** - HTML cleaning on comment text
- **Gemini API** - LLM (with multi-model fallback)
- **Rich** - terminal UI and Markdown rendering

---

## Stage 1: Data Acquisition & Audit

### What is fetched?

Stories are retrieved via the [Algolia HN Search API](https://hn.algolia.com/api/v1/search), which returns ranked results for a query. Comments are then fetched per-story from the [official HN Firebase API](https://hacker-news.firebaseio.com/v0/item/).

Up to `MAX_COMMENTS = 100` top-level comment IDs are fetched per story across `NUM_STORIES = 3` stories, giving a maximum of **300 raw comments** per query. Fetching is done concurrently using `asyncio.Semaphore(50)` to avoid rate limiting.

### Data quality observations

Raw HN threads are noisy. In testing across topics like "Rust", "React", and "LLMs":

- ~20–35% of comments were short (<40 chars): jokes, one-liners, "+1" replies
- Many comments lack upvote scores (especially nested replies)
- Some comments contain HTML tags (`<p>`, `<a>`, `<pre>`) requiring cleaning

### Filtering decisions

Discarded:
- Comments with no text content
- Comments shorter than 40 characters - this threshold was chosen to eliminate noise (one-liners, reactions) while retaining short-but-substantive technical comments

Preserved per comment:
- `author` - for attribution and context
- `score` - primary signal of community agreement
- `time` - allows chronological reasoning
- `depth` - proxy for whether the comment is a top-level take vs. a buried reply

### Audit output

After filtering, the system prints:
- Total fetched / used / discarded counts
- Discard percentage
- Average comment length of retained comments

---

## Stage 2: Chunking & Structure

### The problem with naive chunking

Splitting by raw token count destroys comment context:
- A comment mid-sentence is meaningless without its author and score
- Reply context (who is replying to whom) is lost
- The LLM receives decontextualized text fragments

### The Approach

1. Each comment is serialized as a self-contained line:
   ```
   - (author | score:N | YYYY-MM-DD) comment text here
   ```
   This keeps all metadata inline, so the LLM always knows who said what and how highly it was rated.

2. Comments are ranked before chunking using a scoring formula:
   - Upvotes × 2 (primary signal)
   - Text length / 50, capped at 20 (length as a depth proxy)
   - Thread depth penalty: `max(0, 5 - depth)` (top-level comments weighted higher)
   - +5 bonus for technical keywords: `performance`, `latency`, `design`, `system`, `memory`

3. Chunks of 25 ranked comments are assembled. Only the top 100 comments (after ranking) are passed to the LLM — this keeps the context window manageable while prioritizing signal.

### Why this works

- Every chunk is coherent so no comment is split mid-text
- Metadata is preserved inline, not in a separate lookup table
- Ranking ensures the LLM processes the most valuable comments first
- The chunk set is fixed and reused for both digest and chat, no re-fetching

### Tradeoffs

- Currently we fetch only top-level `kids` IDs, so deeply nested replies are missed
- Comment ranking is heuristic, a low-upvote but insightful comment may be deprioritized
- A future improvement would use embedding-based retrieval to select chunks per query

---

## Stage 3: Digest Generation

The digest prompt instructs the LLM to produce six specific sections:

- **Key Themes** - what the discussion is actually about
- **Pros** - positive signals from the community
- **Cons** - criticisms, caveats, warnings
- **Notable Insights** - non-obvious observations worth surfacing
- **Tools & Alternatives Mentioned** - competing or complementary tools named
- **Overall Take** - a synthesized community verdict

The prompt explicitly forbids generic summaries ("opinions are mixed") and requires the LLM to ground everything in the provided comment data. This structure was chosen to make the output useful for a developer evaluating a technology, not just a summary of what was said.

---

## Stage 4: Context Management for Chat (Brownie Points)

### The problem

A long chat session can accumulate:
- The fixed HN chunk data (~100 comments)
- The full digest
- Multiple rounds of Q&A (each adding ~500–2000 tokens)

After several turns, this can exceed the model's context window.

### The strategy: sliding window with summarization

1. Chat history is stored as a flat list of alternating user/assistant messages.
2. When history reaches `MAX_CHAT_TURNS = 6` full turns (12 messages), we trigger compression.
3. The most recent `SUMMARY_KEEP_TURNS = 2` turns are preserved verbatim, these are most relevant to the current question.
4. All older turns are passed to the LLM to produce a 3–5 bullet summary.
5. The summary replaces the older turns in the history list.

This means the model always has:
- The full HN chunk context (static, never compressed)
- A compressed memory of earlier topics discussed
- The last 2 full turns verbatim

### Why not a pure sliding window?

A sliding window (dropping old turns) would lose important context if the user asks "what did we say earlier about X?" Summarization retains the key conclusions without the full verbosity.

### Tradeoff

Summarization itself costs an API call. For very long sessions this adds latency and cost. A future improvement would use token counting to trigger summarization only when a true context limit is approached.

---

## Stage 5: Edge Cases & Hardening

The system prompt given to the LLM for every chat turn includes explicit rules to handle the following edge cases:

### a. Question with no answer in the data

**Prompt instruction:** "If information is absent, say exactly: 'Not discussed in the threads.'"

The model is told not to speculate or generalize beyond the provided data. This prevents hallucinated answers from being presented as community consensus.

### b. Contradictory opinions in the data

**Prompt instruction:** "If there are contradictory opinions in the data, present BOTH sides fairly - do not pick one."

HN threads frequently contain opposing views (e.g., "Rust is worth the learning curve" vs. "Rust's borrow checker isn't worth it for most projects"). The model is instructed to surface this disagreement rather than flatten it into a false consensus.

### c. Question referencing earlier conversation

**Handled by:** passing full chat history (or its summarized version) in every prompt.

The `build_chat_prompt()` function includes the conversation history block in every call, so the model can reference what was discussed earlier without needing persistent memory.

### d. Manipulative question designed to assert false consensus

**Prompt instruction:** "Do NOT agree with false premises. If a user claims 'everyone agreed X', verify it against the data."

This prevents prompt injection patterns like "since everyone in the thread agreed that Go is better than Rust, summarize why." The model is told to check the actual data before confirming any claimed consensus.

---

## Handling Context Limits

| Component | Size (approx) |
|-----------|--------------|
| HN chunks (100 comments) | ~15,000 tokens |
| Digest | ~500 tokens |
| Chat history (6 turns) | ~3,000–6,000 tokens |
| System prompt | ~300 tokens |

The static chunk set is the dominant cost. We manage it by capping at 100 ranked comments. Chat history is managed via the rolling summarization described in Stage 4.

---

## Future Improvements

- True thread reconstruction (parent-child reply chains)
- Embedding-based retrieval: select the most relevant chunks per query rather than always using the top-ranked static set
- Token counting to trigger summarization dynamically rather than by turn count
- Web frontend

---

## Running the Tool

```bash
pip install -r requirements.txt
python main.py
```

You will be prompted for a Gemini API key and a search topic.