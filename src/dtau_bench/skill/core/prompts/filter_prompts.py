"""Filter prompts for skill quality control."""

# General filter prompt (Stage 1)
GENERAL_FILTER_FUNCTIONAL = """You are a coding expert. Given a predefined skill, evaluate whether its quality is good or bad.

# Evaluation guidelines:
1. **Domain specificity**: Check whether the skill includes domain-specific library names apis, e.g., `apis.spotify.show_playlist_library`.
2. **Over-encapsulation**: Check whether the skill's implementation merely calls a single other skill (i.e., it is just a thin wrapper).
3. **No-Python-libraries**: Check whether additional Python libraries are introduced in the skill.
4. **Reusability**: Check whether there are parameters are specific.
5. **No-Functional style**: Check whether a functional style is being used (e.g., the presence of return).

# Bad Example1:
- No domain-specific library names mentioned
- Additional Python libraries have been introduced
- There are parameters are specific.
- The code is functional style.

```python
import datetime
username = "12345678"
song_ids = set()
for playlist in playlists:
    song_ids.update(playlist.get("song_ids", []))

return song_ids
```

# Bad Example2: Over-encapsulation
```python
apis.supervisor.complete_task(answer=answer)
```

# Good Example
```python
most_liked = None
max_likes = -1
for sid in song_ids:
    song = apis.spotify.show_song(song_id=sid)
    if song["like_count"] > max_likes:
        max_likes = song["like_count"]
        most_liked = song
```

Only return "good" or "bad". Don't return any other words.
"""

GENERAL_FILTER_ATOMIC = """You are evaluating an atomic skill for an agent skill library.

An atomic skill is reusable guidance for how an agent should complete a tool-grounded subtask while solving a user task.
It is not judged as source code or as a standalone function implementation.

# Evaluation guidelines
Return **good** if the skill content is a high-quality reusable atomic skill. Return **bad** otherwise.

Check the following:
1. **Agent usefulness**: The content should help a future agent act correctly on a recurring subtask.
2. **Tool grounding**: The content should be grounded in concrete tool usage, tool ordering, parameter constraints, validation steps, or error recovery behavior.
3. **Workflow clarity**: The content should describe a coherent subtask workflow or guidance pattern, not just vague commentary.
4. **Reusability**: Avoid task-specific hardcoded ids, dates, names, or one-off details.
5. **Non-code style**: Do not treat Python-like code, imports, or function-style implementation as good skill format.
6. **Not mere narration**: Reject content that is mostly a retelling of one task instance without reusable guidance.

# Good atomic skill characteristics
- Explains how an agent should use one or more tools to complete a reusable subtask.
- May include ordered steps, validation requirements, prerequisites, or recovery actions.
- Uses abstract placeholders rather than concrete one-off values.
- Reads like concise workflow guidance for an agent.

# Bad atomic skill characteristics
- Mostly Python code or implementation fragments.
- Mostly hardcoded task details.
- Mostly vague advice with no actionable tool-grounded guidance.
- Mostly narration about a single rollout episode without reusable generalization.

# Good Example
Before retrieving analytics for a content item, first verify that the agent has the correct athlete identity and target content identifier. If ownership or identity checks fail, resolve those prerequisites before calling the analytics tool. When analytics are empty, state that metrics are unavailable and avoid unsupported comparisons.

Only return "good" or "bad". Don't return any other words.
"""

# Tool filter prompt (Stage 2)
TOOL_FILTER_PROMPT = """You are a tool-invocation expert. Based on the tool specifications, verify whether the provided tool invocations are correct.

## Input
1. **Tool invocation content**: may include one or multiple tool calls.
2. **Tool specifications**: including tool description, parameters, return schema, and other usage notes.

## Judging Guidelines
1. **Parameter validation**: Check whether the invocation parameters comply with the specifications (e.g., missing required parameters, unsupported/nonexistent parameters, wrong types or formats, invalid values, etc.).
2. **Call dependency**: For multiple tool calls, verify that their order does not violate logical dependencies. If there is no dependency between the calls, ignore this check.
3. **Comment–function alignment**: Ensure the logic described in any comments matches what the tool is designed to do.
4. **Output Format**: Provide your reasoning and conclude with either 'correct' or 'fail', wrapped in <answer></answer>.

"""

GENERAL_FILTER_PROMPTS = {
    "default": GENERAL_FILTER_FUNCTIONAL,
    "functional": GENERAL_FILTER_FUNCTIONAL,
    "atomic": GENERAL_FILTER_ATOMIC,
}

TOOL_FILTER_PROMPTS = {
    "default": TOOL_FILTER_PROMPT,
}
