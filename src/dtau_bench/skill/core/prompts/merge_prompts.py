"""Skill merging prompts."""

# Functional skill merge prompt
FUNCTIONAL_MERGE_PROMPT = """You are a code expert. Your task is to analyze a list of skills, merge skills that are meaningfully similar, and decompose complex skills into smaller atomic skills while preserving behavior and intent.

# Input Description
The user will provide a list of skills.

---

# Skill Definition Rule
- Skill is a dictionary with four keys: `name`, `document`, `content` and `tools`.
1) `name`: the skill's name.
2) `document`: the skill's functionality, the key parameters, the final output of the skill and any important notes.
3) `content`: the concrete implementation of the skill.
4) `tools`: the key tools used in the skill (list).
- The skill is abstract, modular, and reusable. Specifically, the skill name must be generic under one application (e.g., `spotify get songs by genre` instead of `get pop songs`). The skill must use parameters instead of hard-coded values (e.g., specific email address "jay@gmail.com"). The skill body must be self-contained.
- Explicitly declare the key parameters and the final output data types using type hints. Example: `Parameters: param: str; Outputs: output: list[dict]:`
- Include detailed description of the skill with input and output explanation.
- The skill should not be similar to the existing skills in the skill library.
- The skill must involve multiple processing steps. Simply using the result of an API call without additional logic does not qualify as a valid skill.
- Never call other skills from the skill library or any previously defined skills.
- Do not import any Python packages.
- Avoid a functional style; there's no need to use return.

# Good skill:

{
    "name": "spotify get all user playlists",
    "document": "Retrieve every playlist in the authenticated user's library by paging through results until none remain.\\n\\nParameters\\n----------\\naccess_token : str\\n    Valid Spotify access token for the user.\\n\\nOutputs\\n-------\\nlist[dict]\\n    A list containing every playlist object in the user's library.\\n\\nNotes:\\n-------\\n1. Use a moderate page_limit, avoid setting it too high to prevent exceeding the page limit error.\\n2. Avoid printing the entire retrieved list to prevent overly long outputs; instead, inspect only a small subset to verify the structure.\\n",
    "content": "playlists = []\\n\\npage = 0\\nwhile True:\\n    batch = apis.spotify.show_playlist_library(\\n        access_token=access_token,\\n        page_index=page,\\n        page_limit=20\\n    )\\n    if not batch:          # empty page signals end\\n        break\\n    playlists.extend(batch)\\n    page += 1",
    "tools": ["apis.spotify.show_playlist_library"]
}

---

# Focus
1. Focus on skills with similar names and similar skillality.
2. Carefully analyze the concrete implementation differences between similar skills.

# Merge Guidelines
1. **Generality**: Merge skills that have similar names and similar skillality. The merged skill should use a generic name, and its **Notes** and implementation should cover all plausible variants and edge cases.
2. **Atomicity**: If skills have a containment relationship (one skill's skillality subsumes or builds on another), follow the skill definitions to preserve atomicity and avoid merging.
3. **Merge Constraints**: Any merged skill must comply with the skill definition rules, especially atomicity and reusability-and should avoid being tied to a specific task or scenario.

# Decompose Guidelines
1. **Atomicity**: Only decompose skills whose skillality are overly complex (e.g., they include skillality already covered by other provided skills) into smaller sub-skills.
2. **Generality**: The decomposed skills must follow the skill-definition rules and remain reusable—avoid coupling them to any specific task or scenario.

# Output Format
Output a list containing the skills (with one or multiple skills) from merging and/or decomposing the skills in the input skill list as follows:
<skill>
[
    "skill 1",
    ...
]
</skill>

Note: You don't necessarily need to both merge and decompose. You may choose to only merge them into a single skill.
"""

# Atomic skill merge prompt
ATOMIC_MERGE_PROMPT = """You are a skill expert. Your task is to analyze a set of skills and consolidate their key usage patterns and important notes.

# Input Description
1. The user will provide a list of skills. Each skill describes how to use the same specific tool.
2. The original tool schema: It defines the correct and intended usage of the tool. Ensure the merged skill remains accurate and consistent with it.

---

# Skill Definition Rule
- Skill is a dictionary with four keys: `name`, `document`, `content` and `tools`.
1) `name`: the specific tool's name.
2) `document`: the tool's functionality, the key parameters, the final output of the skill and any important notes.
3) `content`: the tool's usage examples, and examples of combining it with other tools (if applicable).
4) `tools`: the key tools used in the `content` (list).
- The skill is centered around a specific tool, describing its core functionality, important notes, and common usage examples.
- Explicitly declare the key parameters and the final output data types using type hints. Example: `Parameters: param: str; Outputs: output: dict:`
- Include detailed description of the skill with input and output explanation.
- The skill should not be similar to the existing skills in the skill library.
- The parameters used in `content` must be reusable instead of hard-coded values (e.g., specific email address "jay@gmail.com")
- The usage examples of `content` may involve one or more tool uses.
- The `document` must clearly and thoroughly document all relevant details of the specific tool use.
- Never call other skills from the skill library or any previously defined skills.
- Do not import any Python packages.
- Avoid a functional style and Python code style; there's no need to use return.

---

# Good skill:

{
    "name": "get_flight_status",
    "document": "This tool retrieves the real-time status of a specific flight on a given date. It is essential for verifying flight delays, cancellations, or confirming if a flight has landed to address customer inquiries or complaints. \\nParameters: \\n- `flight_number`: str. The unique identifier of the flight (e.g., 'HAT018').\\n- `date`: str. The date of the flight in 'YYYY-MM-DD' format.\\nOutputs: \\n- `status`: str. A string representing the flight's status (e.g., 'delayed', 'landed', 'on-time', 'cancelled').\\nImportant Notes: \\n- To get an accurate status, both the `flight_number` and the correct `date` must be provided, as the same flight number may operate on multiple days.",
    "content": "Example1: A user complains about a past flight being delayed.\\n\\nassistant:\\nFirst, the assistant identifies the user and retrieves their reservation history to locate the relevant flight.\\nget_user_details(user_id='the_user_id')\\nTo investigate the user's claim, first retrieve the details of their reservation to find the flight number and date.\\nget_reservation_details(reservation_id='a_reservation_id')\\nAfter identifying the specific flight number and date from the reservation details, check its status.\\nget_flight_status(flight_number='flight_number_from_reservation', date='flight_date_from_reservation')",
    "tools": ["get_user_details", "get_reservation_details", "get_flight_status"]
}

---、

# Merge Guidelines
1. **Faithfulness**: Ensure the skill's functionality and usage description strictly align with the original tool schema, with no deviation.
2. **Generality**: Keep the usage examples generic and broadly applicable (not tied to a specific task or scenario).
3. **Conciseness**: Do not directly merge other skills. Instead, remove redundant descriptions and examples, keeping only one representative version. Appropriately consolidate key notes where necessary, and avoid verbosity.

# Output Format
Output the merged skill, the format as follows:
<skill>
{
    "name": "The skill name",
    "document": "The merged document",
    "content": "The merged content",
    "tools": ["the tool used in content"]
}
</skill>

"""

SKILL_MERGE_PROMPTS = {
    "default": FUNCTIONAL_MERGE_PROMPT,
    "functional": FUNCTIONAL_MERGE_PROMPT,
    "atomic": ATOMIC_MERGE_PROMPT,
}
