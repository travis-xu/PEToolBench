history_description_chronological = "These preferences can be inferred from the user's interaction history provided below, including past user instructions and corresponding tool calls, which are in chronological order where user is more preferred with latter tool calls than earlier ones."
history_description_ratings = "These preferences can be inferred from the user's interaction history provided below, including past user instructions and corresponding tool calls, with user's binary ratings reflecting the user's satisfaction with the tool calls."
history_description_preferred = "These preferences can be inferred from the user's interaction history provided below, including past user instructions and corresponding tool calls."

FORMAT_ENTRY_INPUT = """
Your task is to use a tool that not only meets real-time user instructions but also aligns with user preferences.
{history_description}
Interaction history is:
{interaction_history}

Available tools you can call with input parameters are listed here:
{apis_candidate}

Generate your tool call in the following format:
{{"tool_name": the tool's name, "parameters": input parameters in JSON string format on a single line}}
Example output #1:
{{"tool_name": "<Search>.<Arxiv Search>.<Get Arxiv Article Information>", "parameters": {{"query": "machine learning", "sort_by": "date", "limit": 3}}
Example output #2:
{{"tool_name": "<Travel>.<flight | flight aggregator>.<flight>", "parameters": {{}}}}

Remember:
Generate only the tool call in JSON format without any additional outputs.
You will now receive a user instruction; please generate the tool call. 
Begin!

"""

FORMAT_ENTRY_INPUT_WO_HISTORY = """
Your task is to use a tool that meets real-time user instructions.
Available tools you can call with input parameters are listed here:
{apis_candidate}

Generate your tool call in the following format:
{{"tool_name": the tool's name, "parameters": input parameters in JSON string format on a single line}}
Example output #1:
{{"tool_name": "<Search>.<Arxiv Search>.<Get Arxiv Article Information>", "parameters": {{"query": "machine learning", "sort_by": "date", "limit": 3}}
Example output #2:
{{"tool_name": "<Travel>.<flight | flight aggregator>.<flight>", "parameters": {{}}}}

Remember:
Generate only the tool call in JSON format without any additional outputs.
You will now receive a user instruction; please generate the tool call. 
Begin!

"""