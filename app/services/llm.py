import json
from openai import OpenAI
from app.core.config import settings
from app.services.github import get_github_repo_contents

client = OpenAI(api_key=settings.openai_api_key, project=settings.project_id)


def run_conversation(prompt: str):
    # Step 1: send the conversation and available functions to the model
    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_github_repo_contents",
                "description": "Get the contents of a given github repository",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "owner": {
                            "type": "string",
                            "description": "Owner of the github repository",
                        },
                        "repo": {
                            "type": "string",
                            "description": "Github repository name",
                        },
                    },
                    "required": ["owner", "repo"],
                },
            },
        }
    ]
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls
    # Step 2: check if the model wanted to call a function
    if tool_calls:
        # Step 3: call the function
        available_functions = {
            "get_github_repo_contents": get_github_repo_contents,
        }
        # extend conversation with assistant's reply
        messages.append(response_message)
        # Step 4: send the info for each function call and function response to the model
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            function_response = function_to_call(**function_args)
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": json.dumps(function_response),
                }
            )  # extend conversation with function response
        second_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
        )  # get a new response from the model where it can see the function response
        return second_response


