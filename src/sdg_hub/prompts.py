# Local
from .registry import PromptRegistry


@PromptRegistry.register("blank")
def blank_chat_template():
    return """{{ messages }}"""


@PromptRegistry.register("instructlab")
def instructlab_chat_template():
    return """{% for message in messages %}{% if message['role'] == 'pretraining' %}{{ '<|pretrain|>' + message['content'] + '<|endoftext|>' + '<|/pretrain|>' }}{% elif message['role'] == 'system' %}{{ '<|system|>' + '\n' + message['content'] + '\n' }}{% elif message['role'] == 'user' %}{{ '<|user|>' + '\n' + message['content'] + '\n' }}{% elif message['role'] == 'assistant' %}{{ '<|assistant|>' + '\n' + message['content'] + '<|endoftext|>' + ('' if loop.last else '\n') }}{% endif %}{% if loop.last and add_generation_prompt %}{{ '<|assistant|>' + '\n' }}{% endif %}{% endfor %}"""


@PromptRegistry.register("mistralai/Mixtral")
def mistral_chat_template():
    return """{%- if messages[0]['role'] == 'system' %}\n    {%- set system_message = messages[0]['content'] %}\n    {%- set loop_messages = messages[1:] %}\n{%- else %}\n    {%- set loop_messages = messages %}\n{%- endif %}\n\n<s>\n{%- for message in loop_messages %}\n    {%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}\n        {{- raise_exception('After the optional system message, conversation roles must alternate user/assistant/user/assistant/...') }}\n    {%- endif %}\n    {%- if message['role'] == 'user' %}\n        {%- if loop.first and system_message is defined %}\n            {{- ' [INST] ' + system_message + '\\n\\n' + message['content'] + ' [/INST]' }}\n        {%- else %}\n            {{- ' [INST] ' + message['content'] + ' [/INST]' }}\n        {%- endif %}\n    {%- elif message['role'] == 'assistant' %}\n        {{- ' ' + message['content'] + '</s>'}}\n    {%- else %}\n        {{- raise_exception('Only user and assistant roles are supported, with the exception of an initial optional system message!') }}\n    {%- endif %}\n{%- endfor %}\n"""


@PromptRegistry.register("meta-llama/Llama-3.3")
def meta_llama_chat_template():
    return """{{- bos_token }}\n{%- if custom_tools is defined %}\n    {%- set tools = custom_tools %}\n{%- endif %}\n{%- if not tools_in_user_message is defined %}\n    {%- set tools_in_user_message = true %}\n{%- endif %}\n{%- if not date_string is defined %}\n    {%- set date_string = \"26 Jul 2024\" %}\n{%- endif %}\n{%- if not tools is defined %}\n    {%- set tools = none %}\n{%- endif %}\n\n{#- This block extracts the system message, so we can slot it into the right place. #}\n{%- if messages[0]['role'] == 'system' %}\n    {%- set system_message = messages[0]['content']|trim %}\n    {%- set messages = messages[1:] %}\n{%- else %}\n    {%- set system_message = \"\" %}\n{%- endif %}\n\n{#- System message + builtin tools #}\n{{- \"<|start_header_id|>system<|end_header_id|>\\n\\n\" }}\n{%- if builtin_tools is defined or tools is not none %}\n    {{- \"Environment: ipython\\n\" }}\n{%- endif %}\n{%- if builtin_tools is defined %}\n    {{- \"Tools: \" + builtin_tools | reject('equalto', 'code_interpreter') | join(\", \") + \"\\n\\n\"}}\n{%- endif %}\n{{- \"Cutting Knowledge Date: December 2023\\n\" }}\n{{- \"Today Date: \" + date_string + \"\\n\\n\" }}\n{%- if tools is not none and not tools_in_user_message %}\n    {{- \"You have access to the following functions. To call a function, please respond with JSON for a function call.\" }}\n    {{- 'Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.' }}\n    {{- \"Do not use variables.\\n\\n\" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- \"\\n\\n\" }}\n    {%- endfor %}\n{%- endif %}\n{{- system_message }}\n{{- \"<|eot_id|>\" }}\n\n{#- Custom tools are passed in a user message with some extra guidance #}\n{%- if tools_in_user_message and not tools is none %}\n    {#- Extract the first user message so we can plug it in here #}\n    {%- if messages | length != 0 %}\n        {%- set first_user_message = messages[0]['content']|trim %}\n        {%- set messages = messages[1:] %}\n    {%- else %}\n        {{- raise_exception(\"Cannot put tools in the first user message when there's no first user message!\") }}\n{%- endif %}\n    {{- '<|start_header_id|>user<|end_header_id|>\\n\\n' -}}\n    {{- \"Given the following functions, please respond with a JSON for a function call \" }}\n    {{- \"with its proper arguments that best answers the given prompt.\\n\\n\" }}\n    {{- 'Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.' }}\n    {{- \"Do not use variables.\\n\\n\" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- \"\\n\\n\" }}\n    {%- endfor %}\n    {{- first_user_message + \"<|eot_id|>\"}}\n{%- endif %}\n\n{%- for message in messages %}\n    {%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}\n        {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n'+ message['content'] | trim + '<|eot_id|>' }}\n    {%- elif 'tool_calls' in message %}\n        {%- if not message.tool_calls|length == 1 %}\n            {{- raise_exception(\"This model only supports single tool-calls at once!\") }}\n        {%- endif %}\n        {%- set tool_call = message.tool_calls[0].function %}\n        {%- if builtin_tools is defined and tool_call.name in builtin_tools %}\n            {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' -}}\n            {{- \"<|python_tag|>\" + tool_call.name + \".call(\" }}\n            {%- for arg_name, arg_val in tool_call.arguments | items %}\n                {{- arg_name + '=\"' + arg_val + '\"' }}\n                {%- if not loop.last %}\n                    {{- \", \" }}\n                {%- endif %}\n                {%- endfor %}\n            {{- \")\" }}\n        {%- else  %}\n            {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' -}}\n            {{- '{\"name\": \"' + tool_call.name + '\", ' }}\n            {{- '\"parameters\": ' }}\n            {{- tool_call.arguments | tojson }}\n            {{- \"}\" }}\n        {%- endif %}\n        {%- if builtin_tools is defined %}\n            {#- This means we're in ipython mode #}\n            {{- \"<|eom_id|>\" }}\n        {%- else %}\n            {{- \"<|eot_id|>\" }}\n        {%- endif %}\n    {%- elif message.role == \"tool\" or message.role == \"ipython\" %}\n        {{- \"<|start_header_id|>ipython<|end_header_id|>\\n\\n\" }}\n        {%- if message.content is mapping or message.content is iterable %}\n            {{- message.content | tojson }}\n        {%- else %}\n            {{- message.content }}\n        {%- endif %}\n        {{- \"<|eot_id|>\" }}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}\n{%- endif %}\n"""


@PromptRegistry.register("microsoft/phi-4")
def microsoft_phi_chat_template():
    return """{% for message in messages %}{% if (message['role'] == 'system') %}{{'<|im_start|>system<|im_sep|>' + message['content'] + '<|im_end|>'}}{% elif (message['role'] == 'user') %}{{'<|im_start|>user<|im_sep|>' + message['content'] + '<|im_end|>'}}{% elif (message['role'] == 'assistant') %}{{'<|im_start|>assistant<|im_sep|>' + message['content'] + '<|im_end|>'}}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant<|im_sep|>' }}{% endif %}"""


@PromptRegistry.register("nvidia/Llama-3_3-Nemotron-Super-49B-v1")
def nemotron_chat_template():
    """
    Format chat messages for the Nemotron model, including a system prompt and structured message headers.

    The template starts with a system message containing "detailed thinking on", then iterates over messages, wrapping each with start and end header tokens and an end-of-text token. For assistant messages containing a `</think>` tag, only the content after this tag is included. Optionally appends an assistant prompt if generation is requested.
    """
    return """{{- bos_token }}
{{- "<|start_header_id|>system<|end_header_id|>\n\n" }}detailed thinking on{{- "<|eot_id|>" }}
{%- for message in messages %}
  {%- if message['role'] == 'assistant' and '</think>' in message['content'] %}
    {%- set content = message['content'].split('</think>')[-1].lstrip() %}
  {%- else %}
    {%- set content = message['content'] %}
  {%- endif %}
  {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + content | trim + '<|eot_id|>' }}
{%- endfor %}
{%- if add_generation_prompt %}
  {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
{%- endif %}"""


@PromptRegistry.register("Qwen/Qwen2.5")
def qwen_2_5_chat_template():
    """
    Formats chat messages into the prompt structure required by the Qwen 2.5 model family, supporting system messages, tool descriptions, function call instructions, and role-based message formatting.

    If tools are provided, includes tool signatures and instructions for function calls in the system prompt. User, assistant, and tool messages are wrapped with special tokens, and assistant tool calls are serialized as JSON within XML tags. Optionally appends a generation prompt for the assistant.
    """
    return """{%- if tools %}\n    {{- \'<|im_start|>system\\n\' }}\n    {%- if messages[0][\'role\'] == \'system\' %}\n        {{- messages[0][\'content\'] }}\n    {%- else %}\n        {{- \'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\' }}\n    {%- endif %}\n    {{- "\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>" }}\n    {%- for tool in tools %}\n        {{- "\\n" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- "\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\"name\\": <function-name>, \\"arguments\\": <args-json-object>}\\n</tool_call><|im_end|>\\n" }}\n{%- else %}\n    {%- if messages[0][\'role\'] == \'system\' %}\n        {{- \'<|im_start|>system\\n\' + messages[0][\'content\'] + \'<|im_end|>\\n\' }}\n    {%- else %}\n        {{- \'<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n\' }}\n    {%- endif %}\n{%- endif %}\n{%- for message in messages %}\n    {%- if (message.role == "user") or (message.role == "system" and not loop.first) or (message.role == "assistant" and not message.tool_calls) %}\n        {{- \'<|im_start|>\' + message.role + \'\\n\' + message.content + \'<|im_end|>\' + \'\\n\' }}\n    {%- elif message.role == "assistant" %}\n        {{- \'<|im_start|>\' + message.role }}\n        {%- if message.content %}\n            {{- \'\\n\' + message.content }}\n        {%- endif %}\n        {%- for tool_call in message.tool_calls %}\n            {%- if tool_call.function is defined %}\n                {%- set tool_call = tool_call.function %}\n            {%- endif %}\n            {{- \'\\n<tool_call>\\n{"name": "\' }}\n            {{- tool_call.name }}\n            {{- \'", "arguments": \' }}\n            {{- tool_call.arguments | tojson }}\n            {{- \'}\\n</tool_call>\' }}\n        {%- endfor %}\n        {{- \'<|im_end|>\\n\' }}\n    {%- elif message.role == "tool" %}\n        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != "tool") %}\n            {{- \'<|im_start|>user\' }}\n        {%- endif %}\n        {{- \'\\n<tool_response>\\n\' }}\n        {{- message.content }}\n        {{- \'\\n</tool_response>\' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}\n            {{- \'<|im_end|>\\n\' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- \'<|im_start|>assistant\\n\' }}\n{%- endif %}\n"""


@PromptRegistry.register("Qwen/Qwen3")
def qwen_3_chat_template():
    """
    Formats chat messages for the Qwen 3 model family, supporting multi-step tool usage, reasoning content, and special XML tags for tool calls and responses.

    This template handles system messages, user and assistant roles, and tool interactions. When tools are provided, it outputs their signatures and instructions for function calls. It tracks the last user query to determine where to insert assistant reasoning content within `<think>` tags. Assistant tool calls are serialized as JSON within `<tool_call>` tags, and tool responses are grouped inside `<tool_response>` tags. Optionally, a generation prompt and empty reasoning block can be added.

    Parameters:
        tools (optional): List of tool signature objects to be included in the prompt.
        messages: List of message objects, each with a role and content, and optionally tool_calls or reasoning_content.
        add_generation_prompt (optional): If true, appends an assistant prompt for generation.
        enable_thinking (optional): If false, inserts an empty reasoning block in the assistant prompt.
    """
    return """{%- if tools %}\n    {{- \'<|im_start|>system\\n\' }}\n    {%- if messages[0].role == \'system\' %}\n        {{- messages[0].content + \'\\n\\n\' }}\n    {%- endif %}\n    {{- "# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>" }}\n    {%- for tool in tools %}\n        {{- "\\n" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- "\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\"name\\": <function-name>, \\"arguments\\": <args-json-object>}\\n</tool_call><|im_end|>\\n" }}\n{%- else %}\n    {%- if messages[0].role == \'system\' %}\n        {{- \'<|im_start|>system\\n\' + messages[0].content + \'<|im_end|>\\n\' }}\n    {%- endif %}\n{%- endif %}\n{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}\n{%- for message in messages[::-1] %}\n    {%- set index = (messages|length - 1) - loop.index0 %}\n    {%- if ns.multi_step_tool and message.role == "user" and message.content is string and not(message.content.startswith(\'<tool_response>\') and message.content.endswith(\'</tool_response>\')) %}\n        {%- set ns.multi_step_tool = false %}\n        {%- set ns.last_query_index = index %}\n    {%- endif %}\n{%- endfor %}\n{%- for message in messages %}\n    {%- if message.content is string %}\n        {%- set content = message.content %}\n    {%- else %}\n        {%- set content = \'\' %}\n    {%- endif %}\n    {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}\n        {{- \'<|im_start|>\' + message.role + \'\\n\' + content + \'<|im_end|>\' + \'\\n\' }}\n    {%- elif message.role == "assistant" %}\n        {%- set reasoning_content = \'\' %}\n        {%- if message.reasoning_content is string %}\n            {%- set reasoning_content = message.reasoning_content %}\n        {%- else %}\n            {%- if \'</think>\' in content %}\n                {%- set reasoning_content = content.split(\'</think>\')[0].rstrip(\'\\n\').split(\'<think>\')[-1].lstrip(\'\\n\') %}\n                {%- set content = content.split(\'</think>\')[-1].lstrip(\'\\n\') %}\n            {%- endif %}\n        {%- endif %}\n        {%- if loop.index0 > ns.last_query_index %}\n            {%- if loop.last or (not loop.last and reasoning_content) %}\n                {{- \'<|im_start|>\' + message.role + \'\\n<think>\\n\' + reasoning_content.strip(\'\\n\') + \'\\n</think>\\n\\n\' + content.lstrip(\'\\n\') }}\n            {%- else %}\n                {{- \'<|im_start|>\' + message.role + \'\\n\' + content }}\n            {%- endif %}\n        {%- else %}\n            {{- \'<|im_start|>\' + message.role + \'\\n\' + content }}\n        {%- endif %}\n        {%- if message.tool_calls %}\n            {%- for tool_call in message.tool_calls %}\n                {%- if (loop.first and content) or (not loop.first) %}\n                    {{- \'\\n\' }}\n                {%- endif %}\n                {%- if tool_call.function %}\n                    {%- set tool_call = tool_call.function %}\n                {%- endif %}\n                {{- \'<tool_call>\\n{"name": "\' }}\n                {{- tool_call.name }}\n                {{- \'", "arguments": \' }}\n                {%- if tool_call.arguments is string %}\n                    {{- tool_call.arguments }}\n                {%- else %}\n                    {{- tool_call.arguments | tojson }}\n                {%- endif %}\n                {{- \'}\\n</tool_call>\' }}\n            {%- endfor %}\n        {%- endif %}\n        {{- \'<|im_end|>\\n\' }}\n    {%- elif message.role == "tool" %}\n        {%- if loop.first or (messages[loop.index0 - 1].role != "tool") %}\n            {{- \'<|im_start|>user\' }}\n        {%- endif %}\n        {{- \'\\n<tool_response>\\n\' }}\n        {{- content }}\n        {{- \'\\n</tool_response>\' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}\n            {{- \'<|im_end|>\\n\' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- \'<|im_start|>assistant\\n\' }}\n    {%- if enable_thinking is defined and enable_thinking is false %}\n        {{- \'<think>\\n\\n</think>\\n\\n\' }}\n    {%- endif %}\n{%- endif %}"""


@PromptRegistry.register("mistralai/Mistral-Small-3")
def mistral_small_3_chat_template():
    return """{%- if not date_string is defined %}\n    {%- set date_string = \"2025-01-01\" %}\n{%- endif %}\n{%- set default_system_message = \"You are Mistral Small 3, a Large Language Model (LLM) created by Mistral AI, a French startup headquartered in Paris.\\nYour knowledge base was last updated on 2023-10-01. The current date is \" + date_string + \".\\n\\nWhen you're not sure about some information, you say that you don't have the information and don't make up anything.\\nIf the user's question is not clear, ambiguous, or does not provide enough context for you to accurately answer the question, you do not try to answer it right away and you rather ask the user to clarify their request (e.g. \\\"What are some good restaurants around me?\\\" => \\\"Where are you?\\\" or \\\"When is the next flight to Tokyo\\\" => \\\"Where do you travel from?\\\")\" %}\n\n<s>\n\n{%- if messages[0]['role'] == 'system' %}\n    {%- set system_message = messages[0]['content'] %}\n    {%- set loop_messages = messages[1:] %}\n{%- else %}\n    {%- set system_message = default_system_message %}\n    {%- set loop_messages = messages %}\n{%- endif %}\n{{- '[SYSTEM_PROMPT]' + system_message + '[/SYSTEM_PROMPT]' }}\n\n{%- for message in loop_messages %}\n    {%- if message['role'] == 'user' %}\n        {{- '[INST]' + message['content'] + '[/INST]' }}\n    {%- elif message['role'] == 'system' %}\n        {{- '[SYSTEM_PROMPT]' + message['content'] + '[/SYSTEM_PROMPT]' }}\n    {%- elif message['role'] == 'assistant' %}\n        {{- message['content'] + '</s>' }}\n    {%- else %}\n        {{- raise_exception('Only user, system and assistant roles are supported!') }}\n    {%- endif %}\n{%- endfor %}"""

  
@PromptRegistry.register("ibm-granite/granite-3.3")
def granite_3_3_chat_template():
    """
    Granite 3.3 2B and 8B Chat Template
    """
    return """
{# Alias tools -> available_tools #}
{%- if tools and not available_tools -%}
    {%- set available_tools = tools -%}
{%- endif -%}

{%- if messages[0]['role'] == 'system' %}
     {%- set system_message = messages[0]['content'] %}
     {%- set loop_messages = messages[1:] %}
 {%- else %}
     {%- set system_message = " Knowledge Cutoff Date: April 2024.\n. You are Granite, developed by IBM." %}
     {%- if available_tools and documents %}
         {%- set system_message = system_message + " You are a helpful assistant with access to the following tools. When a tool is required to answer the user's query, respond only with <|tool_call|> followed by a JSON list of tools used. If a tool does not exist in the provided list of tools, notify the user that you do not have the ability to fulfill the request. \nWrite the response to the user's input by strictly aligning with the facts in the provided documents. If the information needed to answer the question is not available in the documents, inform the user that the question cannot be answered based on the available data." %}
     {%- elif available_tools %}
         {%- set system_message = system_message + " You are a helpful assistant with access to the following tools. When a tool is required to answer the user's query, respond only with <|tool_call|> followed by a JSON list of tools used. If a tool does not exist in the provided list of tools, notify the user that you do not have the ability to fulfill the request." %}
     {%- elif documents %}
         {%- set system_message = system_message + " Write the response to the user's input by strictly aligning with the facts in the provided documents. If the information needed to answer the question is not available in the documents, inform the user that the question cannot be answered based on the available data." %}
    {%- elif thinking %}
    {%- set system_message = system_message + " You are a helpful AI assistant.\nRespond to every user query in a comprehensive and detailed way. You can write down your thoughts and reasoning process before responding. In the thought process, engage in a comprehensive cycle of analysis, summarization, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. In the response section, based on various attempts, explorations, and reflections from the thoughts section, systematically present the final solution that you deem correct. The response should summarize the thought process. Write your thoughts between <think></think> and write your response between <response></response> for each user query." %}
     {%- else %}
         {%- set system_message = system_message + " You are a helpful AI assistant." %}
     {%- endif %}
     {%- if 'citations' in controls and documents %}
         {%- set system_message = system_message + ' \nUse the symbols <|start_of_cite|> and <|end_of_cite|> to indicate when a fact comes from a document in the search result, e.g <|start_of_cite|> {document_id: 1}my fact <|end_of_cite|> for a fact from document 1. Afterwards, list all the citations with their corresponding documents in an ordered list.' %}
     {%- endif %}
     {%- if 'hallucinations' in controls and documents %}
         {%- set system_message = system_message + ' \nFinally, after the response is written, include a numbered list of sentences from the response with a corresponding risk value that are hallucinated and not based in the documents.' %}
     {%- endif %}
     {%- set loop_messages = messages %}
 {%- endif %}
 {{- '<|start_of_role|>system<|end_of_role|>' + system_message + '<|end_of_text|>\n' }}
 {%- if available_tools %}
     {{- '<|start_of_role|>available_tools<|end_of_role|>' }}
     {{- available_tools | tojson(indent=4) }}
     {{- '<|end_of_text|>\n' }}
 {%- endif %}
 {%- if documents %}
     {%- for document in documents %}
         {{- '<|start_of_role|>document {"document_id": "' + document['doc_id'] | string + '"}<|end_of_role|>\n' }}
         {{- document['text'] }}
         {{- '<|end_of_text|>\n' }}
              {%- endfor %}
 {%- endif %}
 {%- for message in loop_messages %}
     {{- '<|start_of_role|>' + message['role'] + '<|end_of_role|>' + message['content'] + '<|end_of_text|>\n' }}
     {%- if loop.last and add_generation_prompt %}
         {{- '<|start_of_role|>assistant' }}
             {%- if controls %}
                 {{- ' ' + controls | tojson()}}
             {%- endif %}
         {{- '<|end_of_role|>' }}
     {%- endif %}
 {%- endfor %}
"""
