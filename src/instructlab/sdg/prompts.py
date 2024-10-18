# Local
from .registry import PromptRegistry


@PromptRegistry.register("blank")
def blank_chat_template():
    return """{{ messages }}"""


@PromptRegistry.register("instructlab")
def instructlab_chat_template():
    return """{% for message in messages %}{% if message['role'] == 'pretraining' %}{{ '<|pretrain|>' + message['content'] + '<|endoftext|>' + '<|/pretrain|>' }}{% elif message['role'] == 'system' %}{{ '<|system|>' + '\n' + message['content'] + '\n' }}{% elif message['role'] == 'user' %}{{ '<|user|>' + '\n' + message['content'] + '\n' }}{% elif message['role'] == 'assistant' %}{{ '<|assistant|>' + '\n' + message['content'] + '<|endoftext|>' + ('' if loop.last else '\n') }}{% endif %}{% if loop.last and add_generation_prompt %}{{ '<|assistant|>' + '\n' }}{% endif %}{% endfor %}"""


@PromptRegistry.register("mistralai")
def mistral_chat_template():
    return """{%- if messages[0]['role'] == 'system' %}\n    {%- set system_message = messages[0]['content'] %}\n    {%- set loop_messages = messages[1:] %}\n{%- else %}\n    {%- set loop_messages = messages %}\n{%- endif %}\n\n<s>\n{%- for message in loop_messages %}\n    {%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}\n        {{- raise_exception('After the optional system message, conversation roles must alternate user/assistant/user/assistant/...') }}\n    {%- endif %}\n    {%- if message['role'] == 'user' %}\n        {%- if loop.first and system_message is defined %}\n            {{- ' [INST] ' + system_message + '\\n\\n' + message['content'] + ' [/INST]' }}\n        {%- else %}\n            {{- ' [INST] ' + message['content'] + ' [/INST]' }}\n        {%- endif %}\n    {%- elif message['role'] == 'assistant' %}\n        {{- ' ' + message['content'] + '</s>'}}\n    {%- else %}\n        {{- raise_exception('Only user and assistant roles are supported, with the exception of an initial optional system message!') }}\n    {%- endif %}\n{%- endfor %}\n"""
