from langchain_community.chat_models.vertexai import ChatVertexAI

# TODO: Remove this once Gemini supports system messages.
from langchain.adapters.openai import convert_dict_to_message
from langchain.schema.messages import SystemMessage
# TODO: Remove this once Gemini supports system messages.


from lwe.core.provider import Provider, PresetValue

class CustomChatVertexAI(ChatVertexAI):

    @property
    def _llm_type(self):
        """Return type of llm."""
        return "chat_vertexai"

class ProviderChatVertexai(Provider):
    """
    Access to chat Vertex AI models
    """

    @property
    def capabilities(self):
        return {
            "chat": True,
            'validate_models': True,
            'models': {
                'chat-bison': {
                    'max_tokens': 4096,
                },
                'codechat-bison': {
                    'max_tokens': 6144,
                },
                'gemini-pro': {
                    'max_tokens': 32768,
                },
            },
        }

    @property
    def default_model(self):
        return 'chat-bison'

    # TODO: Remove this once Gemini supports system messages.
    def prepare_messages_for_llm_chat(self, messages):
        messages = [convert_dict_to_message(m) for m in messages]
        if self.get_model() == 'gemini-pro':
            messages = [m for m in messages if not isinstance(m, SystemMessage)]
        return messages

    def prepare_messages_method(self):
        return self.prepare_messages_for_llm_chat

    def llm_factory(self):
        return CustomChatVertexAI

    def customization_config(self):
        return {
            'model_name': PresetValue(str, options=self.available_models),
            'temperature': PresetValue(float, min_value=0.0, max_value=1.0),
            'max_output_tokens': PresetValue(int, min_value=1, max_value=2048, include_none=True),
            'top_k': PresetValue(int, min_value=1, max_value=40),
            'top_p': PresetValue(float, min_value=0.0, max_value=1.0),
            'project': PresetValue(str, include_none=True),
            'location': PresetValue(str),
            'request_parallelism': PresetValue(int, min_value=1),
            'max_retries': PresetValue(int, min_value=1),
        }
