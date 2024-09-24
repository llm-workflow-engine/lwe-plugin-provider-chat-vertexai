from langchain_google_vertexai import ChatVertexAI, HarmBlockThreshold, HarmCategory

from lwe.core.provider import Provider, PresetValue

DEFAULT_GOOGLE_VERTEXAI_MODEL = 'gemini-1.0-pro'
HARM_BLOCK_THRESHOLD_OPTIONS = [
    'BLOCK_NONE',
    'BLOCK_LOW_AND_ABOVE',
    'BLOCK_MEDIUM_AND_ABOVE',
    'BLOCK_ONLY_HIGH',
    'HARM_BLOCK_THRESHOLD_UNSPECIFIED',
]


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
            'validate_models': False,
        }

    @property
    def default_model(self):
        return DEFAULT_GOOGLE_VERTEXAI_MODEL

    @property
    def static_models(self):
        return {
            'gemini-1.5-pro': {
                "max_tokens": 2097152,
            },
            'gemini-1.5-pro-001': {
                "max_tokens": 2097152,
            },
            'gemini-1.5-pro-002': {
                "max_tokens": 2097152,
            },
            'gemini-1.5-flash': {
                "max_tokens": 2097152,
            },
            'gemini-1.5-flash-001': {
                "max_tokens": 2097152,
            },
            'gemini-1.5-flash-002': {
                "max_tokens": 2097152,
            },
        }

    def prepare_messages_method(self):
        return self.prepare_messages_for_llm_chat

    def llm_factory(self):
        return CustomChatVertexAI

    def configure_safety_settings(self, configured_safety_settings):
        safety_settings = {}
        for category, threshold in configured_safety_settings.items():
            try:
                harm_category = getattr(HarmCategory, category)
                harm_block_threshold = getattr(HarmBlockThreshold, threshold)
            except AttributeError as e:
                raise ValueError(f"Invalid HarmCategory or HarmBlockThreshold: {e}")
            safety_settings[harm_category] = harm_block_threshold
        return safety_settings

    def make_llm(self, customizations=None, tools=None, tool_choice=None, use_defaults=False):
        customizations = customizations or {}
        final_customizations = self.get_customizations()
        final_customizations.update(customizations)
        configured_safety_settings = final_customizations.pop('safety_settings', None)
        if configured_safety_settings:
            final_customizations['safety_settings'] = self.configure_safety_settings(configured_safety_settings)
        return super().make_llm(final_customizations, tools=tools, tool_choice=tool_choice, use_defaults=use_defaults)

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
            'convert_system_message_to_human': PresetValue(bool),
            'safety_settings': {
                'HARM_CATEGORY_DANGEROUS_CONTENT': PresetValue(str, options=HARM_BLOCK_THRESHOLD_OPTIONS, include_none=True),
                'HARM_CATEGORY_HATE_SPEECH': PresetValue(str, options=HARM_BLOCK_THRESHOLD_OPTIONS, include_none=True),
                'HARM_CATEGORY_HARASSMENT': PresetValue(str, options=HARM_BLOCK_THRESHOLD_OPTIONS, include_none=True),
                'HARM_CATEGORY_SEXUALLY_EXPLICIT': PresetValue(str, options=HARM_BLOCK_THRESHOLD_OPTIONS, include_none=True),
            },
            "tools": None,
            "tool_choice": None,
        }
