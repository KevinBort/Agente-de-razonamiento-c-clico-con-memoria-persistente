from langchain_core.language_models import BaseChatModel

from agente.config import Settings


def crear_llm(settings: Settings) -> BaseChatModel:
    if settings.proveedor == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(  # type: ignore[call-arg]
            model=settings.modelo,
            api_key=settings.api_key,  # type: ignore[arg-type]
            temperature=0,
        )

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=settings.modelo,
        api_key=settings.api_key,  # type: ignore[arg-type]
        temperature=0,
    )
