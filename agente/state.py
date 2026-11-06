import operator
from typing import Annotated

from langgraph.graph import MessagesState


class AgentState(MessagesState):
    llamadas_herramientas: Annotated[int, operator.add]
