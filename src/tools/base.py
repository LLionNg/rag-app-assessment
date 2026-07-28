from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from src.core.types import ToolOutput, ToolSpec


class Tool(ABC):
    """Base tool exposed to an agent.

    `name`, `description` and `parameters` form the JSON-schema declaration the
    model sees; `run` performs the work and returns both the text the model
    reads and any structured payload the calling agent needs.
    """

    name: ClassVar[str]
    description: ClassVar[str]
    parameters: ClassVar[dict[str, Any]]

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
        )

    @abstractmethod
    async def run(self, **kwargs: Any) -> ToolOutput: ...
