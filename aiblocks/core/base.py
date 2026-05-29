"""Abstract base classes for all aiblocks modules and their configs."""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class BaseConfig(BaseModel):
    """Base configuration model for all aiblocks modules."""

    class Config:
        extra = "forbid"
        validate_assignment = True


class BaseModule(ABC):
    """Abstract base class that every aiblocks module must implement."""

    def __init__(self, config: BaseConfig) -> None:
        self.config = config

    @abstractmethod
    def build(self) -> Any:
        """Build and return the underlying pipeline or component."""

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the module's primary action."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(config={self.config})"
