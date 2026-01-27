"""Base handler for Chain of Responsibility pattern."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class DataContext:
    """Context object passed through the chain of handlers.

    Attributes:
        input_path: Path to the input CSV file.
        output_dir: Directory where output files will be saved.
        df: DataFrame being processed (populated during pipeline execution).
        x_data: Feature matrix (populated by splitter handler).
        y_data: Target vector (populated by splitter handler).
        metadata: Additional metadata from processing steps.
    """

    input_path: str
    output_dir: str
    df: pd.DataFrame | None = None
    x_data: Any = None
    y_data: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def log(self, handler_name: str, message: str) -> None:
        """Log a message from a handler.

        Args:
            handler_name: Name of the handler logging the message.
            message: Message to log.
        """
        print(f"[{handler_name}] {message}")


class BaseHandler(ABC):
    """Abstract base handler for Chain of Responsibility pattern.

    Each handler processes data and passes it to the next handler in the chain.
    """

    def __init__(self) -> None:
        """Initialize the handler."""
        self._next_handler: BaseHandler | None = None

    @property
    def name(self) -> str:
        """Return the handler name."""
        return self.__class__.__name__

    def set_next(self, handler: BaseHandler) -> BaseHandler:
        """Set the next handler in the chain.

        Args:
            handler: The next handler to process data.

        Returns:
            The handler that was set as next (for chaining).
        """
        self._next_handler = handler
        return handler

    def handle(self, context: DataContext) -> DataContext:
        """Process the data and pass to next handler.

        Args:
            context: The data context to process.

        Returns:
            The processed data context.
        """
        context = self.process(context)

        if self._next_handler:
            return self._next_handler.handle(context)

        return context

    @abstractmethod
    def process(self, context: DataContext) -> DataContext:
        """Process the data context.

        Args:
            context: The data context to process.

        Returns:
            The processed data context.
        """
        pass
