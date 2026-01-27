"""Pipeline builder for constructing handler chains."""

from src.handlers.base import BaseHandler, DataContext
from src.handlers.cleaner import DataCleanerHandler
from src.handlers.encoder import EncoderHandler
from src.handlers.feature_extractor import FeatureExtractorHandler
from src.handlers.loader import DataLoaderHandler
from src.handlers.splitter import DataSplitterHandler


class Pipeline:
    """Builder class for constructing a chain of handlers.

    This class provides a fluent interface for building processing pipelines
    using the Chain of Responsibility pattern.
    """

    def __init__(self) -> None:
        """Initialize an empty pipeline."""
        self._first_handler: BaseHandler | None = None
        self._last_handler: BaseHandler | None = None
        self._handlers: list[BaseHandler] = []

    def add(self, handler: BaseHandler) -> "Pipeline":
        """Add a handler to the pipeline.

        Args:
            handler: The handler to add.

        Returns:
            Self for method chaining.
        """
        self._handlers.append(handler)

        if self._first_handler is None:
            self._first_handler = handler
            self._last_handler = handler
        else:
            if self._last_handler is not None:
                self._last_handler.set_next(handler)
            self._last_handler = handler

        return self

    def execute(self, context: DataContext) -> DataContext:
        """Execute the pipeline on the given context.

        Args:
            context: The data context to process.

        Returns:
            The processed data context.

        Raises:
            ValueError: If the pipeline is empty.
        """
        if self._first_handler is None:
            raise ValueError("Pipeline is empty. Add handlers before executing.")

        print("=" * 60)
        print("Starting pipeline execution")
        print("=" * 60)

        result = self._first_handler.handle(context)

        print("=" * 60)
        print("Pipeline execution completed")
        print("=" * 60)

        return result

    def __len__(self) -> int:
        """Return the number of handlers in the pipeline."""
        return len(self._handlers)


def create_default_pipeline() -> Pipeline:
    """Create the default data processing pipeline.

    The default pipeline includes:
    1. DataLoaderHandler - Load CSV data
    2. DataCleanerHandler - Clean and preprocess
    3. FeatureExtractorHandler - Extract features from raw text
    4. EncoderHandler - Encode categorical features
    5. DataSplitterHandler - Split into X/y and save

    Returns:
        Configured Pipeline instance.
    """
    pipeline = Pipeline()

    pipeline.add(DataLoaderHandler())
    pipeline.add(DataCleanerHandler())
    pipeline.add(FeatureExtractorHandler())
    pipeline.add(EncoderHandler())
    pipeline.add(DataSplitterHandler())

    return pipeline
