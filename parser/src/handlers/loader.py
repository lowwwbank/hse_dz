"""Data loader handler for reading CSV files."""

import pandas as pd

from src.handlers.base import BaseHandler, DataContext


class DataLoaderHandler(BaseHandler):
    """Handler for loading data from CSV files.

    This is typically the first handler in the chain.
    """

    def __init__(self, encoding: str = "utf-8") -> None:
        """Initialize the loader handler.

        Args:
            encoding: Character encoding for reading the CSV file.
        """
        super().__init__()
        self.encoding = encoding

    def process(self, context: DataContext) -> DataContext:
        """Load data from the CSV file specified in context.

        Args:
            context: The data context containing the input path.

        Returns:
            The context with loaded DataFrame.

        Raises:
            FileNotFoundError: If the input file doesn't exist.
            pd.errors.ParserError: If the CSV cannot be parsed.
        """
        context.log(self.name, f"Loading data from {context.input_path}")

        context.df = pd.read_csv(
            context.input_path,
            encoding=self.encoding,
            index_col=0,
            low_memory=False,
        )

        rows, cols = context.df.shape
        context.log(self.name, f"Loaded {rows:,} rows and {cols} columns")
        context.metadata["original_shape"] = (rows, cols)
        context.metadata["original_columns"] = list(context.df.columns)

        return context
