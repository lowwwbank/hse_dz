"""Data cleaner handler for preprocessing raw data."""

import pandas as pd

from src.handlers.base import BaseHandler, DataContext


class DataCleanerHandler(BaseHandler):
    """Handler for cleaning and preprocessing data.

    Performs the following operations:
    - Removes duplicates
    - Handles missing values
    - Standardizes column names
    - Removes invalid entries
    """

    def __init__(
        self,
        drop_duplicates: bool = True,
        dropna_thresh: float = 0.5,
    ) -> None:
        """Initialize the cleaner handler.

        Args:
            drop_duplicates: Whether to remove duplicate rows.
            dropna_thresh: Minimum fraction of non-null values required to keep a row.
        """
        super().__init__()
        self.drop_duplicates = drop_duplicates
        self.dropna_thresh = dropna_thresh

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names.

        Args:
            df: DataFrame to process.

        Returns:
            DataFrame with standardized column names.
        """
        column_mapping = {
            "Пол, возраст": "gender_age",
            "ЗП": "salary",
            "Ищет работу на должность:": "position",
            "Город": "city",
            "Занятость": "employment",
            "График": "schedule",
            "Опыт (двойное нажатие для полной версии)": "experience",
            "Последенее/нынешнее место работы": "last_workplace",
            "Последеняя/нынешняя должность": "last_position",
            "Образование и ВУЗ": "education",
            "Обновление резюме": "resume_updated",
            "Авто": "has_car",
        }

        df = df.rename(columns=column_mapping)
        return df

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows.

        Args:
            df: DataFrame to process.

        Returns:
            DataFrame without duplicates.
        """
        initial_count = len(df)
        df = df.drop_duplicates()
        removed = initial_count - len(df)

        if removed > 0:
            print(f"  Removed {removed:,} duplicate rows")

        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the DataFrame.

        Args:
            df: DataFrame to process.

        Returns:
            DataFrame with handled missing values.
        """
        # Calculate threshold for minimum non-null values per row
        min_count = int(self.dropna_thresh * len(df.columns))

        initial_count = len(df)
        df = df.dropna(thresh=min_count)
        removed = initial_count - len(df)

        if removed > 0:
            print(f"  Removed {removed:,} rows with too many missing values")

        return df

    def process(self, context: DataContext) -> DataContext:
        """Clean and preprocess the data.

        Args:
            context: The data context containing the DataFrame.

        Returns:
            The context with cleaned DataFrame.

        Raises:
            ValueError: If context.df is None.
        """
        if context.df is None:
            raise ValueError("No DataFrame to clean. Run DataLoaderHandler first.")

        context.log(self.name, "Cleaning data...")

        df = context.df.copy()

        # Standardize column names
        df = self._standardize_columns(df)

        # Remove duplicates if requested
        if self.drop_duplicates:
            df = self._remove_duplicates(df)

        # Handle missing values
        df = self._handle_missing_values(df)

        # Remove rows where salary is missing (target variable)
        initial_count = len(df)
        df = df.dropna(subset=["salary"])
        removed = initial_count - len(df)
        if removed > 0:
            print(f"  Removed {removed:,} rows with missing salary")

        context.df = df
        context.log(self.name, f"After cleaning: {len(df):,} rows")
        context.metadata["cleaned_shape"] = df.shape

        return context
