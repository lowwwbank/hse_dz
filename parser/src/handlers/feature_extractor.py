"""Feature extractor handler for creating ML-ready features."""

import re
from typing import Any

import numpy as np
import pandas as pd

from src.handlers.base import BaseHandler, DataContext


class FeatureExtractorHandler(BaseHandler):
    """Handler for extracting features from raw text data.

    Extracts the following features:
    - Gender (from gender_age column)
    - Age (from gender_age column)
    - Salary (parsed to numeric)
    - City (extracted and cleaned)
    - Experience in months (parsed from text)
    - Education level (encoded)
    - Has car (boolean)
    """

    def _extract_gender(self, gender_age: str) -> int:
        """Extract gender from combined gender_age string.

        Args:
            gender_age: String containing gender and age info.

        Returns:
            1 for male, 0 for female, -1 for unknown.
        """
        if pd.isna(gender_age):
            return -1

        gender_age_lower = str(gender_age).lower()

        if "мужчина" in gender_age_lower:
            return 1
        elif "женщина" in gender_age_lower:
            return 0

        return -1

    def _extract_age(self, gender_age: str) -> int:
        """Extract age from combined gender_age string.

        Args:
            gender_age: String containing gender and age info.

        Returns:
            Age as integer, or -1 if not found.
        """
        if pd.isna(gender_age):
            return -1

        # Pattern: число + год/лет/года
        match = re.search(r"(\d+)\s*(?:год|лет|года)", str(gender_age))
        if match:
            return int(match.group(1))

        return -1

    def _parse_salary(self, salary: str) -> float:
        """Parse salary string to numeric value.

        Args:
            salary: Salary string (e.g., "60 000 руб.")

        Returns:
            Salary as float in rubles, or NaN if cannot be parsed.
        """
        if pd.isna(salary):
            return np.nan

        # Remove spaces to get the full number (e.g., "60 000" -> "60000")
        salary_str = str(salary).replace(" ", "").replace("\xa0", "")

        # Extract the number
        match = re.search(r"(\d+)", salary_str)

        if match:
            return float(match.group(1))

        return np.nan

    def _extract_city(self, city_str: str) -> str:
        """Extract city name from city string.

        Args:
            city_str: City string (may include relocation info).

        Returns:
            Cleaned city name.
        """
        if pd.isna(city_str):
            return "unknown"

        # Take the part before the first comma
        city = str(city_str).split(",")[0].strip()
        return city if city else "unknown"

    def _extract_experience_months(self, experience: str) -> int:
        """Extract total experience in months from experience text.

        Args:
            experience: Experience description text.

        Returns:
            Total experience in months, or 0 if not found.
        """
        if pd.isna(experience):
            return 0

        exp_str = str(experience)

        # Pattern: "Опыт работы X лет Y месяц"
        years_match = re.search(r"(\d+)\s*(?:год|лет|года)", exp_str)
        months_match = re.search(r"(\d+)\s*(?:месяц|месяцев|месяца)", exp_str)

        years = int(years_match.group(1)) if years_match else 0
        months = int(months_match.group(1)) if months_match else 0

        return years * 12 + months

    def _extract_education_level(self, education: str) -> int:
        """Extract education level as ordinal encoding.

        Args:
            education: Education description text.

        Returns:
            Education level (0-4):
                0 - Unknown
                1 - Secondary/Среднее
                2 - Specialized secondary/Среднее специальное
                3 - Incomplete higher/Неоконченное высшее
                4 - Higher/Высшее
        """
        if pd.isna(education):
            return 0

        edu_lower = str(education).lower()

        if "высшее образование" in edu_lower or "высшее" in edu_lower:
            return 4
        elif "неоконченное высшее" in edu_lower or "незаконченное высшее" in edu_lower:
            return 3
        elif "среднее специальное" in edu_lower or "средне-специальное" in edu_lower:
            return 2
        elif "среднее" in edu_lower:
            return 1

        return 0

    def _extract_has_car(self, has_car: Any) -> int:
        """Extract boolean flag for car ownership.

        Args:
            has_car: Car ownership info.

        Returns:
            1 if has car, 0 otherwise.
        """
        if pd.isna(has_car):
            return 0

        car_str = str(has_car).lower()

        if "имеется" in car_str or "есть" in car_str or "собственный" in car_str:
            return 1

        return 0

    def process(self, context: DataContext) -> DataContext:
        """Extract features from the raw data.

        Args:
            context: The data context containing the DataFrame.

        Returns:
            The context with feature-engineered DataFrame.

        Raises:
            ValueError: If context.df is None.
        """
        if context.df is None:
            raise ValueError(
                "No DataFrame to process. Run previous handlers first."
            )

        context.log(self.name, "Extracting features...")

        df = context.df.copy()

        # Extract features
        context.log(self.name, "  Extracting gender and age...")
        df["gender"] = df["gender_age"].apply(self._extract_gender)
        df["age"] = df["gender_age"].apply(self._extract_age)

        context.log(self.name, "  Parsing salary...")
        df["salary_numeric"] = df["salary"].apply(self._parse_salary)

        context.log(self.name, "  Extracting city...")
        df["city_clean"] = df["city"].apply(self._extract_city)

        context.log(self.name, "  Extracting experience...")
        df["experience_months"] = df["experience"].apply(
            self._extract_experience_months
        )

        context.log(self.name, "  Extracting education level...")
        df["education_level"] = df["education"].apply(self._extract_education_level)

        context.log(self.name, "  Extracting car ownership...")
        df["has_car_flag"] = df["has_car"].apply(self._extract_has_car)

        # Store feature columns for later use
        context.metadata["feature_columns"] = [
            "gender",
            "age",
            "city_clean",
            "experience_months",
            "education_level",
            "has_car_flag",
        ]
        context.metadata["target_column"] = "salary_numeric"

        context.df = df
        context.log(self.name, f"Extracted {len(context.metadata['feature_columns'])} features")

        return context
