"""Data processor for filtering IT developers and extracting features."""

from __future__ import annotations

import csv
import re

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# Column mapping: original CSV header -> internal name
_COLUMN_MAP = {
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
    "Обновление резюме": "resume_update",
    "Авто": "has_car",
}


class DataProcessor:
    """Processor for HH.ru resume data.

    Filters IT developers and extracts features for classification.
    """

    # Keywords for identifying IT developers in the position field.
    DEVELOPER_KEYWORDS: list[str] = [
        # Developers
        "разработчик", "developer", "программист", "programmer",
        "backend", "frontend", "fullstack", "full-stack", "full stack",
        "devops", "sre",
        # Data / ML
        "data scientist", "data engineer", "ml engineer",
        "аналитик данных",
        # QA
        "qa engineer", "qa ", "тестировщик",
        # General software
        "software engineer", "software developer",
        # 1C
        "1с", "1c",
        # Web
        "web-программист", "web программист", "web-разработчик",
        "веб-разработчик", "верстальщик", "html",
        # Embedded / system
        "embedded", "встраиваемых", "системный программист",
        # DBA
        "dba",
    ]

    LANGUAGE_KEYWORDS: list[str] = [
        "python", "java", "javascript", "c++", "c#", "golang",
        "php", "ruby", "swift", "kotlin", "scala", "rust",
        "typescript", ".net", "react", "angular", "vue", "node.js",
        "laravel", "django", "delphi", "oracle", "axapta",
    ]

    # ----- Level detection keywords -----

    JUNIOR_KEYWORDS: list[str] = [
        "junior", "джуниор", "джун", "стажер", "стажёр", "intern",
        "начинающий", "trainee", "помощник программиста",
        "помощник системного", "младший",
    ]

    MIDDLE_KEYWORDS: list[str] = [
        "middle", "миддл", "мидл",
    ]

    SENIOR_KEYWORDS: list[str] = [
        "senior", "сеньор", "синьор", "lead", "лид",
        "principal", "staff", "архитектор",
        "team lead", "teamlead", "тимлид", "tech lead",
        "head of", "director", "директор по ит",
        "директор по информационным", "cto", "cio",
        "начальник ит", "начальник отдела it",
        "начальник отдела информационных",
        "руководитель it", "руководитель ит",
        "ведущий программист", "ведущий разработчик",
        "ведущий инженер",
        "старший программист", "старший разработчик",
    ]

    # Experience thresholds (months) for fallback labeling
    JUNIOR_MAX_MONTHS: int = 24   # <= 2 years
    SENIOR_MIN_MONTHS: int = 72   # >= 6 years

    def __init__(self) -> None:
        """Initialize the data processor."""
        self._label_encoder = LabelEncoder()
        self._city_encoder = LabelEncoder()

    def load_data(self, path: str) -> pd.DataFrame:
        """Load data from CSV file using csv.reader for robust multiline handling.

        Args:
            path: Path to CSV file.

        Returns:
            Loaded DataFrame.
        """
        rows: list[list[str]] = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            n_cols = len(header)
            for row in reader:
                if len(row) == n_cols:
                    rows.append(row)

        df = pd.DataFrame(rows, columns=header)

        # Drop unnamed index column
        if "" in df.columns:
            df = df.drop(columns=[""])

        # Rename columns using the mapping
        df = df.rename(columns=_COLUMN_MAP)

        return df

    def filter_it_developers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter only IT developers from the dataset.

        Checks the ``position`` field for developer-related keywords
        and programming language names.

        Args:
            df: Input DataFrame.

        Returns:
            Filtered DataFrame with IT developers only.
        """
        positions = df["position"].fillna("").str.lower()

        has_dev = positions.apply(
            lambda p: any(kw in p for kw in self.DEVELOPER_KEYWORDS)
        )
        has_lang = positions.apply(
            lambda p: any(kw in p for kw in self.LANGUAGE_KEYWORDS)
        )

        return df[has_dev | has_lang].copy()

    def determine_level(self, row: pd.Series) -> str | None:
        """Determine developer level from position keywords and experience.

        Strategy:
        1. Look for explicit level keywords in position / last_position.
        2. If no keywords found, fall back to experience-based heuristic.

        Args:
            row: DataFrame row.

        Returns:
            Level string ('junior', 'middle', 'senior') or None.
        """
        position = str(row.get("position", "")).lower() if pd.notna(row.get("position")) else ""
        last_position = str(row.get("last_position", "")).lower() if pd.notna(row.get("last_position")) else ""
        combined = f"{position} {last_position}"

        # 1) Explicit keywords — priority: junior > senior > middle
        for kw in self.JUNIOR_KEYWORDS:
            if kw in combined:
                return "junior"

        for kw in self.SENIOR_KEYWORDS:
            if kw in combined:
                return "senior"

        for kw in self.MIDDLE_KEYWORDS:
            if kw in combined:
                return "middle"

        # 2) Fallback: experience-based heuristic
        exp = str(row.get("experience", "")).lower() if pd.notna(row.get("experience")) else ""
        months = self._extract_experience_months(exp)
        if months is not None:
            if months <= self.JUNIOR_MAX_MONTHS:
                return "junior"
            if months >= self.SENIOR_MIN_MONTHS:
                return "senior"
            return "middle"

        return None

    def extract_features(
        self, df: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Extract features and target from DataFrame.

        Args:
            df: DataFrame with IT developers.

        Returns:
            Tuple of (X features, y labels, feature names).
        """
        df = df.copy()
        df["level"] = df.apply(self.determine_level, axis=1)
        df = df[df["level"].notna()].copy()

        features = pd.DataFrame(index=df.index)

        features["gender"] = df["gender_age"].apply(self._extract_gender)
        features["age"] = df["gender_age"].apply(self._extract_age)
        features["salary"] = df["salary"].apply(self._parse_salary)

        cities = df["city"].apply(self._extract_city)
        self._city_encoder.fit(cities.astype(str))
        features["city_encoded"] = self._city_encoder.transform(cities.astype(str))

        features["experience_months"] = df["experience"].apply(
            lambda x: self._extract_experience_months(str(x).lower()) if pd.notna(x) else 0
        ).fillna(0)

        features["is_full_time"] = df["employment"].apply(
            lambda x: 1 if pd.notna(x) and "полная" in str(x).lower() else 0
        )

        features["education_level"] = df["education"].apply(
            self._extract_education_level
        )

        features["has_car"] = df["has_car"].apply(
            lambda x: 1 if pd.notna(x) and "имеется" in str(x).lower() else 0
        )

        features = features.fillna(0)

        self._label_encoder.fit(["junior", "middle", "senior"])
        y = self._label_encoder.transform(df["level"])

        feature_names = list(features.columns)
        X = features.values.astype(np.float32)

        return X, y, feature_names

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_experience_months(text: str) -> int | None:
        """Extract total experience in months from text."""
        if not text or text == "nan":
            return None

        years_match = re.search(r"опыт работы\s+(\d+)\s*(?:год|лет|года)", text)
        months_match = re.search(r"опыт работы\s+(?:\d+\s*(?:год|лет|года)\s+)?(\d+)\s*(?:месяц)", text)

        if not years_match and not months_match:
            years_match = re.search(r"(\d+)\s*(?:год|лет|года)", text)
            months_match = re.search(r"(\d+)\s*(?:месяц)", text)

        years = int(years_match.group(1)) if years_match else 0
        months = int(months_match.group(1)) if months_match else 0
        total = years * 12 + months

        return total if total > 0 else None

    @staticmethod
    def _extract_gender(value: object) -> int:
        """Extract gender: 1 = male, 0 = female, -1 = unknown."""
        if pd.isna(value):
            return -1
        text = str(value).lower()
        if "мужчина" in text:
            return 1
        if "женщина" in text:
            return 0
        return -1

    @staticmethod
    def _extract_age(value: object) -> int:
        """Extract age in years from ``gender_age`` string."""
        if pd.isna(value):
            return 0
        match = re.search(r"(\d+)\s*(?:год|лет|года)", str(value))
        return int(match.group(1)) if match else 0

    @staticmethod
    def _parse_salary(value: object) -> float:
        """Parse salary string to numeric value."""
        if pd.isna(value):
            return 0.0
        cleaned = str(value).replace(" ", "").replace("\xa0", "")
        match = re.search(r"(\d+)", cleaned)
        return float(match.group(1)) if match else 0.0

    @staticmethod
    def _extract_city(value: object) -> str:
        """Extract city name (first comma-separated token)."""
        if pd.isna(value):
            return "unknown"
        city = str(value).split(",")[0].strip()
        return city if city else "unknown"

    @staticmethod
    def _extract_education_level(value: object) -> int:
        """Ordinal-encode education level (0-4)."""
        if pd.isna(value):
            return 0
        text = str(value).lower()
        # Check more specific patterns first to avoid false matches.
        if "неоконченное высшее" in text or "незаконченное высшее" in text:
            return 3
        if "высшее" in text:
            return 4
        if "среднее специальное" in text or "средне-специальное" in text:
            return 2
        if "среднее" in text:
            return 1
        return 0

    @property
    def label_encoder(self) -> LabelEncoder:
        """Return the label encoder for target variable."""
        return self._label_encoder

    @property
    def city_encoder(self) -> LabelEncoder:
        """Return the city encoder."""
        return self._city_encoder
