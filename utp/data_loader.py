"""
Data loading and type inference module.

This module handles CSV file loading, column type detection,
and data preprocessing for visualization.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from enum import Enum
import warnings


class ColumnType(Enum):
    """Enumeration of detected column types."""
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    TEXT = "text"
    UNKNOWN = "unknown"


class DataLoader:
    """
    CSV data loader with automatic type inference.
    
    Loads CSV files and automatically detects column types
    (numeric, categorical, datetime) for appropriate visualization.
    """
    
    # Thresholds for type detection
    CATEGORICAL_THRESHOLD = 0.05  # Max unique ratio to be categorical
    MAX_CATEGORIES = 50  # Max unique values for categorical
    
    def __init__(self, filepath: Union[str, Path]):
        """
        Initialize data loader with a CSV file path.
        
        Args:
            filepath: Path to the CSV file
        """
        self.filepath = Path(filepath)
        self.data: Optional[pd.DataFrame] = None
        self.column_types: Dict[str, ColumnType] = {}
        self._original_columns: List[str] = []
    
    def load(
        self,
        encoding: str = "utf-8",
        fallback_encodings: List[str] = ["latin-1", "cp1252", "gbk"]
    ) -> pd.DataFrame:
        """
        Load the CSV file with automatic encoding detection.
        
        Args:
            encoding: Primary encoding to try
            fallback_encodings: List of encodings to try if primary fails
            
        Returns:
            Loaded DataFrame
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file cannot be parsed
        """
        if not self.filepath.exists():
            raise FileNotFoundError(f"File not found: {self.filepath}")
        
        # Try primary encoding
        encodings_to_try = [encoding] + fallback_encodings
        
        last_error = None
        for enc in encodings_to_try:
            try:
                self.data = pd.read_csv(
                    self.filepath,
                    encoding=enc,
                    low_memory=False
                )
                self._original_columns = list(self.data.columns)
                self._infer_column_types()
                return self.data
            except UnicodeDecodeError as e:
                last_error = e
                continue
            except Exception as e:
                raise ValueError(f"Error parsing CSV: {e}")
        
        raise ValueError(f"Could not decode file with any encoding. Last error: {last_error}")
    
    def _infer_column_types(self) -> None:
        """Infer types for all columns in the loaded data."""
        if self.data is None:
            return
        
        for col in self.data.columns:
            self.column_types[col] = self._detect_column_type(col)
    
    def _detect_column_type(self, column: str) -> ColumnType:
        """
        Detect the type of a single column.
        
        Args:
            column: Column name
            
        Returns:
            Detected ColumnType
        """
        if self.data is None:
            return ColumnType.UNKNOWN
        
        series = self.data[column]
        
        # Check for datetime
        if self._is_datetime(series):
            return ColumnType.DATETIME
        
        # Check for numeric
        if self._is_numeric(series):
            # Check if it's actually categorical (small set of integers)
            if self._is_categorical_numeric(series):
                return ColumnType.CATEGORICAL
            return ColumnType.NUMERIC
        
        # Check for categorical
        if self._is_categorical(series):
            return ColumnType.CATEGORICAL
        
        # Default to text
        return ColumnType.TEXT
    
    def _is_datetime(self, series: pd.Series) -> bool:
        """Check if series contains datetime values."""
        # Already datetime type
        if pd.api.types.is_datetime64_any_dtype(series):
            return True
        
        # Try to parse as datetime
        if series.dtype == object:
            try:
                # Sample for efficiency
                sample = series.dropna().head(100)
                if len(sample) == 0:
                    return False
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    parsed = pd.to_datetime(sample, errors="coerce")
                
                # If most values parse successfully, it's datetime
                success_rate = parsed.notna().sum() / len(sample)
                return success_rate > 0.8
            except Exception:
                return False
        
        return False
    
    def _is_numeric(self, series: pd.Series) -> bool:
        """Check if series is numeric."""
        # Already numeric type
        if pd.api.types.is_numeric_dtype(series):
            return True
        
        # Try to convert
        if series.dtype == object:
            try:
                numeric_series = pd.to_numeric(series, errors="coerce")
                success_rate = numeric_series.notna().sum() / max(len(series), 1)
                return success_rate > 0.8
            except Exception:
                return False
        
        return False
    
    def _is_categorical(self, series: pd.Series) -> bool:
        """Check if series should be treated as categorical."""
        # Already categorical type
        if pd.api.types.is_categorical_dtype(series):
            return True
        
        n_unique = series.nunique()
        n_total = len(series)
        
        if n_total == 0:
            return False
        
        # Check unique ratio and absolute count
        unique_ratio = n_unique / n_total
        return unique_ratio <= self.CATEGORICAL_THRESHOLD or n_unique <= self.MAX_CATEGORIES
    
    def _is_categorical_numeric(self, series: pd.Series) -> bool:
        """Check if a numeric series should be treated as categorical."""
        if not pd.api.types.is_numeric_dtype(series):
            return False
        
        n_unique = series.nunique()
        n_total = len(series)
        
        if n_total == 0:
            return False
        
        # Small set of unique integers
        if n_unique <= 10:
            # Check if all values are integers
            non_null = series.dropna()
            if len(non_null) == 0:
                return False
            if (non_null == non_null.astype(int)).all():
                return True
        
        return False
    
    def get_column_type(self, column: Union[str, int]) -> ColumnType:
        """
        Get the detected type of a column.
        
        Args:
            column: Column name or index
            
        Returns:
            ColumnType of the column
        """
        col_name = self._resolve_column(column)
        return self.column_types.get(col_name, ColumnType.UNKNOWN)
    
    def _resolve_column(self, column: Union[str, int]) -> str:
        """
        Resolve column identifier to column name.
        
        Args:
            column: Column name or index
            
        Returns:
            Column name as string
            
        Raises:
            ValueError: If column cannot be resolved
        """
        if self.data is None:
            raise ValueError("No data loaded")
        
        if isinstance(column, int):
            if 0 <= column < len(self.data.columns):
                return self.data.columns[column]
            raise ValueError(f"Column index {column} out of range")
        
        if column in self.data.columns:
            return column
        
        raise ValueError(f"Column '{column}' not found")
    
    def get_columns(self) -> List[str]:
        """Get list of all column names."""
        if self.data is None:
            return []
        return list(self.data.columns)
    
    def get_numeric_columns(self) -> List[str]:
        """Get list of numeric column names."""
        return [col for col, ctype in self.column_types.items()
                if ctype == ColumnType.NUMERIC]
    
    def get_categorical_columns(self) -> List[str]:
        """Get list of categorical column names."""
        return [col for col, ctype in self.column_types.items()
                if ctype == ColumnType.CATEGORICAL]
    
    def get_datetime_columns(self) -> List[str]:
        """Get list of datetime column names."""
        return [col for col, ctype in self.column_types.items()
                if ctype == ColumnType.DATETIME]
    
    def get_column_values(
        self,
        column: Union[str, int],
        as_numeric: bool = False,
        as_datetime: bool = False
    ) -> np.ndarray:
        """
        Get values of a column as a numpy array.
        
        Args:
            column: Column name or index
            as_numeric: Force conversion to numeric
            as_datetime: Force conversion to datetime
            
        Returns:
            Column values as numpy array
        """
        if self.data is None:
            raise ValueError("No data loaded")
        
        col_name = self._resolve_column(column)
        series = self.data[col_name]
        
        if as_numeric:
            return pd.to_numeric(series, errors="coerce").values
        
        if as_datetime:
            return pd.to_datetime(series, errors="coerce").values
        
        return series.values
    
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the loaded data.
        
        Returns:
            Dictionary with data summary information
        """
        if self.data is None:
            return {"loaded": False}
        
        return {
            "loaded": True,
            "filepath": str(self.filepath),
            "rows": len(self.data),
            "columns": len(self.data.columns),
            "column_names": list(self.data.columns),
            "column_types": {col: ctype.value for col, ctype in self.column_types.items()},
            "memory_usage": self.data.memory_usage(deep=True).sum(),
        }
    
    def validate_columns(
        self,
        required_columns: List[Union[str, int]],
        optional_columns: Optional[List[Union[str, int]]] = None
    ) -> Tuple[List[str], List[str]]:
        """
        Validate that required columns exist.
        
        Args:
            required_columns: List of required column names/indices
            optional_columns: List of optional column names/indices
            
        Returns:
            Tuple of (resolved required columns, resolved optional columns)
            
        Raises:
            ValueError: If any required column doesn't exist
        """
        resolved_required = []
        for col in required_columns:
            resolved_required.append(self._resolve_column(col))
        
        resolved_optional = []
        if optional_columns:
            for col in optional_columns:
                try:
                    resolved_optional.append(self._resolve_column(col))
                except ValueError:
                    pass  # Optional columns that don't exist are ignored
        
        return resolved_required, resolved_optional
    
    def auto_select_columns(
        self,
        chart_type: str,
        x_col: Optional[Union[str, int]] = None,
        y_cols: Optional[List[Union[str, int]]] = None,
        group_col: Optional[Union[str, int]] = None
    ) -> Dict[str, Any]:
        """
        Automatically select appropriate columns for a chart type.
        
        Args:
            chart_type: Type of chart
            x_col: User-specified X column (optional)
            y_cols: User-specified Y columns (optional)
            group_col: User-specified grouping column (optional)
            
        Returns:
            Dictionary with selected column names
        """
        if self.data is None:
            raise ValueError("No data loaded")
        
        result = {"x": None, "y": [], "group": None}
        
        numeric_cols = self.get_numeric_columns()
        categorical_cols = self.get_categorical_columns()
        datetime_cols = self.get_datetime_columns()
        
        # Resolve user-specified columns
        if x_col is not None:
            result["x"] = self._resolve_column(x_col)
        
        if y_cols is not None:
            result["y"] = [self._resolve_column(c) for c in y_cols]
        
        if group_col is not None:
            result["group"] = self._resolve_column(group_col)
        
        # Auto-select based on chart type
        if chart_type in ["line", "scatter", "area"]:
            if result["x"] is None:
                # Prefer datetime, then first column
                if datetime_cols:
                    result["x"] = datetime_cols[0]
                else:
                    result["x"] = self.data.columns[0]
            
            if not result["y"]:
                # All numeric columns except x
                result["y"] = [c for c in numeric_cols if c != result["x"]]
                if not result["y"] and len(self.data.columns) > 1:
                    result["y"] = [self.data.columns[1]]
        
        elif chart_type == "bar":
            if result["x"] is None:
                # Prefer categorical
                if categorical_cols:
                    result["x"] = categorical_cols[0]
                else:
                    result["x"] = self.data.columns[0]
            
            if not result["y"]:
                result["y"] = [c for c in numeric_cols if c != result["x"]]
        
        elif chart_type in ["hist", "box", "violin"]:
            if not result["y"]:
                result["y"] = numeric_cols if numeric_cols else [self.data.columns[0]]
            
            if result["group"] is None and categorical_cols:
                # Auto-select grouping column
                candidates = [c for c in categorical_cols if c not in result["y"]]
                if candidates:
                    result["group"] = candidates[0]
        
        elif chart_type == "heatmap":
            # Need row index, column index, and value
            if result["x"] is None and len(self.data.columns) >= 3:
                result["x"] = self.data.columns[0]
            if not result["y"] and len(self.data.columns) >= 3:
                result["y"] = [self.data.columns[1]]
            if result["group"] is None and len(self.data.columns) >= 3:
                result["group"] = self.data.columns[2]  # Use as value column
        
        return result
    
    @staticmethod
    def load_multiple(
        filepaths: List[Union[str, Path]],
        **kwargs
    ) -> List[Tuple[Path, pd.DataFrame, Dict[str, ColumnType]]]:
        """
        Load multiple CSV files.
        
        Args:
            filepaths: List of file paths
            **kwargs: Arguments to pass to load()
            
        Returns:
            List of tuples (filepath, dataframe, column_types)
        """
        results = []
        for fp in filepaths:
            loader = DataLoader(fp)
            df = loader.load(**kwargs)
            results.append((loader.filepath, df, loader.column_types))
        return results


def interactive_column_selection(
    data: pd.DataFrame,
    column_types: Dict[str, ColumnType],
    chart_type: str
) -> Dict[str, Any]:
    """
    Interactive column selection when user doesn't specify columns.
    
    Args:
        data: Loaded DataFrame
        column_types: Dictionary of column types
        chart_type: Type of chart
        
    Returns:
        Dictionary with selected columns
    """
    columns = list(data.columns)
    n_cols = len(columns)
    
    print(f"\n检测到 {n_cols} 列：{columns}")
    print(f"列类型：{', '.join(f'{c}({t.value})' for c, t in column_types.items())}\n")
    
    result = {"x": None, "y": [], "group": None}
    
    # X column selection
    if chart_type in ["line", "scatter", "bar", "area", "heatmap"]:
        x_input = input(f"请选择 X 轴列 [0-{n_cols-1} 或列名，默认 0]: ").strip()
        if x_input:
            result["x"] = int(x_input) if x_input.isdigit() else x_input
        else:
            result["x"] = 0
    
    # Y column selection
    if chart_type in ["line", "scatter", "bar", "area"]:
        default_y = "1" if n_cols > 1 else "0"
        y_input = input(f"请选择 Y 轴列 [可多选，逗号分隔，默认 {default_y}]: ").strip()
        if y_input:
            parts = [p.strip() for p in y_input.split(",")]
            result["y"] = [int(p) if p.isdigit() else p for p in parts]
        else:
            result["y"] = [int(default_y)]
    elif chart_type in ["hist", "box", "violin"]:
        y_input = input("请选择数值列 [可多选，逗号分隔，默认全部数值列]: ").strip()
        if y_input:
            parts = [p.strip() for p in y_input.split(",")]
            result["y"] = [int(p) if p.isdigit() else p for p in parts]
    
    # Group column selection
    if chart_type in ["scatter", "box", "violin"]:
        group_input = input("请选择分组列 [可选，回车跳过]: ").strip()
        if group_input:
            result["group"] = int(group_input) if group_input.isdigit() else group_input
    
    return result
