import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime

# Tool 1: Load and Clean Dataset
def load_and_clean_data(file_path: str) -> dict:
    try:
        print(f"Attempting to load data from {file_path}")
        df = pd.read_csv(file_path)
        initial_rows = df.shape[0]

        # Cleaning
        df = df.drop_duplicates()  # Remove duplicates
        df = df.dropna(how='all')  # Drop rows with all NaN
        # Impute missing values for numeric columns with median
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
        # Convert object columns to categorical if unique values < 50%
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() / len(df) < 0.5:
                df[col] = df[col].astype('category')
        # Remove outliers using IQR method
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            df = df[~((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR)))]

        cleaned_rows = df.shape[0]
        return {
            "data": df,
            "status": "success",
            "message": f"Loaded and cleaned dataset from {file_path}. Rows: {initial_rows} -> {cleaned_rows}. Removed duplicates, imputed missing values, and handled outliers."
        }
    except Exception as e:
        return {"data": None, "status": "error", "message": f"Failed to load/clean data: {str(e)}"}

# Tool 2: Perform EDA
def perform_eda(df: pd.DataFrame) -> dict:
    try:
        if df is None or df.empty:
            return {"status": "error", "message": "No data available for EDA.", "summary_stats": {}, "skewness": {}, "missing_values": {}, "outliers": {}}

        # Summary statistics
        summary_stats = df.describe(include='all').to_dict()

        # Missing values
        missing_values = df.isnull().sum().to_dict()

        # Skewness for numeric columns
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        skewness = {col: float(stats.skew(df[col].dropna())) for col in numeric_cols}

        # Outlier detection using IQR
        outliers = {}
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers[col] = len(df[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))])

        # Correlation matrix for numeric columns
        corr_matrix = df[numeric_cols].corr().to_dict()

        return {
            "status": "success",
            "summary_stats": summary_stats,
            "skewness": skewness,
            "missing_values": missing_values,
            "outliers": outliers,
            "corr_matrix": corr_matrix,
            "message": "Comprehensive EDA completed."
        }
    except Exception as e:
        return {"status": "error", "message": f"EDA failed: {str(e)}", "summary_stats": {}, "skewness": {}, "missing_values": {}, "outliers": {}, "corr_matrix": {}}

# Tool 3: Generate Visualizations
def generate_plot(df: pd.DataFrame, plot_type: str = "auto", x_col: str = None, y_col: str = None, annot: bool = False, **kwargs) -> dict:
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = df.select_dtypes(include=['category', 'object']).columns

    # Auto-select plot type if not specified
    if plot_type == "auto":
        if x_col and y_col:
            plot_type = "scatter" if y_col in numeric_cols else "bar"
        elif x_col:
            plot_type = "histogram" if x_col in numeric_cols else "count"
        else:
            plot_type = "heatmap"  # Default to correlation heatmap

    # Validate columns
    if x_col and x_col not in df.columns:
        return {"status": "error", "message": f"Column '{x_col}' not found in dataset."}
    if y_col and y_col not in df.columns:
        return {"status": "error", "message": f"Column '{y_col}' not found in dataset."}

    plt.figure(figsize=(16, 10))  # Increased size for better visibility
    try:
        if plot_type.lower() == "histogram" and x_col:
            sns.histplot(data=df, x=x_col, kde=True)
        elif plot_type.lower() == "scatter" and x_col and y_col:
            sns.scatterplot(data=df, x=x_col, y=y_col, hue=categorical_cols[0] if len(categorical_cols) > 0 else None)
        elif plot_type.lower() == "line" and x_col and y_col:
            sns.lineplot(data=df, x=x_col, y=y_col)
        elif plot_type.lower() == "box" and x_col:
            # Use y=x_col for vertical box plot to show distribution
            sns.boxplot(data=df, y=x_col, color='skyblue')
            # Add grid for better readability
            plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        elif plot_type.lower() == "count" and x_col:
            sns.countplot(data=df, x=x_col)
        elif plot_type.lower() == "heatmap":
            if x_col or y_col:
                return {"status": "error", "message": "Heatmap does not use specific x_col or y_col; it shows correlation matrix."}

            # Compute correlation
            corr_matrix = df[numeric_cols].corr()

            # Generate heatmap
            sns.heatmap(corr_matrix, annot=annot, cmap="coolwarm", linewidths=0.5, vmin=-1, vmax=1,
                        square=True, cbar_kws={"shrink": 0.75})

            # Fix x and y labels for better readability
            plt.xticks(rotation=45, ha="right", fontsize=10)
            plt.yticks(fontsize=10)

        else:
            return {"status": "error", "message": f"Invalid plot type '{plot_type}' or missing required columns."}

        plt.title(f"{plot_type.capitalize()} Plot: {x_col if x_col else 'Heatmap'} {f'vs {y_col}' if y_col else ''}", fontsize=14, pad=20)
        plt.tight_layout()

        # Save plot
        file_name = f"plot_{plot_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(file_name)
        plt.close()

        return {"status": "success", "message": f"Generated {plot_type} plot.", "file": file_name}

    except Exception as e:
        return {"status": "error", "message": f"Plot generation failed: {str(e)}"}

# Tool 4: Correlation Analysis
def calculate_correlation(df: pd.DataFrame, col1: str = None, col2: str = None, **kwargs) -> dict:
    try:
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        if not col1 or not col2:
            col1, col2 = numeric_cols[:2] if len(numeric_cols) >= 2 else (None, None)
        if col1 not in numeric_cols or col2 not in numeric_cols:
            return {"status": "error", "message": "Columns must be numeric for correlation."}

        corr = df[col1].corr(df[col2])
        corr_matrix = df[numeric_cols].corr().to_dict()
        return {
            "correlation": float(corr),
            "col1": col1,
            "col2": col2,
            "corr_matrix": corr_matrix,
            "status": "success",
            "message": f"Correlation between {col1} and {col2} calculated with full matrix."
        }
    except Exception as e:
        return {"status": "error", "message": f"Correlation failed: {str(e)}", "correlation": None, "corr_matrix": {}}

# Tool 5: Comprehensive Report Generation
def generate_report(df: pd.DataFrame, eda_results: dict, analysis_results: dict, plot_results: list) -> dict:
    report_lines = [
        "=== Data Science Report ===",
        f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "\n1. Dataset Overview:",
        f"  - Rows: {df.shape[0]}, Columns: {df.shape[1]}",
        f"  - Columns: {', '.join(df.columns)}",
        "\n2. Exploratory Data Analysis:",
        f"  - Summary Stats: {json.dumps(eda_results.get('summary_stats', {}), indent=2)}",
        f"  - Missing Values: {json.dumps(eda_results.get('missing_values', {}), indent=2)}",
        f"  - Skewness: {json.dumps(eda_results.get('skewness', {}), indent=2)}",
        f"  - Outliers: {json.dumps(eda_results.get('outliers', {}), indent=2)}",
        "\n3. Correlation Analysis:",
        f"  - Specific Correlation: {analysis_results.get('correlation', 'N/A')} between {analysis_results.get('col1', 'N/A')} and {analysis_results.get('col2', 'N/A')}",
        f"  - Full Correlation Matrix: {json.dumps(analysis_results.get('corr_matrix', {}), indent=2)}",
        "\n4. Visualizations:",
        "\n  - " + "\n  - ".join(plot_results or ["No plots generated."]),
        "\n=== End of Report ==="
    ]
    return {"report": "\n".join(report_lines), "status": "success", "message": "Comprehensive report generated."}
