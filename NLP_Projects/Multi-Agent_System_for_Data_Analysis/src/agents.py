from pydantic import BaseModel
from typing import Callable, Dict
from src.tools import load_and_clean_data, perform_eda, generate_plot, calculate_correlation, generate_report
from src.utils import extract_json

## Defining Pydantic Models

# Data Ingestion Agent Output
class DataIngestionOutput(BaseModel):
    status: str
    message: str
    rows: int | None = None
    columns: int | None = None

# EDA Agent Output
class EDAOutput(BaseModel):
    status: str
    message: str
    summary_stats: dict = {}
    skewness: dict = {}
    missing_values: dict = {}
    outliers: dict = {}
    corr_matrix: dict = {}

# Visualization Agent Output
class VisualizationOutput(BaseModel):
    status: str
    message: str
    plot_type: str | None = None
    x_column: str | None = None
    y_column: str | None = None

# Analysis Agent Output
class AnalysisOutput(BaseModel):
    status: str
    message: str
    correlation: float | None = None
    col1: str | None = None
    col2: str | None = None
    corr_matrix: dict = {}

# Report Generation Agent Output
class ReportOutput(BaseModel):
    status: str
    message: str
    report_text: str

# Orchestrator Agent Output
class OrchestratorOutput(BaseModel):
    status: str
    message: str
    target_agent: str
    parameters: dict

## Implementing Agents

# Create agent function
def create_agent(role: str, tools: Dict[str, Callable], output_model: type[BaseModel]) -> Callable:
    def agent(query: str, context: dict) -> BaseModel:
        tool_names = list(tools.keys())
        parameters = context.get("parameters", {})

        prompt = (
            f"You are a {role} agent. Available tools: {', '.join(tool_names)}.\n"
            f"User query: {query}\n\n"
            f"Decide which tool to use based on the query. Respond with a JSON object:\n"
            f"- 'action': the name of the tool to use\n"
            f"- 'parameters': dict of parameters needed for the tool\n"
            f"- 'message': explanation of your decision\n"
        )
        llm_response = call_llm(prompt)
        json_response = extract_json(llm_response)

        if json_response and "action" in json_response:
            action = json_response.get("action")
            tool_params = {**parameters, **json_response.get("parameters", {})}

            if action in tools:
                try:
                    if role == "data ingestion" and action == "load_and_clean_data":
                        file_path = tool_params.get("file_path")
                        if file_path:
                            result = load_and_clean_data(file_path)
                            if result["status"] == "success" and result["data"] is not None:
                                context["df"] = result["data"]
                            return DataIngestionOutput(
                                status=result["status"],
                                message=result["message"],
                                rows=result["data"].shape[0] if result["status"] == "success" else None,
                                columns=result["data"].shape[1] if result["status"] == "success" else None
                            )
                    elif role == "EDA" and action == "perform_eda":
                        if "df" in context and context["df"] is not None:
                            result = perform_eda(context["df"])
                            context["eda_results"] = result
                            return EDAOutput(**result)
                    else:
                        result = tools[action](**tool_params)
                        return output_model(**result)
                except Exception as e:
                    return output_model(status="error", message=f"Error executing {action}: {str(e)}")
        return output_model(status="error", message="Failed to process request.")

    return agent

# Define agents
data_agent = create_agent("data ingestion", {"load_and_clean_data": load_and_clean_data}, DataIngestionOutput)
eda_agent = create_agent("EDA", {"perform_eda": perform_eda}, EDAOutput)
visualization_agent = create_agent("visualization", {"generate_plot": generate_plot}, VisualizationOutput)
analysis_agent = create_agent("analysis", {"calculate_correlation": calculate_correlation}, AnalysisOutput)
report_agent = create_agent("report generation", {"generate_report": generate_report}, ReportOutput)
