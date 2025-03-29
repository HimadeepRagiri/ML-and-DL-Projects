from src.agent_outputs import OrchestratorOutput
from src.utils import extract_json
from src.models import call_llm
import re

# Implement the Orchestrator Agent
def orchestrator_agent(query: str, context: dict) -> OrchestratorOutput:
    has_data = context.get("df") is not None

    prompt = (
        f"You are a data science orchestration system. Based on the user query, determine which "
        f"specialized agent should handle the request.\n\n"
        f"User query: \"{query}\"\n"
        f"Data loaded: {'Yes' if has_data else 'No'}\n\n"
        f"Available agents:\n"
        f"- 'data': For loading datasets\n"
        f"- 'eda': For exploratory data analysis\n"
        f"- 'visualization': For creating plots\n"
        f"- 'analysis': For statistical analysis\n"
        f"- 'report': For generating reports (automatically perform prior steps if needed)\n\n"
        f"Respond with a JSON object: {{'target_agent': 'name', 'message': 'explanation'}}"
    )

    llm_response = call_llm(prompt)
    json_response = extract_json(llm_response)

    if not json_response or "target_agent" not in json_response:
        return OrchestratorOutput(status="error", message="Invalid request.", target_agent="none", parameters={})

    target_agent = json_response.get("target_agent", "none").lower()
    message = json_response.get("message", "Processing...")
    parameters = {}

    if target_agent == "data":
        file_path_match = re.search(r'(/\S+\.\w+)', query)
        if file_path_match:
            parameters["file_path"] = file_path_match.group(1)
    elif target_agent == "visualization":
        # Extract plot type from query
        plot_type = "auto"
        if "scatter" in query.lower():
            plot_type = "scatter"
        elif "line" in query.lower():
            plot_type = "line"
        elif "histogram" in query.lower():
            plot_type = "histogram"
        elif "box" in query.lower():
            plot_type = "box"
        elif "count" in query.lower():
            plot_type = "count"
        elif "heatmap" in query.lower():
            plot_type = "heatmap"

        parameters["plot_type"] = plot_type

        # Extract column names from query (case-insensitive)
        df_cols = context.get("df", pd.DataFrame()).columns
        query_lower = query.lower()
        columns = []
        for col in df_cols:
            if col.lower() in query_lower:
                columns.append(col)

        if len(columns) >= 1:
            parameters["x_col"] = columns[0]
        if len(columns) >= 2:
            parameters["y_col"] = columns[1]

        # If no columns specified but plot type requires them, use defaults
        if not parameters.get("x_col") and plot_type in ["scatter", "line", "histogram", "box", "count"]:
            numeric_cols = context["df"].select_dtypes(include=['float64', 'int64']).columns
            parameters["x_col"] = numeric_cols[0] if len(numeric_cols) > 0 else None
            if plot_type in ["scatter", "line"] and len(numeric_cols) > 1:
                parameters["y_col"] = numeric_cols[1]

    elif target_agent == "analysis":
        # Extract column names from query for correlation
        df_cols = context.get("df", pd.DataFrame()).columns
        query_lower = query.lower()
        columns = [col for col in df_cols if col.lower() in query_lower]
        if len(columns) >= 2:
            parameters["col1"] = columns[0]
            parameters["col2"] = columns[1]

    return OrchestratorOutput(status="success", message=message, target_agent=target_agent, parameters=parameters)
