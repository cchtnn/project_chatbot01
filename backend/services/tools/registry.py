"""Tool registry with descriptions for orchestrator."""
from typing import Dict, List, Any
from pydantic import BaseModel
from models.schemas import ToolResult

class ToolSchema(BaseModel):
    name: str
    description: str
    parameters: Dict[str, str]  # param_name: description

TOOLS: List[ToolSchema] = [
    ToolSchema(
    name="TranscriptTool",
    description="Answers questions about student transcripts using intelligent query reformulation and CSV analysis. Handles all transcript queries: student GPA, course enrollment, grade analysis, rankings, comparisons, and aggregate statistics. Automatically interprets natural language questions.",
    parameters={
        # No parameters needed - query reformulation handles everything
    }
    ),
    ToolSchema(
        name="PayrollTool",
        description="Answers payroll calendar questions. Finds check dates by period, periods by check date, counts periods per year. Works for any year with payroll data.",
        parameters={
            "year": "Optional: filter by year (e.g. 2026)",
            "start_date": "Optional: period start date",
            "end_date": "Optional: period end date",
            "check_date": "Optional: check date to find period",
        }
    ),
    ToolSchema(
        name="BorPlannerTool",
        description="Answers Board of Regents (BOR) meeting schedule questions. Finds next meeting, full year schedule, committee meetings.",
        parameters={
            "year": "Optional: filter by year",
            "committee": "Optional: filter by committee name",
        }
    ),
    ToolSchema(
        name="GenericRagTool",
        description="Fallback for policy, handbook, catalog, or any unstructured text question. Uses hybrid retrieval (vector+BM25+rerank) over all documents. Always cite sources.",
        parameters={
            "question": "The full question to answer",
            "domain_filter": "Optional: policy, hr, academic, etc.",
        }
    ),
]

def get_tool_descriptions() -> str:
    """Formatted tool descriptions for orchestrator prompt."""
    return "\n\n".join([
        f"**{tool.name}**\n{tool.description}\nParameters: {', '.join(f'{k}: {v}' for k, v in tool.parameters.items())}"
        for tool in TOOLS
    ])
