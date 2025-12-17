# services/orchestrator.py

from typing import Dict, Any, List
import json
import re
from datetime import datetime

from pydantic import BaseModel

from core import get_logger
from config import get_settings
from services.tools.registry import get_tool_descriptions
from services.tools.transcript_tool import answer as transcript_answer
from services.tools.payroll_tool import answer as payroll_answer
from services.tools.bor_planner_tool import answer as bor_answer
from services.tools.generic_rag_tool import answer as rag_answer
from models.schemas import ToolResult
from services.rag_pipeline import RAGPipeline

logger = get_logger(__name__)
settings = get_settings()


TOOL_MAP = {
    "TranscriptTool": transcript_answer,
    "PayrollTool": payroll_answer,
    "BorPlannerTool": bor_answer,
    "GenericRagTool": rag_answer,
}


class OrchestratorRequest(BaseModel):
    query: str
    conversation_history: List[Dict[str, str]] = []


class OrchestratorResponse(BaseModel):
    answer: str
    tools_used: List[str]
    confidence: float
    sources: List[Dict[str, Any]]


class Orchestrator:
    def __init__(self) -> None:
        # Use existing RAGPipeline only for small planning prompts
        self.planner_pipeline = RAGPipeline()

    def handle_query(self, request: OrchestratorRequest) -> OrchestratorResponse:
        """Main orchestrator entry point."""
        tool_plan = self._plan_tools(request.query)

        tool_results: List[ToolResult] = []
        tools_used: List[str] = []

        for tool_call in tool_plan.get("tools", []):
            tool_name = tool_call.get("name")
            params = tool_call.get("parameters") or {}
            if tool_name in TOOL_MAP:
                try:
                    # Transcript-specific parameter extraction
                    if tool_name == "TranscriptTool":
                        q_lower = request.query.lower()
                        name = None
                        if " of " in q_lower:
                            parts = request.query.split(" of ")
                            name = parts[-1].strip(" .?")
                        if name:
                            params["student_name"] = name

                    # Payroll-specific parameter extraction
                    if tool_name == "PayrollTool":
                        q = request.query.lower()

                        # Year detection
                        m_year = re.search(r"\b(20[0-9]{2})\b", q)
                        if m_year:
                            params["year"] = int(m_year.group(1))

                        # Check-date detection: e.g., 2/6/2026 or 02/06/26
                        m_date = re.search(r"(\d{1,2}/\d{1,2}/\d{2,4})", q)
                        if m_date:
                            params["check_date"] = m_date.group(1)

                        # Period detection: e.g., "from 1/17/2026 to 1/30/2026"
                        m_period = re.search(
                            r"from\s+(\d{1,2}/\d{1,2}/\d{2,4})\s+(to|through|till|-)\s+(\d{1,2}/\d{1,2}/\d{2,4})",
                            q,
                        )
                        if m_period:
                            params["start_date"] = m_period.group(1)
                            params["end_date"] = m_period.group(3)

                    result = TOOL_MAP[tool_name](request.query, params)
                    tool_results.append(result)
                    tools_used.append(tool_name)
                except Exception as e:
                    logger.error(f"Tool {tool_name} failed: {e}", exc_info=True)

        if not tool_results:
            return OrchestratorResponse(
                answer="I could not route this question to any tool.",
                tools_used=[],
                confidence=0.0,
                sources=[],
            )

        final_answer = self._synthesize(tool_results, request.query)
        avg_conf = sum(t.confidence for t in tool_results) / max(len(tool_results), 1)

        # TODO: aggregate real sources from each ToolResult.citations
        return OrchestratorResponse(
            answer=final_answer,
            tools_used=tools_used,
            confidence=avg_conf,
            sources=[],
        )

    def _plan_tools(self, query: str) -> Dict[str, Any]:
        """LLM tool selection using existing pipeline as a planner."""
        tools_desc = get_tool_descriptions()
        prompt = f"""
You are a routing assistant for a multi-tool system.

Available tools:
{tools_desc}

Routing rules:
- Use TranscriptTool for student transcript questions (GPA, courses, terms, students).
- Use PayrollTool for payroll calendar questions (pay periods, check dates, number of periods).
- Use BorPlannerTool for Board of Regents (BOR) meeting schedule and ACCT/NLS event questions.
- Use GenericRagTool for all other policy/handbook/catalog questions.

User query: "{query}"

Respond with ONLY valid JSON in this format:

{{
  "tools": [
    {{"name": "ToolName", "parameters": {{"param": "value"}}}}
  ]
}}

Prefer TranscriptTool for transcript questions, PayrollTool for payroll calendar,
BorPlannerTool for Board of Regents meeting schedule or ACCT event questions,
and GenericRagTool otherwise.
"""

        # Use RAGPipeline.generate_answer as a thin LLM wrapper for planning
        plan_str = self.planner_pipeline._generate_answer(context="", question=prompt)
        try:
            return json.loads(plan_str)
        except Exception:
            logger.warning("Tool planning JSON parse failed, falling back to GenericRagTool.")
            return {"tools": [{"name": "GenericRagTool", "parameters": {}}]}

    def _synthesize(self, tool_results: List[ToolResult], query: str) -> str:
        """Format final response."""
        parts: List[str] = []
        for tr in tool_results:
            parts.append(tr.explanation)
        return "\n\n".join(parts)
