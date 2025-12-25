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
        
        # ============ DEBUGGING: Log incoming query ============
        logger.info("=" * 80)
        logger.info("ORCHESTRATOR: New query received")
        logger.info(f"Query: {request.query}")
        logger.info(f"Conversation history length: {len(request.conversation_history)}")
        logger.info("=" * 80)
        
        tool_plan = self._plan_tools(request.query)
        
        # ============ DEBUGGING: Log tool classification result ============
        logger.info("-" * 80)
        logger.info("TOOL CLASSIFICATION RESULT:")
        logger.info(f"Selected tools: {json.dumps(tool_plan, indent=2)}")
        logger.info("-" * 80)

        tool_results: List[ToolResult] = []
        tools_used: List[str] = []

        for tool_call in tool_plan.get("tools", []):
            tool_name = tool_call.get("name")
            params = tool_call.get("parameters") or {}
            
            # ============ DEBUGGING: Log tool execution ============
            logger.info(f"\n{'='*80}")
            logger.info(f"EXECUTING TOOL: {tool_name}")
            logger.info(f"Initial parameters: {json.dumps(params, indent=2)}")
            
            if tool_name in TOOL_MAP:
                try:
                    # Payroll-specific parameter extraction
                    if tool_name == "PayrollTool":
                        logger.info(">>> PayrollTool parameter extraction started")
                        q = request.query
                        q_lower = q.lower()

                        # Year detection
                        m_year = re.search(r"\b(20[0-9]{2})\b", q_lower)
                        if m_year:
                            params["year"] = int(m_year.group(1))
                            logger.info(f">>> Extracted year: {params['year']}")

                        # Check-date detection: e.g., 2/6/2026 or 02/06/26
                        m_date = re.search(r"(\d{1,2}/\d{1,2}/\d{2,4})", q_lower)
                        if m_date:
                            params["check_date"] = m_date.group(1)
                            logger.info(f">>> Extracted check_date: {params['check_date']}")

                        # Period detection: e.g., "from 1/17/2026 to 1/30/2026"
                        m_period = re.search(
                            r"from\s+(\d{1,2}/\d{1,2}/\d{2,4})\s+(to|through|till|-)\s+(\d{1,2}/\d{1,2}/\d{2,4})",
                            q_lower,
                        )
                        if m_period:
                            params["start_date"] = m_period.group(1)
                            params["end_date"] = m_period.group(3)
                            logger.info(f">>> Extracted period: {params['start_date']} to {params['end_date']}")

                        # Pay period / payroll number detection → payroll_no
                        m_payroll_no = re.search(
                            r"(pay\s*period|payroll|payroll\s*number|pay\s*period\s*number)\s*(no\.?|number)?\s*(\d+)",
                            q_lower,
                        )
                        if m_payroll_no:
                            params["payroll_no"] = int(m_payroll_no.group(3))
                            logger.info(f">>> Extracted payroll_no: {params['payroll_no']}")

                        # Ask for date difference
                        if "difference" in q_lower and "day" in q_lower:
                            params["ask_days_diff"] = True
                            logger.info(">>> Detected request for day difference calculation")
                        
                        logger.info(f">>> Final PayrollTool parameters: {json.dumps(params, indent=2)}")

                    # ============ DEBUGGING: Tool execution ============
                    logger.info(f">>> Calling {tool_name} with parameters: {json.dumps(params, indent=2)}")
                    result = TOOL_MAP[tool_name](request.query, params)
                    
                    # ============ DEBUGGING: Log tool result ============
                    logger.info(f">>> {tool_name} execution completed")
                    logger.info(f">>> Result confidence: {result.confidence}")
                    logger.info(f">>> Result format_hint: {result.format_hint}")
                    logger.info(f">>> Result explanation length: {len(result.explanation)} chars")
                    logger.info(f">>> Result data keys: {list(result.data.keys()) if isinstance(result.data, dict) else 'N/A'}")
                    logger.info(f">>> Citations: {result.citations}")
                    logger.info(f"{'='*80}\n")
                    
                    tool_results.append(result)
                    tools_used.append(tool_name)
                except Exception as e:
                    logger.error(f"Tool {tool_name} failed: {e}", exc_info=True)
                    logger.error(f"Failed parameters were: {json.dumps(params, indent=2)}")

        if not tool_results:
            logger.warning("No tool results generated - returning error response")
            return OrchestratorResponse(
                answer="I could not route this question to any tool.",
                tools_used=[],
                confidence=0.0,
                sources=[],
            )

        # ============ DEBUGGING: Log synthesis ============
        logger.info("-" * 80)
        logger.info("SYNTHESIZING FINAL ANSWER")
        logger.info(f"Number of tool results to synthesize: {len(tool_results)}")
        
        final_answer = self._synthesize(tool_results, request.query)
        avg_conf = sum(t.confidence for t in tool_results) / max(len(tool_results), 1)

        logger.info(f"Final answer length: {len(final_answer)} chars")
        logger.info(f"Average confidence: {avg_conf:.3f}")
        logger.info(f"Tools used: {tools_used}")
        logger.info("-" * 80)

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
        
        # ============ DEBUGGING: Log planning prompt ============
        logger.info("\n" + "="*80)
        logger.info("TOOL PLANNING PHASE")
        logger.info(f"Planning prompt length: {len(prompt)} chars")
        logger.debug(f"Full planning prompt:\n{prompt}")
        
        # Use RAGPipeline.generate_answer as a thin LLM wrapper for planning
        plan_str = self.planner_pipeline._generate_answer(context="", question=prompt)
        
        # ============ DEBUGGING: Log raw LLM response ============
        logger.info(f"Raw LLM planning response:\n{plan_str}")
        logger.info("="*80 + "\n")
        
        try:
            parsed_plan = json.loads(plan_str)
            logger.info(f"Successfully parsed tool plan: {json.dumps(parsed_plan, indent=2)}")
            return parsed_plan
        except Exception as e:
            logger.warning(f"Tool planning JSON parse failed: {e}")
            logger.warning("Falling back to GenericRagTool")
            return {"tools": [{"name": "GenericRagTool", "parameters": {}}]}

    def _synthesize(self, tool_results: List[ToolResult], query: str) -> str:
        """Format final response."""
        parts: List[str] = []
        for idx, tr in enumerate(tool_results):
            logger.info(f"Synthesizing result {idx+1}/{len(tool_results)}")
            logger.debug(f"Result {idx+1} explanation: {tr.explanation[:200]}...")
            parts.append(tr.explanation)
        
        final = "\n\n".join(parts)
        logger.info(f"Synthesis complete. Final answer length: {len(final)} chars")
        return final