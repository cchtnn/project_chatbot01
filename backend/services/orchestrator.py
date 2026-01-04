# services/orchestrator.py
# ENTERPRISE-GRADE SEMANTIC ROUTING ORCHESTRATOR (3-LAYER CONSERVATIVE)
# Semantic Router + Parallel Execution for Bias Protection
# ENHANCED: Conversational Layer + Source Formatting
# BACKWARD COMPATIBLE with existing Orchestrator interface


from typing import Dict, Any, List, Tuple, Optional
import json
import re
import logging
import numpy as np
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed


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
from services.conversational_handler import ConversationalHandler  # NEW


logger = get_logger(__name__)
settings = get_settings()


TOOL_MAP = {
    "TranscriptTool": transcript_answer,
    "PayrollTool": payroll_answer,
    "BorPlannerTool": bor_answer,
    "GenericRagTool": rag_answer,
}



# =============================================================================
# SEMANTIC ROUTER - Embedded in Orchestrator
# =============================================================================


class SemanticRouter:
    """
    Semantic routing using embeddings for bias-free intent classification.
    Replaces brittle keyword matching with semantic similarity.
    """
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """Initialize with tool descriptions and examples."""
        
        # Import here to avoid startup dependency if not used
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(embedding_model)
            logger.info(f"[SemanticRouter] Loaded model: {embedding_model}")
        except ImportError:
            logger.error(
                "[SemanticRouter] sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
            self.model = None
            return
        
        # Tool descriptions with balanced examples
        self.tool_contexts = {
            "TranscriptTool": {
                "description": """
                Handles individual student academic records and performance data.
                Queries about specific student grades, courses taken by students,
                GPA calculations, academic standing, student enrollment in courses,
                credit hours earned, student performance, transcript data.
                """,
                "examples": [
                    "What is Trista Barrett's GPA?",
                    "Show me courses taken by Leslie Bright",
                    "Which students are enrolled in Business Math?",
                    "List top 5 students by GPA",
                    "How many credit hours did John complete?",
                    "Display students in tabular format sorted by GPA",
                    "What courses is Leslie enrolled in?",
                    "Show me all students with GPA above 3.5",
                    "What is Trista's GPA?",
                    "Show courses for John",
                    "Trista Barrett GPA",
                    "Leslie courses enrolled",
                    "GPA for Arnoldo",
                    "Courses taken by Blen",
                ]
            },
            
            "PayrollTool": {
                "description": """
                Handles payroll calendar, pay periods, check dates, payment schedules.
                Queries about when payments are made, specific pay period information,
                payroll deadlines, payment processing dates, salary schedules.
                """,
                "examples": [
                    "When is the check date for pay period 5?",
                    "What is the payroll schedule for 2024?",
                    "When do I get paid for p03?",
                    "Show me all pay periods in 2025",
                    "What date does payroll period 12 end?",
                    "When is the next payroll date?",
                ]
            },
            
            "BorPlannerTool": {
                "description": """
                Handles Board of Regents meetings, committee schedules, governance events.
                Queries about meeting dates, committee meetings, board calendars,
                governance event schedules, board agendas.
                """,
                "examples": [
                    "When is the next Board of Regents meeting?",
                    "Finance committee meeting schedule",
                    "Show me all BOR meetings for 2024",
                    "When does the audit committee meet?",
                    "What's on the agenda for the board meeting?",
                    "When is the Finance/Audit/Investment Committee meeting?",
                ]
            },
            
            "GenericRagTool": {
                "description": """
                Handles institutional policies, handbooks, procedures, general information.
                Academic calendar, housing rules, enrollment procedures (NOT student enrollment data),
                benefits information, health services, conduct policies, institutional calendars,
                general questions about policies and procedures.
                """,
                "examples": [
                    "What are the housing rules in the residence handbook?",
                    "What is the enrollment policy?",
                    "Show me the academic calendar for Fall 2024",
                    "What are the student conduct policies?",
                    "How do I register for classes?",
                    "What health insurance options are available?",
                    "What is the refund policy?",
                    "List the 2024-2025 academic calendar",
                ]
            }
        }
        
        # Pre-compute embeddings
        self.tool_embeddings = self._precompute_tool_embeddings()
        logger.info(f"[SemanticRouter] Ready with {len(self.tool_embeddings)} tools")
    
    def _precompute_tool_embeddings(self) -> Dict[str, np.ndarray]:
        """Pre-compute embeddings for all tool contexts."""
        if not self.model:
            return {}
        
        tool_embeddings = {}
        
        for tool_name, context in self.tool_contexts.items():
            # Combine description + examples
            full_context = (
                context["description"] + "\n\nExamples:\n" + 
                "\n".join(context["examples"])
            )
            
            # Embed
            embedding = self.model.encode(full_context, convert_to_tensor=False)
            tool_embeddings[tool_name] = embedding
            
        return tool_embeddings
    
    def get_top_k_candidates(
        self, 
        query: str, 
        k: int = 2
    ) -> List[Tuple[str, float]]:
        """
        Get top K candidate tools for a query.
        
        Args:
            query: User query
            k: Number of top candidates to return
        
        Returns:
            List of (tool_name, similarity_score) tuples, sorted descending
        """
        if not self.model:
            return []
        
        # Embed query
        query_embedding = self.model.encode(query, convert_to_tensor=False)
        
        # Compute similarities
        similarities = {}
        for tool_name, tool_embedding in self.tool_embeddings.items():
            similarity = self._cosine_similarity(query_embedding, tool_embedding)
            similarities[tool_name] = similarity
        
        # Sort and return top k
        sorted_tools = sorted(
            similarities.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return sorted_tools[:k]
    
    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))



# =============================================================================
# FEEDBACK TRACKING
# =============================================================================


class RoutingFeedback:
    """Track routing decisions for monitoring."""
    
    def __init__(self):
        self.routing_log = []
        self.parallel_count = 0
        self.total_count = 0
    
    def log_decision(
        self,
        query: str,
        routed_tool: str,
        confidence: float,
        routing_source: str
    ):
        """Log routing decision."""
        entry = {
            "timestamp": datetime.now(),
            "query": query,
            "routed_tool": routed_tool,
            "confidence": confidence,
            "routing_source": routing_source,
        }
        self.routing_log.append(entry)
        self.total_count += 1
        
        if routing_source == "parallel_judge":
            self.parallel_count += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get routing performance metrics."""
        return {
            "total_routed": self.total_count,
            "parallel_executions": self.parallel_count,
            "parallel_rate": (
                self.parallel_count / max(self.total_count, 1)
            ),
        }



# =============================================================================
# PYDANTIC MODELS
# =============================================================================


class OrchestratorRequest(BaseModel):
    query: str
    conversation_history: List[Dict[str, str]] = []
    session_file_hashes: List[str] = []  # NEW - for session-scoped filtering


class OrchestratorResponse(BaseModel):
    answer: str
    tools_used: List[str]
    confidence: float
    sources: List[Dict[str, Any]]



# =============================================================================
# MAIN ORCHESTRATOR CLASS
# =============================================================================


class Orchestrator:
    """
    4-Layer Enhanced Semantic Routing Orchestrator with Conversational Support.
    
    Architecture:
    - Layer -1: Conversational detection (NEW - greetings/thanks/help)
    - Layer 0: Explicit format markers (optional, < 1ms)
    - Layer 1: Semantic router (candidate generator, 50-100ms)
    - Layer 2: Conservative decision + parallel execution (200-600ms when needed)
    
    Bias Protection: Parallel execution for close contests (margin < 0.20)
    """
    
    def __init__(self, rag_pipeline: Optional[RAGPipeline] = None) -> None:
        """Initialize orchestrator with semantic router and conversational handler."""
        self.rag_pipeline = rag_pipeline or RAGPipeline()
        self.planner_pipeline = self.rag_pipeline
        self.feedback = RoutingFeedback()
        self._cached_parallel_result = None
        self._cached_conversational_result = None  # NEW
        
        # Initialize semantic router
        try:
            self.semantic_router = SemanticRouter()
            logger.info(
                "[Orchestrator] Initialized with Semantic Router "
                "+ Parallel Execution (3-layer conservative)"
            )
        except Exception as e:
            logger.error(f"[Orchestrator] Semantic router failed to init: {e}")
            self.semantic_router = None
        
        # Initialize conversational handler (NEW)
        try:
            self.conversational_handler = ConversationalHandler()
            logger.info("[Orchestrator] Conversational handler initialized")
        except Exception as e:
            logger.error(f"[Orchestrator] Conversational handler failed: {e}")
            self.conversational_handler = None
    
    def handle_query(self, request: OrchestratorRequest) -> OrchestratorResponse:
        """Main orchestrator entry point."""
        
        # Route query
        tool_name, confidence, routing_source = self._route_query(request.query)
        
        logger.info(
            f"[Orchestrator] Routed to {tool_name} "
            f"(confidence={confidence:.2f}, source={routing_source})"
        )
        
        # NEW: Check if conversational
        if hasattr(self, '_cached_conversational_result') and self._cached_conversational_result:
            result = self._cached_conversational_result
            self._cached_conversational_result = None
            
            return OrchestratorResponse(
                answer=result.explanation,
                tools_used=["ConversationalHandler"],
                confidence=result.confidence,
                sources=[],
            )
        
        # Check if we have cached parallel result
        if self._cached_parallel_result:
            result = self._cached_parallel_result
            self._cached_parallel_result = None
            
            return OrchestratorResponse(
                answer=result.explanation,
                tools_used=[tool_name],
                confidence=result.confidence,
                sources=result.sources if hasattr(result, 'sources') else [],
            )
        
        # Execute tool
        tool_results: List[ToolResult] = []
        tools_used: List[str] = []
        
        # Prepare filters for session-scoped queries
        query_filters = None
        if request.session_file_hashes:
            # ChromaDB filter format: match any of these file hashes
            query_filters = {"file_hash": {"$in": request.session_file_hashes}}
            logger.info(f"[Orchestrator] Applying session filter: {len(request.session_file_hashes)} documents")

        if tool_name in TOOL_MAP:
            try:
                # Extract parameters
                params = {}
                
                if tool_name == "TranscriptTool":
                    # Extract student name
                    student_name = None
                    match = re.search(
                        r'(?:courses|grades?|gpa|transcript).*?(?:for|of|is)\s+([A-Z][a-z]{3,}(?:\s+[A-Z][a-z]+)*)\b',
                        request.query
                    )
                    if match:
                        student_name = match.group(1).strip()
                    
                    if not student_name:
                        match = re.search(
                            r'(?:of|for)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})',
                            request.query
                        )
                        if match:
                            student_name = match.group(1).strip()
                    
                    if student_name:
                        params["student_name"] = student_name
                
                elif tool_name == "PayrollTool":
                    q_lower = request.query.lower()
                    m_year = re.search(r"\b(20[0-9]{2})\b", q_lower)
                    if m_year:
                        params["year"] = int(m_year.group(1))
                    m_payroll = re.search(r"(pay\s*period|payroll|p)\s*(\d+)", q_lower)
                    if m_payroll:
                        params["payroll_no"] = int(m_payroll.group(2))
                
                # Execute with session filters
                params["filters"] = query_filters  # Pass filters to tool
                result = TOOL_MAP[tool_name](request.query, params)
                tool_results.append(result)
                tools_used.append(tool_name)
                
                # Log feedback
                self.feedback.log_decision(
                    query=request.query,
                    routed_tool=tool_name,
                    confidence=confidence,
                    routing_source=routing_source
                )
                
            except Exception as e:
                logger.error(f"Tool execution failed: {e}", exc_info=True)
        
        # Synthesize response
        if not tool_results:
            return OrchestratorResponse(
                answer="Unable to process query.",
                tools_used=[],
                confidence=0.0,
                sources=[],
            )
        
        final_answer = "\n\n".join([
            tr.explanation if hasattr(tr, 'explanation') 
            else tr.get('explanation', '') 
            for tr in tool_results
        ])
        avg_conf = sum(t.confidence for t in tool_results) / len(tool_results)
        
        # NEW: Universal source formatting (Phase 4)
        formatted_sources = self._format_sources(tool_results, tools_used)
        
        return OrchestratorResponse(
            answer=final_answer,
            tools_used=tools_used,
            confidence=avg_conf,
            sources=formatted_sources,
        )
    
    def _format_sources(self, tool_results: List[ToolResult], tools_used: List[str]) -> List[Dict[str, Any]]:
        """
        Universal source formatter - converts tool citations to UI format.
        
        Handles:
        - CSV sources (transcript, payroll)
        - PDF sources (policies, handbooks)
        - JSON sources (BOR meetings)
        """
        formatted_sources = []
        
        for tool_result, tool_name in zip(tool_results, tools_used):
            # Priority 1: Use tool_result.sources if available (GenericRAG provides this)
            if hasattr(tool_result, 'sources') and tool_result.sources:
                formatted_sources.extend(tool_result.sources)
                continue
            
            # Priority 2: Convert citations to source format
            if hasattr(tool_result, 'citations') and tool_result.citations:
                for citation in tool_result.citations:
                    # Determine source type from file extension
                    ext = citation.split('.')[-1].lower() if '.' in citation else 'unknown'
                    
                    source_type_map = {
                        'csv': 'Structured Data',
                        'pdf': 'Document',
                        'json': 'Structured Data',
                        'docx': 'Document',
                        'txt': 'Document'
                    }
                    
                    formatted_sources.append({
                        "filename": citation,
                        "type": source_type_map.get(ext, 'File'),
                        "tool": tool_name,
                        "relevance": tool_result.confidence if hasattr(tool_result, 'confidence') else 0.8
                    })
        
        # Deduplicate by filename
        seen = set()
        unique_sources = []
        for src in formatted_sources:
            key = src.get("filename", "")
            if key and key not in seen:
                seen.add(key)
                unique_sources.append(src)
        
        return unique_sources
    
    def _route_query(self, query: str) -> Tuple[str, float, str]:
        """
        ENHANCED HYBRID ROUTING: Conversational -->LLM Description-Based -->Semantic Validation
        
        Architecture:
        - Layer -1: Conversational detection (NEW - < 1ms)
        - Layer 0: Fast heuristics (< 1ms) - explicit markers
        - Layer 1: LLM with tool descriptions (150ms) - PRIMARY, understands intent
        - Layer 2: Semantic router (80ms) - VALIDATION
        - Layer 3: Parallel execution (400ms) - disagreement resolution
        
        Returns: (tool_name, confidence, routing_source)
        """
        q_norm = query.lower()
        
        # =================================================================
        # LAYER -1: CONVERSATIONAL DETECTION (NEW - HIGHEST PRIORITY)
        # =================================================================
        if self.conversational_handler:
            conv_intent, conv_conf = self.conversational_handler.detect(query)
            if conv_intent:
                logger.info(f"[Layer -1] Conversational intent '{conv_intent}' -->ConversationalHandler")
                # Cache the result so handle_query can use it
                self._cached_conversational_result = self.conversational_handler.handle(conv_intent)
                return ("ConversationalHandler", conv_conf, "conversational")
        
        # =================================================================
        # LAYER 0: Explicit Format Markers (< 1ms)
        # =================================================================
        
        # Payroll code format (p01, p05, etc.)
        if re.search(r'\bp\d{2}\b', q_norm):
            logger.info("[Layer 0] Explicit payroll code --> PayrollTool")
            return ("PayrollTool", 1.0, "explicit_format")
        
        # Exact BOR phrase
        if 'board of regents' in q_norm:
            logger.info("[Layer 0] Explicit BOR phrase --> BorPlannerTool")
            return ("BorPlannerTool", 1.0, "explicit_phrase")
        
        # =================================================================
        # LAYER 1: LLM Description-Based Routing (PRIMARY - 150ms)
        # =================================================================
        
        llm_tool, llm_conf = self._llm_description_route(query)
        
        # =================================================================
        # LAYER 2: Semantic Router (VALIDATION - 80ms)
        # =================================================================
        
        semantic_candidates = []
        sem_tool, sem_conf = None, 0.0
        
        if self.semantic_router and self.semantic_router.model:
            semantic_candidates = self.semantic_router.get_top_k_candidates(query, k=2)
            
            if semantic_candidates and semantic_candidates[0][1] >= 0.50:
                sem_tool, sem_conf = semantic_candidates[0]
                logger.info(f"[Semantic] Top: {sem_tool}={sem_conf:.3f}")
            else:
                logger.info("[Semantic] All scores < 0.50")
        else:
            logger.warning("[Semantic] Router unavailable")
        
        # =================================================================
        # DECISION LOGIC: Combine LLM + Semantic
        # =================================================================
        
        # Case 1: LLM high confidence + Semantic agrees
        if llm_tool and llm_conf >= 0.75 and llm_tool == sem_tool:
            logger.info(f"[Route] LLM + Semantic agree ({llm_conf:.2f}) --> {llm_tool}")
            return (llm_tool, max(llm_conf, sem_conf), "llm_semantic_agree")
        
        # Case 2: LLM very confident (trust it even if semantic disagrees)
        if llm_tool and llm_conf >= 0.85:
            logger.info(f"[Route] LLM very confident ({llm_conf:.2f}) --> {llm_tool}")
            if sem_tool and sem_tool != llm_tool:
                logger.info(f"[Route] Overriding semantic ({sem_tool})")
            return (llm_tool, llm_conf, "llm_high_confidence")
        
        # Case 3: Semantic confident + LLM agrees or absent
        if sem_tool and sem_conf >= 0.70 and (not llm_tool or llm_tool == sem_tool):
            logger.info(f"[Route] Semantic confident ({sem_conf:.2f}) --> {sem_tool}")
            return (sem_tool, sem_conf, "semantic_confident")
        
        # Case 4: DISAGREEMENT - Run parallel to resolve
        if llm_tool and sem_tool and llm_tool != sem_tool:
            logger.info(
                f"[Route] LLM vs Semantic disagreement: {llm_tool}({llm_conf:.2f}) "
                f"vs {sem_tool}({sem_conf:.2f}) --> PARALLEL"
            )
            
            try:
                candidates = [(llm_tool, llm_conf), (sem_tool, sem_conf)]
                selected_tool, result = self._execute_parallel_with_judge(query, candidates)
                self._cached_parallel_result = result
                return (selected_tool, 0.85, "parallel_disagreement")
            except Exception as e:
                logger.error(f"[Route] Parallel failed: {e}, using LLM decision")
                return (llm_tool or sem_tool, 0.70, "parallel_failed")
        
        # Case 5: LLM medium confidence, no semantic validation
        if llm_tool and llm_conf >= 0.60:
            logger.info(f"[Route] LLM medium confidence ({llm_conf:.2f}) --> {llm_tool}")
            return (llm_tool, llm_conf, "llm_medium_confidence")
        
        # Case 6: Only semantic has answer (LLM failed)
        if sem_tool:
            logger.info(f"[Route] LLM unavailable, using semantic --> {sem_tool}")
            return (sem_tool, sem_conf, "semantic_fallback")
        
        # Case 7: Both failed - fallback to GenericRagTool
        logger.warning("[Route] Both LLM and Semantic uncertain --> GenericRagTool fallback")
        return ("GenericRagTool", 0.30, "low_confidence_fallback")


    def _execute_parallel_with_judge(
        self, 
        query: str, 
        candidate_tools: List[Tuple[str, float]]
    ) -> Tuple[str, ToolResult]:
        """
        Execute top 2 candidate tools in parallel with LLM judge.
        ONLY called when ambiguous (margin < 0.20).
        """
        logger.info(f"[Parallel] Executing {len(candidate_tools)} tools")
        
        results = {}
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {}
            
            for tool_name, semantic_conf in candidate_tools:
                if tool_name in TOOL_MAP:
                    future = executor.submit(TOOL_MAP[tool_name], query, {})
                    futures[future] = (tool_name, semantic_conf)
            
            # Collect results with timeout
            for future in as_completed(futures, timeout=30):
                tool_name, semantic_conf = futures[future]
                try:
                    result = future.result()
                    results[tool_name] = {
                        "result": result,
                        "semantic_confidence": semantic_conf
                    }
                    logger.debug(f"[Parallel] {tool_name} completed")
                except Exception as e:
                    logger.error(f"[Parallel] {tool_name} failed: {e}")
        
        if not results:
            # Both failed
            return ("GenericRagTool", ToolResult(
                data={},
                explanation="Tools failed to execute.",
                confidence=0.0,
                format_hint="text",
                citations=[]
            ))
        
        if len(results) == 1:
            # Only one succeeded
            tool_name = list(results.keys())[0]
            return (tool_name, results[tool_name]["result"])
        
        # LLM Judge - pick best answer
        tool_names = list(results.keys())
        judge_prompt = f"""You are judging which answer is better for this query.


Query: "{query}"


Answer A ({tool_names[0]}):
{results[tool_names[0]]["result"].explanation[:300]}


Answer B ({tool_names[1]}):
{results[tool_names[1]]["result"].explanation[:300]}


Which answer is more relevant and accurate? Respond with ONLY: A or B
"""
        
        try:
            decision = self.rag_pipeline._generate_answer("", judge_prompt).strip().upper()
            
            if decision == "A":
                selected = tool_names[0]
            elif decision == "B":
                selected = tool_names[1]
            else:
                # Fallback to highest confidence
                selected = max(
                    results.items(), 
                    key=lambda x: x[1]["result"].confidence
                )[0]
            
            logger.info(f"[Judge] Selected {selected}")
            return (selected, results[selected]["result"])
            
        except Exception as e:
            logger.error(f"[Judge] Failed: {e}")
            # Fallback to highest confidence
            selected = max(
                results.items(), 
                key=lambda x: x[1]["result"].confidence
            )[0]
            return (selected, results[selected]["result"])
    
    def get_routing_metrics(self) -> Dict[str, Any]:
        """Get routing performance metrics."""
        return self.feedback.get_metrics()


    def _llm_description_route(self, query: str) -> Tuple[Optional[str], float]:
        """
        LLM-based routing using tool DESCRIPTIONS (not examples).
        Understands query INTENT and matches to tool PURPOSE.
        
        This is the PRIMARY router - generalizes to any entity names.
        """
        
        prompt = f"""You are a query routing expert. Analyze this query and determine which tool can answer it based on PURPOSE.


Query: "{query}"


Available Tools:


1. **TranscriptTool**
   Purpose: Handles questions about SPECIFIC STUDENT academic records - grades, GPA, courses taken by individual students, transcript data, enrollment status of named students in courses.
   Key capability: Queries data about individual students by name.


2. **PayrollTool**
   Purpose: Handles payroll schedules, pay periods, check dates, payment processing dates.
   Key capability: Payroll calendar and payment timing information.


3. **BorPlannerTool**
   Purpose: Handles Board of Regents meetings, committee schedules, governance events.
   Key capability: Meeting schedules and board governance calendars.


4. **GenericRagTool**
   Purpose: Handles institutional policies, procedures, handbooks, general information, academic calendar, enrollment procedures (NOT individual student data).
   Key capability: Policy documents and general institutional information.


Task: Which tool's PURPOSE best matches this query's INTENT?


Rules:
- If query asks about a SPECIFIC STUDENT's data (GPA, courses, grades) --> TranscriptTool
- If query asks about payroll dates/schedules --> PayrollTool
- If query asks about board meetings --> BorPlannerTool
- If query asks about policies/procedures/general info --> GenericRagTool


Respond with JSON only (no explanation):
{{"tool": "ToolName", "confidence": 0.85}}


Where confidence: 0.90-1.0 = very clear, 0.75-0.89 = clear, 0.60-0.74 = probable, <0.60 = uncertain
"""
        
        try:
            # Use RAG pipeline's LLM
            response = self.rag_pipeline._generate_answer("", prompt)
            
            # Parse JSON response
            import json
            # Extract JSON from response (handle if LLM adds extra text)
            json_match = re.search(r'\{[^}]+\}', response)
            if json_match:
                result = json.loads(json_match.group(0))
            else:
                result = json.loads(response)
            
            tool = result.get("tool")
            confidence = float(result.get("confidence", 0.0))
            
            # Validate tool exists
            if tool in TOOL_MAP and confidence >= 0.60:
                logger.info(f"[LLM Route] {tool} (conf={confidence:.2f})")
                return (tool, confidence)
            else:
                logger.warning(f"[LLM Route] Invalid tool or low confidence: {tool}={confidence:.2f}")
                return (None, 0.0)
        
        except Exception as e:
            logger.error(f"[LLM Route] Failed: {e}")
            return (None, 0.0)
