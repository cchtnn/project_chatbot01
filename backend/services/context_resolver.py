"""
backend/services/context_resolver.py
Resolves ambiguous references in follow-up questions using session history.
Handles pronouns, demonstratives, and contextual references across all domains.
"""

import re
from typing import List, Dict, Optional
from core import get_logger

logger = get_logger(__name__)


class ContextResolver:
    """
    Enriches queries with context from conversation history.
    Resolves: she/he/they, this/that, the course/student, etc.
    """
    
    # Patterns indicating need for context
    AMBIGUOUS_PATTERNS = [
        r'\b(she|he|they|his|her|their|him|them)\b',  # Pronouns
        r'\b(this|that|these|those)\s+(course|student|policy|meeting|period)\b',  # Demonstratives
        r'^(which|what|when|how)\s+(course|student)',  # Question words without entity
        r'\b(the same|another|more|also)\b',  # References
    ]
    
    def needs_context(self, query: str) -> bool:
        """Check if query likely needs context resolution."""
        query_lower = query.lower()
        
        for pattern in self.AMBIGUOUS_PATTERNS:
            if re.search(pattern, query_lower):
                logger.info(f"[ContextResolver] Ambiguous pattern detected: {pattern}")
                return True
        
        return False
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities from text (simple heuristic-based).
        Returns: {"students": [...], "courses": [...], "dates": [...]}
        """
        entities = {
            "students": [],
            "courses": [],
            "dates": [],
            "periods": []
        }
        
        # Student names (capitalized First Last)
        student_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b'
        students = re.findall(student_pattern, text)
        entities["students"] = list(set(students))[:3]  # Top 3 unique
        
        # Course titles (often have numbers or multiple capitals)
        course_pattern = r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+\d+|[A-Z]{2,}\s*\d+)'
        courses = re.findall(course_pattern, text)
        entities["courses"] = list(set(courses))[:3]
        
        # Dates (simple)
        date_pattern = r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b'
        dates = re.findall(date_pattern, text)
        entities["dates"] = list(set(dates))[:2]
        
        # Pay periods
        period_pattern = r'\b(pay\s*period|p)\s*(\d+)\b'
        periods = re.findall(period_pattern, text.lower())
        entities["periods"] = [f"pay period {p[1]}" for p in periods][:2]
        
        return entities
    
    def resolve(self, query: str, history: List[Dict[str, str]]) -> str:
        """
        Resolve ambiguous query using conversation history.
        
        Args:
            query: Current user query
            history: List of {"question": "...", "answer": "..."} from session
        
        Returns:
            Enriched query with context injected
        """
        if not self.needs_context(query):
            logger.info("[ContextResolver] No ambiguous patterns - query is self-contained")
            return query
        
        if not history or len(history) == 0:
            logger.info("[ContextResolver] No history available")
            return query
        
        # Extract entities from last 3 exchanges (most recent context)
        recent_history = history[-3:] if len(history) >= 3 else history
        
        all_entities = {
            "students": [],
            "courses": [],
            "dates": [],
            "periods": []
        }
        
        for exchange in recent_history:
            question_entities = self.extract_entities(exchange.get("question", ""))
            answer_entities = self.extract_entities(exchange.get("answer", ""))
            
            for key in all_entities.keys():
                all_entities[key].extend(question_entities.get(key, []))
                all_entities[key].extend(answer_entities.get(key, []))
        
        # Deduplicate, keep order (most recent first)
        for key in all_entities.keys():
            seen = set()
            unique = []
            for item in reversed(all_entities[key]):  # Reverse to prioritize recent
                if item not in seen:
                    seen.add(item)
                    unique.append(item)
            all_entities[key] = list(reversed(unique))
        
        logger.info(f"[ContextResolver] Extracted entities: {all_entities}")
        
        # Build context string
        enriched_query = query
        context_parts = []
        
        if all_entities["students"]:
            context_parts.append(f"Student(s) mentioned recently: {', '.join(all_entities['students'][:2])}")
        
        if all_entities["courses"]:
            context_parts.append(f"Course(s) mentioned recently: {', '.join(all_entities['courses'][:2])}")
        
        if all_entities["periods"]:
            context_parts.append(f"Pay period(s) mentioned recently: {', '.join(all_entities['periods'][:2])}")
        
        if context_parts:
            context_hint = "\n[Context from conversation: " + "; ".join(context_parts) + "]"
            enriched_query = query + context_hint
            logger.info(f"[ContextResolver] Enriched query: {enriched_query}")
        
        return enriched_query
