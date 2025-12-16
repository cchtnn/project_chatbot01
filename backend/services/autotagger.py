"""
Enterprise AutoTagger - LLM-powered document classification.

Extracts domain, type, keywords from full document context (not chunks).
Tags ONCE per document → stored in ChromaDB metadata → fast filtering.
"""

import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import ollama

from core import get_logger, Language
from config import get_settings
from services.text_processor import TextChunk
from services.document_parser import ParsedDocument

logger = get_logger(__name__)


class AutoTagger:
    """LLM-powered document classification and metadata extraction."""
    
    DOMAINS = [
        "policy", "hr", "finance", "legal", "academic", "governance", 
        "student_services", "healthcare", "technical", "general"
    ]
    
    TYPES = [
        "handbook", "policy_doc", "form", "report", "transcript", 
        "schedule", "guide", "contract", "manual", "presentation"
    ]
    
    def __init__(self):
        self.settings = get_settings()
        self.model = self.settings.ollama_model or "llama3.2:3b"
        logger.info(f"AutoTagger ready: {self.model}")

    def tag_document(self, parsed_doc: ParsedDocument) -> Dict[str, Any]:
        """
        Tag entire document with domain, type, keywords.
        
        Args:
            parsed_doc: Full document from parser
            
        Returns:
            Tags for ALL chunks (shared document-level metadata)
        """
        if not parsed_doc.content:
            return {"domain": "general", "type": "unknown", "keywords": []}
        
        # Sample first 1000 chars + filename for context
        sample = " ".join(parsed_doc.content[:3])[:1000]
        filename_hint = parsed_doc.filename.lower()
        
        tags = self._classify_with_llm(sample, filename_hint)
        tags.update({
            "filename": parsed_doc.filename,
            "document_type": parsed_doc.document_type.value,
            "language": self._detect_language(sample),
            "user_id": parsed_doc.user_id,
            "file_hash": parsed_doc.file_hash,
            "is_public": parsed_doc.user_id is None,
        })
        
        logger.info(f"Tagged {parsed_doc.filename}: {tags['domain']} ({tags['type']})")
        return tags

    def tag_chunks(self, chunks: List[TextChunk], doc_tags: Dict[str, Any]) -> List[TextChunk]:
        """Attach document tags to all chunks."""
        for chunk in chunks:
            chunk.metadata.update(doc_tags)
        return chunks

    def _classify_with_llm(self, sample: str, filename: str) -> Dict[str, Any]:
        """LLM classification prompt."""
        prompt = f"""
Analyze this document and classify it precisely.

FILENAME: {filename}
CONTENT SAMPLE: {sample[:800]}...

AVAILABLE DOMAINS: {', '.join(self.DOMAINS)}
AVAILABLE TYPES: {', '.join(self.TYPES)}

Return ONLY valid JSON:
{{
  "domain": "one_domain_from_list",
  "type": "one_type_from_list", 
  "keywords": ["3-5 keywords"],
  "confidence": 0.0-1.0
}}
"""
        
        try:
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                options={"temperature": 0.1, "num_predict": 150}
            )
            return self._parse_tags(response['response'])
        except Exception as e:
            logger.warning(f"LLM tagging failed: {e}")
            return self._fallback_tags(filename)

    def _parse_tags(self, llm_response: str) -> Dict[str, Any]:
        """Extract JSON from LLM response."""
        try:
            # Extract JSON block
            start = llm_response.find('{')
            end = llm_response.rfind('}') + 1
            json_str = llm_response[start:end]
            import json
            tags = json.loads(json_str)
            return {
                "domain": tags.get("domain", "general"),
                "type": tags.get("type", "unknown"),
                "keywords": tags.get("keywords", []),
                "confidence": tags.get("confidence", 0.5)
            }
        except:
            return self._fallback_tags("unknown")

    def _fallback_tags(self, filename: str) -> Dict[str, Any]:
        """Rule-based fallback."""
        filename_lower = filename.lower()
        if any(x in filename_lower for x in ["policy", "guideline"]):
            return {"domain": "policy", "type": "policy_doc", "keywords": ["policy"], "confidence": 0.7}
        elif any(x in filename_lower for x in ["hr", "employee", "staff"]):
            return {"domain": "hr", "type": "handbook", "keywords": ["hr"], "confidence": 0.7}
        elif any(x in filename_lower for x in ["finance", "budget", "invoice"]):
            return {"domain": "finance", "type": "report", "keywords": ["finance"], "confidence": 0.7}
        return {"domain": "general", "type": "unknown", "keywords": [], "confidence": 0.3}

    def _detect_language(self, sample: str) -> str:
        """Simple language detection."""
        sample_lower = sample.lower()
        if any(word in sample_lower for word in ["policy", "employee", "finance"]):
            return "en"
        # Add Spanish/Navajo detection later
        return "en"


class AutoTagger:
    # existing __init__ and tag_document / tag methods ...

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        """
        Generic text generation wrapper so other services (RAG) can reuse
        the same LLM client for answering questions.
        """
        # Reuse whatever client AutoTagger already uses internally
        response = self.llm.generate(prompt, max_tokens=max_tokens)
        # If your llm client returns a dict or list, adapt this line accordingly.
        return response
