"""
backend/services/conversational_handler.py
Conversational intent detection for non-informational queries.
Enterprise-grade chitchat handler inspired by Perplexity/Claude AI.
"""

import re
from typing import Tuple, Optional
from models.schemas import ToolResult
from core import get_logger

logger = get_logger(__name__)


class ConversationalHandler:
    """
    Detects and handles conversational queries (greetings, thanks, help).
    Returns None if query is informational (should route to domain tools).
    """
    
    # Conversational patterns (order matters - check specific before generic)
    PATTERNS = {
        "greeting": [
            r"^(hi|hello|hey|good morning|good afternoon|good evening)[\s\.,!?]*$",
            r"^(hi|hello|hey)\s+(there|everyone|team|jericho)[\s\.,!?]*$",
        ],
        "thanks": [
            r"^(thank you|thanks|thank|thx|ty)[\s\.,!?]*$",
            r"^(thank you|thanks)\s+(so much|very much|a lot)[\s\.,!?]*$",
        ],
        "help": [
            r"^(help|help me|how do i use this|what can you do|what can i ask)[\s\.,!?]*$",
            r"^what\s+(can|could)\s+(you|i)\s+(do|ask)[\s\.,!?]*$",
        ],
        "goodbye": [
            r"^(bye|goodbye|see you|farewell|take care)[\s\.,!?]*$",
        ]
    }
    
    # Professional responses (Markdown formatted)
    RESPONSES = {
        "greeting": """Hello! I'm the **Diné College Assistant**, powered by Jericho.

I can help you with:
- **📊 Student Transcripts** - GPA, courses, academic records
- **💰 Payroll Calendar** - Pay periods, check dates
- **📅 Board Meetings** - BOR schedules, committee meetings
- **📚 Policies & Handbooks** - HR policies, procedures, rules

What would you like to know?""",
        
        "thanks": """You're welcome! Feel free to ask if you have more questions about Diné College. 😊""",
        
        "help": """I'm here to answer questions about **Diné College** across multiple domains:

### 📊 Student Transcripts
- "What's Trista Barrett's GPA?"
- "Which courses is Leslie enrolled in?"
- "Show top 5 students by GPA"

### 💰 Payroll Calendar
- "When is the check date for pay period 5?"
- "Show payroll schedule for 2024"

### 📅 Board of Regents
- "When is the next Board meeting?"
- "Finance committee meeting schedule"

### 📚 Policies & Handbooks
- "What are the housing rules?"
- "Show me the sick leave policy"
- "Academic calendar for Fall 2024"

Just ask your question naturally!""",
        
        "goodbye": """Goodbye! Come back anytime you need assistance with Diné College information. 👋"""
    }
    
    def detect(self, query: str) -> Tuple[Optional[str], float]:
        """
        Detect if query is conversational.
        
        Returns:
            (intent, confidence) - e.g., ("greeting", 0.95) or (None, 0.0)
        """
        query_clean = query.strip().lower()
        
        # Empty query
        if not query_clean:
            return (None, 0.0)
        
        # Check patterns
        for intent, patterns in self.PATTERNS.items():
            for pattern in patterns:
                if re.match(pattern, query_clean, re.IGNORECASE):
                    logger.info(f"[Conversational] Detected '{intent}' intent")
                    return (intent, 0.95)
        
        # Not conversational
        return (None, 0.0)
    
    def handle(self, intent: str) -> ToolResult:
        """Generate response for conversational intent."""
        response = self.RESPONSES.get(intent, self.RESPONSES["help"])
        
        return ToolResult(
            data={"intent": intent},
            explanation=response,
            confidence=0.95,
            format_hint="text",
            citations=[]
        )
