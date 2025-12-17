from core import init_logger, get_logger
init_logger()
from services.tools import bor_planner_tool

questions = [
    "When is the Board meetings due?",
    "when is ACCT NLS '26 event scheduled",
]

for q in questions:
    res = bor_planner_tool.answer(q, {})
    print("Q:", q)
    print("A:", res.explanation)
    print("----")
