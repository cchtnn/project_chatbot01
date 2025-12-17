from core import init_logger, get_logger
init_logger()
from services.tools import payroll_tool


logger = get_logger(__name__)

if __name__ == "__main__":
    q = "What is the payroll period for check date 2/6/2026?"
    result = payroll_tool.answer(
        question=q,
        params={"check_date": "2/6/2026", "year": 2026},
    )
    print("\n=== TOOL RESULT ===")
    print("Explanation:", result.explanation)
    print("Confidence:", result.confidence)
    print("Format hint:", result.format_hint)
    print("Data keys:", result.data.keys())
