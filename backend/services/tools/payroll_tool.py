"""PayrollTool - generic payroll calendar agent."""

from typing import Dict, Any
import pandas as pd
from datetime import datetime
from core import get_logger
from services.data_views import get_payroll_df
from models.schemas import ToolResult
from services.df_agent import run_df_agent


logger = get_logger(__name__)


def _normalize_date(value: str) -> str:
    """Normalize date strings to YYYY-MM-DD if possible."""
    if not value:
        return ""
    for fmt in ("%m/%d/%Y", "%m/%d/%y", "%Y-%m-%d"):
        try:
            return datetime.strptime(value.strip(), fmt).strftime("%Y-%m-%d")
        except Exception:
            continue
    return value.strip()


def _extract_date_from_text(text: str) -> str:
    """Very simple date pattern extractor."""
    import re

    m = re.search(r"(\d{1,2}/\d{1,2}/\d{4})", text)
    if m:
        return _normalize_date(m.group(1))
    return ""


def answer(question: str, params: Dict[str, Any] | None = None) -> ToolResult:
    df = get_payroll_df()
    if df.empty:
        return ToolResult(
            data={},
            explanation="No payroll calendar data available.",
            confidence=0.0,
            format_hint="text",
            citations=[],
        )

    params = params or {}
    year = params.get("year")
    start_date = _normalize_date(params.get("start_date", "")) if params.get("start_date") else ""
    end_date = _normalize_date(params.get("end_date", "")) if params.get("end_date") else ""
    check_date = _normalize_date(params.get("check_date", "")) if params.get("check_date") else ""

    q_lower = question.lower()

    df_work = df.copy()

    if year:
        if "year" in df_work.columns:
            df_work = df_work[df_work["year"] == int(year)]
        elif "start_date" in df_work.columns:
            df_work["year"] = pd.to_datetime(df_work["start_date"], errors="coerce").dt.year
            df_work = df_work[df_work["year"] == int(year)]

    # 1) Period → check date
    if (start_date and end_date) or ("between" in q_lower and "check date" in q_lower):
        for col in ("start_date", "end_date", "check_date"):
            if col in df_work.columns:
                df_work[col] = df_work[col].astype(str).apply(_normalize_date)

        mask = (df_work["start_date"] == start_date) & (df_work["end_date"] == end_date)
        match = df_work[mask]
        if match.empty:
            return ToolResult(
                data={},
                explanation=(
                    f"No payroll period found between {start_date or 'N/A'} and "
                    f"{end_date or 'N/A'} in the payroll calendar."
                ),
                confidence=0.4,
                format_hint="text",
                citations=["payroll_calendar"],
            )

        rows = match.to_dict("records")
        explanation = "For the requested payroll period, the check date(s) are:\n"
        for row in rows:
            explanation += (
                f"- Payroll #{row.get('payroll_no', '')}: "
                f"{row.get('start_date', '')}–{row.get('end_date', '')} → "
                f"check date {row.get('check_date', '')}\n"
            )

        return ToolResult(
            data={"rows": rows},
            explanation=explanation.strip(),
            confidence=0.95,
            format_hint="text",
            citations=["payroll_calendar"],
        )

    # 2) Check date → period
    if check_date or ("check date" in q_lower and "payroll period" in q_lower):
        for col in ("start_date", "end_date", "check_date"):
            if col in df_work.columns:
                df_work[col] = df_work[col].astype(str).apply(_normalize_date)

        target = check_date or _extract_date_from_text(q_lower)
        if not target:
            return ToolResult(
                data={},
                explanation=(
                    "Please provide the check date (e.g., 2/6/2026 or 2026-02-06) "
                    "to find the corresponding payroll period."
                ),
                confidence=0.2,
                format_hint="text",
                citations=[],
            )

        match = df_work[df_work["check_date"] == target]
        if match.empty:
            return ToolResult(
                data={},
                explanation=f"No payroll period found with check date {target}.",
                confidence=0.4,
                format_hint="text",
                citations=["payroll_calendar"],
            )

        rows = match.to_dict("records")
        explanation = "For the requested check date, the payroll period(s) are:\n"
        for row in rows:
            explanation += (
                f"- Payroll #{row.get('payroll_no', '')}: "
                f"{row.get('start_date', '')}–{row.get('end_date', '')} "
                f"(check date {row.get('check_date', '')})\n"
            )
        return ToolResult(
            data={"rows": rows},
            explanation=explanation.strip(),
            confidence=0.95,
            format_hint="text",
            citations=["payroll_calendar"],
        )

    # 3) How many periods in a year?
    if "how many" in q_lower and "payroll" in q_lower and ("year" in q_lower or year):
        if "year" not in df_work.columns and "start_date" in df_work.columns:
            df_work["year"] = pd.to_datetime(df_work["start_date"], errors="coerce").dt.year
        if year:
            df_year = df_work[df_work["year"] == int(year)]
        else:
            df_year = df_work

        count = len(df_year)
        explanation = f"There are {count} payroll periods in {year or 'the available data'}."
        rows = df_year[["payroll_no", "start_date", "end_date", "check_date"]].to_dict("records")
        return ToolResult(
            data={"year": int(year) if year else None, "count": count, "rows": rows},
            explanation=explanation,
            confidence=0.9,
            format_hint="table",
            citations=["payroll_calendar"],
        )

    # Fallback → dataframe agent
    df_answer = run_df_agent(question, df_work, df_name="payroll calendar")
    return ToolResult(
        data={"raw": df_answer.get("raw")},
        explanation=df_answer["answer"],
        confidence=0.75,
        format_hint="text",
        citations=["payroll_calendar"],
    )
