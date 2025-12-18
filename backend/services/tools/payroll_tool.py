"""PayrollTool - generic payroll calendar agent."""

from typing import Dict, Any
import re
from datetime import datetime

import pandas as pd

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
    start_date = (
        _normalize_date(params.get("start_date", ""))
        if params.get("start_date")
        else ""
    )
    end_date = (
        _normalize_date(params.get("end_date", ""))
        if params.get("end_date")
        else ""
    )
    check_date = (
        _normalize_date(params.get("check_date", ""))
        if params.get("check_date")
        else ""
    )

    payroll_no = params.get("payroll_no")
    ask_days_diff = params.get("ask_days_diff", False)

    q_lower = question.lower()

    df_work = df.copy()

    # Normalize columns to lower for robustness
    df_work.columns = [c.strip().lower() for c in df_work.columns]

    # Ensure key columns exist
    required_cols = ["payroll_no", "start_date", "end_date", "check_date"]
    missing = [c for c in required_cols if c not in df_work.columns]
    if missing:
        return ToolResult(
            data={},
            explanation=(
                "The payroll calendar is missing expected columns: "
                + ", ".join(missing)
            ),
            confidence=0.0,
            format_hint="text",
            citations=[],
        )

    # Year filtering (if available)
    if year:
        if "year" in df_work.columns:
            df_work = df_work[df_work["year"] == int(year)]
        elif "start_date" in df_work.columns:
            df_work["year"] = pd.to_datetime(
                df_work["start_date"], errors="coerce"
            ).dt.year
            df_work = df_work[df_work["year"] == int(year)]

    # Parse dates for downstream calculations
    for col in ("start_date", "end_date", "check_date"):
        if col in df_work.columns:
            df_work[col] = pd.to_datetime(
                df_work[col].astype(str).apply(_normalize_date), errors="coerce"
            )

    # ------------------------------------------------------------------
    # NEW LOGIC 1: Which pay period has least date difference
    # ------------------------------------------------------------------
    # "which Pay period number has the least date difference between the End date and Check Date?"
    if (
        "least" in q_lower
        and "difference" in q_lower
        and "end date" in q_lower
        and "check date" in q_lower
    ):
        df_work["end_check_diff_days"] = (
            df_work["check_date"] - df_work["end_date"]
        ).abs().dt.days
        df_valid = df_work.dropna(subset=["end_check_diff_days"])
        if df_valid.empty:
            return ToolResult(
                data={},
                explanation=(
                    "Could not compute date differences; the payroll calendar has "
                    "invalid or missing end or check dates."
                ),
                confidence=0.0,
                format_hint="text",
                citations=["payroll_calendar"],
            )
        best = df_valid.loc[df_valid["end_check_diff_days"].idxmin()]
        best_no = int(best["payroll_no"])
        diff = int(best["end_check_diff_days"])
        explanation = (
            f"Pay period {best_no} has the least date difference between the "
            f"end date and the check date: {diff} day(s)."
        )
        return ToolResult(
            data={
                "payroll_no": best_no,
                "difference_days": diff,
            },
            explanation=explanation,
            confidence=0.95,
            format_hint="text",
            citations=["payroll_calendar"],
        )

    # ------------------------------------------------------------------
    # NEW LOGIC 2: Days difference between start_date and check_date
    # ------------------------------------------------------------------
    # "What is the days difference between the Pay Period start date and Check date for Pay roll number 10"
    if ask_days_diff and payroll_no is not None:
        match = df_work[df_work["payroll_no"] == int(payroll_no)]
        if match.empty:
            return ToolResult(
                data={},
                explanation=f"No payroll period found with number {payroll_no}.",
                confidence=0.4,
                format_hint="text",
                citations=["payroll_calendar"],
            )
        row = match.iloc[0]
        if pd.isna(row["start_date"]) or pd.isna(row["check_date"]):
            return ToolResult(
                data={},
                explanation=(
                    f"Dates are missing for payroll period {payroll_no}, "
                    "so the day difference cannot be computed."
                ),
                confidence=0.3,
                format_hint="text",
                citations=["payroll_calendar"],
            )
        diff_days = (row["check_date"] - row["start_date"]).days
        explanation = (
            f"For payroll period {payroll_no}, the difference between the "
            f"start date and the check date is {diff_days} day(s)."
        )
        return ToolResult(
            data={
                "payroll_no": int(payroll_no),
                "difference_days": diff_days,
            },
            explanation=explanation,
            confidence=0.95,
            format_hint="text",
            citations=["payroll_calendar"],
        )

    # ------------------------------------------------------------------
    # NEW LOGIC 3: Check date for a specific payroll_no
    # ------------------------------------------------------------------
    # "What is the check date for Pay period 3"
    if payroll_no is not None and "check date" in q_lower:
        match = df_work[df_work["payroll_no"] == int(payroll_no)]
        if match.empty:
            return ToolResult(
                data={},
                explanation=f"No payroll period found with number {payroll_no}.",
                confidence=0.4,
                format_hint="text",
                citations=["payroll_calendar"],
            )
        row = match.iloc[0]
        if pd.isna(row["check_date"]):
            return ToolResult(
                data={},
                explanation=(
                    f"The check date for payroll period {payroll_no} is not available."
                ),
                confidence=0.3,
                format_hint="text",
                citations=["payroll_calendar"],
            )
        check_date_str = row["check_date"].strftime("%Y-%m-%d")
        start_str = (
            row["start_date"].strftime("%Y-%m-%d")
            if not pd.isna(row["start_date"])
            else "N/A"
        )
        end_str = (
            row["end_date"].strftime("%Y-%m-%d")
            if not pd.isna(row["end_date"])
            else "N/A"
        )
        explanation = (
            f"For payroll period {payroll_no}, the check date is {check_date_str}. "
            f"The work period runs from {start_str} to {end_str}."
        )
        return ToolResult(
            data={
                "payroll_no": int(payroll_no),
                "check_date": check_date_str,
                "start_date": start_str,
                "end_date": end_str,
            },
            explanation=explanation,
            confidence=0.95,
            format_hint="text",
            citations=["payroll_calendar"],
        )

    # ------------------------------------------------------------------
    # EXISTING LOGIC 1: Period (start_date/end_date) → check date
    # ------------------------------------------------------------------
    if (start_date and end_date) or ("between" in q_lower and "check date" in q_lower):
        for col in ("start_date", "end_date", "check_date"):
            if col in df_work.columns:
                df_work[col] = df_work[col].astype(str).apply(_normalize_date)

        mask = (df_work["start_date"] == start_date) & (
            df_work["end_date"] == end_date
        )
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

    # ------------------------------------------------------------------
    # EXISTING LOGIC 2: Check date → period
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # EXISTING LOGIC 3: How many periods in a year?
    # ------------------------------------------------------------------
    if "how many" in q_lower and "payroll" in q_lower and ("year" in q_lower or year):
        if "year" not in df_work.columns and "start_date" in df_work.columns:
            df_work["year"] = pd.to_datetime(
                df_work["start_date"], errors="coerce"
            ).dt.year
        if year:
            df_year = df_work[df_work["year"] == int(year)]
        else:
            df_year = df_work

        count = len(df_year)
        explanation = (
            f"There are {count} payroll periods in {year or 'the available data'}."
        )
        rows = df_year[
            ["payroll_no", "start_date", "end_date", "check_date"]
        ].to_dict("records")
        return ToolResult(
            data={
                "year": int(year) if year else None,
                "count": count,
                "rows": rows,
            },
            explanation=explanation,
            confidence=0.9,
            format_hint="table",
            citations=["payroll_calendar"],
        )

    # ------------------------------------------------------------------
    # Fallback → dataframe agent
    # ------------------------------------------------------------------
    df_answer = run_df_agent(question, df_work, df_name="payroll calendar")
    return ToolResult(
        data={"raw": df_answer.get("raw")},
        explanation=df_answer["answer"],
        confidence=0.75,
        format_hint="text",
        citations=["payroll_calendar"],
    )
