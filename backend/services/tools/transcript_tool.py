from typing import Dict, Any
import pandas as pd
from core import get_logger
from services.data_views import get_transcript_df
from models.schemas import ToolResult
from services.df_agent import run_df_agent

logger = get_logger(__name__)


def _find_student_name_col(df: pd.DataFrame) -> str | None:
    cols = [c for c in df.columns]
    for c in cols:
        if c.strip().lower() == "student name":
            return c
    for c in cols:
        cl = c.strip().lower()
        if "student" in cl and "name" in cl:
            return c
    return None


def _find_gpa_col(df: pd.DataFrame) -> str | None:
    cols = [c for c in df.columns]
    for c in cols:
        if c.strip().lower() == "gpa":
            return c
    for c in cols:
        if "gpa" in c.strip().lower():
            return c
    return None


def answer(question: str, params: Dict[str, Any] | None = None) -> ToolResult:
    df = get_transcript_df()
    if df.empty:
        return ToolResult(
            data={},
            explanation="No transcript data available.",
            confidence=0.0,
            format_hint="text",
            citations=[],
        )

    params = params or {}
    student_name = params.get("student_name")

    name_col = _find_student_name_col(df)
    gpa_col = _find_gpa_col(df)

    if not name_col:
        return ToolResult(
            data={},
            explanation="Transcript data is loaded but no student name column could be identified.",
            confidence=0.4,
            format_hint="text",
            citations=[],
        )

    # Single-student questions
    if student_name:
        target = student_name.strip().lower()

        mask = (
            df[name_col]
            .astype(str)
            .str.lower()
            .str.contains(target, na=False)
        )
        student_df = df[mask]

        if student_df.empty:
            parts = target.split()
            if len(parts) >= 2:
                last_name = parts[-1]
                mask = (
                    df[name_col]
                    .astype(str)
                    .str.lower()
                    .str.contains(last_name, na=False)
                )
                student_df = df[mask]

        if student_df.empty:
            return ToolResult(
                data={},
                explanation=f"No transcript records found for '{student_name}'.",
                confidence=0.5,
                format_hint="text",
                citations=[],
            )

        row = student_df.iloc[0].to_dict()
        gpa_val = None
        if gpa_col and gpa_col in row:
            try:
                gpa_val = float(row[gpa_col])
            except Exception:
                gpa_val = row[gpa_col]

        explanation = f"GPA details for {row.get(name_col, student_name)}: GPA = {gpa_val}."
        return ToolResult(
            data={"student": row, "gpa": gpa_val},
            explanation=explanation,
            confidence=0.9,
            format_hint="text",
            citations=["merged_transcripts.csv"],
        )

    # Aggregate questions
    q_lower = question.lower()

    if "how many" in q_lower and "student" in q_lower:
        count = df[name_col].dropna().nunique()
        explanation = f"There are {count} unique students in the transcript data."
        return ToolResult(
            data={"student_count": count},
            explanation=explanation,
            confidence=0.9,
            format_hint="text",
            citations=["merged_transcripts.csv"],
        )

    if "top" in q_lower and gpa_col:
        try:
            df_num = df.copy()
            df_num[gpa_col] = pd.to_numeric(df_num[gpa_col], errors="coerce")
            top = (
                df_num.dropna(subset=[gpa_col])
                .sort_values(gpa_col, ascending=False)
            )
            top = top[[name_col, gpa_col]].drop_duplicates(subset=[name_col]).head(5)
            explanation = (
                "**Top students by GPA:**\n"
                + top.rename(columns={name_col: "Student Name", gpa_col: "GPA"}).to_markdown(index=False)
            )
            return ToolResult(
                data={"rows": top.to_dict("records")},
                explanation=explanation,
                confidence=0.9,
                format_hint="table",
                citations=["merged_transcripts.csv"],
            )
        except Exception as e:
            logger.warning(f"Top GPA computation failed: {e}")

    # Fallback → dataframe agent
    df_answer = run_df_agent(question, df, df_name="transcript records")
    return ToolResult(
        data={"raw": df_answer.get("raw")},
        explanation=df_answer["answer"],
        confidence=0.75,
        format_hint="text",
        citations=["merged_transcripts.csv"],
    )
