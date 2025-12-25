from typing import Dict, Any, Optional, Tuple, List
import pandas as pd
import re, os

from core import get_logger
from services.data_views import get_transcript_df
from models.schemas import ToolResult
from services.df_agent import run_df_agent

logger = get_logger(__name__)


def _find_col(df: pd.DataFrame, target: str) -> Optional[str]:
    """Find a column whose normalized name matches target exactly or contains it."""
    cols = list(df.columns)
    target_l = target.strip().lower()

    for c in cols:
        if c.strip().lower() == target_l:
            return c
    for c in cols:
        cl = c.strip().lower()
        if target_l in cl:
            return c
    return None


def _find_student_name_col(df: pd.DataFrame) -> Optional[str]:
    cols = list(df.columns)
    for c in cols:
        if c.strip().lower() == "student name":
            return c
    for c in cols:
        cl = c.strip().lower()
        if "student" in cl and "name" in cl:
            return c
    return None


def _find_gpa_col(df: pd.DataFrame) -> Optional[str]:
    return _find_col(df, "gpa")


def _find_quality_points_col(df: pd.DataFrame) -> Optional[str]:
    return _find_col(df, "quality points")


def _find_hours_gpa_col(df: pd.DataFrame) -> Optional[str]:
    # Hours GPA is often the credit hours used in GPA calc
    return _find_col(df, "hours gpa")


def _compute_student_gpa(
    sdf: pd.DataFrame,
    gpa_col: Optional[str],
    qp_col: Optional[str],
    hrs_col: Optional[str],
) -> Tuple[Optional[float], str]:
    """
    Compute a student's GPA from all their transcript rows.

    Priority:
    1) If a non-null GPA value exists in gpa_col, use its mean.
    2) Else, if quality points & hours-gpa present, compute sum(QP)/sum(Hours).
    3) Else, return None.
    """
    # 1) Direct GPA column
    if gpa_col and gpa_col in sdf.columns:
        gpa_series = pd.to_numeric(sdf[gpa_col], errors="coerce")
        gpa_series = gpa_series.dropna()
        if not gpa_series.empty:
            return float(gpa_series.mean()), "from GPA column"

    # 2) Derived from quality points / hours
    if qp_col and hrs_col and qp_col in sdf.columns and hrs_col in sdf.columns:
        qp = pd.to_numeric(sdf[qp_col], errors="coerce")
        hrs = pd.to_numeric(sdf[hrs_col], errors="coerce")
        mask = (hrs > 0) & qp.notna()
        if mask.any():
            total_qp = float(qp[mask].sum())
            total_hrs = float(hrs[mask].sum())
            if total_hrs > 0:
                return total_qp / total_hrs, "from Quality Points / Hours GPA"

    return None, "no numeric GPA available"


def _normalize_name(s: str) -> str:
    return " ".join(s.strip().lower().split())


def _filter_student(df: pd.DataFrame, name_col: str, student_name: str) -> pd.DataFrame:
    """Robust matching: full-name then last-name fallback."""
    target = _normalize_name(student_name)
    
    logger.info(f"    Filtering for student: '{student_name}' (normalized: '{target}')")
    
    # Show sample of available names for debugging
    sample_names = df[name_col].dropna().unique()[:5]
    logger.info(f"    Sample available student names: {list(sample_names)}")
    logger.info(f"    Total unique students in data: {df[name_col].nunique()}")

    # Full-name contains
    mask = df[name_col].astype(str).apply(_normalize_name).str.contains(target, na=False)
    sdf = df[mask]
    
    logger.info(f"    Full-name match found: {len(sdf)} records")

    if sdf.empty:
        parts = target.split()
        if len(parts) >= 2:
            last_name = parts[-1]
            logger.info(f"    Trying last-name fallback: '{last_name}'")
            mask = df[name_col].astype(str).str.lower().str.contains(last_name, na=False)
            sdf = df[mask]
            logger.info(f"    Last-name match found: {len(sdf)} records")
            
            # If still no match, show similar names
            if sdf.empty:
                logger.warning(f"    No match found for '{student_name}'")
                # Try to find similar names using fuzzy matching on last name
                all_names = df[name_col].astype(str).str.lower()
                similar = all_names[all_names.str.contains(last_name[:3], na=False)].unique()[:5]
                if len(similar) > 0:
                    logger.info(f"    Similar names found (by last name prefix): {list(similar)}")

    return sdf


def _build_courses_table(sdf: pd.DataFrame) -> List[Dict[str, Any]]:
    cols = list(sdf.columns)
    course_num_col = None
    course_title_col = None
    term_col = None
    subterm_col = None
    grade_col = None

    for c in cols:
        cl = c.strip().lower()
        if "course number" == cl or ("course" in cl and "number" in cl):
            course_num_col = c
        if "course title" == cl or ("course" in cl and "title" in cl):
            course_title_col = c
        if cl == "term":
            term_col = c
        if "subterm" in cl:
            subterm_col = c
        if cl == "grade":
            grade_col = c

    logger.info(f"    Course table columns identified:")
    logger.info(f"      - Course Number: {course_num_col}")
    logger.info(f"      - Course Title: {course_title_col}")
    logger.info(f"      - Term: {term_col}")
    logger.info(f"      - Subterm: {subterm_col}")
    logger.info(f"      - Grade: {grade_col}")

    subset_cols = [
        c for c in [course_num_col, course_title_col, term_col, subterm_col, grade_col] if c
    ]
    if not subset_cols:
        logger.warning("    No course-related columns found")
        return []

    subset = (
        sdf[subset_cols]
        .rename(
            columns={
                course_num_col or "Course Number": "Course Number",
                course_title_col or "Course Title": "Course Title",
                term_col or "Term": "Term",
                subterm_col or "Subterm": "Subterm",
                grade_col or "Grade": "Grade",
            }
        )
        .drop_duplicates()
    )
    
    logger.info(f"    Built courses table: {len(subset)} unique course records")
    
    return subset.to_dict("records")


def answer(question: str, params: Dict[str, Any] | None = None) -> ToolResult:
    logger.info("\n" + "="*80)
    logger.info("TRANSCRIPT TOOL EXECUTION")
    logger.info(f"Question: {question}")
    logger.info(f"Parameters: {params}")
    logger.info("="*80)
    
    try:
        # Import the CSV handler
        from services.tools.student_transcript_csv_handler import get_csv_transcript_handler
        
        # Get the transcript CSV path - this should point to merged CSV after ZIP upload
        from services.data_views import get_transcript_csv_path
        csv_path = get_transcript_csv_path()
        
        # Additional check: Look for session-specific merged CSV if params contain session_id
        if params and 'session_id' in params:
            session_id = params['session_id']
            # Try user-specific path first
            if params.get('username'):
                session_csv = f"data/user_uploads/{params['username']}/session_{session_id}/csv_files/merged_transcripts.csv"
            else:
                session_csv = f"data/public_uploads/session_{session_id}/csv_files/merged_transcripts.csv"
            
            if os.path.exists(session_csv):
                csv_path = session_csv
                logger.info(f"Using session-specific merged CSV: {csv_path}")
        
        if not csv_path or not os.path.exists(csv_path):
            logger.error(f"Transcript CSV not found at: {csv_path}")
            return ToolResult(
                data={},
                explanation="Transcript data file is not available.",
                confidence=0.0,
                format_hint="text",
                citations=[],
            )
        
        logger.info(f"Using CSV handler for transcript query")
        logger.info(f"CSV path: {csv_path}")
        
        # Get the CSV handler instance
        handler = get_csv_transcript_handler(csv_path)
        
        # Process the query through the CSV agent with reformulation
        answer_text = handler.process_query(
            user_query=question,
            language='English',
            use_summarizer=True,
            format_type="auto"
        )
        
        logger.info(f"CSV handler response length: {len(answer_text)} chars")
        
        # Determine confidence based on response quality
        confidence = 0.9
        if "error" in answer_text.lower() or "not available" in answer_text.lower():
            confidence = 0.5
        elif "no data found" in answer_text.lower():
            confidence = 0.3
        
        return ToolResult(
            data={"answer": answer_text},
            explanation=answer_text,
            confidence=confidence,
            format_hint="text",  # Can be "text" or "table" based on content
            citations=["merged_transcripts.csv"],
        )
        
    except Exception as e:
        logger.error(f"Transcript tool failed: {e}", exc_info=True)
        return ToolResult(
            data={},
            explanation=f"I encountered an error while processing your transcript query: {str(e)}",
            confidence=0.0,
            format_hint="text",
            citations=[],
        )