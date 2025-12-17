# services/df_agent.py

from typing import Any, Dict
import pandas as pd

from core import get_logger
from services.llm_factory import get_llm

logger = get_logger(__name__)


def run_df_agent(question: str, df: pd.DataFrame, df_name: str) -> Dict[str, Any]:
    """
    Use a LangChain pandas dataframe agent to answer a question over a dataframe.
    Returns {"answer": str, "raw": optional raw agent output}.
    """
    from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

    if df.empty:
        return {"answer": f"No data available for {df_name}.", "raw": None}

    llm = get_llm()

    agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=False,
        allow_dangerous_code=True,
    )

    prompt = (
        f"You are a data analysis assistant working on the {df_name} table. "
        f"Use pandas operations correctly and answer strictly from the data.\n\n"
        f"Columns: {list(df.columns)}\n\n"
        f"Question: {question}"
    )

    logger.info("DF_AGENT %s question=%s", df_name, question)
    result = agent.invoke(prompt)

    if isinstance(result, str):
        return {"answer": result, "raw": None}
    if isinstance(result, dict) and "output" in result:
        return {"answer": result["output"], "raw": result}
    return {"answer": str(result), "raw": result}
