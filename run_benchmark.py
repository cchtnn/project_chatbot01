
import csv
import json
from pathlib import Path
import requests

# Paths
QUESTIONS_PATH = Path("benchmark_questions.csv")
OUTPUT_PATH = Path("benchmark_answers_new.json")

API_URL = "http://localhost:8000/query"

DUMMY_SESSION_ID = 1  # any int; backend just needs it present


def ask_backend(question: str) -> dict:
    """
    Call /query with form-encoded body:
      query: str
      session_id: int (as string in form)
      private: bool (as 'true'/'false')
    """
    body = f"query={requests.utils.quote(question)}&session_id={DUMMY_SESSION_ID}&private=false"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    resp = requests.post(API_URL, data=body, headers=headers, timeout=60)
    resp.raise_for_status()
    return resp.json()


def main() -> None:
    results = []

    with QUESTIONS_PATH.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = row["id"]
            domain = row["domain"]
            question = row["question"]

            try:
                answer_obj = ask_backend(question)
                answer_text = (
                    answer_obj.get("answer")
                    or answer_obj.get("response")
                    or answer_obj.get("message")
                    or ""
                )

                results.append(
                    {
                        "id": qid,
                        "domain": domain,
                        "question": question,
                        "answer_text": answer_text,
                        "raw_response": answer_obj,
                    }
                )

            except Exception as e:
                results.append(
                    {
                        "id": qid,
                        "domain": domain,
                        "question": question,
                        "answer_text": "",
                        "raw_response": {"error": str(e)},
                    }
                )

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
