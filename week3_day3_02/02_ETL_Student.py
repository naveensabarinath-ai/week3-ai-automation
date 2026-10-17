"""
Robust End-to-End ETL Automation Pipeline with Hybrid Fallback
Step 1: Load the file
Step 2: Read the file
Step 3: OpenAI clean with deterministic Python regex/rule fallback
Step 4: Create the final CSV file
"""

import json
import os
import re
from datetime import datetime
from typing import List, Optional, Any
import pandas as pd

# Load environment variables from .env file
from dotenv import load_dotenv

# Safe import for OpenAI and Pydantic
try:
    from pydantic import BaseModel, Field
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    BaseModel = object
    def Field(*args, **kwargs): return None

# ---------------------------------------------------------
# Configuration & Environment Setup
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Explicitly load .env from the script's directory (or project root if adjacent)
env_path = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path=env_path)

INPUT_CSV_PATH = os.path.join(BASE_DIR, "students_dirty_50.csv")
OUTPUT_CSV_PATH = os.path.join(BASE_DIR, "students_cleaned_50.csv")
BATCH_SIZE = 25

api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    print("[Env] OPENAI_API_KEY detected successfully.")
else:
    print("[Env Warning] OPENAI_API_KEY not found in .env; fallback mode will be used.")

client = OpenAI(api_key=api_key) if (OPENAI_AVAILABLE and api_key) else None


# ---------------------------------------------------------
# Pydantic Schema (Used if OpenAI + Pydantic are installed)
# ---------------------------------------------------------
if OPENAI_AVAILABLE:
    class CleanStudent(BaseModel):
        student_id: int
        full_name: str = Field(description="Normalized full name in Title Case, trimmed of whitespace.")
        email: Optional[str] = Field(description="Normalized valid email address in lowercase, or None.")
        dob: str = Field(description="Standardized date of birth in strict YYYY-MM-DD format.")
        enrollment_date: str = Field(description="Standardized enrollment date in strict YYYY-MM-DD format.")
        score: int = Field(description="Normalized score clamped between 0 and 100.")

    class CleanDataset(BaseModel):
        students: List[CleanStudent]


# ---------------------------------------------------------
# Step 1 & 2: Load and Read File
# ---------------------------------------------------------
def step1_and_2_read_data(file_path: str) -> pd.DataFrame:
    """Step 1 & 2: Load and read the raw input file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    print(f"[Step 1 & 2] Loading '{file_path}'...")
    df = pd.read_csv(file_path)
    df = df.fillna("")
    print(f"Loaded {len(df)} records.")
    return df


# ---------------------------------------------------------
# Deterministic Fallback Cleaning Logic (Regex & Date rules)
# ---------------------------------------------------------
def parse_date_fallback(date_val: Any) -> str:
    """Robust date parser handling common dirty formats."""
    s = str(date_val).strip()
    if not s:
        return "1970-01-01"
    
    formats = [
        "%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d",
        "%d-%m-%Y", "%Y.%m.%d", "%m-%d-%Y",
        "%B %d, %Y", "%B %d %Y", "%b %d %Y", "%b %d, %Y"
    ]
    for fmt in formats:
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue

    match = re.search(r"(\d{4})[-/. ](\d{1,2})[-/. ](\d{1,2})", s)
    if match:
        y, m, d = match.groups()
        return f"{int(y):04d}-{int(m):02d}-{int(d):02d}"

    return s

def clean_record_fallback(row: dict) -> dict:
    """Cleans a single record using regex and rule-based logic."""
    # 1. Clean full_name
    name = str(row.get("full_name", "")).strip()
    name = re.sub(r"[_\s]+", " ", name)
    name = name.title()

    # 2. Clean email
    email = str(row.get("email", "")).strip().lower()
    email = email.replace("_at_", "@").replace("@@", "@").replace("#", "@")
    email_regex = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"
    clean_email = email if re.match(email_regex, email) else None

    # 3. Clean dates
    dob = parse_date_fallback(row.get("dob", ""))
    enrollment_date = parse_date_fallback(row.get("enrollment_date", ""))

    # 4. Clean score
    try:
        raw_score = float(row.get("score", 0))
        raw_score = abs(raw_score)  # e.g., -12 -> 12
        if raw_score > 100:
            raw_score = raw_score / 10.0  # e.g., 950 -> 95
        score = int(round(min(100, max(0, raw_score))))
    except (ValueError, TypeError):
        score = 0

    return {
        "student_id": int(row.get("student_id", 0)),
        "full_name": name,
        "email": clean_email,
        "dob": dob,
        "enrollment_date": enrollment_date,
        "score": score,
    }


# ---------------------------------------------------------
# Step 3: Clean Data (OpenAI with Automatic Fallback)
# ---------------------------------------------------------
def step3_clean_with_openai_and_fallback(df: pd.DataFrame, batch_size: int = 25) -> pd.DataFrame:
    """Step 3: Cleans with OpenAI if configured, otherwise falls back gracefully."""
    print(f"[Step 3] Processing data (Batch size: {batch_size})...")

    system_prompt = (
        "You are an expert data cleaning agent.\n"
        "Normalize and fix dirty student records:\n"
        "1. full_name: Clean whitespace, convert to Title Case, remove underscores.\n"
        "2. email: Standardize to lowercase. Replace '_at_' or '@@' with standard '@'. Set invalid to null.\n"
        "3. dob & enrollment_date: Parse dates into strict YYYY-MM-DD format.\n"
        "4. score: Valid range is 0 to 100. Fix negative numbers (-12 -> 12) or trailing extra zeros (950 -> 95)."
    )

    cleaned_records = []
    total_rows = len(df)

    for start_idx in range(0, total_rows, batch_size):
        end_idx = min(start_idx + batch_size, total_rows)
        chunk = df.iloc[start_idx:end_idx]
        payload = chunk.to_dict(orient="records")

        batch_cleaned = False
        if OPENAI_AVAILABLE and client is not None:
            try:
                print(f" -> [OpenAI] Sending rows {start_idx + 1} to {end_idx}...")
                response = client.beta.chat.completions.parse(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Clean records:\n{json.dumps(payload)}"}
                    ],
                    response_format=CleanDataset,
                    timeout=30.0,
                )
                for student in response.choices[0].message.parsed.students:
                    cleaned_records.append(student.model_dump())
                batch_cleaned = True
                print(f"    [OpenAI] Success for rows {start_idx + 1} to {end_idx}.")
            except Exception as e:
                print(f"    [OpenAI Warning] {e}. Switching to rule-based fallback...")

        # Run fallback when OpenAI is not available, fails, or has no key
        if not batch_cleaned:
            print(f"    [Fallback Engine] Cleaning rows {start_idx + 1} to {end_idx}...")
            for row in payload:
                cleaned_records.append(clean_record_fallback(row))

    return pd.DataFrame(cleaned_records)


# ---------------------------------------------------------
# Step 4: Write Final CSV
# ---------------------------------------------------------
def step4_save_final_csv(df_cleaned: pd.DataFrame, output_path: str) -> None:
    """Step 4: Create the final CSV file."""
    print(f"[Step 4] Writing cleaned data to: {output_path}")
    df_cleaned.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Pipeline complete! {len(df_cleaned)} records saved successfully.")


# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------
def main():
    df_raw = step1_and_2_read_data(INPUT_CSV_PATH)
    df_clean = step3_clean_with_openai_and_fallback(df_raw, batch_size=BATCH_SIZE)
    step4_save_final_csv(df_clean, OUTPUT_CSV_PATH)


if __name__ == "__main__":
    main()