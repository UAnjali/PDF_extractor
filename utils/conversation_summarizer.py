import os
import yaml
import csv
import json
from pathlib import Path
from config.config import load_config
from langchain_google_genai import ChatGoogleGenerativeAI

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "conversation_summary_prompt.txt"

# Load the prompt template
with open(PROMPT_PATH, "r") as f:
    PROMPT_TEMPLATE = f.read()

# CSV columns
CSV_COLUMNS = [
    "session_id", "name", "email", "contact_number", "status", "ctc", "company", "role", "summary"
]

# LLM setup (Gemini, can be adapted for OpenAI)
def get_llm():
    config = load_config()
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-preview-05-20",
        google_api_key=config["GOOGLE_API_KEY"],
        temperature=0.1,
        convert_system_message_to_human=True,
        generation_config={
            "max_output_tokens": 1024,
            "top_p": 0.8,
            "top_k": 40
        }
    )

def summarize_and_append_to_csv(yaml_path: str, csv_path: str):
    # Read YAML conversation
    with open(yaml_path, "r") as f:
        conversation_yaml = f.read()
    session_id = Path(yaml_path).stem

    # Prepare prompt
    prompt = PROMPT_TEMPLATE.replace("{conversation_yaml}", conversation_yaml)

    # Call LLM
    llm = get_llm()
    response = llm.invoke(prompt)
    try:
        result = json.loads(response.content if hasattr(response, 'content') else response)
    except Exception as e:
        # Fallback: write NA for all fields except session_id and summary
        result = {
            "summary": response.content if hasattr(response, 'content') else str(response),
            "name": "NA", "email": "NA", "contact_number": "NA", "status": "NA", "ctc": "NA", "company": "NA", "role": "NA"
        }

    # Prepare row
    row = [
        session_id,
        result.get("name", "NA"),
        result.get("email", "NA"),
        result.get("contact_number", "NA"),
        result.get("status", "NA"),
        result.get("ctc", "NA"),
        result.get("company", "NA"),
        result.get("role", "NA"),
        result.get("summary", "NA")
    ]

    # Write to CSV (append, create if not exists)
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(CSV_COLUMNS)
        writer.writerow(row) 