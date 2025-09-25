from json import dumps, loads
from os import getenv

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=getenv("OPENAI_API_KEY"))

INPUT_FILE = "transcripts.jsonl"
OUTPUT_FILE = "adversarial.jsonl"


def classify(transcript):
    # Prepare the conversation in plain text
    conversation_text = "\n".join(
        [f"{turn['role']}: {turn['content']}" for turn in transcript["transcript"]]
    )

    # Send to GPT-4o for classification with structured JSON output
    response = client.chat.completions.with_structured_output.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a classifier. Read the conversation and classify it as adversarial: true if the customer uses adversarial or hostile language, otherwise adversarial: false."},
            {"role": "user", "content": conversation_text},
        ],
        temperature=0,
        schema={
            "type": "object",
            "properties": {
                "adversarial": {"type": "boolean"}
            },
            "required": ["adversarial"]
        }
    )

    # Extract only the structured output (the JSON classification)
    return response.output


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as infile, open(
        OUTPUT_FILE, "w", encoding="utf-8"
    ) as outfile:
        for line in infile:
            transcript = loads(line)
            result = classify(transcript)
            outfile.write(dumps(result) + "\n")


if __name__ == "__main__":
    main()