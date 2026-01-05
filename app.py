import ollama
import re
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from Assistant import emergency_bot

app = FastAPI()

# Allow frontend to call API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your frontend's URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/incident")
def get_incident():

    #read incident_reports\incident_report_2025-09-19 11-40.txt file
    with open("incident_reports/incident_report_2025-09-19 11-40.txt", "r", encoding="utf-8") as f:
        conversation = f.read()


    # report = emergency_bot()
    # conversation = report

    sys_prompt = f"""
    You are an emergency dispatcher.
    Given the Caller - Assistant conversation, decide if Ambulance and Fire Engine are required.
    Return only List format [Summary of incident, Say "Ambulance is required" if there is need of Ambulance otherwise "Ambulance is not required", say "Fire Engine is required" if there is need of Fire Engine otherwise "Fire Engine is not required", Number of injured people: ?, Number of dead people: ?]

    Caller -Assistant conversation: {conversation}
    """

    response = ollama.chat(
        model="gemma3:4b",
        messages=[{"role": "user", "content": sys_prompt}]
    )

    # Extract only the list content from response
    match = re.search(r'\[(.*?)\]', response['message']['content'])
    if match:
        content_within_brackets = match.group(1).split(",")
        content_within_brackets = [item.strip() for item in content_within_brackets]

        # Map to structured JSON
        incident_data = {
            "summary": content_within_brackets[0],
            "ambulance": content_within_brackets[1],
            "fire_engine": content_within_brackets[2],
            "injured": content_within_brackets[3].split(":")[-1].strip(),
            "dead": content_within_brackets[4].split(":")[-1].strip(),
        }
        return incident_data

    return {"error": "Failed to parse incident"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
