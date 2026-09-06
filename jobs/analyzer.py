from ai.prompts import JD_ANALYSIS_JSON_PROMPT
from utils.ai_json import generate_structured_json

SYSTEM_PROMPT = (
    "You are CareerPilot AI's job description analysis engine. Extract only "
    "what is explicitly stated or clearly implied in the job description — "
    "never invent requirements that aren't there."
)


def analyze_job_description(manager, jd_text: str) -> dict:
    """
    Returns a dict with keys: job_title, required_skills, preferred_skills,
    experience_required, education_required, responsibilities, keywords,
    soft_skills.
    """
    prompt = JD_ANALYSIS_JSON_PROMPT.format(jd_text=jd_text)
    return generate_structured_json(manager, prompt, system_prompt=SYSTEM_PROMPT, max_tokens=1200)
