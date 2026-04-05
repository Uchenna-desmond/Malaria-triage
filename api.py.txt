import os
import hashlib
from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_community.tools import WikipediaQueryRun, PubmedQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

os.environ["ANTHROPIC_API_KEY"] = "your-api-key-here"

# ── Tools ─────────────────────────────────────────────────────────────────────

@tool
def classify_fever(temperature: float, age_months: int) -> str:
    """Classify fever severity based on temperature in Celsius and age in months.
    Args:
        temperature: Body temperature in Celsius.
        age_months: Age of the child in months.
    """
    if age_months < 3 and temperature >= 38.0:
        return "CRITICAL: Any fever in a child under 3 months requires immediate emergency referral."
    if temperature >= 39.5:
        return "HIGH FEVER: Urgent clinic referral needed. Administer paracetamol if available."
    elif temperature >= 38.5:
        return "MODERATE FEVER: Monitor closely. Refer to clinic if no improvement in 24 hours."
    elif temperature >= 37.5:
        return "LOW-GRADE FEVER: Monitor at home. Keep child hydrated. Reassess in 6 hours."
    return "NO FEVER: Temperature is normal. Investigate other causes of illness."

@tool
def check_danger_signs(symptoms: str) -> str:
    """Check for WHO IMCI danger signs. Keywords: convulsions, unable to drink,
    vomits everything, lethargic, stiff neck, bulging fontanelle, unconscious.
    Args:
        symptoms: Free-text description of the child symptoms.
    """
    danger_signs = ["convulsions", "unable to drink", "vomits everything",
                    "lethargic", "stiff neck", "bulging fontanelle",
                    "unable to breastfeed", "unconscious", "sunken eyes"]
    detected = [s for s in danger_signs if s in symptoms.lower()]
    if detected:
        return f"DANGER SIGNS DETECTED: {', '.join(detected)}. Refer to emergency services immediately."
    return "No WHO IMCI danger signs detected. Continue assessment."

@tool
def assess_malaria_risk(region: str, fever_days: int, has_chills: bool, has_sweats: bool) -> str:
    """Assess malaria likelihood based on region, fever duration and classic symptoms.
    Args:
        region: Country or region where the child lives.
        fever_days: Number of days the child has had fever.
        has_chills: Whether the child has chills.
        has_sweats: Whether the child has sweating episodes.
    """
    high_risk_regions = ["nigeria", "ghana", "kenya", "uganda", "tanzania",
                         "mali", "burkina faso", "mozambique", "malawi",
                         "zambia", "ethiopia", "cameroon", "sub-saharan africa"]
    risk_score = sum([
        3 if any(r in region.lower() for r in high_risk_regions) else 0,
        2 if has_chills else 0,
        2 if has_sweats else 0,
        1 if 1 <= fever_days <= 7 else 0,
    ])
    if risk_score >= 6:
        return "HIGH MALARIA RISK: Perform RDT test immediately. Treat as malaria if test unavailable."
    elif risk_score >= 3:
        return "MODERATE MALARIA RISK: RDT testing strongly recommended."
    return "LOW MALARIA RISK: Consider other diagnoses."

@tool
def screen_differential(symptoms: str, fever_days: int) -> str:
    """Differentiate between malaria, typhoid, meningitis, and viral fever.
    Args:
        symptoms: Free-text description of all symptoms.
        fever_days: Number of days the child has had fever.
    """
    s = symptoms.lower()
    findings = []
    if any(x in s for x in ["chills", "sweats", "shivering", "rigors"]):
        findings.append("MALARIA likely: classic chills/sweats pattern.")
    if any(x in s for x in ["headache", "abdominal pain", "rose spots"]) and fever_days > 5:
        findings.append("TYPHOID possible: prolonged fever with abdominal symptoms.")
    if any(x in s for x in ["stiff neck", "photophobia", "bulging fontanelle"]):
        findings.append("MENINGITIS suspected: EMERGENCY referral required.")
    if any(x in s for x in ["runny nose", "cough", "sore throat"]) and fever_days < 5:
        findings.append("VIRAL FEVER likely: upper respiratory symptoms present.")
    return " | ".join(findings) if findings else "Unclear pattern. Refer to clinic for testing."

@tool
def recommend_referral(fever_severity: str, danger_signs_present: bool, age_months: int) -> str:
    """Recommend home management, clinic referral, or emergency evacuation.
    Args:
        fever_severity: Output from classify_fever tool.
        danger_signs_present: Whether WHO IMCI danger signs were detected.
        age_months: Age of the child in months.
    """
    if danger_signs_present or age_months < 3:
        return "EMERGENCY EVACUATION: Transport child to hospital immediately."
    if "critical" in fever_severity.lower() or "high" in fever_severity.lower():
        return "URGENT CLINIC REFERRAL: Child must be seen by a clinician today."
    if "moderate" in fever_severity.lower():
        return "CLINIC REFERRAL: Visit clinic within 24 hours if no improvement."
    return "HOME MANAGEMENT: Keep child hydrated, monitor temperature every 6 hours. Return if worsens."

# ── External tools ────────────────────────────────────────────────────────────

wikipedia = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=1000)
)
pubmed = PubmedQueryRun()

tools = [classify_fever, check_danger_signs, assess_malaria_risk,
         screen_differential, recommend_referral, wikipedia, pubmed]

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a paediatric fever triage assistant supporting community 
health workers in low-income countries. Assess febrile children under 5 years old.
Collect age in months, temperature, fever duration in days, region, and symptoms.
Run all triage tools once you have the information. Use simple clear language."""

# ── Model and agent ───────────────────────────────────────────────────────────

llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0.3)
memory = MemorySaver()
agent = create_react_agent(llm, tools, checkpointer=memory, prompt=SYSTEM_PROMPT)

# ── Authentication ────────────────────────────────────────────────────────────

REGISTERED_KEYS = {
    "chw-nigeria-001": "Amina Bello - Kano State",
    "chw-nigeria-002": "Emeka Okafor - Lagos State",
    "chw-ghana-001":   "Kofi Mensah - Accra Region",
}

security = HTTPBearer()

def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    if token not in REGISTERED_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or unregistered API key. Contact admin to register.",
        )
    return REGISTERED_KEYS[token]

# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(title="Paediatric Malaria Triage API")

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default-session"

class ChatResponse(BaseModel):
    response: str
    session_id: str
    chw_name: str

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, chw_name: str = Depends(verify_api_key)):
    config = {"configurable": {"thread_id": request.session_id}}
    result = agent.invoke(
        {"messages": [HumanMessage(content=request.message)]},
        config=config,
    )
    return ChatResponse(
        response=result["messages"][-1].content,
        session_id=request.session_id,
        chw_name=chw_name,
    )

@app.post("/register")
def register_chw(name: str, region: str, admin_password: str):
    if admin_password != os.environ.get("ADMIN_PASSWORD", "changeme"):
        raise HTTPException(status_code=403, detail="Invalid admin password.")
    new_key = hashlib.sha256(f"{name}{region}".encode()).hexdigest()[:20]
    REGISTERED_KEYS[new_key] = f"{name} - {region}"
    return {"api_key": new_key, "chw": f"{name} - {region}"}

@app.get("/health")
def health():
    return {"status": "running", "tool": "Paediatric Malaria Triage API"}