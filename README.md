# 🥗 NutriCoach AI — Nutrition Agentic AI Application

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0.3-green?logo=flask&logoColor=white)
![IBM Granite](https://img.shields.io/badge/IBM%20Granite-13B%20Instruct-0f62fe?logo=ibm&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Platform](https://img.shields.io/badge/Platform-Windows-informational?logo=windows)

**Personalised 7-Day Nutrition Planning powered by IBM Granite LLM**
**Agentic AI · ReAct Pattern · HAM Ethics · Full SDLC**

[Live Demo](#-quick-start) · [Architecture](#-agentic-ai-architecture) · [API Docs](#-rest-api) · [Ethics](#-ethics--guardrails)

</div>

---

## 📖 Overview

**NutriCoach AI** is a full-stack **Agentic AI web application** that collects user details (name, age, diet preference, medical history, location) and generates a personalised 7-day nutrition plan using the **IBM Granite LLM** (`ibm/granite-13b-instruct-v2`) via IBM watsonx.ai.

The agent follows the **ReAct (Reason + Act)** pattern — calling specialised tools for local food data, medical dietary adjustments, and calorie estimation before generating a contextually-rich nutrition plan.

### ✨ Key Features

| Feature | Details |
|---------|---------|
| 🤖 **IBM Granite LLM** | `ibm/granite-13b-instruct-v2` via watsonx.ai |
| 🧠 **Agentic ReAct Loop** | THINK → ACT (3 tools) → REASON → GENERATE → VALIDATE → OUTPUT |
| 🌍 **20+ Countries** | Region-specific food databases |
| 🏥 **12 Medical Conditions** | Diabetes, hypertension, cholesterol, thyroid, kidney, gout, and more |
| 🛡️ **HAM Guardrails** | Helpful · Accurate · Mindful — 4-layer safety pipeline |
| 💬 **Multi-turn Q&A** | Context-aware NutriCoach chatbot |
| 📁 **Save Plans as .md** | Auto-saved Markdown files with YAML frontmatter |
| 🔌 **Offline Mode** | Fully functional without IBM API credentials |
| 🌐 **Web UI** | Single-page app with tabs, tables, reasoning trace |

---

## 🚀 Quick Start

### Windows — One Click

```batch
# Navigate to the project folder, then:
double-click  run.bat
```

The launcher automatically:
1. ✅ Detects Python (3.9+)
2. ✅ Creates a virtual environment
3. ✅ Installs all dependencies from `requirements.txt`
4. ✅ Creates `.env` from template
5. ✅ Starts Flask server on `http://localhost:5000`
6. ✅ Opens your browser automatically

### Manual Setup

```bash
# 1. Clone the repository
git clone https://github.com/aishavaghasiya33/Nutrition-Agentic-AI-Application-.git
cd Nutrition-Agentic-AI-Application-

# 2. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment (optional — for live IBM Granite responses)
cp .env.example .env
# Edit .env and add your IBM watsonx.ai credentials

# 5. Run the application
python app.py
```

Open **http://localhost:5000** in your browser.

---

## ⚙️ IBM watsonx.ai Configuration

> The app works **immediately in Offline Mode** without any credentials — a full rule-based plan generator provides complete 7-day meal plans.

To enable live **IBM Granite LLM** responses:

1. Get credentials from [IBM Cloud](https://cloud.ibm.com):
   - **API Key**: https://cloud.ibm.com/iam/apikeys
   - **Project ID**: From your watsonx.ai project settings

2. Edit `.env`:

```env
WATSONX_API_KEY=your_actual_api_key_here
WATSONX_PROJECT_ID=your_actual_project_id_here
WATSONX_URL=https://us-south.ml.cloud.ibm.com
FLASK_SECRET_KEY=your-secret-key-here
FLASK_PORT=5000
```

### IBM Granite Model Parameters

| Parameter | Value |
|-----------|-------|
| Model ID | `ibm/granite-13b-instruct-v2` |
| Platform | IBM watsonx.ai |
| Decoding | Greedy |
| Max Tokens | 1,200 |
| Min Tokens | 80 |
| Repetition Penalty | 1.1 |
| Stop Sequences | `###`, `User:`, `Human:` |

---

## 🏗️ Project Structure

```
Nutrition-Agentic-AI-Application/
│
├── app.py                  # Flask web server — 6 REST API endpoints
├── granite_agent.py        # IBM Granite ReAct agent + 3 tools + Q&A engine
├── guardrails.py           # HAM ethics, safety guardrails, medical disclaimer
├── plan_saver.py           # Markdown file persistence (.md plans)
├── requirements.txt        # Python package dependencies
├── run.bat                 # Windows one-click launcher
├── _test_run.py            # Smoke test suite (5 profiles, Q&A, guardrails)
├── .env.example            # Environment variable template
├── .gitignore              # Git ignore rules
│
├── templates/
│   └── index.html          # Single-page web app (~1,100 lines HTML/CSS/JS)
│
└── saved_plans/            # Auto-generated .md nutrition plan files
    └── nutrition_plan_*.md
```

---

## 🧠 Agentic AI Architecture

### ReAct (Reason + Act) Agent Loop

```
┌──────────────────────────────────────────────────────────────────┐
│                      NutriCoach AI Agent                         │
│                                                                  │
│   User Input (name, age, diet, location, medical history)        │
│          │                                                       │
│          ▼                                                       │
│   ┌─────────────┐                                                │
│   │    THINK    │  Analyse profile for personalisation signals   │
│   └──────┬──────┘                                                │
│          │                                                       │
│          ▼                                                       │
│   ┌─────────────┐   Tool 1: get_local_foods(city, country, diet) │
│   │     ACT     │   Tool 2: get_medical_adjustments(history)     │
│   │  (3 Tools)  │   Tool 3: calculate_bmr(age, gender)           │
│   └──────┬──────┘                                                │
│          │                                                       │
│          ▼                                                       │
│   ┌─────────────┐                                                │
│   │   REASON    │  Synthesise tool outputs → enriched prompt     │
│   └──────┬──────┘                                                │
│          │                                                       │
│          ▼                                                       │
│   ┌─────────────┐                                                │
│   │  GENERATE   │  IBM Granite LLM  (or structured offline)      │
│   └──────┬──────┘                                                │
│          │                                                       │
│          ▼                                                       │
│   ┌─────────────┐                                                │
│   │  VALIDATE   │  HAM guardrails — 9 forbidden pattern scans    │
│   └──────┬──────┘                                                │
│          │                                                       │
│          ▼                                                       │
│   Personalised 7-Day Plan  →  Auto-saved as .md file             │
└──────────────────────────────────────────────────────────────────┘
```

### Agent Tool Registry

| Tool | Parameters | Purpose |
|------|-----------|---------|
| `get_local_foods` | city, country, diet | Returns region-specific foods (20+ countries, 4 diet types) |
| `get_medical_adjustments` | medical_history | Evidence-based dietary rules for 9 conditions |
| `calculate_bmr` | age, gender | Mifflin-St Jeor BMR + 3 activity-level calorie targets |

### Supported Locations

India · Pakistan · Bangladesh · China · Japan · Indonesia · Thailand · Saudi Arabia · Turkey · Nigeria · Ethiopia · USA · UK · Germany · France · Brazil · Mexico · Australia · *(+ universal default)*

### Medical Conditions Handled

Diabetes · Hypertension · Cholesterol · Thyroid · Kidney Disease · Celiac/Gluten · Gout · Heart Disease · Cancer · Eating Disorders · Pregnancy · General Health

---

## 🛡️ Ethics & Guardrails

**HAM = Helpful · Accurate · Mindful**

```
Input Layer          Processing Layer       Output Layer
─────────────        ────────────────       ────────────
• Prompt injection   • IBM Granite with     • 9 forbidden
  detection (8         domain-constrained     pattern scans
  patterns)            system prompt        • Condition-specific
• Input sanitisation • Structured offline     medical warnings
• Profile validation   fallback             • Mandatory medical
• Field length         (always available)     disclaimer on
  limits                                      every plan
```

| Guardrail | Description |
|-----------|-------------|
| Injection Detection | Blocks `ignore instructions`, `jailbreak`, `DAN mode`, `you are now`, etc. |
| Input Sanitisation | Strips HTML tags, removes SQL chars, enforces length limits |
| Output Safety | Rejects any output suggesting stopping medications, guaranteeing cures, or extreme diets |
| Medical Warnings | Auto-generates specialist referral notices for detected conditions |
| Medical Disclaimer | Appended to **every** generated plan — non-negotiable |
| No Harmful Advice | Agent is prompted never to advise replacing medical care |

---

## 🌐 REST API

Base URL: `http://localhost:5000`

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Web UI — full single-page application |
| `POST` | `/api/generate-plan` | Generate personalised 7-day nutrition plan |
| `POST` | `/api/ask` | Multi-turn Q&A with NutriCoach AI |
| `GET` | `/api/saved-plans` | List all saved `.md` plan files |
| `GET` | `/api/load-plan/<filename>` | Load content of a specific saved plan |
| `GET` | `/api/health` | Health check + model information |

### POST /api/generate-plan

**Request:**
```json
{
  "name": "Priya Sharma",
  "age": 34,
  "gender": "female",
  "diet_preference": "vegetarian",
  "city": "Mumbai",
  "country": "India",
  "medical_history": "Type 2 Diabetes, High Blood Pressure"
}
```

**Response:**
```json
{
  "success": true,
  "plan": "# 🥗 Personalised 7-Day Nutrition Plan\n...",
  "calorie_info": { "bmr": 1382, "sedentary": 1658, "moderate": 2141, "active": 2383 },
  "local_foods": "Dal, Sabzi, Roti, Brown Rice, Paneer, Sprouts, Curd",
  "source": "ibm_granite",
  "reasoning_trace": ["THINK: ...", "ACT: ...", "GENERATE: ...", "VALIDATE: ...", "DONE"],
  "saved_file": "nutrition_plan_Priya_Sharma_20241015_143022.md",
  "generated_at": "2024-10-15T14:30:22.123456"
}
```

### POST /api/ask

**Request:**
```json
{ "question": "What foods should I avoid with diabetes?" }
```

**Response:**
```json
{
  "success": true,
  "answer": "For diabetes management, focus on low-GI foods...",
  "source": "ibm_granite"
}
```

---

## 📋 Agentic AI SDLC Phases

| Phase | What Was Built |
|-------|----------------|
| **1. Requirements** | User profiling spec, medical ethics rules, HAM principles, output format |
| **2. Design** | ReAct loop diagram, tool registry, layered guardrail architecture, prompt templates |
| **3. Development** | Flask API, IBM Granite SDK, 3 agent tools, Q&A engine, plan persistence, web UI |
| **4. Testing** | 5-profile smoke test, Q&A validation, guardrail injection tests, safety scans |
| **5. Deployment** | One-click `run.bat`, venv auto-setup, auto-install, auto browser launch |
| **6. Monitoring** | Agent reasoning trace UI, source tracking, plan history, `/api/health` endpoint |

---

## 🖥️ Web Application Screenshots

| Tab | Description |
|-----|-------------|
| **📋 Nutrition Plan** | Step-by-step form → 7-day plan with calorie cards + reasoning trace |
| **💬 Ask NutriCoach** | Multi-turn Q&A chat with suggestion chips |
| **📁 Saved Plans** | Browse, view, and download all `.md` plan files |
| **ℹ️ About** | Architecture, SDLC phases, tools, ethics documentation |

---

## 🧪 Running Tests

```bash
# Activate virtual environment first
venv\Scripts\activate

# Run the smoke test suite
python _test_run.py
```

Expected output:
```
✓ Priya Sharma         | vegetarian        | India      | OK
✓ James Carter         | non-vegetarian    | USA        | OK
✓ Sophie Laurent       | vegan             | France     | OK
✓ Kenji Tanaka         | pescatarian       | Japan      | OK
✓ Amara Okafor         | vegetarian        | Nigeria    | OK
ALL TESTS PASSED
```

---

## 📦 Dependencies

```
flask==3.0.3              # Web framework
flask-cors==4.0.1         # Cross-Origin Resource Sharing
requests==2.32.3          # HTTP library
python-dotenv==1.0.1      # Environment variable management
markdown==3.6             # Markdown rendering
ibm-watsonx-ai==1.1.2    # IBM Granite LLM SDK
```

---

## ⚕️ Medical Disclaimer

> **This application is for informational and educational purposes only.**
> It does **not** constitute medical advice, diagnosis, or treatment.
> Nutritional needs are highly individual. Always consult a qualified
> healthcare professional or registered dietitian before making significant
> dietary changes — especially if you have medical conditions or take
> prescription medications.
> The creators of this application bear no liability for health outcomes
> based on this AI-generated output.

---

## 📄 License

This project is licensed under the MIT License.

---

<div align="center">

*Built with ❤️ using **IBM Granite LLM** · **Flask** · **Python 3.12** · **Agentic AI SDLC***

</div>
