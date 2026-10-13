"""
granite_agent.py — IBM Granite LLM Agentic AI Core
Handles: plan generation, Q&A resolution, tool calling simulation,
         multi-turn memory, and ReAct-style reasoning loop.

Agentic AI SDLC Phase: Model Integration & Agent Loop
IBM Model: ibm/granite-13b-instruct-v2 via watsonx.ai
"""

import os
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ── Try to import IBM watsonx.ai SDK ────────────────────────────────────────
try:
    from ibm_watsonx_ai import Credentials
    from ibm_watsonx_ai.foundation_models import ModelInference
    from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as Params
    IBM_SDK_AVAILABLE = True
except ImportError:
    IBM_SDK_AVAILABLE = False

from guardrails import (
    check_input_safety,
    check_output_safety,
    get_condition_warnings,
    MEDICAL_DISCLAIMER,
)

# ── IBM Granite Model Configuration ─────────────────────────────────────────
GRANITE_MODEL_ID  = "ibm/granite-13b-instruct-v2"
WATSONX_URL       = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
WATSONX_API_KEY   = os.getenv("WATSONX_API_KEY", "")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID", "")

_GEN_PARAMS_RAW = {
    "decoding_method":   "greedy",
    "max_new_tokens":    1200,
    "min_new_tokens":    80,
    "repetition_penalty": 1.1,
    "stop_sequences":    ["###", "User:", "Human:"],
}


def _get_model():
    """Initialise and return the IBM Granite model client."""
    if not IBM_SDK_AVAILABLE:
        return None
    if not WATSONX_API_KEY or WATSONX_API_KEY in ("your_ibm_cloud_api_key_here", ""):
        return None
    try:
        creds = Credentials(url=WATSONX_URL, api_key=WATSONX_API_KEY)
        params = {
            Params.DECODING_METHOD:    _GEN_PARAMS_RAW["decoding_method"],
            Params.MAX_NEW_TOKENS:     _GEN_PARAMS_RAW["max_new_tokens"],
            Params.MIN_NEW_TOKENS:     _GEN_PARAMS_RAW["min_new_tokens"],
            Params.REPETITION_PENALTY: _GEN_PARAMS_RAW["repetition_penalty"],
            Params.STOP_SEQUENCES:     _GEN_PARAMS_RAW["stop_sequences"],
        }
        return ModelInference(
            model_id=GRANITE_MODEL_ID,
            credentials=creds,
            project_id=WATSONX_PROJECT_ID,
            params=params,
        )
    except Exception:
        return None


def _call_granite(prompt: str) -> str:
    """
    Call IBM Granite LLM. Falls back gracefully to structured offline generator
    when credentials are absent, so the app is always runnable for demos.
    """
    model = _get_model()
    if model:
        try:
            response = model.generate_text(prompt=prompt)
            return response.strip() if response else "__OFFLINE__"
        except Exception:
            pass
    return "__OFFLINE__"


# ═══════════════════════════════════════════════════════════════════════════
#  AGENT TOOL REGISTRY  (ReAct pattern — Reason + Act)
# ═══════════════════════════════════════════════════════════════════════════

TOOL_REGISTRY = {}


def register_tool(name):
    def decorator(fn):
        TOOL_REGISTRY[name] = fn
        return fn
    return decorator


@register_tool("get_local_foods")
def get_local_foods(city: str, country: str, diet: str) -> str:
    """Returns locally popular foods for the given location and diet preference."""
    local_db = {
        # ── South Asia ──────────────────────────────────────────────────────
        "india": {
            "vegetarian":     "Dal, Sabzi, Roti, Brown Rice, Paneer, Sprouts, Curd, Idli, Dosa",
            "non-vegetarian": "Tandoori Chicken, Fish Curry, Eggs, Dal, Roti, Brown Rice",
            "vegan":          "Dal, Tofu Bhurji, Brown Rice, Roti, Seasonal Vegetables, Sprouts",
            "pescatarian":    "Fish Curry, Prawns, Dal, Rice, Seasonal Vegetables",
        },
        "pakistan": {
            "vegetarian":     "Daal, Sabzi, Roti, Rice, Lassi, Chana, Seasonal Fruits",
            "non-vegetarian": "Chicken Karahi, Mutton Biryani, Daal, Roti, Eggs",
            "vegan":          "Daal, Vegetable Curry, Brown Rice, Roti, Seasonal Vegetables",
            "pescatarian":    "Fish Tikka, Prawn Curry, Daal, Rice",
        },
        "bangladesh": {
            "vegetarian":     "Rice, Lentil Soup, Saag, Chana, Seasonal Vegetables",
            "non-vegetarian": "Hilsa Fish Curry, Chicken, Rice, Lentils, Eggs",
            "vegan":          "Rice, Lentils, Seasonal Vegetables, Banana Flower Curry",
            "pescatarian":    "Hilsa Fish, Prawn Bhuna, Rice, Lentils",
        },
        # ── East Asia ───────────────────────────────────────────────────────
        "china": {
            "vegetarian":     "Tofu Stir-fry, Edamame, Brown Rice, Bok Choy, Congee",
            "non-vegetarian": "Steamed Fish, Chicken Broth, Brown Rice, Dumplings",
            "vegan":          "Mapo Tofu (vegan), Edamame, Brown Rice, Stir-fried Greens",
            "pescatarian":    "Steamed Fish, Prawn Dumplings, Brown Rice, Bok Choy",
        },
        "japan": {
            "vegetarian":     "Miso Soup, Tofu, Brown Rice, Edamame, Seaweed Salad",
            "non-vegetarian": "Grilled Salmon, Chicken Teriyaki, Brown Rice, Miso Soup",
            "vegan":          "Miso Soup (kombu), Brown Rice, Pickled Vegetables, Tofu",
            "pescatarian":    "Sushi (salmon/tuna), Miso Soup, Edamame, Brown Rice",
        },
        # ── Southeast Asia ──────────────────────────────────────────────────
        "indonesia": {
            "vegetarian":     "Tempeh, Tofu Satay, Brown Rice, Gado-Gado, Vegetables",
            "non-vegetarian": "Chicken Satay, Beef Rendang, Brown Rice, Gado-Gado",
            "vegan":          "Tempeh, Tofu Stir-fry, Brown Rice, Seasonal Vegetables",
            "pescatarian":    "Grilled Fish, Prawn Curry, Brown Rice, Gado-Gado",
        },
        "thailand": {
            "vegetarian":     "Tofu Pad Thai, Brown Rice, Thai Green Curry (veg), Papaya Salad",
            "non-vegetarian": "Chicken Pad Thai, Fish Cakes, Brown Rice, Tom Kha",
            "vegan":          "Veg Pad Thai, Brown Rice, Stir-fried Vegetables, Mango Salad",
            "pescatarian":    "Prawn Pad Thai, Fish Cakes, Brown Rice, Tom Yum",
        },
        # ── Middle East ─────────────────────────────────────────────────────
        "saudi arabia": {
            "vegetarian":     "Falafel, Hummus, Whole-wheat Pita, Fattoush, Lentil Soup",
            "non-vegetarian": "Kabsa (Chicken/Lamb), Hummus, Whole-wheat Pita, Lentil Soup",
            "vegan":          "Falafel, Hummus, Whole-wheat Pita, Tabbouleh, Lentil Soup",
            "pescatarian":    "Grilled Fish, Hummus, Whole-wheat Pita, Fattoush",
        },
        "turkey": {
            "vegetarian":     "Menemen (egg-based), Lentil Soup, Whole-grain Bread, Ayran, Salad",
            "non-vegetarian": "Grilled Chicken Kebab, Lentil Soup, Bulgur Pilaf, Salad",
            "vegan":          "Falafel, Hummus, Lentil Soup, Bulgur Pilaf, Fresh Salad",
            "pescatarian":    "Grilled Sea Bass, Lentil Soup, Bulgur Pilaf, Salad",
        },
        # ── Africa ──────────────────────────────────────────────────────────
        "nigeria": {
            "vegetarian":     "Jollof Rice, Moi Moi, Vegetable Soup, Plantains, Beans",
            "non-vegetarian": "Jollof Rice, Grilled Chicken, Egusi Soup, Plantains",
            "vegan":          "Jollof Rice, Bean Stew, Plantains, Okra Soup, Vegetables",
            "pescatarian":    "Jollof Rice, Fish Pepper Soup, Plantains, Beans",
        },
        "ethiopia": {
            "vegetarian":     "Injera, Misir Wot (lentils), Gomen, Fosolia (beans)",
            "non-vegetarian": "Injera, Doro Wot, Lentil Stew, Greens",
            "vegan":          "Injera, Misir Wot, Gomen, Fosolia, Red Lentil Stew",
            "pescatarian":    "Injera, Grilled Fish, Lentil Stew, Greens",
        },
        # ── Europe ──────────────────────────────────────────────────────────
        "usa": {
            "vegetarian":     "Quinoa Bowl, Greek Salad, Lentil Soup, Whole-Grain Bread, Beans",
            "non-vegetarian": "Grilled Chicken, Salmon, Turkey Wrap, Egg White Omelette",
            "vegan":          "Tempeh Bowl, Chickpea Salad, Oat Milk Smoothie, Avocado Toast",
            "pescatarian":    "Tuna Salad, Grilled Salmon, Shrimp Stir-fry, Quinoa Bowl",
        },
        "uk": {
            "vegetarian":     "Jacket Potato, Bean Stew, Porridge, Cheese Sandwich, Garden Salad",
            "non-vegetarian": "Grilled Chicken, Smoked Salmon, Scrambled Eggs, Baked Beans",
            "vegan":          "Oat Porridge, Lentil Soup, Falafel Wrap, Chia Pudding",
            "pescatarian":    "Fish and Salad, Prawn Stir-fry, Smoked Mackerel, Baked Potato",
        },
        "germany": {
            "vegetarian":     "Rye Bread, Quark, Lentil Soup, Potato Salad, Vegetable Stew",
            "non-vegetarian": "Grilled Chicken, Rye Bread, Lentil Soup, Egg Dishes",
            "vegan":          "Rye Bread, Hummus, Lentil Soup, Roasted Vegetables",
            "pescatarian":    "Grilled Trout, Rye Bread, Lentil Soup, Potato Salad",
        },
        "france": {
            "vegetarian":     "Ratatouille, Whole-grain Baguette, Salade Niçoise (no fish), Lentils",
            "non-vegetarian": "Poulet Rôti, Salade Niçoise, Whole-grain Bread, Lentils",
            "vegan":          "Ratatouille, Whole-grain Bread, Chickpea Salad, Fruit Plate",
            "pescatarian":    "Salade Niçoise, Grilled Salmon, Whole-grain Bread, Ratatouille",
        },
        # ── South America ───────────────────────────────────────────────────
        "brazil": {
            "vegetarian":     "Feijão (beans), Brown Rice, Salad, Açaí Bowl, Pão de Queijo (moderate)",
            "non-vegetarian": "Grilled Chicken, Feijão, Brown Rice, Salad, Tropical Fruits",
            "vegan":          "Feijão, Brown Rice, Salad, Açaí Bowl, Seasonal Vegetables",
            "pescatarian":    "Grilled Fish, Feijão, Brown Rice, Tropical Fruit Salad",
        },
        "mexico": {
            "vegetarian":     "Black Beans, Brown Rice, Avocado, Corn Tortillas, Salsa",
            "non-vegetarian": "Grilled Chicken Tacos, Black Beans, Brown Rice, Avocado",
            "vegan":          "Black Bean Tacos, Brown Rice, Guacamole, Corn Tortillas",
            "pescatarian":    "Fish Tacos, Black Beans, Brown Rice, Avocado, Salsa",
        },
        # ── Oceania ─────────────────────────────────────────────────────────
        "australia": {
            "vegetarian":     "Avocado Toast, Lentil Soup, Quinoa Salad, Whole-grain Bread",
            "non-vegetarian": "Grilled Barramundi, Chicken Salad, Whole-grain Bread, Eggs",
            "vegan":          "Avocado Toast, Lentil Soup, Quinoa Bowl, Tropical Fruits",
            "pescatarian":    "Grilled Barramundi, Quinoa Salad, Whole-grain Bread",
        },
        # ── Default (any unrecognised country) ──────────────────────────────
        "default": {
            "vegetarian":  "Whole Grains, Legumes, Vegetables, Fruits, Dairy or Dairy Alternatives",
            "non-vegetarian": "Lean Protein (chicken/fish/eggs), Vegetables, Whole Grains",
            "vegan":       "Legumes, Tofu/Tempeh, Whole Grains, Nuts, Seasonal Vegetables",
            "pescatarian": "Fish, Seafood, Legumes, Whole Grains, Seasonal Vegetables",
        },
    }
    country_key = country.lower().strip()
    diet_key = diet.lower().strip()
    db = local_db.get(country_key, local_db["default"])
    # normalise diet key
    if diet_key not in db:
        diet_key = "vegetarian"
    return db.get(diet_key, db.get("vegetarian", "Balanced whole foods"))


@register_tool("get_medical_adjustments")
def get_medical_adjustments(medical_history: str) -> dict:
    """Returns evidence-based dietary adjustments for known medical conditions."""
    adjustments = {}
    lower = (medical_history or "").lower()

    if "diabetes" in lower:
        adjustments["diabetes"] = {
            "avoid":  "refined sugar, white rice, sugary drinks, white bread, fruit juices",
            "prefer": "low-GI foods: whole grains, leafy greens, legumes, nuts, cinnamon",
            "tip":    "Eat smaller meals every 3–4 hours to stabilise blood glucose. Target carbs: 45–60g per meal.",
        }
    if "blood pressure" in lower or "hypertension" in lower:
        adjustments["hypertension"] = {
            "avoid":  "excess sodium (salt > 2,300 mg/day), processed foods, pickles, alcohol, caffeine",
            "prefer": "DASH diet: bananas, leafy greens, berries, low-fat dairy, whole grains, legumes",
            "tip":    "Target < 1,500 mg sodium/day if BP is very high. Potassium (bananas, sweet potato) helps counter sodium.",
        }
    if "cholesterol" in lower:
        adjustments["cholesterol"] = {
            "avoid":  "saturated fats (red meat, butter, full-fat dairy), trans fats (fried/processed foods)",
            "prefer": "oats, beans, avocado, olive oil, fatty fish (omega-3), flaxseeds, walnuts",
            "tip":    "Soluble fibre (oats, legumes) actively lowers LDL. Target: 5–10g/day of soluble fibre.",
        }
    if "thyroid" in lower:
        adjustments["thyroid"] = {
            "avoid":  "raw cruciferous vegetables in large quantities, excess soy products",
            "prefer": "iodine-rich foods (seaweed, fish, dairy), selenium sources (Brazil nuts, eggs)",
            "tip":    "Take thyroid medication on an empty stomach; avoid calcium supplements within 4 hrs of medication.",
        }
    if "kidney" in lower:
        adjustments["kidney"] = {
            "avoid":  "high-potassium foods (bananas, oranges, potatoes), high-phosphorus foods (dairy, nuts), excess protein",
            "prefer": "white bread, white rice, cauliflower, cabbage, apples, berries, egg whites",
            "tip":    "Fluid and protein intake must be guided by your nephrologist. Do NOT self-adjust.",
        }
    if "celiac" in lower or "gluten" in lower:
        adjustments["celiac_gluten"] = {
            "avoid":  "wheat, barley, rye, malt, triticale — all gluten-containing grains",
            "prefer": "rice, quinoa, oats (certified GF), corn, potatoes, legumes, fresh fruits",
            "tip":    "Check all food labels. Cross-contamination risk is real — use dedicated cookware.",
        }
    if "gout" in lower:
        adjustments["gout"] = {
            "avoid":  "red meat, organ meats, shellfish, beer, high-fructose corn syrup, sugary drinks",
            "prefer": "low-fat dairy, cherries, berries, plenty of water (3+ litres/day), vegetables",
            "tip":    "Stay well-hydrated (water is the best uric-acid flush). Vitamin C supplements may help.",
        }
    if "heart" in lower:
        adjustments["heart_disease"] = {
            "avoid":  "saturated and trans fats, high sodium, excess alcohol, sugar-sweetened beverages",
            "prefer": "Mediterranean diet: olive oil, fish (omega-3), whole grains, legumes, nuts, vegetables",
            "tip":    "Follow AHA guidelines: < 6% calories from saturated fat, < 1,500 mg/day sodium.",
        }

    if not adjustments:
        adjustments["general"] = {
            "avoid":  "ultra-processed foods, excess added sugar (< 25g/day), excess sodium",
            "prefer": "whole foods, a rainbow of fruits and vegetables, lean proteins, healthy fats",
            "tip":    "Drink at least 8 glasses (2 litres) of water daily. Move for 30 min most days.",
        }
    return adjustments


@register_tool("calculate_bmr")
def calculate_bmr(age: int, weight_kg: float = 65, height_cm: float = 170,
                  gender: str = "unspecified") -> dict:
    """
    Estimates Basal Metabolic Rate using the Mifflin-St Jeor equation.
    Returns BMR and activity-adjusted calorie targets.
    """
    g = gender.lower()
    if g in ("male", "m"):
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
    elif g in ("female", "f"):
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age - 161
    else:
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age - 78   # sex-neutral average
    return {
        "bmr":       round(bmr),
        "sedentary": round(bmr * 1.2),   # desk job, no exercise
        "moderate":  round(bmr * 1.55),  # 3–5 days/week exercise (plan target)
        "active":    round(bmr * 1.725), # 6–7 days/week hard exercise
    }


# ═══════════════════════════════════════════════════════════════════════════
#  PROMPT BUILDERS
# ═══════════════════════════════════════════════════════════════════════════

def _build_plan_prompt(profile: dict, local_foods: str,
                       med_adjustments: dict, calorie_info: dict) -> str:
    med_text = json.dumps(med_adjustments, indent=2)
    return f"""### System
You are NutriCoach, a professional AI nutrition advisor built on IBM Granite.
Your role is to create safe, personalised, science-backed 7-day nutrition plans.
Tone: warm, encouraging, and professional (HAM — Helpful, Accurate, Mindful).

Rules:
- Never recommend stopping, reducing, or replacing medication or medical treatment.
- Never guarantee cures, rapid weight loss, or unrealistic health outcomes.
- Always suggest consulting a healthcare professional for any medical condition.
- Focus on whole foods, nutritional balance, and sustainable long-term habits.
- Use markdown formatting with tables and headers.

### User Profile
- Name: {profile["name"]}
- Age: {profile["age"]}
- Gender: {profile.get("gender", "unspecified")}
- Diet Preference: {profile["diet_preference"]}
- Location: {profile["city"]}, {profile["country"]}
- Medical History: {profile.get("medical_history", "None")}

### Tool Outputs (Agent gathered this data via tool calls)
Local Popular Foods: {local_foods}
Medical Dietary Adjustments:
{med_text}
Estimated Daily Calories: {calorie_info["moderate"]} kcal (moderate activity level)

### Task
Generate a complete 7-day personalised nutrition plan for {profile["name"]}. Include:
1. Brief nutritional assessment (2–3 sentences)
2. Daily calorie target and macro breakdown (carbs/protein/fat %)
3. Day-by-day meal plan (Breakfast, Mid-Morning Snack, Lunch, Afternoon Snack, Dinner)
4. Hydration and lifestyle tips
5. Foods to AVOID based on medical history
6. Weekly grocery list (grouped by category)

### NutriCoach Response
"""


def _build_qa_prompt(question: str, profile: dict, conversation_history: list) -> str:
    history_text = ""
    for turn in conversation_history[-4:]:
        history_text += f"User: {turn['user']}\nNutriCoach: {turn['assistant']}\n"

    return f"""### System
You are NutriCoach, a friendly AI nutrition advisor built on IBM Granite.
Answer the user's nutrition/health question helpfully and accurately.
If the question is outside nutrition/health, politely redirect.
Never provide dangerous medical advice. Always recommend consulting professionals for serious health issues.
User profile context: Age {profile.get("age","N/A")}, Diet: {profile.get("diet_preference","N/A")}, \
Location: {profile.get("city","N/A")}, {profile.get("country","N/A")}, \
Medical: {profile.get("medical_history","None")}.

### Conversation History
{history_text}

### User Question
{question}

### NutriCoach Answer
"""


# ═══════════════════════════════════════════════════════════════════════════
#  STRUCTURED PLAN GENERATOR  (always-available, no API key required)
# ═══════════════════════════════════════════════════════════════════════════

def _generate_structured_plan(profile: dict, local_foods: str,
                               med_adjustments: dict, calorie_info: dict) -> str:
    """
    Generates a comprehensive, fully-structured 7-day nutrition plan
    using rule-based logic. Ensures the app is always demonstrable.
    """
    name    = profile["name"]
    age     = int(profile["age"])
    diet    = profile["diet_preference"]
    city    = profile["city"]
    country = profile["country"]
    medical = profile.get("medical_history", "None")
    is_veg  = "non-vegetarian" not in diet.lower() and "pescatarian" not in diet.lower()
    is_vegan = "vegan" in diet.lower()
    is_pesc  = "pescatarian" in diet.lower()

    # ── Protein sources ─────────────────────────────────────────────────────
    if is_vegan:
        protein_src = "Tofu, Tempeh, Legumes (Dal/Chickpeas/Rajma), Edamame, Nuts, Seeds"
    elif is_pesc:
        protein_src = "Fish (Salmon/Tuna/Tilapia), Prawns, Eggs, Dal, Legumes, Tofu"
    elif is_veg:
        protein_src = "Dal, Paneer, Eggs, Legumes (Chickpeas/Rajma), Tofu, Greek Yogurt, Nuts"
    else:
        protein_src = "Chicken (grilled/baked), Fish, Eggs, Dal, Legumes, Greek Yogurt"

    # ── Dairy / dairy-alternative note ──────────────────────────────────────
    dairy_note = "Soy Yogurt / Oat Milk / Almond Milk" if is_vegan else "Low-fat Curd/Yogurt, Skimmed Milk, Paneer"

    # ── 7-day meal plans by diet type ────────────────────────────────────────
    if is_vegan:
        days = [
            ("Foundation Day",     "Oat Porridge with chia seeds, berries, and almond milk",   "Soaked almonds + orange",     "Brown Rice + Dal + Stir-fried Vegetables",              "Roasted chickpeas",              "Vegetable Stew + Quinoa"),
            ("Protein Focus",      "Tofu Scramble + whole-grain toast + avocado",              "Soy yogurt + mixed seeds",    "Chickpea Curry + Brown Rice + Kachumber Salad",         "Apple + peanut butter",          "Lentil Soup + Whole-grain Bread"),
            ("Fibre & Gut Health", "Whole-grain toast + almond butter + banana",              "Chia pudding with plant milk","Rajma (kidney beans) + Brown Rice + Green Salad",       "Carrot sticks + hummus",         "Tofu Stir-fry + Quinoa + Greens"),
            ("Antioxidant Boost",  "Berry smoothie (oat milk, berries, flaxseeds)",           "Walnuts + dark chocolate (1 sq)", "Quinoa Bowl + Roasted Vegetables + Avocado",         "Green tea + almonds",            "Veg Curry + Brown Rice"),
            ("Hydration Day",      "Methi Roti + avocado spread + herbal tea",               "Coconut water + banana",      "Chole (chickpea curry) + Roti + Salad",                 "Roasted makhana (fox nuts)",      "Palak (spinach) Dal + Roti"),
            ("Light & Detox",      "Moong Dal Chilla + mint chutney",                        "Fresh fruit salad",           "Vegetable Khichdi + pickles",                           "Herbal tea + rice crackers",      "Clear Vegetable Soup + Whole-grain Bread"),
            ("Nourish & Restore",  "Idli + Sambar + coconut chutney",                        "Sprouts chaat with lemon",    "Mixed Dal + Brown Rice + Mixed Vegetables",             "Plant-based smoothie",           "Dal Tadka + Roti + Cucumber Raita (soy curd)"),
        ]
    elif is_pesc:
        days = [
            ("Foundation Day",     "Scrambled eggs + whole-grain toast + avocado",           "Mixed nuts + 1 fruit",        "Grilled Fish + Brown Rice + Green Salad",                "Soy yogurt + flaxseeds",         "Prawn Stir-fry + Quinoa + Steamed Broccoli"),
            ("Protein Focus",      "Greek yogurt + granola + mixed berries",                 "Roasted chickpeas",           "Tuna Salad Wrap + Vegetable Soup",                       "Apple + almond butter",          "Baked Salmon + Steamed Sweet Potato + Salad"),
            ("Fibre & Gut Health", "Oats + chia seeds + banana + nuts",                     "Yogurt + berries",            "Prawn & Lentil Soup + Whole-grain Bread",                "Carrot sticks + hummus",         "Tilapia Curry + Brown Rice + Greens"),
            ("Antioxidant Boost",  "Smoothie bowl with berries + granola",                  "Walnuts + dark chocolate",    "Smoked Salmon + Quinoa + Roasted Vegetables",            "Green tea + almonds",            "Fish Tacos + Avocado + Salsa"),
            ("Hydration Day",      "Poached eggs + whole-grain toast + tomatoes",           "Coconut water + banana",      "Prawn Fried Rice (brown) + Salad",                       "Trail mix",                       "Fish Curry + Brown Rice + Cucumber Raita"),
            ("Light & Detox",      "Oat porridge + honey + berries",                        "Fresh fruit salad",           "Clear Fish Broth + Whole-grain Bread + Salad",           "Herbal tea + rice crackers",      "Grilled Shrimp + Steamed Vegetables + Quinoa"),
            ("Nourish & Restore",  "Egg Omelette + whole-grain toast + avocado",            "Sprouts + lemon",             "Tuna + Brown Rice Bowl + Roasted Vegetables",            "Fruit smoothie",                  "Pan-seared Fish + Mashed Sweet Potato + Greens"),
        ]
    elif not is_veg:  # non-vegetarian
        days = [
            ("Foundation Day",     "Scrambled eggs (2) + whole-grain toast + avocado",      "Mixed nuts + 1 fruit",        "Grilled Chicken + Brown Rice + Salad",                   "Greek yogurt + berries",          "Baked Fish + Steamed Broccoli + Quinoa"),
            ("Protein Focus",      "Greek yogurt + granola + mixed berries",                "Boiled egg + apple",          "Chicken Wrap (whole-wheat) + Vegetable Soup",            "Roasted chickpeas",              "Grilled Salmon + Sweet Potato + Asparagus"),
            ("Fibre & Gut Health", "Oats + chia seeds + banana",                           "Yogurt + flaxseeds",          "Turkey/Chicken Sandwich (whole-grain) + Salad",          "Carrot sticks + hummus",         "Stir-fried Chicken + Brown Rice + Mixed Greens"),
            ("Antioxidant Boost",  "Smoothie (spinach, banana, protein, almond milk)",     "Walnuts + dark chocolate",    "Chicken Quinoa Bowl + Roasted Vegetables",               "Green tea + almonds",            "Baked Chicken + Roasted Vegetables + Brown Rice"),
            ("Hydration Day",      "Egg Omelette + whole-grain toast + tomatoes",          "Coconut water + banana",      "Chicken Biryani (small portion) + Raita + Salad",        "Trail mix",                       "Fish Tikka + Brown Rice + Dal"),
            ("Light & Detox",      "Poached eggs + avocado toast",                        "Fresh fruit salad",           "Clear Chicken Broth + Whole-grain Bread + Salad",        "Herbal tea + rice crackers",      "Grilled Chicken + Steamed Vegetables"),
            ("Nourish & Restore",  "Omelette (2 eggs + spinach) + whole-grain bread",     "Sprouts chaat",               "Chicken + Quinoa + Mixed Vegetables",                    "Fruit smoothie",                  "Dal + Roti + Grilled Fish + Cucumber Raita"),
        ]
    else:  # vegetarian
        days = [
            ("Foundation Day",     "Oats Porridge + berries + almonds + honey",            "Mixed nuts + 1 fruit",        "Brown Rice + Dal + Sabzi + Curd",                        "Buttermilk + sprouts",            "Vegetable Khichdi + Raita"),
            ("Protein Focus",      "Vegetable Upma / Poha + green chutney",               "Roasted chickpeas",           "Whole-wheat Roti + Rajma + Salad",                       "Apple + peanut butter",           "Paneer Stir-fry + Roti + Soup"),
            ("Fibre & Gut Health", "Whole-grain toast + avocado + chia seeds",            "Curd with flaxseeds",         "Lentil Soup + Salad + Whole-wheat Bread",                "Carrot & cucumber + hummus",      "Tofu Stir-fry + Brown Rice + Greens"),
            ("Antioxidant Boost",  "Berry Smoothie + whole-grain cereal",                 "Walnuts + 1 sq dark choco",   "Quinoa + Roasted Veg + Paneer",                          "Green tea + almonds",             "Mixed Vegetable Curry + Brown Rice"),
            ("Hydration Day",      "Methi Thepla + mint chutney + curd",                 "Coconut water + banana",      "Chole Bhature (1 portion) / Chole + Roti + Salad",       "Roasted makhana",                 "Palak Paneer + Roti + Salad"),
            ("Light & Detox",      "Moong Dal Chilla + coconut chutney",                 "Fresh fruit salad",           "Vegetable Khichdi + Kadhi",                              "Herbal tea + rice crackers",      "Clear Veg Soup + Whole-wheat Bread"),
            ("Nourish & Restore",  "Idli (3) + Sambar + coconut chutney",               "Sprouts Chaat",               "Brown Rice + Sambar + Mixed Veg + Papad",               "Fruit Smoothie",                  "Dal Tadka + Roti + Cucumber Raita"),
        ]

    # ── Build avoid section ──────────────────────────────────────────────────
    avoid_section = ""
    for cond, info in med_adjustments.items():
        if cond != "general":
            avoid_section += (
                f"\n### {cond.replace('_', ' ').title()}\n"
                f"- **Avoid:** {info['avoid']}\n"
                f"- **Prefer:** {info['prefer']}\n"
                f"- 💡 **Tip:** {info['tip']}\n"
            )
    if not avoid_section:
        avoid_section = "\n- No specific restrictions beyond general healthy eating guidelines.\n- Minimise ultra-processed foods, added sugars, and excessive sodium.\n"

    # ── Macro calculation ───────────────────────────────────────────────────
    cal = calorie_info["moderate"]
    carb_g   = round(cal * 0.50 / 4)
    prot_g   = round(cal * 0.25 / 4)
    fat_g    = round(cal * 0.25 / 9)

    # ── Build day-by-day table ──────────────────────────────────────────────
    day_tables = ""
    for i, (theme, bfast, snack1, lunch, snack2, dinner) in enumerate(days, 1):
        day_tables += f"""
### Day {i} — {theme}
| Meal | Food | Approx. Calories |
|------|------|-----------------|
| 🌅 Breakfast | {bfast} | ~350 kcal |
| 🍎 Mid-Morning Snack | {snack1} | ~150 kcal |
| ☀️ Lunch | {lunch} | ~550 kcal |
| 🫐 Afternoon Snack | {snack2} | ~100 kcal |
| 🌙 Dinner | {dinner} | ~450 kcal |
"""

    plan = f"""# 🥗 Personalised 7-Day Nutrition Plan
**Prepared for:** {name} | **Age:** {age} | **Diet:** {diet.title()} | **Gender:** {profile.get("gender","Unspecified").title()}
**Location:** {city}, {country} | **Medical History:** {medical}
**Generated by:** NutriCoach AI — IBM Granite LLM | **Date:** {datetime.now().strftime("%B %d, %Y at %H:%M")}

---

## 📊 Nutritional Assessment

Based on your profile, a balanced and sustainable nutrition approach is recommended. At age {age}, your body requires adequate protein ({protein_src.split(",")[0].strip()} are excellent choices) alongside complex carbohydrates and healthy fats to sustain energy and metabolic health throughout the day. Incorporating locally available foods from {city} ensures cultural familiarity and supports long-term adherence to the plan.

---

## 🎯 Daily Nutritional Targets

| Metric | Value |
|--------|-------|
| **Basal Metabolic Rate (BMR)** | {calorie_info["bmr"]} kcal/day |
| **Sedentary Calorie Target** | {calorie_info["sedentary"]} kcal/day |
| **Moderate Activity Target** ✅ | {calorie_info["moderate"]} kcal/day |
| **Active Lifestyle Target** | {calorie_info["active"]} kcal/day |
| **Carbohydrates (50%)** | ~{carb_g}g/day |
| **Protein (25%)** | ~{prot_g}g/day |
| **Healthy Fats (25%)** | ~{fat_g}g/day |

> 💡 *Targets are calculated using the Mifflin-St Jeor equation. Moderate Activity (1,550 kcal multiplier) is your recommended daily intake.*

---

## 📅 7-Day Meal Plan
{day_tables}

---

## 💧 Hydration & Lifestyle Tips

- **Water Intake:** Drink **8–10 glasses (2.0–2.5 litres)** of plain water daily.
- **Morning Ritual:** Start with warm lemon water (skip if you have acidity/GERD).
- **Beverages:** Replace sugary drinks with herbal teas (chamomile, green tea, tulsi).
- **Sleep:** Aim for **7–8 hours** per night — sleep deprivation disrupts hunger hormones (ghrelin/leptin).
- **Exercise:** At least **30 minutes of moderate activity** (brisk walking, yoga, cycling) 5 days/week.
- **Meal Timing:** Eat every **3–4 hours** to maintain stable blood sugar and prevent overeating.
- **Mindful Eating:** Eat slowly, chew thoroughly (20+ chews per bite), avoid screens during meals.
- **Alcohol:** Limit to 0–1 unit/day for women and 0–2 units/day for men (if at all).

---

## 🚫 Foods to Avoid — Based on Your Health Profile
{avoid_section}

---

## 🛒 Weekly Grocery List

### 🥬 Vegetables & Greens
Spinach, Broccoli, Carrots, Cucumbers, Tomatoes, Bell Peppers, Cauliflower, Cabbage, Bottle Gourd, Zucchini, Sweet Potato

### 🍎 Fruits
Apples, Bananas (1/day), Berries (blueberry/strawberry), Oranges, Guava, Papaya, Pomegranate

### 🌾 Grains & Cereals
Brown Rice, Whole-Wheat Flour (Atta), Rolled Oats, Quinoa, Whole-Wheat Bread, Multigrain Roti

### 🫘 Proteins
{protein_src}, Chickpeas, Kidney Beans (Rajma), Green Moong Dal, Masoor Dal

### 🥛 Dairy / Dairy Alternatives
{dairy_note}, Chia Seeds (calcium source)

### 🌰 Healthy Fats & Nuts
Almonds (10–15/day), Walnuts (4–5/day), Flaxseeds, Chia Seeds, Olive Oil (extra virgin), Peanut Butter (no added sugar)

### 🌿 Herbs & Spices
Turmeric, Ginger, Garlic, Cinnamon, Fenugreek Seeds, Coriander, Cumin, Black Pepper

---

*🌍 Local food inspiration for {city}, {country}: {local_foods}*
"""
    return plan


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN AGENT FUNCTION — Generate Nutrition Plan  (ReAct Loop)
# ═══════════════════════════════════════════════════════════════════════════

def generate_nutrition_plan(profile: dict) -> dict:
    """
    Agentic AI ReAct Loop:
    1. THINK  — analyse user profile for personalisation signals
    2. ACT    — call registered tools (get_local_foods, get_medical_adjustments, calculate_bmr)
    3. REASON — build a context-rich, structured prompt
    4. GENERATE — call IBM Granite LLM (or structured offline generator)
    5. VALIDATE — run output safety guardrails
    6. RETURN — final plan dict with metadata
    """
    reasoning_trace = []

    # ── STEP 1: Think ────────────────────────────────────────────────────────
    reasoning_trace.append("THINK: Analysing user profile for personalisation signals.")

    # ── STEP 2: Act — Tool Calls ─────────────────────────────────────────────
    reasoning_trace.append(f"ACT: Calling tool [get_local_foods] → city={profile['city']}, country={profile['country']}, diet={profile['diet_preference']}")
    local_foods = TOOL_REGISTRY["get_local_foods"](
        profile["city"], profile["country"], profile["diet_preference"]
    )

    reasoning_trace.append("ACT: Calling tool [get_medical_adjustments] → processing health conditions.")
    med_adjustments = TOOL_REGISTRY["get_medical_adjustments"](
        profile.get("medical_history", "none")
    )

    reasoning_trace.append(f"ACT: Calling tool [calculate_bmr] → age={profile['age']}, gender={profile.get('gender','unspecified')}")
    calorie_info = TOOL_REGISTRY["calculate_bmr"](
        age=int(profile["age"]),
        gender=profile.get("gender", "unspecified"),
    )

    # ── STEP 3: Reason ───────────────────────────────────────────────────────
    reasoning_trace.append("REASON: Synthesising tool outputs into personalised context prompt.")
    prompt = _build_plan_prompt(profile, local_foods, med_adjustments, calorie_info)

    # ── STEP 4: Generate ─────────────────────────────────────────────────────
    reasoning_trace.append("GENERATE: Calling IBM Granite LLM (ibm/granite-13b-instruct-v2) via watsonx.ai.")
    raw_output = _call_granite(prompt)

    if raw_output == "__OFFLINE__":
        reasoning_trace.append("GENERATE: IBM Granite API not configured — using structured offline plan generator.")
        plan_text = _generate_structured_plan(profile, local_foods, med_adjustments, calorie_info)
        source = "structured_offline"
    else:
        plan_text = raw_output
        source = "ibm_granite"

    # ── STEP 5: Validate ─────────────────────────────────────────────────────
    reasoning_trace.append("VALIDATE: Running output safety guardrails (HAM principles).")
    safety = check_output_safety(plan_text)
    if not safety["safe"]:
        reasoning_trace.append(f"GUARDRAIL: Unsafe pattern detected [{safety['flagged']}] — falling back to safe structured plan.")
        plan_text = _generate_structured_plan(profile, local_foods, med_adjustments, calorie_info)
        source = "guardrail_fallback"

    # ── Append condition-specific health warnings ────────────────────────────
    warnings = get_condition_warnings(profile.get("medical_history", ""))
    if warnings:
        plan_text += "\n\n## ⚠️ Health Condition Notices\n"
        for w in warnings:
            plan_text += f"- {w}\n"
        plan_text += "\n> These notices are automatically generated to ensure your safety. Please consult the relevant specialist.\n"

    # ── Append mandatory medical disclaimer ──────────────────────────────────
    plan_text += MEDICAL_DISCLAIMER

    reasoning_trace.append("DONE: Personalised nutrition plan generated and validated successfully.")

    return {
        "plan":            plan_text,
        "calorie_info":    calorie_info,
        "local_foods":     local_foods,
        "source":          source,
        "reasoning_trace": reasoning_trace,
        "generated_at":    datetime.now().isoformat(),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Q&A AGENT — Resolve common user nutrition queries
# ═══════════════════════════════════════════════════════════════════════════

OFFLINE_QA = {
    "calorie": (
        "Your calorie needs depend on age, weight, height, and activity level. "
        "Sedentary adults generally need ~1,600–2,000 kcal/day; moderately active adults ~2,000–2,400 kcal/day. "
        "Use the calorie targets displayed in your generated nutrition plan for a personalised figure."
    ),
    "protein": (
        "Adults generally need **0.8–1.2g of protein per kg of body weight** per day. "
        "For a 65 kg person, that's approximately 52–78g. "
        "Excellent sources include dal, eggs, chicken, paneer, tofu, tempeh, legumes, and Greek yogurt."
    ),
    "sugar": (
        "Added sugars should be kept under **25g (6 tsp) per day** for women and **36g (9 tsp)** for men (WHO guidelines). "
        "Opt for natural sugars from whole fruits, which come packaged with fibre and micronutrients."
    ),
    "water": (
        "The general recommendation is **8–10 glasses (2–2.5 litres)** of water per day. "
        "This increases with physical activity, hot weather, fever, or breastfeeding. "
        "A good rule: drink until your urine is pale yellow."
    ),
    "diabetes": (
        "For type 2 diabetes management, focus on low-GI foods: whole grains, legumes, non-starchy vegetables, nuts. "
        "Avoid refined carbs, sugary drinks, white rice/bread. "
        "Eat smaller, more frequent meals. **Always follow your doctor's and dietitian's specific guidance** — "
        "medication and diet work together."
    ),
    "blood pressure": (
        "The DASH diet is evidence-based for blood pressure reduction: "
        "plenty of fruits, vegetables, whole grains, low-fat dairy, legumes, and nuts. "
        "Limit sodium to < 2,300 mg/day (ideally < 1,500 mg). "
        "Reduce alcohol, avoid processed meats, and exercise regularly. "
        "Always monitor BP and consult your cardiologist."
    ),
    "weight loss": (
        "Sustainable weight loss involves a modest **calorie deficit of 300–500 kcal/day** below maintenance, "
        "high-protein intake (to preserve muscle), adequate fibre (keeps you fuller), and regular exercise. "
        "Aim for 0.5–1 kg/week — faster loss typically means muscle loss. Crash diets are not recommended."
    ),
    "vitamin": (
        "Key vitamins to monitor: "
        "**Vitamin D** (sunlight 15–20 min/day + dairy/eggs/fatty fish or supplements); "
        "**B12** (essential for vegans — supplement or fortified foods); "
        "**Iron** (leafy greens, legumes, red meat — pair with Vitamin C for absorption); "
        "**Calcium** (dairy, sesame seeds, leafy greens). "
        "Get a blood test to identify any actual deficiencies before supplementing."
    ),
    "cholesterol": (
        "To lower LDL ('bad') cholesterol: increase soluble fibre (oats, beans, lentils), eat fatty fish (omega-3), "
        "use olive oil instead of butter, eat more avocados and nuts. "
        "Avoid saturated fats (red meat, butter, full-fat dairy) and trans fats (fried/packaged foods). "
        "Consult a cardiologist if readings are significantly elevated."
    ),
    "thyroid": (
        "For hypothyroidism (underactive), ensure adequate iodine (seaweed, fish, dairy) and selenium (Brazil nuts, eggs). "
        "Limit raw cruciferous vegetables in large quantities. "
        "Take your medication on an empty stomach. "
        "Always follow your endocrinologist's dietary recommendations."
    ),
    "fibre": (
        "Adults need **25–38g of dietary fibre per day**. "
        "Good sources: whole grains (oats, quinoa, brown rice), legumes (dal, beans), "
        "vegetables (broccoli, carrots), fruits (apples, berries), nuts, and seeds (flax, chia). "
        "Increase fibre gradually and drink plenty of water to avoid bloating."
    ),
    "iron": (
        "Iron-rich foods: red meat, liver, chicken, fish (haem iron — well absorbed); "
        "lentils, spinach, tofu, pumpkin seeds (non-haem iron — pair with Vitamin C to boost absorption). "
        "Avoid tea/coffee with iron-rich meals as tannins block absorption. "
        "Supplement only if blood test confirms deficiency."
    ),
    "meal timing": (
        "Eating every **3–4 hours** helps stabilise blood sugar and prevents excessive hunger. "
        "Try: breakfast within 1 hr of waking, mid-morning snack, lunch, afternoon snack, dinner. "
        "Avoid large meals late at night (after 8 PM). "
        "For weight management, total daily intake matters more than exact timing."
    ),
    "snack": (
        "Healthy snack options: a handful of mixed nuts (almonds/walnuts), an apple with peanut butter, "
        "Greek yogurt with berries, roasted chickpeas, hummus with carrot/cucumber sticks, "
        "a boiled egg, a small bowl of sprouts, or a piece of dark chocolate (70%+) with green tea. "
        "Aim for snacks that combine protein + fibre to keep you satisfied."
    ),
    "default": (
        "That's a great question! For personalised nutrition advice, I recommend consulting a registered dietitian "
        "who can properly assess your individual needs. "
        "Generally speaking, focus on whole, minimally-processed foods, adequate protein (0.8–1.2g/kg), "
        "plenty of colourful vegetables, healthy fats, and staying well-hydrated (8–10 glasses/day). "
        "Is there anything specific about nutrition or diet I can help you explore?"
    ),
}


def answer_query(question: str, profile: dict, conversation_history: list) -> dict:
    """
    Resolves user nutrition Q&A using IBM Granite or offline knowledge base.
    Includes input safety validation and output safety checking.
    """
    # Input safety check
    safety = check_input_safety(question)
    if not safety["safe"]:
        return {
            "answer": f"⚠️ {safety['reason']}",
            "source": "guardrail",
        }

    prompt = _build_qa_prompt(question, profile, conversation_history)
    raw = _call_granite(prompt)

    if raw == "__OFFLINE__":
        # Search offline knowledge base by keyword
        q_lower = question.lower()
        answer = OFFLINE_QA["default"]
        for key, val in OFFLINE_QA.items():
            if key != "default" and key in q_lower:
                answer = val
                break
        return {"answer": answer, "source": "offline_kb"}

    # Output safety check
    safety_out = check_output_safety(raw)
    if not safety_out["safe"]:
        return {
            "answer": (
                "I want to make sure I give you safe and responsible advice. "
                "For this specific question, I'd recommend consulting a qualified healthcare professional "
                "or a registered dietitian who can give you personalised and clinically validated guidance."
            ),
            "source": "guardrail",
        }

    return {"answer": raw, "source": "ibm_granite"}
