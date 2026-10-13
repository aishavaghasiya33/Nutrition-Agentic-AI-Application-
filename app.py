"""
app.py — Flask Web Server for Nutrition Agentic AI Application
IBM Granite LLM | Agentic AI SDLC | HAM Principles
"""

import os
import json
from flask import Flask, request, jsonify, render_template_string, session
from flask_cors import CORS
from dotenv import load_dotenv

from guardrails import validate_user_profile, check_input_safety, sanitize_user_input
from granite_agent import generate_nutrition_plan, answer_query
from plan_saver import save_plan_to_md, list_saved_plans, load_plan

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "nutrition-ai-secret-2024")
CORS(app)

# ── Load the HTML template from file ────────────────────────────────────────
TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "templates", "index.html")

def get_template():
    with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
        return f.read()


# ═══════════════════════════════════════════════════════════════════════════
#  ROUTES
# ═══════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template_string(get_template())


@app.route("/api/generate-plan", methods=["POST"])
def api_generate_plan():
    """
    Endpoint: Generate personalised nutrition plan.
    Accepts JSON profile, runs agentic pipeline, returns plan.
    """
    try:
        data = request.get_json(force=True)

        # Sanitize all string inputs
        for field in ["name", "city", "country", "medical_history"]:
            if field in data:
                data[field] = sanitize_user_input(data[field], field)

        # Validate profile
        validation = validate_user_profile(data)
        if not validation["valid"]:
            return jsonify({"success": False, "errors": validation["errors"]}), 400

        # Input safety check on medical history / free text
        safety = check_input_safety(data.get("medical_history", ""))
        if not safety["safe"]:
            return jsonify({"success": False, "errors": [safety["reason"]]}), 400

        # Store profile in session
        session["profile"] = data
        session["conversation_history"] = []

        # Run the agentic pipeline
        result = generate_nutrition_plan(data)

        # Auto-save to .md
        save_result = save_plan_to_md(data, result["plan"])

        return jsonify({
            "success": True,
            "plan": result["plan"],
            "calorie_info": result["calorie_info"],
            "local_foods": result["local_foods"],
            "source": result["source"],
            "reasoning_trace": result["reasoning_trace"],
            "saved_file": save_result["filename"],
            "generated_at": result["generated_at"],
        })

    except Exception as e:
        return jsonify({"success": False, "errors": [f"Server error: {str(e)}"]}), 500


@app.route("/api/ask", methods=["POST"])
def api_ask():
    """
    Endpoint: Answer user's nutrition Q&A query.
    Maintains multi-turn conversation context.
    """
    try:
        data = request.get_json(force=True)
        question = sanitize_user_input(data.get("question", ""), "question")

        if not question or len(question.strip()) < 3:
            return jsonify({"success": False, "error": "Please enter a valid question."}), 400

        profile = session.get("profile", {
            "name": "User", "age": 30, "diet_preference": "vegetarian",
            "city": "Unknown", "country": "Unknown", "medical_history": "None"
        })
        history = session.get("conversation_history", [])

        result = answer_query(question, profile, history)

        # Update conversation history
        history.append({"user": question, "assistant": result["answer"]})
        if len(history) > 10:
            history = history[-10:]
        session["conversation_history"] = history

        return jsonify({
            "success": True,
            "answer": result["answer"],
            "source": result["source"],
        })

    except Exception as e:
        return jsonify({"success": False, "error": f"Server error: {str(e)}"}), 500


@app.route("/api/saved-plans", methods=["GET"])
def api_saved_plans():
    """Returns list of saved nutrition plan filenames."""
    plans = list_saved_plans()
    return jsonify({"success": True, "plans": plans})


@app.route("/api/load-plan/<filename>", methods=["GET"])
def api_load_plan(filename):
    """Loads a previously saved nutrition plan."""
    # Security: prevent path traversal
    if ".." in filename or "/" in filename or "\\" in filename:
        return jsonify({"success": False, "error": "Invalid filename."}), 400
    content = load_plan(filename)
    if content is None:
        return jsonify({"success": False, "error": "Plan not found."}), 404
    return jsonify({"success": True, "content": content, "filename": filename})


@app.route("/api/health", methods=["GET"])
def api_health():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "app": "Nutrition Agentic AI",
        "model": "IBM Granite (ibm/granite-13b-instruct-v2)",
        "version": "1.0.0",
    })


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    port = int(os.getenv("FLASK_PORT", 5000))
    print("=" * 60)
    print("  Nutrition Agentic AI -- IBM Granite LLM")
    print("=" * 60)
    print(f"  Open your browser: http://localhost:{port}")
    print(f"  Plans saved to:    saved_plans/")
    print(f"  Model:             IBM Granite 13B Instruct v2")
    print("=" * 60)
    app.run(debug=False, host="0.0.0.0", port=port)
