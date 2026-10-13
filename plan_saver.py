"""
plan_saver.py — Save generated nutrition plans to .md files
Phase: Output Management & Persistence in Agentic AI SDLC
"""

import os
import re
from datetime import datetime


PLANS_DIR = os.path.join(os.path.dirname(__file__), "saved_plans")


def _safe_filename(name: str) -> str:
    """Sanitise name for use in filenames."""
    safe = re.sub(r"[^a-zA-Z0-9_\-]", "_", name)
    return safe[:30]


def save_plan_to_md(profile: dict, plan_text: str) -> dict:
    """
    Saves the nutrition plan as a Markdown (.md) file.
    Returns {"success": bool, "filepath": str, "filename": str}
    """
    os.makedirs(PLANS_DIR, exist_ok=True)

    safe_name = _safe_filename(profile.get("name", "user"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"nutrition_plan_{safe_name}_{timestamp}.md"
    filepath = os.path.join(PLANS_DIR, filename)

    # Build the full markdown document
    header = f"""---
title: "Personalised Nutrition Plan"
user: "{profile.get('name', 'N/A')}"
age: {profile.get('age', 'N/A')}
diet: "{profile.get('diet_preference', 'N/A')}"
location: "{profile.get('city', 'N/A')}, {profile.get('country', 'N/A')}"
medical_history: "{profile.get('medical_history', 'None')}"
generated_at: "{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
generated_by: "Nutrition Agentic AI — IBM Granite LLM"
---

"""

    full_content = header + plan_text

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(full_content)

    return {
        "success": True,
        "filepath": filepath,
        "filename": filename,
        "plans_dir": PLANS_DIR,
    }


def list_saved_plans() -> list:
    """Returns a list of all saved plan filenames."""
    if not os.path.exists(PLANS_DIR):
        return []
    files = [f for f in os.listdir(PLANS_DIR) if f.endswith(".md")]
    files.sort(reverse=True)  # newest first
    return files


def load_plan(filename: str) -> str:
    """Loads a saved plan by filename."""
    filepath = os.path.join(PLANS_DIR, filename)
    if not os.path.exists(filepath):
        return None
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()
