from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, Optional

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import pandas as pd
from forecast_math import Assumptions, build_forecast_df, df_to_excel_bytes, df_to_json_table
from io import BytesIO
from dotenv import load_dotenv
import tempfile
import os
from excel_processor import process_excel_file

load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Gemini configuration
_GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
_GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


def _call_gemini_parse_assumptions(query: str) -> Dict[str, Any]:
    """Call Gemini to parse user query into structured assumptions."""
    if not _GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set. Set it in your environment to enable parsing via Gemini.")

    import google.generativeai as genai

    genai.configure(api_key=_GEMINI_API_KEY)
    model = genai.GenerativeModel(
        model_name=_GEMINI_MODEL_NAME,
        generation_config={
            "response_mime_type": "application/json",
            "temperature": 0.2,
        },
        system_instruction=(
            "You are a Finance Copilot API. Your ONLY job is to take any messy, vague, or informal "
            "user query about forecasting revenue/sales growth and convert it into a STRICT JSON object "
            "with the following exact keys (no extras, no missing): "
            "months, initial_salespeople, salespeople_added_per_month, deals_per_salesperson, "
            "large_customer_revenue_per_month, marketing_spend_per_month, average_cac, conversion_rate, "
            "sme_customer_revenue_per_month, overrides. "
            "Rules:\n"
            "1. Always return ONLY valid JSON — never natural language outside the JSON.\n"
            "2. Normalize user input: convert vague phrases like '2k ads' → 2000, '4 percent' → 0.04, etc.\n"
            "3. If the user specifies sudden changes (e.g. 'in month 6 hire 100 salespeople'), put them in "
            "an array under 'overrides' with objects like {\"month\": 6, \"field\": \"salespeople\", \"op\": \"add\", \"value\": 100}.\n"
            "4. If the user provides contradictory info (e.g. adds 1 salesperson each month but also says "
            "'hire 100 in month 6'), keep BOTH — use normal flow plus overrides.\n"
            "5. If the user omits critical fields (e.g. average_cac, conversion_rate, months, etc.), "
            "fill them with a string 'MISSING_FIELDNAME' instead of guessing, e.g. 'MISSING_average_cac'.\n"
            "6. Units: assume USD for money, months are integers, percentages are decimals (0.04 not 4).\n"
            "7. Be extremely fault-tolerant: handle bad grammar, shorthand, and incomplete sentences.\n"
            "8. Output structure must always match the schema with all required keys present."
        )
    )

    response = model.generate_content(query)
    text = response.text if hasattr(response, "text") else str(response)
    
    # Check if Gemini is asking for missing data
    if "MISSING_" in text and ("We still need" in text or "Could you provide" in text):
        json_start = text.find("{")
        json_end = text.rfind("}") + 1
        if json_start != -1 and json_end > json_start:
            json_part = text[json_start:json_end]
            try:
                parsed = json.loads(json_part)
                return parsed
            except json.JSONDecodeError:
                pass
    
    # Attempt to extract JSON payload
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            parsed = json.loads(text[start : end + 1])
        else:
            raise RuntimeError(f"Gemini returned a non-JSON response: {text[:200]}...")

    return parsed


def _coerce_assumptions(payload: Dict[str, Any]) -> Assumptions:
    """Convert parsed payload to Assumptions object with proper types and defaults."""
    
    def _num(value: Any, default: float) -> float:
        if value is None:
            return float(default)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            s = value.strip().replace(",", "").replace("$", "")
            # Handle percentages like "4%"
            if s.endswith("%"):
                try:
                    return float(s[:-1]) / 100.0
                except ValueError:
                    return float(default)
            # Handle shorthand like 200k, 1.5m, 2b
            try:
                lower = s.lower()
                multiplier = 1.0
                if lower.endswith("k"):
                    multiplier = 1_000.0
                    lower = lower[:-1]
                elif lower.endswith("m"):
                    multiplier = 1_000_000.0
                    lower = lower[:-1]
                elif lower.endswith("b"):
                    multiplier = 1_000_000_000.0
                    lower = lower[:-1]
                return float(lower) * multiplier
            except ValueError:
                # Best-effort: extract first number from the string (e.g., "8 months")
                import re
                m = re.search(r"-?\d+(?:\.\d+)?", s)
                if m:
                    try:
                        return float(m.group(0))
                    except ValueError:
                        return float(default)
                return float(default)
        return float(default)

    def _int(value: Any, default: int) -> int:
        v = _num(value, default)
        return int(v)

    # Extract and normalize values
    months_v = _int(payload.get("months"), 12)
    initial_salespeople_v = _int(payload.get("initial_salespeople"), 2)
    salespeople_added_per_month_v = _int(payload.get("salespeople_added_per_month"), 1)
    deals_per_salesperson_v = _num(payload.get("deals_per_salesperson"), 1)
    large_customer_revenue_per_month_v = _num(payload.get("large_customer_revenue_per_month"), 16667)
    marketing_spend_per_month_v = _num(payload.get("marketing_spend_per_month"), 200000)
    average_cac_v = _num(payload.get("average_cac"), 1250)
    conversion_rate_v = _num(payload.get("conversion_rate"), 0.45)
    sme_customer_revenue_per_month_v = _num(payload.get("sme_customer_revenue_per_month"), 5000)

    # Normalize conversion rate: if > 1, treat as percentage
    if conversion_rate_v > 1.0:
        conversion_rate_v = conversion_rate_v / 100.0
    # Clamp to [0,1]
    conversion_rate_v = max(0.0, min(1.0, conversion_rate_v))

    return Assumptions(
        months=max(1, months_v),
        initial_salespeople=max(0, initial_salespeople_v),
        salespeople_added_per_month=max(0, salespeople_added_per_month_v),
        deals_per_salesperson=max(0.0, deals_per_salesperson_v),
        large_customer_revenue_per_month=max(0.0, large_customer_revenue_per_month_v),
        marketing_spend_per_month=max(0.0, marketing_spend_per_month_v),
        average_cac=max(0.0001, average_cac_v),
        conversion_rate=conversion_rate_v,
        sme_customer_revenue_per_month=max(0.0, sme_customer_revenue_per_month_v),
        overrides=payload.get("overrides", []),
    )


def _merge_assumptions(previous: Dict[str, Any] | None, new: Dict[str, Any]) -> Dict[str, Any]:
    """Merge parsed assumptions into the previous baseline."""
    prev = dict(previous or {})
    merged: Dict[str, Any] = dict(prev)

    keys = [
        "months",
        "initial_salespeople", 
        "salespeople_added_per_month",
        "deals_per_salesperson",
        "large_customer_revenue_per_month",
        "marketing_spend_per_month",
        "average_cac",
        "conversion_rate",
        "sme_customer_revenue_per_month",
    ]

    for k in keys:
        if k in new and new[k] is not None and not (isinstance(new[k], str) and new[k].startswith("MISSING_")):
            merged[k] = new[k]

    # Overrides: merge with previous ones (accumulate)
    if "overrides" not in merged:
        merged["overrides"] = []
    
    if "overrides" in new and new["overrides"] is not None:
        # Merge new overrides with existing ones
        existing_overrides = merged["overrides"]
        new_overrides = new["overrides"]
        
        # Combine all overrides
        all_overrides = existing_overrides + new_overrides
        
        # Remove duplicates (same month + field combination, keep the latest)
        seen = {}
        for override in all_overrides:
            key = (override.get("month"), override.get("field"))
            seen[key] = override  # This will keep the latest override for each month+field
        
        merged["overrides"] = list(seen.values())

    return merged


@app.route("/forecast", methods=["POST"])
def forecast() -> Any:
    """Generate forecast from user query."""
    data = request.get_json(silent=True) or {}
    query = data.get("query") if isinstance(data, dict) else None
    if not query:
        # Also support raw text/plain body
        query = request.data.decode("utf-8") if request.data else ""
    if not query:
        return jsonify({"error": "Missing query in request body"}), 400

    try:
        # Parse query with Gemini
        parsed = _call_gemini_parse_assumptions(query)
        
        # Dashboard may pass the prior accepted baseline
        prev_assumptions = None
        if isinstance(data, dict):
            ctx = data.get("context") or {}
            prev_assumptions = ctx.get("previous_assumptions")
        
        # Merge into previous baseline so we keep editing the same state
        merged_payload = _merge_assumptions(prev_assumptions, parsed)
        
        # Convert to Assumptions object
        assumptions = _coerce_assumptions(merged_payload)
        
        # Build forecast
        df = build_forecast_df(assumptions)
        table = df_to_json_table(df)
        
        return jsonify({
            "assumptions": merged_payload, 
            "assumptions_used": assumptions.__dict__, 
            "forecast": table
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/download", methods=["GET"])
def download() -> Any:
    """Download forecast as Excel file."""
    params: Dict[str, Any] = {
        "months": int(request.args.get("months", 12)),
        "initial_salespeople": int(request.args.get("initial_salespeople", 2)),
        "salespeople_added_per_month": int(request.args.get("salespeople_added_per_month", 1)),
        "deals_per_salesperson": float(request.args.get("deals_per_salesperson", 1)),
        "large_customer_revenue_per_month": float(request.args.get("large_customer_revenue_per_month", 16667)),
        "marketing_spend_per_month": float(request.args.get("marketing_spend_per_month", 200000)),
        "average_cac": float(request.args.get("average_cac", 1250)),
        "conversion_rate": float(request.args.get("conversion_rate", 0.45)),
        "sme_customer_revenue_per_month": float(request.args.get("sme_customer_revenue_per_month", 5000)),
    }

    assumptions = _coerce_assumptions(params)
    df = build_forecast_df(assumptions)
    excel_bytes = df_to_excel_bytes(df)

    filename = f"forecast_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.xlsx"
    return send_file(
        path_or_file=BytesIO(excel_bytes),
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True,
        download_name=filename,
    )


@app.route("/forecast-excel", methods=["POST"])
def forecast_excel() -> Any:
    """Generate forecast from uploaded Excel file."""
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if not (file.filename.endswith('.xlsx') or file.filename.endswith('.xls')):
        return jsonify({"error": "File must be an Excel file (.xlsx or .xls)"}), 400
    
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
            file.save(tmp_file.name)
            tmp_file_path = tmp_file.name
        
        try:
            # Process the Excel file (returns same format as Gemini)
            result = process_excel_file(tmp_file_path)
            assumptions_dict = result['assumptions']
            
            # Convert to Assumptions object
            assumptions = _coerce_assumptions(assumptions_dict)
            
            # Build forecast
            df = build_forecast_df(assumptions)
            table = df_to_json_table(df)
            
            return jsonify({
                "assumptions": assumptions_dict,
                "assumptions_used": assumptions.__dict__,
                "forecast": table,
                "extracted_data": result.get('extracted_data', {}),
                "source": result.get('source', 'excel_processor'),
                "message": result.get('message', 'Excel file processed successfully')
            })
            
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
                
    except Exception as exc:
        return jsonify({"error": f"Error processing Excel file: {str(exc)}"}), 500


@app.route("/health", methods=["GET"])
def health() -> Any:
    """Health check endpoint."""
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8000)))


