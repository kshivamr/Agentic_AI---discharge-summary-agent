"""
Discharge Summary Agent
=======================
Provider : Groq (free — groq.com)
Run      : python code.py
PDF      : patients_details/patient .pdf
Key      : .env  →  GROQ_API_KEY=gsk_...
Outputs  : discharge_summary.txt | trace.txt | raw_output.json
"""

import os, json, datetime, textwrap, io
from pathlib import Path

# ── Load .env ─────────────────────────────────────────────────────────────────
env_file = Path(".env")
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL   = "llama-3.3-70b-versatile"
MAX_STEPS    = 20

# Auto-detect PDF in patients_details/ folder (handles any filename / spaces)
_folder   = Path("patients_details")
_pdfs     = list(_folder.glob("*.pdf")) if _folder.exists() else []
PDF_PATH  = _pdfs[0] if _pdfs else Path("patients_details/patient.pdf")

OUT_TXT   = Path("discharge_summary.txt")
OUT_TRACE = Path("trace.txt")
OUT_JSON  = Path("raw_output.json")

# ── Check requirements early ──────────────────────────────────────────────────
if not GROQ_API_KEY:
    print("\nERROR: GROQ_API_KEY not found.")
    print("Add this line to your .env file:  GROQ_API_KEY=gsk_...")
    print("Get a free key at: https://console.groq.com\n")
    exit(1)

try:
    from openai import OpenAI
except ImportError:
    print("\nERROR: openai package missing.  Run:  pip install openai\n")
    exit(1)

try:
    import fitz
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False
    print("WARNING: pymupdf missing. Run:  pip install pymupdf")

try:
    import pytesseract
    from PIL import Image
    HAS_OCR = True
except ImportError:
    HAS_OCR = False
    print("WARNING: pytesseract/Pillow missing — OCR disabled. Run:  pip install pytesseract Pillow")

# ── Groq client ───────────────────────────────────────────────────────────────
client = OpenAI(
    api_key  = GROQ_API_KEY,
    base_url = "https://api.groq.com/openai/v1"
)

# ══════════════════════════════════════════════════════════════════════════════
#  TOOL 1 — Extract PDF text  (OCR fallback for scanned documents)
# ══════════════════════════════════════════════════════════════════════════════

def extract_pdf_text(pdf_path: str, max_chars: int = 60000) -> dict:
    path = Path(pdf_path)
    if not path.exists():
        return {"success": False, "error": f"File not found: {pdf_path}"}
    if not HAS_FITZ:
        return {"success": False, "error": "pymupdf not installed. Run: pip install pymupdf"}
    try:
        doc   = fitz.open(str(path))
        pages = []
        for i, page in enumerate(doc):
            text = page.get_text("text").strip()
            if not text and HAS_OCR:
                # Scanned image page — run OCR
                mat = fitz.Matrix(2.0, 2.0)
                pix = page.get_pixmap(matrix=mat)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                text = pytesseract.image_to_string(img, config="--psm 6").strip()
            if text:
                pages.append(f"[PAGE {i+1}]\n{text}")
        doc.close()
        full = "\n\n".join(pages)
        return {
            "success"  : True,
            "file"     : path.name,
            "num_pages": len(pages),
            "text"     : full[:max_chars],
            "truncated": len(full) > max_chars
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

# ══════════════════════════════════════════════════════════════════════════════
#  TOOL 2 — Drug interaction checker
# ══════════════════════════════════════════════════════════════════════════════

DRUG_DB = {
    ("oflox",      "meftal")    : "MODERATE: NSAIDs increase fluoroquinolone CNS toxicity risk.",
    ("ofloxacin",  "meftal")    : "MODERATE: NSAIDs increase fluoroquinolone CNS toxicity risk.",
    ("loperamide", "emeset")    : "LOW: Antidiarrhoeal + antiemetic — monitor constipation.",
    ("lopiramide", "emeset")    : "LOW: Antidiarrhoeal + antiemetic — monitor constipation.",
    ("tramadol",   "emeset")    : "LOW: Additive serotonergic effect; monitor.",
    ("meropenem",  "pan")       : "LOW: No significant interaction.",
    ("insulin",    "ofloxacin") : "LOW: Fluoroquinolones may affect blood glucose — monitor.",
    ("metformin",  "contrast")  : "HIGH: Hold metformin before contrast procedures.",
    ("warfarin",   "cipro")     : "HIGH: Fluoroquinolones significantly increase INR.",
}

def check_drug_interactions(medications: list) -> dict:
    ml    = [m.lower() for m in medications]
    found = []
    for i, a in enumerate(ml):
        for j, b in enumerate(ml[i+1:], i+1):
            for (ka, kb), msg in DRUG_DB.items():
                if (ka in a and kb in b) or (ka in b and kb in a):
                    found.append({
                        "pair"    : [medications[i], medications[j]],
                        "severity": msg.split(":")[0],
                        "detail"  : msg
                    })
    return {
        "medications_checked": medications,
        "interactions_found" : found,
        "count"              : len(found)
    }

# ══════════════════════════════════════════════════════════════════════════════
#  TOOL 3 — Flag for clinician review
# ══════════════════════════════════════════════════════════════════════════════

_flags = []

def flag_for_clinician_review(field: str, reason: str, details: str = "") -> dict:
    entry = {
        "field"  : field,
        "reason" : reason,
        "details": details,
        "time"   : datetime.datetime.now().strftime("%H:%M:%S")
    }
    _flags.append(entry)
    return {"flagged": True, "total_flags": len(_flags), "entry": entry}

# ══════════════════════════════════════════════════════════════════════════════
#  TOOL 4 — Compile final summary
# ══════════════════════════════════════════════════════════════════════════════

REQUIRED_FIELDS = [
    "patient_demographics", "admission_date", "discharge_date",
    "principal_diagnosis", "secondary_diagnoses", "hospital_course",
    "procedures", "discharge_medications", "medication_changes",
    "allergies", "follow_up_instructions", "pending_results", "discharge_condition"
]

def compile_summary(summary_dict: dict) -> dict:
    validated    = {}
    auto_flagged = []
    for field in REQUIRED_FIELDS:
        val   = summary_dict.get(field)
        empty = (val is None or
                 (isinstance(val, str) and not val.strip()) or
                 val == [])
        if empty:
            validated[field] = "MISSING — flag for clinician review"
            auto_flagged.append(field)
        else:
            validated[field] = val
    # keep any extra keys the agent added
    for k, v in summary_dict.items():
        if k not in validated:
            validated[k] = v
    return {
        "success"             : True,
        "validated_summary"   : validated,
        "auto_flagged_missing": auto_flagged,
        "clinician_flags"     : _flags
    }

# ══════════════════════════════════════════════════════════════════════════════
#  TOOL SCHEMAS  (OpenAI / Groq format)
# ══════════════════════════════════════════════════════════════════════════════

TOOLS = [
    {
        "type": "function",
        "function": {
            "name"       : "extract_pdf_text",
            "description": (
                "Extract all text from a patient PDF file. "
                "Automatically uses OCR for scanned/image PDFs. "
                "Call this FIRST before doing anything else."
            ),
            "parameters": {
                "type"      : "object",
                "properties": {
                    "pdf_path" : {"type": "string",  "description": "Full path to the PDF file"},
                    "max_chars": {"type": "integer", "description": "Max characters to return (default 60000)"}
                },
                "required": ["pdf_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name"       : "check_drug_interactions",
            "description": (
                "Check a list of medication names for known drug-drug interactions. "
                "Always call this after identifying the discharge medications."
            ),
            "parameters": {
                "type"      : "object",
                "properties": {
                    "medications": {
                        "type" : "array",
                        "items": {"type": "string"},
                        "description": "List of medication names"
                    }
                },
                "required": ["medications"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name"       : "flag_for_clinician_review",
            "description": (
                "Flag a field or safety concern for mandatory clinician review. "
                "Call this for: missing data, conflicting information between documents, "
                "medications changed with no documented reason, pending results, "
                "or any drug interaction found."
            ),
            "parameters": {
                "type"      : "object",
                "properties": {
                    "field"  : {"type": "string", "description": "Which field or topic needs review"},
                    "reason" : {"type": "string", "description": "Why it needs review"},
                    "details": {"type": "string", "description": "Additional context or conflicting values"}
                },
                "required": ["field", "reason"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name"       : "compile_summary",
            "description": (
                "Compile the final structured discharge summary. "
                "Call this as the LAST step after reading all documents, "
                "raising all flags, and checking drug interactions. "
                "Any field left empty is automatically marked MISSING."
            ),
            "parameters": {
                "type"      : "object",
                "properties": {
                    "summary_dict": {
                        "type"      : "object",
                        "description": "All discharge summary fields",
                        "properties": {
                            "patient_demographics" : {"type": "string"},
                            "admission_date"        : {"type": "string"},
                            "discharge_date"        : {"type": "string"},
                            "principal_diagnosis"   : {"type": "string"},
                            "secondary_diagnoses"   : {"type": "string"},
                            "hospital_course"       : {"type": "string"},
                            "procedures"            : {"type": "string"},
                            "discharge_medications" : {"type": "array", "items": {"type": "object"}},
                            "medication_changes"    : {"type": "string"},
                            "allergies"             : {"type": "string"},
                            "follow_up_instructions": {"type": "string"},
                            "pending_results"       : {"type": "string"},
                            "discharge_condition"   : {"type": "string"},
                            "conflicts_detected"    : {"type": "string"},
                            "agent_notes"           : {"type": "string"}
                        }
                    }
                },
                "required": ["summary_dict"]
            }
        }
    }
]

# ══════════════════════════════════════════════════════════════════════════════
#  TOOL DISPATCHER
# ══════════════════════════════════════════════════════════════════════════════

def dispatch(name: str, inputs: dict) -> str:
    try:
        if   name == "extract_pdf_text":
            r = extract_pdf_text(inputs["pdf_path"], inputs.get("max_chars", 60000))
        elif name == "check_drug_interactions":
            r = check_drug_interactions(inputs["medications"])
        elif name == "flag_for_clinician_review":
            r = flag_for_clinician_review(
                    inputs["field"], inputs["reason"], inputs.get("details", ""))
        elif name == "compile_summary":
            r = compile_summary(inputs["summary_dict"])
        else:
            r = {"error": f"Unknown tool: {name}"}
    except Exception as e:
        r = {"error": f"Tool '{name}' failed: {e}"}
    return json.dumps(r, indent=2)

# ══════════════════════════════════════════════════════════════════════════════
#  SYSTEM PROMPT
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM = """You are a clinical AI that produces discharge summary DRAFTS for clinician review.

ABSOLUTE RULES — never break these:
1. NEVER invent, guess, or assume any clinical fact not present in the source documents.
2. If a required field cannot be sourced → leave it blank and call flag_for_clinician_review.
3. If two documents disagree on the same fact → flag the conflict, do NOT silently pick one.
4. If a medication was added, stopped, or changed with no documented reason → flag for reconciliation.
5. Always call check_drug_interactions after identifying discharge medications.
6. This output is always a DRAFT — never auto-finalized.

WORKFLOW — follow in this exact order:
Step 1 → Call extract_pdf_text with the patient PDF path
Step 2 → Read carefully: identify demographics, dates, diagnoses, labs, medications, procedures
Step 3 → For every missing field, conflict, pending result, or unexplained med change:
         call flag_for_clinician_review
Step 4 → Call check_drug_interactions with the full discharge medication list
Step 5 → Call compile_summary with everything found (leave missing fields blank/empty)
Step 6 → Done

The PDF is likely a scanned handwritten document — OCR text may be imperfect.
When something is unclear, note it as unclear and flag it — never guess the value."""

# ══════════════════════════════════════════════════════════════════════════════
#  TRACE LOGGER
# ══════════════════════════════════════════════════════════════════════════════

def log(step: int, kind: str, content: str, trace_lines: list):
    line = f"\n{'='*65}\nSTEP {step} [{kind}]\n{'='*65}\n{content}\n"
    print(line)
    trace_lines.append(line)

# ══════════════════════════════════════════════════════════════════════════════
#  AGENT LOOP
# ══════════════════════════════════════════════════════════════════════════════

def run_agent() -> dict:
    global _flags
    _flags = []

    if not PDF_PATH.exists():
        print(f"\nERROR: PDF not found at '{PDF_PATH}'")
        print(f"Files in patients_details/: {list(_folder.glob('*'))}")
        exit(1)

    trace         = []
    step          = 0
    final_summary = None

    messages = [
        {
            "role"   : "user",
            "content": (
                f"Patient PDF is at: {PDF_PATH}\n\n"
                "Produce a complete structured discharge summary draft. "
                "Follow your workflow — start by extracting the PDF."
            )
        }
    ]

    log(0, "START", f"Model: {GROQ_MODEL}  |  PDF: {PDF_PATH}", trace)

    while step < MAX_STEPS:
        step += 1

        # ── Call Groq ──────────────────────────────────────────────────────────
        try:
            resp = client.chat.completions.create(
                model      = GROQ_MODEL,
                max_tokens = 4096,
                messages   = [{"role": "system", "content": SYSTEM}] + messages,
                tools      = TOOLS,
                tool_choice= "auto"
            )
        except Exception as e:
            log(step, "API ERROR", str(e), trace)
            break

        msg        = resp.choices[0].message
        stop       = resp.choices[0].finish_reason
        reasoning  = msg.content or ""
        tool_calls = msg.tool_calls or []

        if reasoning:
            log(step, "REASONING", reasoning, trace)

        # Add assistant message to history
        messages.append({"role": "assistant", "content": msg})

        # ── Check if finished ──────────────────────────────────────────────────
        if stop == "stop" and not tool_calls:
            log(step, "DONE", "Agent finished successfully.", trace)
            break
        if not tool_calls:
            log(step, "DONE", "No more tool calls.", trace)
            break

        # ── Execute each tool call ─────────────────────────────────────────────
        for tc in tool_calls:
            name   = tc.function.name
            inputs = json.loads(tc.function.arguments)

            log(step, f"TOOL CALL → {name}", json.dumps(inputs, indent=2), trace)

            result_str = dispatch(name, inputs)
            result_obj = json.loads(result_str)

            # Capture the final summary when compile_summary succeeds
            if name == "compile_summary" and result_obj.get("success"):
                final_summary = result_obj

            display = result_str[:2000] + ("\n...[truncated in trace]" if len(result_str) > 2000 else "")
            log(step, f"RESULT ← {name}", display, trace)

            # Feed result back to the model
            messages.append({
                "role"        : "tool",
                "tool_call_id": tc.id,
                "name"        : name,
                "content"     : result_str
            })

    else:
        log(step, "CAP REACHED", f"Hit {MAX_STEPS}-step limit. Stopping.", trace)

    return {
        "final_summary": final_summary,
        "flags"        : _flags,
        "trace"        : "\n".join(trace),
        "steps_used"   : step
    }

# ══════════════════════════════════════════════════════════════════════════════
#  OUTPUT FORMATTER
# ══════════════════════════════════════════════════════════════════════════════

def format_output(result: dict) -> str:
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    fs  = result.get("final_summary")

    L = [
        "=" * 68,
        "     DISCHARGE SUMMARY DRAFT — FOR CLINICIAN REVIEW ONLY",
        f"     Generated : {now}  |  Groq / {GROQ_MODEL}",
        "     WARNING   : AI draft. Do NOT use without clinician verification.",
        "=" * 68,
        ""
    ]

    if not fs:
        L.append("COMPILATION FAILED — check trace.txt for details.")
        L.append("")
    else:
        vs = fs.get("validated_summary", {})

        def section(title: str, key: str):
            val = vs.get(key, "MISSING")
            L += ["─" * 68, f"  {title.upper()}", "─" * 68]
            if isinstance(val, list):
                for item in val:
                    if isinstance(item, dict):
                        L.append("  * " + "  |  ".join(f"{k}: {v}" for k, v in item.items()))
                    else:
                        L.append(f"  * {item}")
            else:
                for ln in textwrap.wrap(str(val), 66):
                    L.append(f"  {ln}")
            L.append("")

        section("Patient Demographics",              "patient_demographics")
        section("Admission Date",                    "admission_date")
        section("Discharge Date",                    "discharge_date")
        section("Principal Diagnosis",               "principal_diagnosis")
        section("Secondary Diagnoses",               "secondary_diagnoses")
        section("Hospital Course",                   "hospital_course")
        section("Procedures",                        "procedures")

        # Medications — formatted as a table
        L += ["─" * 68, "  DISCHARGE MEDICATIONS", "─" * 68]
        meds = vs.get("discharge_medications", [])
        if isinstance(meds, list) and meds:
            L.append(f"  {'#':<3} {'Medication':<22} {'Dose':<10} {'Frequency':<14} {'Duration'}")
            L.append("  " + "-" * 60)
            for i, m in enumerate(meds, 1):
                if isinstance(m, dict):
                    L.append(
                        f"  {i:<3} "
                        f"{str(m.get('name','?'))[:21]:<22} "
                        f"{str(m.get('dose','?'))[:9]:<10} "
                        f"{str(m.get('frequency','?'))[:13]:<14} "
                        f"{m.get('duration','?')}"
                    )
                else:
                    L.append(f"  {i}. {m}")
        else:
            L.append(f"  {meds}")
        L.append("")

        section("Medication Changes from Admission", "medication_changes")
        section("Allergies",                         "allergies")
        section("Follow-up Instructions",            "follow_up_instructions")
        section("Pending Results",                   "pending_results")
        section("Discharge Condition",               "discharge_condition")

        # Conflicts
        if vs.get("conflicts_detected"):
            L += ["─" * 68, "  CONFLICTS — CLINICIAN MUST RESOLVE", "─" * 68]
            for ln in textwrap.wrap(vs["conflicts_detected"], 66):
                L.append(f"  {ln}")
            L.append("")

        # Auto-flagged missing fields
        missing = fs.get("auto_flagged_missing", [])
        if missing:
            L += ["─" * 68, "  AUTO-FLAGGED MISSING FIELDS", "─" * 68]
            for f in missing:
                L.append(f"  * {f}")
            L.append("")

    # Clinician review flags
    flags = result.get("flags", [])
    if flags:
        L += ["─" * 68, f"  CLINICIAN REVIEW FLAGS  ({len(flags)} total)", "─" * 68]
        for i, fl in enumerate(flags, 1):
            L.append(f"  [{i}] {fl['field']}")
            L.append(f"       Reason  : {fl['reason']}")
            if fl.get("details"):
                L.append(f"       Details : {fl['details']}")
            L.append("")

    L += [
        "=" * 68,
        f"  Steps used : {result.get('steps_used', 0)} / {MAX_STEPS}",
        f"  Flags      : {len(result.get('flags', []))}",
        "=" * 68
    ]
    return "\n".join(L)

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n  Discharge Summary Agent")
    print(f"  Provider  : Groq")
    print(f"  Model     : {GROQ_MODEL}")
    print(f"  PDF       : {PDF_PATH}")
    print(f"  Max steps : {MAX_STEPS}\n")

    result    = run_agent()
    formatted = format_output(result)

    OUT_TXT.write_text(formatted,                                  encoding="utf-8")
    OUT_TRACE.write_text(result["trace"],                          encoding="utf-8")
    OUT_JSON.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    print(formatted)
    print(f"\n  Files saved:")
    print(f"  {OUT_TXT}      — readable discharge summary")
    print(f"  {OUT_TRACE}         — full agent step trace")
    print(f"  {OUT_JSON}    — raw JSON output\n")