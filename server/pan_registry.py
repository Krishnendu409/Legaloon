"""
PAN Registry for LegaLoom-Env — FY 2025-26
Complete vendor PAN database covering all 260 invoices.
Inoperative PANs trigger 20% TDS under Section 206AA.
"""

from typing import Optional

PAN_DB = {
    # ── Inoperative PANs (not linked to Aadhaar) ────────────────────────────
    "AAXCC1234Z": {"name": "CloudMatrix Infrastructure Pvt Ltd", "status": "inoperative", "pan_type": "company"},
    "AAXCC5678A": {"name": "Alpha Tech Services Pvt Ltd",        "status": "inoperative", "pan_type": "company"},
    "AAXFF9012B": {"name": "Beta Consulting LLP",                "status": "inoperative", "pan_type": "llp"},
    "AAXCC3456C": {"name": "Gamma Security Pvt Ltd",             "status": "inoperative", "pan_type": "company"},

    # ── Professional services ─────────────────────────────────────────────────
    "AADFS1234F": {"name": "Sharma & Associates LLP",            "status": "operative",   "pan_type": "llp"},
    "AAFFK5678G": {"name": "Kapoor & Mehta Chartered Accountants","status": "operative",  "pan_type": "firm"},
    "ABCPG2345H": {"name": "Priya Gupta & Co (CS)",              "status": "operative",   "pan_type": "individual"},
    "AAFFV3456J": {"name": "Verma Architects LLP",               "status": "operative",   "pan_type": "llp"},
    "AADPM4567K": {"name": "Dr Suresh Mehta (Physician)",        "status": "operative",   "pan_type": "individual"},
    "AAFFT5678L": {"name": "Joshi Tax Consultants LLP",          "status": "operative",   "pan_type": "llp"},
    "AABPI6789M": {"name": "Iyer Interior Design Studio",        "status": "operative",   "pan_type": "individual"},

    # ── Technical services / IT companies ────────────────────────────────────
    "AABCT1234C": {"name": "TechServ Solutions Pvt Ltd",         "status": "operative",   "pan_type": "company"},
    "AABCI2345D": {"name": "Infovision Consulting Pvt Ltd",      "status": "operative",   "pan_type": "company"},
    "AABCD3456E": {"name": "DataPro Analytics Pvt Ltd",          "status": "operative",   "pan_type": "company"},
    "AABCN4567F": {"name": "Nexus BPO Services Pvt Ltd",         "status": "operative",   "pan_type": "company"},
    "AABCW5678G": {"name": "WebCraft Technologies Pvt Ltd",      "status": "operative",   "pan_type": "company"},
    "AABCS6789H": {"name": "SysNet IT Solutions Pvt Ltd",        "status": "operative",   "pan_type": "company"},
    "AABCP7890J": {"name": "CloudPeak Services Pvt Ltd",         "status": "operative",   "pan_type": "company"},

    # ── Contractors ───────────────────────────────────────────────────────────
    "AAQCS1122A": {"name": "QuickCater Services",                "status": "operative",   "pan_type": "company"},
    "AABCG1234B": {"name": "Guardian Security Solutions Pvt Ltd","status": "operative",   "pan_type": "company"},
    "AAQCC2345B": {"name": "CleanPro Facility Management",       "status": "operative",   "pan_type": "company"},
    "AABCE3456C": {"name": "EventCraft Pvt Ltd",                 "status": "operative",   "pan_type": "company"},
    "AAQFP4567C": {"name": "PrintWell Enterprises",              "status": "operative",   "pan_type": "firm"},
    "AABCM5678D": {"name": "Manpower Solutions India Pvt Ltd",   "status": "operative",   "pan_type": "company"},

    # ── Rent / property ───────────────────────────────────────────────────────
    "AABCP8901E": {"name": "Prestige Commercial Properties",     "status": "operative",   "pan_type": "company"},
    "AADPA9012F": {"name": "Anand Kumar Sharma",                 "status": "operative",   "pan_type": "individual"},
    "AABCC0123G": {"name": "ColoSpace Technologies Pvt Ltd",     "status": "operative",   "pan_type": "company"},
    "AABCE1234H": {"name": "EquipHire Solutions Pvt Ltd",        "status": "operative",   "pan_type": "company"},

    # ── Commission / brokerage ────────────────────────────────────────────────
    "AAQFS2345H": {"name": "SalesForce Agents",                  "status": "operative",   "pan_type": "firm"},
    "AAQFR3456J": {"name": "ReferralNet Associates",             "status": "operative",   "pan_type": "firm"},

    # ── 194T partner / 194Q goods ─────────────────────────────────────────────
    "AAFFM6789E": {"name": "Mehta & Sons Trading Co (Partnership)","status": "operative", "pan_type": "firm"},
    "AABCI7890F": {"name": "Industrial Steel Suppliers Pvt Ltd", "status": "operative",   "pan_type": "company"},
}


def lookup_pan(pan: str) -> Optional[dict]:
    return PAN_DB.get(pan.upper().strip())


def is_pan_valid(pan: str) -> bool:
    record = lookup_pan(pan)
    return record is not None and record["status"] == "operative"


def is_company(pan: str) -> bool:
    record = lookup_pan(pan)
    if record is None:
        return True  # safe default
    return record["pan_type"] in ("company", "llp")


def pan_status_message(pan: str) -> str:
    record = lookup_pan(pan)
    if record is None:
        return (f"PAN {pan} not found in registry. "
                "Section 206AA applies (typically 20%, with section-specific exceptions such as 194Q/194O at 5%).")
    if record["status"] == "inoperative":
        return (
            f"PAN {pan} ({record['name']}) is INOPERATIVE "
            f"(not linked to Aadhaar). "
            "Section 206AA applies (typically 20%, with section-specific exceptions such as 194Q/194O at 5%). "
            "CBDT Circular 6/2024 does not provide relief for FY 2025-26."
        )
    pan_type = record['pan_type']
    if pan_type in ('individual', 'firm'):
        rate_note = 'Individual/firm rate applies (e.g. 194C @ 1%, not 2%).'
    else:
        rate_note = 'Company rate applies (e.g. 194C @ 2%).'
    return (
        f"PAN {pan} ({record['name']}) is operative and valid. "
        f"Vendor type: {pan_type}. {rate_note}"
    )
