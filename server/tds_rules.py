"""
TDS Rules Engine for LegaLoom-Env — FY 2025-26 (AY 2026-27)

Sources:
  - Income Tax Act 1961 (as amended by Finance Act 2025)
  - CBDT Circular 6/2024
  - ClearTax, RSM India, TDSRateChart.com (cross-verified)

Last verified: April 2026
"""

# ---------------------------------------------------------------------------
# Section 206AA — PAN rules (applies universally across all sections)
# ---------------------------------------------------------------------------
# When PAN is missing, invalid, or inoperative (not linked to Aadhaar):
#   Rate = higher of (section rate, rate in force, 20%)
#   For most B2B payments this means flat 20%
#   EXCEPTION: 194O and 194Q use 5% instead of 20%
PAN_INVALID_RATE = 20.0
PAN_INVALID_RATE_ECOMMERCE = 5.0   # 194O and 194Q special rule

# Section 206AB removed from April 1, 2025 — no longer applicable

# ---------------------------------------------------------------------------
# Master TDS Section Table — FY 2025-26
# ---------------------------------------------------------------------------
TDS_SECTIONS = {

    # -----------------------------------------------------------------------
    # 194C — Payments to Contractors
    # Covers: labour supply, catering, housekeeping, security, event
    #         management, transportation (if >10 trucks), printing,
    #         advertising production, construction, broadcasting
    # Does NOT cover: standard goods purchase, 194J services (from Oct 2024)
    # Special: Transporter with ≤10 trucks + PAN declaration → NIL TDS
    # -----------------------------------------------------------------------
    "194C": {
        "description": "Payments to Contractors",
        "nature": (
            "Work contracts: labour supply, catering, housekeeping, security, "
            "event management, transportation (>10 trucks), printing, "
            "advertising production, construction, broadcasting/telecasting. "
            "EXCLUDES: standard goods purchase and 194J services."
        ),
        "rate_individual": 1.0,      # Individual / HUF vendor
        "rate_default": 1.0,         # alias for individual
        "rate_company": 2.0,         # Company / Firm / LLP / Co-op society
        "rate_no_pan": 20.0,
        "threshold_single": 30000,   # Single payment threshold (UNCHANGED)
        "threshold_annual": 100000,  # Annual cumulative threshold (UNCHANGED)
        "gst_excluded": True,        # TDS on pre-GST if GST shown separately
        "transporter_exemption": True,  # Nil if ≤10 trucks + PAN declaration
        "examples": [
            "manpower supply", "housekeeping", "facility management",
            "catering services", "security services", "event management",
            "transportation services", "courier services",
            "printing and stationery", "advertising production",
            "construction work", "labour contract",
        ],
    },

    # -----------------------------------------------------------------------
    # 194J — Professional or Technical Services
    # Two sub-rates:
    #   Professional (10%): legal, medical, CA, architect, engineering profession
    #   Technical (2%):  IT support, software dev (company), cloud, data processing
    # KEY RULE: Company vendor → always technical (2%), never professional
    #           Individual/LLP professional → can be 10%
    # Threshold CHANGED: ₹30,000 → ₹50,000 from FY 2025-26
    # Director fees: NO threshold, always deduct
    # -----------------------------------------------------------------------
    "194J": {
        "description": "Fees for Professional or Technical Services",
        "nature": (
            "Professional (10%): legal, medical, CA, architect, CS, engineering profession, "
            "advertising (creative/design), interior decoration, notified professions. "
            "Technical (2%): IT consulting (company), software dev (company), "
            "cloud services, data processing, BPO, call centres, managerial services. "
            "Royalty non-film (10%), Royalty film (2%), Non-compete (10%), "
            "Director fees (10%, no threshold)."
        ),
        "rate_professional": 10.0,   # Legal, medical, CA, architect etc.
        "rate_default": 10.0,        # Default (professional)
        "rate_technical": 2.0,       # Technical services, call centre, BPO
        "rate_royalty_film": 2.0,    # Royalty for cinematographic film
        "rate_royalty_other": 10.0,  # Royalty other than film
        "rate_non_compete": 10.0,    # Non-compete fees
        "rate_director": 10.0,       # Director fees (no threshold)
        "rate_no_pan": 20.0,
        "threshold_single": 50000,   # CHANGED from ₹30,000 to ₹50,000 (FY 2025-26)
        "threshold_annual": 50000,   # CHANGED from ₹30,000 to ₹50,000 (FY 2025-26)
        "threshold_director": 0,     # NO threshold for director fees
        "gst_excluded": True,
        "examples_professional": [
            "legal consultation", "legal advisory", "advocate fees",
            "audit fees", "chartered accountant", "statutory audit",
            "medical consultation", "doctor fees",
            "architectural services", "architectural design",
            "engineering consultancy", "structural engineering",
            "company secretary", "secretarial services",
            "interior design", "interior decoration",
        ],
        "examples_technical": [
            "IT support", "IT maintenance", "technical support",
            "software development", "software development company",
            "cloud services", "cloud hosting", "cloud infrastructure",
            "data processing", "data analytics", "data management",
            "BPO services", "business process outsourcing",
            "call centre", "customer support outsourcing",
            "network maintenance", "system administration",
            "annual maintenance contract AMC", "platform services",
            "SaaS subscription", "managed services",
        ],
    },

    # -----------------------------------------------------------------------
    # 194I — Rent
    # Sub-rates: land/building/furniture (10%), machinery/equipment (2%)
    # Threshold CHANGED: ₹2,40,000 → ₹6,00,000 annual from FY 2025-26
    # Equivalently: ₹50,000 per month
    # NRI landlord: 30% (no threshold, no exemption)
    # Security deposit: NOT rent, not subject to TDS
    # -----------------------------------------------------------------------
    "194I": {
        "description": "Rent",
        "nature": (
            "Rent for land, building (including factory), furniture, fittings: 10%. "
            "Rent for plant, machinery, equipment: 2%. "
            "Covers: office rent, warehouse rent, co-working space, "
            "hotel rooms (regular business use), server rack rental. "
            "Does NOT cover: refundable security deposit."
        ),
        "rate_default": 10.0,        # Land / building / furniture / fittings
        "rate_land_building": 10.0,
        "rate_machinery": 2.0,       # Plant / machinery / equipment
        "rate_no_pan": 20.0,
        "rate_nri": 30.0,            # NRI landlord, no threshold
        "threshold_single": 0,       # No single-payment limit
        "threshold_annual": 600000,  # CHANGED: ₹2,40,000 → ₹6,00,000 (FY 2025-26)
        "threshold_monthly": 50000,  # Equivalent monthly cap
        "gst_excluded": True,
        "examples_building": [
            "office rent", "commercial rent", "warehouse rent",
            "cold storage rent", "co-working space", "business centre",
            "hotel rooms (regular business)", "godown rent",
        ],
        "examples_machinery": [
            "equipment rental", "machinery hire", "plant hire",
            "vehicle hire (without driver)", "server rack rental",
            "crane hire", "generator hire",
        ],
    },

    # -----------------------------------------------------------------------
    # 194IB — Rent by Individuals/HUF not liable for tax audit
    # Different from 194I — applies to individuals/HUF tenants
    # Rate CHANGED: 5% → 2% from October 1, 2024
    # TDS deducted ONCE: last month of year or tenancy, whichever earlier
    # -----------------------------------------------------------------------
    "194IB": {
        "description": "Rent by Individual/HUF (not under audit)",
        "nature": (
            "Applies when payer is individual or HUF NOT required to get tax audit. "
            "Monthly rent exceeds ₹50,000. "
            "TDS deducted once a year, not monthly."
        ),
        "rate_default": 2.0,         # CHANGED from 5% to 2% (Oct 2024)
        "rate_no_pan": 20.0,
        "threshold_single": 50000,   # Per month
        "threshold_annual": 600000,
        "gst_excluded": True,
    },

    # -----------------------------------------------------------------------
    # 194H — Commission or Brokerage
    # Rate CHANGED: 5% → 2% from October 1, 2024
    # Threshold CHANGED: ₹15,000 → ₹20,000 from FY 2025-26
    # -----------------------------------------------------------------------
    "194H": {
        "description": "Commission or Brokerage",
        "nature": (
            "Sales commission, agency commission, referral fees, "
            "dealer margins, distributor commissions, brokerage. "
            "Does NOT cover: insurance commission (194D), "
            "securities brokerage (194H with separate provisions)."
        ),
        "rate_default": 2.0,         # CHANGED from 5% to 2% (Oct 2024)
        "rate_no_pan": 20.0,
        "threshold_single": 20000,   # CHANGED from ₹15,000 (FY 2025-26)
        "threshold_annual": 20000,   # CHANGED from ₹15,000 (FY 2025-26)
        "gst_excluded": True,
        "examples": [
            "sales commission", "agency commission", "brokerage",
            "referral fee", "distribution commission", "dealer margin",
            "channel partner commission", "franchise fee (commission type)",
        ],
    },

    # -----------------------------------------------------------------------
    # 194A — Interest other than Securities
    # -----------------------------------------------------------------------
    "194A": {
        "description": "Interest other than Interest on Securities",
        "nature": (
            "Interest paid by firms, companies (not banks) on loans, "
            "inter-company interest, delayed payment interest. "
            "Banks use separate thresholds."
        ),
        "rate_default": 10.0,
        "rate_no_pan": 20.0,
        "threshold_single": 0,
        "threshold_annual": 10000,    # Non-bank payers
        "threshold_bank": 50000,      # CHANGED: ₹40,000 → ₹50,000 (FY 2025-26)
        "threshold_senior_citizen": 100000,  # CHANGED: ₹50,000 → ₹1,00,000 (FY 2025-26)
        "gst_excluded": False,
        "examples": [
            "loan interest", "inter-company interest",
            "delayed payment interest", "interest on security deposit",
        ],
    },

    # -----------------------------------------------------------------------
    # 194Q — Purchase of Goods
    # Applies ONLY if buyer's turnover > ₹10 crore
    # Threshold: ₹50,00,000 cumulative purchases from one vendor per year
    # No PAN: 5% (SPECIAL — not 20%)
    # -----------------------------------------------------------------------
    "194Q": {
        "description": "Purchase of Goods",
        "nature": (
            "Goods purchase when buyer's turnover exceeds ₹10 crore. "
            "Applies when cumulative purchase from one vendor > ₹50 lakh in year. "
            "Does NOT apply to transactions covered by 194O."
        ),
        "rate_default": 0.1,
        "rate_no_pan": 5.0,          # SPECIAL: 5% not 20% for goods
        "threshold_single": 0,
        "threshold_annual": 5000000,  # ₹50,00,000
        "buyer_turnover_required": 10000000,  # Buyer must have >₹10 crore turnover
        "gst_excluded": True,
        "examples": [
            "raw material purchase", "goods purchase", "merchandise",
            "inventory purchase", "product purchase", "bulk goods",
        ],
    },

    # -----------------------------------------------------------------------
    # 194O — E-Commerce Operator Payments
    # No PAN: 5% (SPECIAL — not 20%)
    # -----------------------------------------------------------------------
    "194O": {
        "description": "TDS on E-Commerce Participants",
        "nature": (
            "Payments by e-commerce operators to participants/sellers "
            "for goods/services facilitated through digital platform."
        ),
        "rate_default": 0.1,
        "rate_no_pan": 5.0,          # SPECIAL: 5% not 20%
        "threshold_single": 0,
        "threshold_annual": 500000,   # ₹5,00,000
        "gst_excluded": True,
    },

    # -----------------------------------------------------------------------
    # 194T — NEW from April 1, 2025
    # TDS on payments by partnership firms/LLPs to partners
    # Does NOT cover: capital repayment, drawings
    # -----------------------------------------------------------------------
    "194T": {
        "description": "Payments by Partnership Firms to Partners",
        "nature": (
            "NEW from April 1, 2025. "
            "Salary, remuneration, commission, bonus, interest paid by "
            "partnership firms or LLPs to their partners. "
            "Does NOT cover: capital repayment, drawings."
        ),
        "rate_default": 10.0,
        "rate_no_pan": 20.0,
        "threshold_single": 0,
        "threshold_annual": 20000,    # ₹20,000 annual
        "gst_excluded": False,
        "examples": [
            "partner salary", "partner remuneration", "partner commission",
            "partner bonus", "interest on partner capital",
        ],
    },

    # -----------------------------------------------------------------------
    # 194D — Insurance Commission
    # -----------------------------------------------------------------------
    "194D": {
        "description": "Insurance Commission",
        "nature": "Commission paid to insurance agents and intermediaries.",
        "rate_individual": 2.0,
        "rate_default": 2.0,
        "rate_company": 10.0,
        "rate_no_pan": 20.0,
        "threshold_single": 15000,
        "threshold_annual": 15000,
        "gst_excluded": False,
    },
}


# ---------------------------------------------------------------------------
# Categories that attract ZERO TDS under normal B2B conditions
# ---------------------------------------------------------------------------
NO_TDS_CATEGORIES = [
    # Standard goods (not covered by 194Q unless buyer turnover >10cr)
    "goods", "products", "merchandise", "hardware", "equipment purchase",
    "raw material", "stationery", "office supplies", "inventory",
    "furniture purchase", "computer purchase", "mobile purchase",

    # Exempt payees
    "government payment", "payment to government", "payment to rbi",
    "payment to lic", "payment to sebi",

    # Transporter exemption
    "transport charges with pan declaration",  # ≤10 trucks, PAN provided

    # Reimbursements at actual
    "reimbursement at actual", "actual expense reimbursement",

    # Below threshold
    "below threshold",
]


# ---------------------------------------------------------------------------
# Exemptions and special cases
# ---------------------------------------------------------------------------
EXEMPTIONS = {
    "government": "No TDS on payments to Central/State Government, RBI, SEBI",
    "form_15g_15h": "Vendor submits self-declaration — no TDS if income below exemption",
    "section_197": "Lower Deduction Certificate from IT dept — deduct at certified rate",
    "transporter_194c": "Nil TDS if transporter owns ≤10 goods carriages and furnishes PAN",
    "nri_dtaa": "Lower of domestic rate or DTAA treaty rate for NRI payments",
}


# ---------------------------------------------------------------------------
# Lookup functions
# ---------------------------------------------------------------------------

def get_rate(section_code: str, vendor_is_company: bool = True,
             pan_valid: bool = True, is_machinery: bool = False,
             is_technical: bool = False, is_director: bool = False,
             is_film_royalty: bool = False) -> float:
    """
    Return the correct TDS rate (%) for a given situation.

    Args:
        section_code      : e.g. "194J", "194C", "194I"
        vendor_is_company : True if vendor is Pvt Ltd / LLP / firm / company
        pan_valid         : False if PAN is missing, invalid, or inoperative
        is_machinery      : True for 194I machinery/equipment (2% sub-rate)
        is_technical      : True for 194J technical services (2% sub-rate)
        is_director       : True for director fees under 194J
        is_film_royalty   : True for royalty on cinematographic films (2%)

    Returns:
        Float percentage, e.g. 10.0 means 10%
    """
    # 206AA override — PAN invalid/inoperative/missing
    if not pan_valid:
        # Special: 194O and 194Q use 5% not 20%
        if section_code in ("194O", "194Q"):
            return PAN_INVALID_RATE_ECOMMERCE
        return PAN_INVALID_RATE

    section = TDS_SECTIONS.get(section_code)
    if not section:
        return 0.0

    # Section-specific rate logic
    if section_code == "194C":
        return section["rate_company"] if vendor_is_company else section["rate_individual"]

    if section_code == "194I":
        return section["rate_machinery"] if is_machinery else section["rate_land_building"]

    if section_code == "194IB":
        return section["rate_default"]

    if section_code == "194J":
        if is_director:
            return section["rate_director"]
        if is_film_royalty:
            return section["rate_royalty_film"]
        # Company vendors can only provide technical services (not professional)
        if vendor_is_company and not is_technical:
            return section["rate_technical"]
        if is_technical:
            return section["rate_technical"]
        return section["rate_professional"]

    if section_code == "194D":
        return section["rate_company"] if vendor_is_company else section["rate_individual"]

    return section.get("rate_default", 0.0)


def threshold_crossed(section_code: str, invoice_amount: float,
                      cumulative_ytd: float = 0.0,
                      is_director: bool = False) -> bool:
    """
    Determine whether TDS is actually due based on threshold rules.

    Args:
        section_code    : TDS section
        invoice_amount  : current invoice value in INR (pre-GST)
        cumulative_ytd  : total paid to this vendor so far this financial year
        is_director     : True for director fees (no threshold)

    Returns:
        True if TDS must be deducted on this invoice
    """
    # Director fees — no threshold, always deduct
    if section_code == "194J" and is_director:
        return True

    section = TDS_SECTIONS.get(section_code)
    if not section:
        return False

    single_limit  = section.get("threshold_single", 0)
    annual_limit  = section.get("threshold_annual", 0)
    running_total = cumulative_ytd + invoice_amount

    # No thresholds at all → TDS always applies
    if single_limit == 0 and annual_limit == 0:
        return True

    # Single payment crosses limit
    if single_limit > 0 and invoice_amount >= single_limit:
        return True

    # Cumulative annual crosses limit
    if annual_limit > 0 and running_total >= annual_limit:
        return True

    return False


def compute_tds(taxable_amount: float, rate_percent: float) -> float:
    """
    Compute TDS deduction amount.

    Args:
        taxable_amount : INR amount (pre-GST if GST shown separately)
        rate_percent   : e.g. 10.0 for 10%

    Returns:
        TDS amount in INR, rounded to 2 decimal places
    """
    return round(taxable_amount * rate_percent / 100, 2)


def get_tds_base(gross_amount: float, gst_amount: float = 0.0,
                 gst_shown_separately: bool = True) -> float:
    """
    Calculate the correct base amount for TDS computation.

    Args:
        gross_amount          : total invoice amount including GST
        gst_amount            : GST component
        gst_shown_separately  : True if GST is itemised separately on invoice

    Returns:
        Taxable base amount for TDS
    """
    if gst_shown_separately and gst_amount > 0:
        return gross_amount - gst_amount
    return gross_amount  # TDS on full amount if GST not broken out


def section_summary(section_code: str) -> str:
    """Return a human-readable description of a section for agent prompts."""
    s = TDS_SECTIONS.get(section_code)
    if not s:
        return f"Unknown section: {section_code}"
    return (
        f"{section_code} — {s['description']} | "
        f"Default rate: {s['rate_default']}% | "
        f"Nature: {s['nature'][:100]}"
    )


def classify_service(description: str, vendor_is_company: bool = True) -> dict:
    """
    Attempt to classify a service description into a TDS section.
    Returns dict with section_code, rate, and confidence.

    This is used by the RAG/lookup action in the environment.
    """
    desc = description.lower().strip()

    # Priority order matters — more specific first
    classification_rules = [
        # 194I — Rent (check first, very distinct)
        ("194I", ["rent", "lease", "office space", "warehouse rent",
                  "co-working", "co working", "coworking",
                  "virtual office", "meeting room", "conference room",
                  "serviced office", "office premises",
                  "server rack", "colocation", "data centre space",
                  "equipment rental", "machinery hire", "vehicle hire",
                  "cold storage rent", "godown rent",
                  "warehouse lease", "property lease"], False),

        # 194T — Partner payments (NEW FY 2025-26) — BEFORE 194H
        # "partner commission" must not trigger 194H's "commission" keyword
        ("194T", ["partner salary", "partner remuneration",
                  "partner commission", "interest on capital partner",
                  "partner bonus", "partner sitting fees",
                  "partner drawings", "sitting fees",
                  "interest on partner capital", "remuneration to partner",
                  "partner capital", "capital contribution",
                  "interest on capital contribution",
                  "partner interest on capital",
                  "interest on capital"], False),

        # 194H — Commission (third-party agents only)
        ("194H", ["commission", "brokerage", "referral fee",
                  "agency fee", "dealer margin", "channel partner fee",
                  "distribution commission", "franchise fee"], False),

        # 194Q — Goods purchase (check BEFORE 194C/194J — very specific keywords)
        ("194Q", ["procurement", "raw material", "raw materials",
                  "steel", "metal", "aluminium", "copper",
                  "packaging material", "electronic components",
                  "goods purchase", "bulk purchase", "bulk order",
                  "inventory purchase", "merchandise purchase",
                  "industrial equipment purchase", "machine purchase"], False),

        # 194C — Contractors (after 194J_PROFESSIONAL to avoid false positives)
        ("194C", ["catering", "housekeeping", "facility management",
                  "security services", "security guard", "manpower supply",
                  "labour supply", "labour contract", "labor contract",
                  "event management", "event organiser", "event coordination",
                  "event coordinator", "product launch event",
                  "launch event", "awards ceremony",
                  "waste management", "waste disposal", "sanitation services",
                  "cleaning services", "pest control",
                  "transportation services", "courier", "printing",
                  "advertising production",
                  "construction work", "construction contract",
                  "contract staffing", "manpower"], False),

        # 194J Professional — individual/LLP (10%) — check BEFORE technical
        # Key: consultancy/advisory keywords must beat 194C construction keyword
        ("194J_PROFESSIONAL", ["legal", "advocate", "law firm",
                                "audit", "chartered accountant", "ca firm",
                                "medical", "doctor", "architect",
                                "engineering consultancy",
                                "company secretary", "interior design",
                                "management consulting", "management consultancy",
                                "project management consultancy",
                                "structural engineering", "mep consultancy",
                                "advisory", "consulting retainer",
                                "project management", "valuation",
                                "due diligence", "tax consulting",
                                "secretarial", "regulatory",
                                "arbitration", "litigation"], False),

        # 194J Technical — company vendors (2%)
        ("194J_TECHNICAL", ["it support", "technical support", "cloud",
                             "software development", "data processing",
                             "bpo", "call centre", "call center",
                             "network maintenance", "system administration",
                             "amc", "annual maintenance",
                             "managed services", "platform",
                             "saas", "infrastructure services"], True),
    ]

    for section, keywords, requires_company in classification_rules:
        if any(kw in desc for kw in keywords):
            if section == "194J_TECHNICAL":
                actual_rate = 2.0
                actual_section = "194J"
            elif section == "194J_PROFESSIONAL":
                # Company vendors → always technical rate
                if vendor_is_company:
                    actual_rate = 2.0
                else:
                    actual_rate = 10.0
                actual_section = "194J"
            else:
                actual_section = section
                s = TDS_SECTIONS.get(section, {})
                if section == "194C":
                    actual_rate = s.get("rate_company" if vendor_is_company
                                       else "rate_individual", 2.0)
                else:
                    actual_rate = s.get("rate_default", 10.0)

            return {
                "section": actual_section,
                "rate": actual_rate,
                "confidence": "high",
                "note": section_summary(actual_section),
            }

    # Default fallback — 194J professional at 10%
    return {
        "section": "194J",
        "rate": 10.0 if not vendor_is_company else 2.0,
        "confidence": "low",
        "note": "Could not classify with certainty. Defaulting to 194J. Verify manually.",
    }
