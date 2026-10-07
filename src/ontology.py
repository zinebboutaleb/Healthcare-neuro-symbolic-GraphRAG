"""Explicit domain ontology (schema) for the clinical knowledge graph.

The ontology is *executable*: every triple loaded into the graph is validated against SCHEMA,
and the retriever reads its traversal policy (INVERSE_TRAVERSAL) from here.
Human-readable version: ontology/ontology.md
"""

ENTITY_TYPES = {
    "Disease": "A clinical condition with symptoms, risk factors, tests and treatments.",
    "Symptom": "An observable or reported manifestation of a disease.",
    "Medication": "A (fictional) drug used as a treatment.",
    "Treatment": "A non-drug intervention (diet, therapy, programme).",
    "RiskFactor": "A factor that raises the likelihood of a disease.",
    "DiagnosticTest": "A test or examination used to assess a disease or monitor therapy.",
    "Contraindication": "A reified record explaining why a medication is restricted for a condition.",
    "SideEffect": "An adverse effect associated with a medication.",
    "PatientCondition": "A patient-level state that modifies treatment suitability.",
    "ClinicalGuideline": "A (fictional) guideline snippet that links entities to required tests.",
}

# Types the question entity-linker may match against free text.
# Contraindication / ClinicalGuideline names embed other entity names, so they are excluded.
LINKABLE_TYPES = [
    "Disease", "Symptom", "Medication", "Treatment", "RiskFactor",
    "DiagnosticTest", "SideEffect", "PatientCondition",
]

# relation -> allowed (domain, range) pairs
SCHEMA = {
    "HAS_SYMPTOM": [("Disease", "Symptom")],
    "HAS_RISK_FACTOR": [("Disease", "RiskFactor")],
    "TREATED_BY": [("Disease", "Medication"), ("Disease", "Treatment")],
    "CONTRAINDICATED_BY": [("Medication", "PatientCondition"), ("Treatment", "PatientCondition")],
    "HAS_SIDE_EFFECT": [("Medication", "SideEffect")],
    "DIAGNOSED_BY": [("Disease", "DiagnosticTest")],
    "ASSOCIATED_WITH": [
        ("Disease", "Disease"),
        ("ClinicalGuideline", "Disease"),
        ("ClinicalGuideline", "Medication"),
        ("Contraindication", "PatientCondition"),
    ],
    "AFFECTS": [("SideEffect", "PatientCondition"), ("Contraindication", "Medication")],
    "REQUIRES": [("ClinicalGuideline", "DiagnosticTest")],
    "ALTERNATIVE_TO": [("Medication", "Medication")],
}

RELATION_DESCRIPTIONS = {
    "HAS_SYMPTOM": "Disease manifests with a symptom.",
    "HAS_RISK_FACTOR": "Disease is made more likely by a risk factor.",
    "TREATED_BY": "Disease is managed with a medication or treatment.",
    "CONTRAINDICATED_BY": "Medication/treatment is not recommended for a patient condition.",
    "HAS_SIDE_EFFECT": "Medication may cause a side effect.",
    "DIAGNOSED_BY": "Disease is assessed with a diagnostic test.",
    "ASSOCIATED_WITH": "Generic association (comorbidity, guideline scope, contraindication scope).",
    "AFFECTS": "Side effect is of concern for a patient condition / contraindication restricts a medication.",
    "REQUIRES": "Guideline requires a diagnostic test (e.g. monitoring).",
    "ALTERNATIVE_TO": "Medication can substitute another medication.",
}

# By default traversal follows edge direction; the first hop may go either way (so a question
# about a symptom can start at the disease). These relations may also be walked backwards at
# any hop (e.g. Medication <-ASSOCIATED_WITH- Guideline).
INVERSE_TRAVERSAL = {"ASSOCIATED_WITH", "ALTERNATIVE_TO"}


def validate_triple(src_type: str, rel: str, dst_type: str) -> None:
    if rel not in SCHEMA:
        raise ValueError(f"Unknown relation: {rel}")
    if (src_type, dst_type) not in SCHEMA[rel]:
        raise ValueError(f"Ontology violation: ({src_type})-[{rel}]->({dst_type}) is not allowed")
