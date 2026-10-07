"""Generate the SYNTHETIC clinical knowledge base (entities, relationships, documents, benchmark).

Everything here is fictional teaching data. Medication names are invented; facts are NOT medically
authoritative. Run:  python -m src.data_generation
"""
from __future__ import annotations

import csv
from pathlib import Path

from src.config import DATA_DIR, DOC_DIR
from src.ontology import ENTITY_TYPES, validate_triple

# --------------------------------------------------------------------------------------
# Entities: (name, aliases, description fragment)
# --------------------------------------------------------------------------------------
DISEASES = [
    ("Hypertension", ["high blood pressure"], "persistently elevated blood pressure"),
    ("Type 2 Diabetes", ["diabetes", "t2d"], "impaired regulation of blood glucose"),
    ("Asthma", [], "recurrent, reversible narrowing of the airways"),
    ("Migraine", [], "recurrent headache attacks with sensory sensitivity"),
    ("Chronic Kidney Disease", ["ckd"], "gradual loss of kidney filtering function"),
    ("Heart Failure", [], "reduced ability of the heart to pump effectively"),
    ("Hypothyroidism", [], "an underactive thyroid gland"),
    ("Major Depression", ["depression"], "a persistent depressive episode"),
    ("Osteoarthritis", [], "degeneration of joint cartilage"),
    ("GERD", ["acid reflux"], "recurrent reflux of stomach contents into the oesophagus"),
]
SYMPTOMS = [
    ("Headache", [], "pain felt in the head"),
    ("Dizziness", ["dizzy"], "a sensation of unsteadiness"),
    ("Blurred Vision", ["blurry vision"], "reduced sharpness of sight"),
    ("Chest Tightness", [], "a feeling of pressure in the chest"),
    ("Increased Thirst", ["excessive thirst"], "unusually strong thirst"),
    ("Frequent Urination", [], "needing to urinate more often than usual"),
    ("Fatigue", ["tiredness"], "persistent tiredness"),
    ("Wheezing", [], "a whistling sound when breathing"),
    ("Shortness of Breath", ["breathlessness", "short of breath"], "difficulty breathing"),
    ("Persistent Cough", [], "a cough lasting for weeks"),
    ("Nausea", [], "a feeling of sickness"),
    ("Light Sensitivity", ["photophobia"], "discomfort in bright light"),
    ("Swelling in Legs", ["leg swelling", "swollen legs"], "fluid accumulation in the legs"),
    ("Palpitations", [], "awareness of the heartbeat"),
    ("Weight Gain", [], "unexplained increase in body weight"),
    ("Cold Intolerance", [], "unusual sensitivity to cold"),
    ("Low Mood", [], "a persistently sad mood"),
    ("Loss of Interest", [], "reduced interest in usual activities"),
    ("Joint Pain", [], "pain in one or more joints"),
    ("Joint Stiffness", [], "reduced ease of joint movement"),
    ("Heartburn", [], "burning discomfort behind the breastbone"),
]
RISK_FACTORS = [
    ("Obesity", [], "excess body weight"),
    ("High Sodium Intake", ["salt intake"], "a diet high in salt"),
    ("Sedentary Lifestyle", [], "low physical activity"),
    ("Smoking", [], "tobacco use"),
    ("Family History", [], "relatives with the condition"),
    ("Advanced Age", ["older age"], "older age"),
    ("Chronic Stress", [], "long-term psychological stress"),
    ("Allergen Exposure", [], "repeated contact with allergens"),
    ("High Alcohol Intake", [], "heavy alcohol consumption"),
    ("Poor Sleep", [], "insufficient or disrupted sleep"),
]
TESTS = [
    ("Blood Pressure Measurement", [], "repeated measurement of arterial blood pressure"),
    ("HbA1c Test", ["hba1c"], "a blood test reflecting average glucose control"),
    ("Spirometry", [], "a breathing test measuring airflow"),
    ("Neurological Examination", [], "a structured examination of nervous-system function"),
    ("Serum Creatinine Test", [], "a blood test of kidney function"),
    ("Echocardiogram", [], "an ultrasound examination of the heart"),
    ("TSH Blood Test", ["tsh test"], "a blood test of thyroid-stimulating hormone"),
    ("Depression Screening Questionnaire", [], "a structured questionnaire on mood"),
    ("Joint X-Ray", [], "imaging of a joint"),
    ("Upper Endoscopy", [], "direct visual inspection of the upper digestive tract"),
]
SIDE_EFFECTS = [
    ("Dry Cough", [], "a persistent tickling, non-productive cough"),
    ("Hyperkalemia", [], "elevated potassium in the blood"),
    ("Lightheadedness", [], "feeling faint on standing"),
    ("Hypoglycemia", [], "abnormally low blood glucose"),
    ("Tremor", [], "involuntary shaking"),
    ("Drowsiness", [], "excessive sleepiness"),
    ("Stomach Irritation", [], "irritation of the stomach lining"),
    ("Fluid Retention", [], "accumulation of body fluid"),
    ("Sleep Disturbance", [], "disrupted sleep"),
    ("Dehydration", [], "loss of body water"),
]
PATIENT_CONDITIONS = [
    ("Pregnancy", ["pregnant"], "a pregnant patient"),
    ("Severe Renal Impairment", ["severe kidney impairment", "renal impairment"], "markedly reduced kidney function"),
    ("Low Blood Pressure", ["hypotension"], "abnormally low blood pressure"),
    ("Liver Impairment", [], "reduced liver function"),
    ("Peptic Ulcer History", ["peptic ulcer"], "a history of stomach or duodenal ulcers"),
    ("Heart Rhythm Disorder", ["arrhythmia"], "an abnormal heart rhythm"),
    ("Glaucoma", [], "raised pressure inside the eye"),
    ("Elderly Frailty", ["frail", "frailty"], "an elderly, frail patient"),
    ("Seizure Disorder", ["epilepsy"], "a tendency to seizures"),
    ("Recurrent Hypoglycemia History", [], "repeated episodes of low blood glucose in the past"),
]
MEDICATIONS = [
    ("Cardivex", [], "a fictional ACE-inhibitor-like agent that relaxes blood vessels"),
    ("Diurex", [], "a fictional diuretic-like agent that increases fluid excretion"),
    ("Glucorin", [], "a fictional biguanide-like agent that reduces glucose production"),
    ("Sulfaglide", [], "a fictional insulin-secretagogue-like agent that stimulates insulin release"),
    ("Respirol", [], "a fictional inhaled bronchodilator-like agent that opens the airways"),
    ("Neurotrip", [], "a fictional triptan-like agent that constricts cranial blood vessels"),
    ("Renoprotect", [], "a fictional agent that lowers pressure inside kidney filtering units"),
    ("Thyrozen", [], "a fictional synthetic thyroid-hormone-like agent"),
    ("Serenomax", [], "a fictional SSRI-like agent that raises serotonin availability"),
    ("Arthrocalm", [], "a fictional NSAID-like anti-inflammatory agent"),
    ("Gastrazole", [], "a fictional proton-pump-inhibitor-like agent that reduces stomach acid"),
]
TREATMENTS = [
    ("Low-Sodium Diet", [], "a dietary plan restricting salt"),
    ("Structured Exercise Program", [], "supervised, graded physical activity"),
    ("Cognitive Behavioral Therapy", ["cbt"], "a structured talking therapy"),
    ("Physical Therapy", [], "guided exercises for movement and strength"),
    ("Weight Management Program", [], "a supervised programme for gradual weight reduction"),
    ("Trigger Avoidance Plan", [], "identifying and avoiding personal triggers"),
]

# --------------------------------------------------------------------------------------
# Relationships
# --------------------------------------------------------------------------------------
HAS_SYMPTOM = {
    "Hypertension": ["Headache", "Dizziness", "Blurred Vision", "Chest Tightness"],
    "Type 2 Diabetes": ["Increased Thirst", "Frequent Urination", "Fatigue", "Blurred Vision"],
    "Asthma": ["Wheezing", "Shortness of Breath", "Chest Tightness", "Persistent Cough"],
    "Migraine": ["Headache", "Nausea", "Light Sensitivity", "Dizziness"],
    "Chronic Kidney Disease": ["Fatigue", "Swelling in Legs", "Nausea", "Frequent Urination"],
    "Heart Failure": ["Shortness of Breath", "Swelling in Legs", "Fatigue", "Palpitations"],
    "Hypothyroidism": ["Fatigue", "Weight Gain", "Cold Intolerance", "Low Mood"],
    "Major Depression": ["Low Mood", "Loss of Interest", "Fatigue"],
    "Osteoarthritis": ["Joint Pain", "Joint Stiffness"],
    "GERD": ["Heartburn", "Persistent Cough", "Nausea", "Chest Tightness"],
}
HAS_RISK_FACTOR = {
    "Hypertension": ["Obesity", "High Sodium Intake", "Sedentary Lifestyle"],
    "Type 2 Diabetes": ["Obesity", "Sedentary Lifestyle", "Family History"],
    "Asthma": ["Smoking", "Allergen Exposure", "Family History"],
    "Migraine": ["Chronic Stress", "Poor Sleep", "Family History"],
    "Chronic Kidney Disease": ["Advanced Age", "Smoking", "High Sodium Intake"],
    "Heart Failure": ["Advanced Age", "High Alcohol Intake", "Sedentary Lifestyle"],
    "Hypothyroidism": ["Advanced Age", "Family History"],
    "Major Depression": ["Chronic Stress", "Poor Sleep", "High Alcohol Intake"],
    "Osteoarthritis": ["Obesity", "Advanced Age"],
    "GERD": ["Obesity", "Smoking", "High Alcohol Intake"],
}
TREATED_BY = {
    "Hypertension": ["Cardivex", "Diurex", "Low-Sodium Diet"],
    "Type 2 Diabetes": ["Glucorin", "Sulfaglide", "Weight Management Program"],
    "Asthma": ["Respirol", "Trigger Avoidance Plan"],
    "Migraine": ["Neurotrip", "Trigger Avoidance Plan"],
    "Chronic Kidney Disease": ["Renoprotect", "Low-Sodium Diet"],
    "Heart Failure": ["Cardivex", "Diurex", "Structured Exercise Program"],
    "Hypothyroidism": ["Thyrozen"],
    "Major Depression": ["Serenomax", "Cognitive Behavioral Therapy"],
    "Osteoarthritis": ["Arthrocalm", "Physical Therapy", "Weight Management Program"],
    "GERD": ["Gastrazole", "Weight Management Program"],
}
DIAGNOSED_BY = {
    "Hypertension": ["Blood Pressure Measurement"],
    "Type 2 Diabetes": ["HbA1c Test"],
    "Asthma": ["Spirometry"],
    "Migraine": ["Neurological Examination"],
    "Chronic Kidney Disease": ["Serum Creatinine Test"],
    "Heart Failure": ["Echocardiogram"],
    "Hypothyroidism": ["TSH Blood Test"],
    "Major Depression": ["Depression Screening Questionnaire"],
    "Osteoarthritis": ["Joint X-Ray"],
    "GERD": ["Upper Endoscopy"],
}
HAS_SIDE_EFFECT = {
    "Cardivex": ["Dry Cough", "Hyperkalemia", "Lightheadedness"],
    "Diurex": ["Dehydration", "Lightheadedness"],
    "Glucorin": ["Stomach Irritation"],
    "Sulfaglide": ["Hypoglycemia", "Fluid Retention"],
    "Respirol": ["Tremor"],
    "Neurotrip": ["Drowsiness", "Lightheadedness"],
    "Renoprotect": ["Hyperkalemia"],
    "Thyrozen": ["Sleep Disturbance"],
    "Serenomax": ["Drowsiness", "Sleep Disturbance"],
    "Arthrocalm": ["Stomach Irritation", "Fluid Retention"],
    "Gastrazole": ["Drowsiness"],
}
# side effect -> patient conditions for which it is of particular concern
SE_AFFECTS = {
    "Hyperkalemia": ["Severe Renal Impairment"],
    "Lightheadedness": ["Elderly Frailty", "Low Blood Pressure"],
    "Dehydration": ["Severe Renal Impairment"],
    "Hypoglycemia": ["Elderly Frailty"],
    "Fluid Retention": ["Severe Renal Impairment"],
    "Stomach Irritation": ["Peptic Ulcer History"],
    "Drowsiness": ["Elderly Frailty"],
    "Tremor": ["Heart Rhythm Disorder"],
    "Sleep Disturbance": ["Seizure Disorder"],
}
# (medication, patient condition, severity, rationale)  -> reified Contraindication nodes
CONTRAINDICATIONS = [
    ("Cardivex", "Pregnancy", "severe", "ACE-inhibitor-like agents are marked as not recommended during pregnancy in this synthetic knowledge base."),
    ("Cardivex", "Severe Renal Impairment", "moderate", "The synthetic entry flags reduced kidney clearance and a risk of potassium build-up."),
    ("Diurex", "Low Blood Pressure", "severe", "Fluid-lowering agents may further reduce blood pressure."),
    ("Diurex", "Elderly Frailty", "moderate", "Fluid loss may raise the risk of falls in frail patients."),
    ("Glucorin", "Severe Renal Impairment", "severe", "The synthetic entry flags impaired clearance of the agent."),
    ("Glucorin", "Liver Impairment", "moderate", "Reduced liver function may impair handling of the agent."),
    ("Sulfaglide", "Recurrent Hypoglycemia History", "severe", "The agent stimulates insulin release and may trigger further low-glucose episodes."),
    ("Sulfaglide", "Elderly Frailty", "moderate", "Low-glucose episodes are harder to detect and manage in frail patients."),
    ("Respirol", "Heart Rhythm Disorder", "moderate", "Bronchodilator-like agents may increase heart rate."),
    ("Neurotrip", "Pregnancy", "severe", "The synthetic entry marks this agent as not recommended during pregnancy."),
    ("Serenomax", "Seizure Disorder", "moderate", "The synthetic entry flags a lowered seizure threshold."),
    ("Serenomax", "Glaucoma", "moderate", "The synthetic entry flags a risk of raised eye pressure."),
    ("Arthrocalm", "Peptic Ulcer History", "severe", "Anti-inflammatory agents may irritate the stomach lining and reopen ulcers."),
]
ALTERNATIVE_TO = [("Diurex", "Cardivex"), ("Sulfaglide", "Glucorin")]
ASSOCIATED_DISEASES = [
    ("Hypertension", "Chronic Kidney Disease"),
    ("Hypertension", "Heart Failure"),
    ("Type 2 Diabetes", "Chronic Kidney Disease"),
    ("Major Depression", "Migraine"),
    ("GERD", "Asthma"),
]
# (guideline name, text, associated entity (Disease|Medication), required test)
GUIDELINES = [
    ("Hypertension Confirmation Guideline", "Confirm elevated readings on repeated measurements before starting long-term therapy.", "Hypertension", "Blood Pressure Measurement"),
    ("ACE-Inhibitor-Like Renal Monitoring Guideline", "Check kidney function before and after starting Cardivex.", "Cardivex", "Serum Creatinine Test"),
    ("Diabetes Monitoring Guideline", "Track long-term glucose control at regular intervals.", "Type 2 Diabetes", "HbA1c Test"),
    ("Insulin-Secretagogue Review Guideline", "Review glucose control when Sulfaglide is started or its dose is changed.", "Sulfaglide", "HbA1c Test"),
    ("Asthma Lung Function Guideline", "Document baseline airflow limitation when asthma is suspected.", "Asthma", "Spirometry"),
    ("Heart Failure Imaging Guideline", "Assess cardiac structure and function when heart failure is suspected.", "Heart Failure", "Echocardiogram"),
    ("CKD Staging Guideline", "Stage kidney disease using repeated kidney function measurements.", "Chronic Kidney Disease", "Serum Creatinine Test"),
    ("Thyroid Replacement Monitoring Guideline", "Monitor thyroid hormone levels after starting Thyrozen.", "Thyrozen", "TSH Blood Test"),
    ("Depression Screening Guideline", "Use a structured questionnaire to screen for depressive symptoms.", "Major Depression", "Depression Screening Questionnaire"),
    ("Gastric Safety Guideline", "Consider endoscopic review in patients on Arthrocalm who develop persistent upper-abdominal symptoms.", "Arthrocalm", "Upper Endoscopy"),
    ("Diuretic Hydration Guideline", "Check kidney function and hydration status when Diurex is used.", "Diurex", "Serum Creatinine Test"),
    ("Migraine Evaluation Guideline", "Perform a neurological examination to exclude other causes of recurrent headache.", "Migraine", "Neurological Examination"),
]

# --------------------------------------------------------------------------------------
# Benchmark: id, question, gold_hops, expected answer entities, required evidence entities, note
# --------------------------------------------------------------------------------------
BENCHMARK = [
    ("Q01", "What symptoms are associated with Asthma?", 1,
     "Wheezing|Shortness of Breath|Chest Tightness|Persistent Cough",
     "Asthma|Wheezing|Shortness of Breath|Chest Tightness|Persistent Cough", "Asthma-HAS_SYMPTOM->Symptom"),
    ("Q02", "Which medications and treatments are used for Type 2 Diabetes?", 1,
     "Glucorin|Sulfaglide|Weight Management Program",
     "Type 2 Diabetes|Glucorin|Sulfaglide|Weight Management Program", "Disease-TREATED_BY->Medication/Treatment"),
    ("Q03", "Which diagnostic test is used for Hypothyroidism?", 1,
     "TSH Blood Test", "Hypothyroidism|TSH Blood Test", "Disease-DIAGNOSED_BY->Test"),
    ("Q04", "What risk factors are associated with Migraine?", 1,
     "Chronic Stress|Poor Sleep|Family History",
     "Migraine|Chronic Stress|Poor Sleep|Family History", "Disease-HAS_RISK_FACTOR->RiskFactor"),
    ("Q05", "Which patient conditions are contraindications for Respirol?", 1,
     "Heart Rhythm Disorder", "Respirol|Heart Rhythm Disorder", "Medication-CONTRAINDICATED_BY->PatientCondition"),
    ("Q06", "Which treatment for Hypertension is contraindicated in a patient with Low Blood Pressure?", 2,
     "Diurex", "Hypertension|Diurex|Low Blood Pressure",
     "Hypertension-TREATED_BY->Diurex-CONTRAINDICATED_BY->Low Blood Pressure"),
    ("Q07", "A pregnant patient has Hypertension. Which medication should not be recommended?", 2,
     "Cardivex", "Hypertension|Cardivex|Pregnancy",
     "Hypertension-TREATED_BY->Cardivex-CONTRAINDICATED_BY->Pregnancy (Neurotrip is also contraindicated in Pregnancy but does not treat Hypertension)"),
    ("Q08", "Which medication for Hypertension has Dry Cough as a side effect and is contraindicated in Pregnancy?", 2,
     "Cardivex", "Hypertension|Cardivex|Dry Cough|Pregnancy",
     "Intersection of two constraints on the medications reachable from Hypertension"),
    ("Q09", "Starting from Type 2 Diabetes, which medication can cause Hypoglycemia, and which patient conditions make it inappropriate?", 2,
     "Sulfaglide|Recurrent Hypoglycemia History|Elderly Frailty",
     "Type 2 Diabetes|Sulfaglide|Hypoglycemia|Recurrent Hypoglycemia History|Elderly Frailty",
     "T2D-TREATED_BY->Sulfaglide-{HAS_SIDE_EFFECT->Hypoglycemia, CONTRAINDICATED_BY->PatientCondition}"),
    ("Q10", "Which disease is a candidate explanation for a patient reporting Shortness of Breath and Swelling in Legs, and which test is used to diagnose it?", 2,
     "Heart Failure|Echocardiogram", "Shortness of Breath|Swelling in Legs|Heart Failure|Echocardiogram",
     "Symptoms<-HAS_SYMPTOM-Disease-DIAGNOSED_BY->Test (symbolic rule R02)"),
    ("Q11", "Starting from Osteoarthritis, identify the treatment with Stomach Irritation as a side effect and the patient condition that this side effect affects.", 3,
     "Arthrocalm|Peptic Ulcer History", "Osteoarthritis|Arthrocalm|Stomach Irritation|Peptic Ulcer History",
     "Osteoarthritis-TREATED_BY->Arthrocalm-HAS_SIDE_EFFECT->Stomach Irritation-AFFECTS->Peptic Ulcer History"),
    ("Q12", "Which diagnostic test do clinical guidelines require for monitoring the medications used for Hypertension?", 3,
     "Serum Creatinine Test", "Hypertension|Cardivex|Diurex|Serum Creatinine Test",
     "Hypertension-TREATED_BY->Medication<-ASSOCIATED_WITH-Guideline-REQUIRES->Test (symbolic rule R05)"),
    ("Q13", "A patient with Severe Renal Impairment has Heart Failure. Which treatment should not be recommended, and which alternative needs additional review?", 3,
     "Cardivex|Diurex", "Heart Failure|Cardivex|Severe Renal Impairment|Diurex|Dehydration",
     "R01 blocks Cardivex; R04 proposes alternative Diurex; R03 flags Diurex (Dehydration AFFECTS Severe Renal Impairment)"),
    ("Q14", "A patient reports Increased Thirst and Frequent Urination. For the candidate disease, which medication has a side effect that affects Elderly Frailty?", 4,
     "Type 2 Diabetes|Sulfaglide|Hypoglycemia",
     "Increased Thirst|Frequent Urination|Type 2 Diabetes|Sulfaglide|Hypoglycemia|Elderly Frailty",
     "Symptom<-Disease-TREATED_BY->Medication-HAS_SIDE_EFFECT->SideEffect-AFFECTS->PatientCondition (4 edges)"),
]

ID_PREFIX = {
    "Disease": "DIS", "Symptom": "SYM", "Medication": "MED", "Treatment": "TRT", "RiskFactor": "RF",
    "DiagnosticTest": "TST", "SideEffect": "SE", "PatientCondition": "PC",
    "Contraindication": "CI", "ClinicalGuideline": "GL",
}


def join(items: list[str]) -> str:
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def build_entities() -> list[dict]:
    groups = {
        "Disease": DISEASES, "Symptom": SYMPTOMS, "RiskFactor": RISK_FACTORS, "DiagnosticTest": TESTS,
        "SideEffect": SIDE_EFFECTS, "PatientCondition": PATIENT_CONDITIONS,
        "Medication": MEDICATIONS, "Treatment": TREATMENTS,
    }
    rows = []
    for type_, items in groups.items():
        for i, (name, aliases, desc) in enumerate(items, 1):
            description = f"{name}: {desc}." if type_ == "Disease" else desc[0].upper() + desc[1:] + "."
            rows.append(dict(id=f"{ID_PREFIX[type_]}{i:02d}", name=name, type=type_, description=description, aliases="|".join(aliases)))
    for i, (med, pc, sev, why) in enumerate(CONTRAINDICATIONS, 1):
        rows.append(dict(id=f"CI{i:02d}", name=f"{med} contraindicated in {pc}", type="Contraindication",
                         description=f"[{sev}] {why}", aliases=""))
    for i, (name, text, _, _) in enumerate(GUIDELINES, 1):
        rows.append(dict(id=f"GL{i:02d}", name=name, type="ClinicalGuideline", description=text, aliases=""))
    names = [r["name"] for r in rows]
    assert len(names) == len(set(names)), "entity names must be globally unique"
    return rows


def build_relationships(entities: list[dict]) -> list[dict]:
    type_of = {e["name"]: e["type"] for e in entities}
    rows: list[dict] = []

    def add(src, rel, dst, note=""):
        validate_triple(type_of[src], rel, type_of[dst])  # ontology check
        rows.append(dict(source=src, source_type=type_of[src], relation=rel, target=dst, target_type=type_of[dst], note=note))

    for table, rel in [(HAS_SYMPTOM, "HAS_SYMPTOM"), (HAS_RISK_FACTOR, "HAS_RISK_FACTOR"), (TREATED_BY, "TREATED_BY"),
                       (DIAGNOSED_BY, "DIAGNOSED_BY"), (HAS_SIDE_EFFECT, "HAS_SIDE_EFFECT")]:
        for src, dsts in table.items():
            for dst in dsts:
                add(src, rel, dst)
    for se, pcs in SE_AFFECTS.items():
        for pc in pcs:
            add(se, "AFFECTS", pc)
    for med, pc, sev, why in CONTRAINDICATIONS:
        ci = f"{med} contraindicated in {pc}"
        add(med, "CONTRAINDICATED_BY", pc, note=f"[{sev}] {why}")
        add(ci, "AFFECTS", med)
        add(ci, "ASSOCIATED_WITH", pc)
    for a, b in ALTERNATIVE_TO:
        add(a, "ALTERNATIVE_TO", b)
    for a, b in ASSOCIATED_DISEASES:
        add(a, "ASSOCIATED_WITH", b)
    for name, _, target, test in GUIDELINES:
        add(name, "ASSOCIATED_WITH", target)
        add(name, "REQUIRES", test)
    return rows


def build_documents() -> list[dict]:
    """Templated synthetic documents. Facts are deliberately spread over several documents:
    disease pages list symptoms/risk factors/tests/treatments; medication monographs hold the safety
    information; guideline bulletins hold monitoring requirements."""
    docs = []
    assoc = {}
    for a, b in ASSOCIATED_DISEASES:
        assoc.setdefault(a, []).append(b)
        assoc.setdefault(b, []).append(a)
    for i, (name, _, frag) in enumerate(DISEASES, 1):
        text = (
            f"{name} is a fictional teaching entry describing {frag}. "
            f"Reported symptoms include {join(HAS_SYMPTOM[name])}. "
            f"Listed risk factors are {join(HAS_RISK_FACTOR[name])}. "
            f"The condition is assessed using {join(DIAGNOSED_BY[name])}. "
            f"Management options listed in this knowledge base are {join(TREATED_BY[name])}."
        )
        if name in assoc:
            text += f" It is listed as associated with {join(assoc[name])}."
        docs.append(dict(doc_id=f"DOC-DIS-{i:02d}", title=f"Overview of {name}", doc_type="disease_overview", text=text))
    for i, (name, _, frag) in enumerate(MEDICATIONS, 1):
        sentences = [f"{name} is {frag}.", f"Reported side effects in this knowledge base include {join(HAS_SIDE_EFFECT[name])}."]
        for se in HAS_SIDE_EFFECT[name]:
            if se in SE_AFFECTS:
                sentences.append(f"{se} is of particular concern for patients with {join(SE_AFFECTS[se])}.")
        cis = [c for c in CONTRAINDICATIONS if c[0] == name]
        if cis:
            sentences.append(f"{name} is marked as contraindicated for patients with {join([c[1] for c in cis])}.")
            sentences += [f"Contraindication note for {c[1]}: {c[3]}" for c in cis]
        for a, b in ALTERNATIVE_TO:
            if a == name:
                sentences.append(f"{name} is listed as an alternative to {b}.")
            if b == name:
                sentences.append(f"{a} is listed as an alternative to {name}.")
        docs.append(dict(doc_id=f"DOC-MED-{i:02d}", title=f"Monograph: {name}", doc_type="medication_monograph", text=" ".join(sentences)))
    for k in range(0, len(GUIDELINES), 2):
        pair = GUIDELINES[k:k + 2]
        text = " ".join(f"Guideline '{n}': {t} It is associated with {tgt} and requires the {test}." for n, t, tgt, test in pair)
        docs.append(dict(doc_id=f"DOC-GL-{k // 2 + 1:02d}", title=f"Guideline bulletin {k // 2 + 1}", doc_type="guideline_bulletin", text=text))
    return docs


def write_all(data_dir: Path = DATA_DIR) -> dict:
    data_dir.mkdir(parents=True, exist_ok=True)
    doc_dir = data_dir / "documents"
    doc_dir.mkdir(exist_ok=True)
    for old in doc_dir.glob("*.txt"):
        old.unlink()
    entities = build_entities()
    rels = build_relationships(entities)
    docs = build_documents()
    with open(data_dir / "entities.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "name", "type", "description", "aliases"])
        w.writeheader(); w.writerows(entities)
    with open(data_dir / "relationships.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["source", "source_type", "relation", "target", "target_type", "note"])
        w.writeheader(); w.writerows(rels)
    for d in docs:
        (doc_dir / f"{d['doc_id']}.txt").write_text(
            f"doc_id: {d['doc_id']}\ntitle: {d['title']}\ndoc_type: {d['doc_type']}\n---\n{d['text']}\n", encoding="utf-8")
    with open(data_dir / "benchmark_questions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "question", "hop_type", "gold_hops", "expected_answer", "required_evidence", "reasoning_note"])
        for qid, q, hops, ans, ev, note in BENCHMARK:
            w.writerow([qid, q, "single" if hops == 1 else "multi", hops, ans, ev, note])
    counts = {t: sum(1 for e in entities if e["type"] == t) for t in ENTITY_TYPES}
    return dict(entities=len(entities), relationships=len(rels), documents=len(docs), questions=len(BENCHMARK), by_type=counts)


if __name__ == "__main__":
    stats = write_all()
    print("Synthetic KB written to", DATA_DIR)
    for k, v in stats.items():
        print(f"  {k}: {v}")
