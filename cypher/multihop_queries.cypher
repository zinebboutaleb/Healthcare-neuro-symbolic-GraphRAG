// Q1 (2 hops): treatments of Hypertension and the patient conditions that contraindicate them
MATCH (d:Disease {name: 'Hypertension'})-[:TREATED_BY]->(m:Medication)-[:CONTRAINDICATED_BY]->(pc:PatientCondition)
RETURN d.name AS disease, m.name AS medication, pc.name AS patient_condition
ORDER BY medication, patient_condition;

// Q2 (3 hops): Osteoarthritis -> treatment -> side effect -> patient condition affected by that side effect
MATCH (d:Disease {name: 'Osteoarthritis'})-[:TREATED_BY]->(m:Medication)-[:HAS_SIDE_EFFECT]->(se:SideEffect)-[:AFFECTS]->(pc:PatientCondition)
RETURN d.name AS disease, m.name AS medication, se.name AS side_effect, pc.name AS affected_condition;

// Q3 (aggregation): candidate diseases for a symptom set, ranked by number of matching symptoms
MATCH (d:Disease)-[:HAS_SYMPTOM]->(s:Symptom)
WHERE s.name IN ['Shortness of Breath', 'Swelling in Legs']
RETURN d.name AS candidate_disease, count(s) AS matched_symptoms, collect(s.name) AS symptoms
ORDER BY matched_symptoms DESC, candidate_disease;

// Q4 (3 hops, mixed direction): monitoring tests required by guidelines for medications that treat Hypertension
MATCH (d:Disease {name: 'Hypertension'})-[:TREATED_BY]->(m:Medication)<-[:ASSOCIATED_WITH]-(g:ClinicalGuideline)-[:REQUIRES]->(t:DiagnosticTest)
RETURN d.name AS disease, m.name AS medication, g.name AS guideline, t.name AS required_test
ORDER BY medication;

// Q5 (variable length): everything reachable from Heart Failure through clinical edges, up to 3 hops
MATCH p = (d:Disease {name: 'Heart Failure'})-[:TREATED_BY|CONTRAINDICATED_BY|HAS_SIDE_EFFECT|AFFECTS*1..3]->(x)
RETURN [n IN nodes(p) | n.name] AS path
ORDER BY size(nodes(p)) DESC, path
LIMIT 15;

// Q6 (negation as failure): alternatives to medications blocked by Severe Renal Impairment that are not themselves blocked
MATCH (m:Medication)-[:CONTRAINDICATED_BY]->(:PatientCondition {name: 'Severe Renal Impairment'})
MATCH (alt:Medication)-[:ALTERNATIVE_TO]-(m)
WHERE NOT (alt)-[:CONTRAINDICATED_BY]->(:PatientCondition {name: 'Severe Renal Impairment'})
RETURN m.name AS blocked_medication, alt.name AS alternative;
