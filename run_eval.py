import pandas as pd
import mysql.connector
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import sys

sys.stdout.reconfigure(encoding='utf-8')

def get_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="Root",
        database="fire"
    )

def evaluate_classifier():
    print("\n" + "="*50)
    print(" 🛠️  EVALUATING CLASSIFIER (TYPE PREDICTION)")
    print("="*50)
    try:
        classifier_data = joblib.load('ml/classifier.pkl')
    except Exception as e:
        print(f"Failed to load classifier.pkl: {e}")
        return

    conn = get_db()
    cur = conn.cursor(dictionary=True)
    cur.execute("""
        SELECT type, building_type, priority_level, description
        FROM applications
        WHERE type IS NOT NULL
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()

    y_true = []
    y_pred = []

    for row in rows:
        actual_type = row['type']

        building = row.get('building_type') or 'Unknown'
        known_buildings = list(classifier_data['le_building'].classes_)
        if building not in known_buildings:
            building = known_buildings[0]
        building_enc = classifier_data['le_building'].transform([building])[0]

        priority = row.get('priority_level') or 'Normal'
        known_priorities = list(classifier_data['le_priority'].classes_)
        if priority not in known_priorities:
            priority = known_priorities[0]
        priority_enc = classifier_data['le_priority'].transform([priority])[0]

        base_features = np.array([[building_enc, priority_enc]], dtype=float)

        if classifier_data['has_tfidf'] and classifier_data['tfidf'] is not None:
            desc_vec = classifier_data['tfidf'].transform(
                            [str(row.get('description') or '')]).toarray()
            X_cls_raw = np.hstack([base_features, desc_vec])
        else:
            X_cls_raw = base_features

        X_cls = classifier_data['scaler'].transform(X_cls_raw)
        pred_label = classifier_data['le_type'].inverse_transform(
                            classifier_data['model'].predict(X_cls))[0]

        y_true.append(actual_type)
        y_pred.append(pred_label)

    acc = accuracy_score(y_true, y_pred)
    print(f"\nTotal Applications evaluated: {len(y_true)}")
    print(f"Deployment Accuracy (on all data): {acc*100:.1f}%")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))


def evaluate_risk_scorer():
    print("\n" + "="*50)
    print(" 🛠️  EVALUATING RISK SCORER")
    print("="*50)
    try:
        risk_data = joblib.load('ml/risk_scorer.pkl')
    except Exception as e:
        print(f"Failed to load risk_scorer.pkl: {e}")
        return

    # ── Load held-out test IDs saved during training ───────
    test_app_ids = set(risk_data.get('test_app_ids', []))
    if not test_app_ids:
        print("No test_app_ids found in risk_scorer.pkl.")
        print("Retrain with the updated risk_scorer.py first.")
        return
    print(f"Held-out test set: {len(test_app_ids)} app_ids")

    conn = get_db()
    cur = conn.cursor(dictionary=True)
    cur.execute("""
        SELECT
            a.app_id,
            a.inspection_score,

            COUNT(DISTINCT i.inspection_id)                        AS total_inspections,
            COALESCE(DATEDIFF(NOW(), MAX(i.date)), 999)            AS days_since_inspection,

            COALESCE(AVG(ic.overall_score), 0)                     AS avg_checklist_score,
            COALESCE(SUM(ic.fire_extinguishers  = 0), 0)           AS missing_extinguishers,
            COALESCE(SUM(ic.fire_alarm_system   = 0), 0)           AS missing_alarm,
            COALESCE(SUM(ic.sprinkler_system    = 0), 0)           AS missing_sprinkler,
            COALESCE(SUM(ic.smoke_detectors     = 0), 0)           AS missing_smoke_detector,
            COALESCE(SUM(ic.fire_hose           = 0), 0)           AS missing_fire_hose,
            COALESCE(SUM(ic.emergency_exits     = 0), 0)           AS missing_emergency_exits,
            COALESCE(SUM(ic.emergency_lighting  = 0), 0)           AS missing_emergency_lighting,
            COALESCE(SUM(ic.evacuation_plan     = 0), 0)           AS missing_evacuation_plan,
            COALESCE(SUM(ic.electrical_safety   = 0), 0)           AS missing_electrical_safety,
            COALESCE(SUM(ic.building_structure  = 0), 0)           AS missing_building_structure,
            COALESCE(MAX(
                (ic.fire_extinguishers  = 0) +
                (ic.fire_alarm_system   = 0) +
                (ic.sprinkler_system    = 0) +
                (ic.smoke_detectors     = 0) +
                (ic.fire_hose           = 0) +
                (ic.emergency_exits     = 0) +
                (ic.emergency_lighting  = 0) +
                (ic.evacuation_plan     = 0) +
                (ic.electrical_safety   = 0) +
                (ic.building_structure  = 0)
            ), 0)                                                   AS max_failed_checks,

            COUNT(CASE WHEN fu.status = 'Pending'       THEN 1 END) AS pending_followups,
            COUNT(CASE WHEN fu.completed_at IS NOT NULL THEN 1 END) AS resolved_followups,
            COUNT(fu.follow_up_id)                                   AS total_followups,

            MAX(CASE WHEN n.status = 'Active' THEN 1 ELSE 0 END)   AS noc_active,
            MAX(CASE WHEN n.noc_id IS NULL    THEN 1 ELSE 0 END)    AS no_noc_ever,
            COALESCE(MIN(DATEDIFF(n.validity, NOW())), -1)          AS days_to_noc_expiry,

            COUNT(ah.history_id)                                    AS status_changes

        FROM applications a
        LEFT JOIN inspections i            ON a.app_id = i.app_id
        LEFT JOIN inspection_checklists ic ON a.app_id = ic.app_id
        LEFT JOIN follow_ups fu            ON a.app_id = fu.app_id
        LEFT JOIN nocs n                   ON a.app_id = n.app_id
        LEFT JOIN application_history ah   ON a.app_id = ah.app_id
        WHERE a.inspection_score IS NOT NULL
        GROUP BY a.app_id, a.inspection_score
    """)
    all_rows = cur.fetchall()
    cur.close()
    conn.close()

    # ── Filter to held-out test rows only ──────────────────
    rows = [r for r in all_rows if r['app_id'] in test_app_ids]
    print(f"Rows matched from DB: {len(rows)} / {len(test_app_ids)} expected")

    if not rows:
        print("No matching rows found. Check that app_ids match the DB.")
        return

    y_true = []
    y_pred = []
    features_list = risk_data['features']

    for row in rows:
        score = float(row['inspection_score'])
        if score <= 40:
            actual_label = 'High'
        elif score <= 70:
            actual_label = 'Medium'
        else:
            actual_label = 'Low'

        # All raw values from query
        raw = {
            'total_inspections':          float(row.get('total_inspections')          or 0),
            'days_since_inspection':      float(row.get('days_since_inspection')      or 0),
            'avg_checklist_score':        float(row.get('avg_checklist_score')        or 0),
            'missing_extinguishers':      float(row.get('missing_extinguishers')      or 0),
            'missing_alarm':              float(row.get('missing_alarm')              or 0),
            'missing_sprinkler':          float(row.get('missing_sprinkler')          or 0),
            'missing_smoke_detector':     float(row.get('missing_smoke_detector')     or 0),
            'missing_fire_hose':          float(row.get('missing_fire_hose')          or 0),
            'missing_emergency_exits':    float(row.get('missing_emergency_exits')    or 0),
            'missing_emergency_lighting': float(row.get('missing_emergency_lighting') or 0),
            'missing_evacuation_plan':    float(row.get('missing_evacuation_plan')    or 0),
            'missing_electrical_safety':  float(row.get('missing_electrical_safety')  or 0),
            'missing_building_structure': float(row.get('missing_building_structure') or 0),
            'max_failed_checks':          float(row.get('max_failed_checks')          or 0),
            'pending_followups':          float(row.get('pending_followups')          or 0),
            'resolved_followups':         float(row.get('resolved_followups')         or 0),
            'total_followups':            float(row.get('total_followups')            or 0),
            'noc_active':                 float(row.get('noc_active')                 or 0),
            'no_noc_ever':                float(row.get('no_noc_ever')                or 0),
            'days_to_noc_expiry':         float(row.get('days_to_noc_expiry')         if row.get('days_to_noc_expiry') is not None else -1),
            'status_changes':             float(row.get('status_changes')             or 0),
        }

        # Engineered features — must exactly match risk_scorer.py
        total_fu = raw['total_followups']
        raw['followup_resolution_rate'] = (
            raw['resolved_followups'] / total_fu if total_fu > 0 else 0.0
        )
        raw['total_equipment_failures'] = (
            raw['missing_extinguishers'] + raw['missing_alarm'] +
            raw['missing_sprinkler'] + raw['missing_smoke_detector'] +
            raw['missing_fire_hose'] + raw['missing_emergency_exits'] +
            raw['missing_emergency_lighting'] + raw['missing_evacuation_plan'] +
            raw['missing_electrical_safety'] + raw['missing_building_structure']
        )
        raw['noc_expiry_risk'] = (
            -1 if raw['noc_active'] == 0
            else (1 if raw['days_to_noc_expiry'] < 30 else 0)
        )

        # Use only features the model was trained on, in the same order
        X_risk = np.array([[raw[f] for f in features_list]])
        pred_label = risk_data['model'].predict(X_risk)[0]

        y_true.append(actual_label)
        y_pred.append(pred_label)

    acc = accuracy_score(y_true, y_pred)
    print(f"\nHeld-out Test Accuracy (Risk Labels): {acc*100:.1f}%")
    print("\nRisk Label Classification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))


if __name__ == '__main__':
    evaluate_classifier()
    evaluate_risk_scorer()
    print("\n✅ Evaluation complete.")