import pandas as pd
import mysql.connector
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import numpy as np

# ── DB Connection ──────────────────────────────────────────
def get_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="Root",
        database="fire"
    )

# ── Load Data ──────────────────────────────────────────────
conn = get_db()
cursor = conn.cursor(dictionary=True)
cursor.execute("""
    SELECT
        a.app_id,
        a.inspection_score,

        -- Inspection features
        COUNT(DISTINCT i.inspection_id)                       AS total_inspections,
        COALESCE(DATEDIFF(NOW(), MAX(i.date)), 999)           AS days_since_inspection,

        -- Checklist features (independent safety signal)
        COALESCE(AVG(ic.overall_score), 0)                    AS avg_checklist_score,
        COALESCE(SUM(ic.fire_extinguishers = 0), 0)           AS missing_extinguishers,
        COALESCE(SUM(ic.fire_alarm_system  = 0), 0)           AS missing_alarm,
        COALESCE(SUM(ic.sprinkler_system   = 0), 0)           AS missing_sprinkler,
        COALESCE(SUM(ic.smoke_detectors    = 0), 0)           AS missing_smoke_detector,
        COALESCE(SUM(ic.fire_hose          = 0), 0)           AS missing_fire_hose,
        COALESCE(SUM(ic.emergency_exits    = 0), 0)           AS missing_emergency_exits,
        COALESCE(SUM(ic.emergency_lighting = 0), 0)           AS missing_emergency_lighting,
        COALESCE(SUM(ic.evacuation_plan    = 0), 0)           AS missing_evacuation_plan,
        COALESCE(SUM(ic.electrical_safety  = 0), 0)           AS missing_electrical_safety,
        COALESCE(SUM(ic.building_structure = 0), 0)           AS missing_building_structure,
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
        ), 0)                                                  AS max_failed_checks,

        -- Follow-up features
        COUNT(CASE WHEN fu.status = 'Pending'  THEN 1 END)    AS pending_followups,
        COUNT(CASE WHEN fu.completed_at IS NOT NULL THEN 1 END) AS resolved_followups,
        COUNT(fu.follow_up_id)                                 AS total_followups,

        -- NOC features
        MAX(CASE WHEN n.status = 'Active'  THEN 1 ELSE 0 END) AS noc_active,
        MAX(CASE WHEN n.noc_id IS NULL     THEN 1 ELSE 0 END) AS no_noc_ever,
        COALESCE(MIN(DATEDIFF(n.validity, NOW())), -1)         AS days_to_noc_expiry,

        -- History features
        COUNT(ah.history_id)                                   AS status_changes

    FROM applications a
    LEFT JOIN inspections i            ON a.app_id = i.app_id
    LEFT JOIN inspection_checklists ic ON a.app_id = ic.app_id
    LEFT JOIN follow_ups fu            ON a.app_id = fu.app_id
    LEFT JOIN nocs n                   ON a.app_id = n.app_id
    LEFT JOIN application_history ah   ON a.app_id = ah.app_id
    WHERE a.inspection_score IS NOT NULL
    GROUP BY a.app_id, a.inspection_score
""")
df = pd.DataFrame(cursor.fetchall()).fillna(0)
cursor.close()
conn.close()

print(f"\nTotal rows fetched: {len(df)}")

if len(df) == 0:
    print("No data found. Make sure inspection_score is filled in applications table.")
    exit()

# ── Create Risk Label ──────────────────────────────────────
df['inspection_score'] = pd.to_numeric(
    df['inspection_score'], errors='coerce').fillna(50)

df['risk_label'] = pd.cut(
    df['inspection_score'],
    bins=[0, 40, 70, 100],
    labels=['High', 'Medium', 'Low'],
    include_lowest=True
)
df = df.dropna(subset=['risk_label'])

print(f"Rows after label assignment: {len(df)}")
print("\nClass distribution:")
print(df['risk_label'].value_counts())
print(df['risk_label'].value_counts(normalize=True).mul(100).round(1).astype(str) + '%')

# ── Engineer Features ──────────────────────────────────────
# Ratio of resolved to total follow-ups (0 if none exist)
df['followup_resolution_rate'] = np.where(
    df['total_followups'] > 0,
    df['resolved_followups'] / df['total_followups'],
    0.0
)

# Total missing equipment across all checklists
df['total_equipment_failures'] = (
    df['missing_extinguishers'] +
    df['missing_alarm'] +
    df['missing_sprinkler'] +
    df['missing_smoke_detector'] +
    df['missing_fire_hose'] +
    df['missing_emergency_exits'] +
    df['missing_emergency_lighting'] +
    df['missing_evacuation_plan'] +
    df['missing_electrical_safety'] +
    df['missing_building_structure']
)

# NOC expiry risk: -1 = no active NOC, 1 = expiring within 30 days, 0 = fine
df['noc_expiry_risk'] = np.where(
    df['noc_active'] == 0, -1,
    np.where(df['days_to_noc_expiry'] < 30, 1, 0)
)

print("\nEngineered features: followup_resolution_rate, total_equipment_failures, noc_expiry_risk")

# ── Features and Target ────────────────────────────────────
features = [
    # Inspection
    'total_inspections',
    'days_since_inspection',
    'avg_checklist_score',

    # Equipment safety (independent signal from recency)
    'max_failed_checks',
    'total_equipment_failures',
    'missing_sprinkler',
    'missing_alarm',
    'missing_fire_hose',
    'missing_emergency_exits',
    'missing_evacuation_plan',
    'missing_electrical_safety',

    # Follow-ups
    'pending_followups',
    'followup_resolution_rate',

    # NOC
    'noc_active',
    'no_noc_ever',
    'noc_expiry_risk',

    # History
    'status_changes',
]

# Only keep features that actually exist in the dataframe
features = [f for f in features if f in df.columns]
print(f"\nFeatures used ({len(features)}): {features}")

X = df[features].values.astype(float)
y = df['risk_label']

min_class = int(df['risk_label'].value_counts().min())

# ── Train Test Split ───────────────────────────────────────
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
except ValueError:
    print("Warning: Stratify failed. Using random split.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

# Record held-out test IDs for honest evaluation
test_indices = y_test.index.tolist()
test_app_ids = df.loc[test_indices, 'app_id'].tolist()

print(f"\nTraining samples : {len(X_train)}")
print(f"Testing samples  : {len(X_test)}")

# ── Compare Models ─────────────────────────────────────────
cv_folds = min(5, min_class)
skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

candidates = {
    'GradientBoosting': GradientBoostingClassifier(
        n_estimators=100,
        max_depth=3,
        min_samples_leaf=3,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    ),
    'RandomForest (balanced)': RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        min_samples_leaf=3,
        class_weight='balanced',   # helps High risk not get ignored
        random_state=42,
    ),
}

print(f"\n── {cv_folds}-fold Cross-Validation ──────────────────────")
best_name, best_score, best_model = None, -1, None
for name, clf in candidates.items():
    scores = cross_val_score(clf, X, y, cv=skf, scoring='f1_macro')
    print(f"  {name:<35} macro-F1 = {scores.mean():.3f} ± {scores.std():.3f}")
    if scores.mean() > best_score:
        best_score, best_name, best_model = scores.mean(), name, clf

print(f"\n✅ Best: {best_name}  (macro-F1 = {best_score:.3f})")

# ── Train and Evaluate ─────────────────────────────────────
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

print("\n" + "="*50)
print("  MODEL EVALUATION — Risk Scorer (improved)")
print("="*50)
print(f"\nTest Accuracy : {accuracy_score(y_test, y_pred)*100:.1f}%")

cv_acc = cross_val_score(best_model, X, y, cv=skf, scoring='accuracy')
print(f"Cross-val ({cv_folds}-fold acc): {cv_acc.mean()*100:.1f}% ± {cv_acc.std()*100:.1f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("Confusion Matrix:")
labels_order = ['High', 'Medium', 'Low']
cm = confusion_matrix(y_test, y_pred, labels=labels_order)
print(pd.DataFrame(cm, index=labels_order, columns=labels_order))

# ── Feature Importance ─────────────────────────────────────
print("\nFeature Importances:")
imp = pd.Series(best_model.feature_importances_, index=features).sort_values(ascending=False)
for feat, score in imp.items():
    bar = '█' * int(score * 50)
    print(f"  {feat:<30} {bar} {score:.3f}")

# ── Save ───────────────────────────────────────────────────
joblib.dump({
    'model':        best_model,
    'features':     features,
    'test_app_ids': test_app_ids,
}, 'ml/risk_scorer.pkl')
print("\nSaved ml/risk_scorer.pkl")
print("Done.")