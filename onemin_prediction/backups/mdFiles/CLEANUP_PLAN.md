# CLEANUP PLAN - Remove Unnecessary Files

**Date:** December 7, 2025  
**Purpose:** Organize workspace by removing duplicate/outdated documentation and shell scripts

---

## FILES TO REMOVE (Duplicates & Outdated)

### 📄 Documentation Files (Duplicates/Summaries)

These are summaries or duplicates of essential documentation:

```
REMOVE:
  ❌ 00_START_HERE.txt
     └─ Duplicate: Use README_NEW_DOCUMENTATION.md instead (cleaner, newer)
  
  ❌ README_CODE_REVIEW.md
     └─ Duplicate: Content merged into CODE_REVIEW_REPORT.md
  
  ❌ REVIEW_INDEX.md
     └─ Duplicate: Use DOCUMENTATION_INDEX.md instead (comprehensive)
  
  ❌ REVIEW_SUMMARY.md
     └─ Duplicate: Use CODE_REVIEW_REPORT.md summary
  
  ❌ README_NEW_DOCUMENTATION.md
     └─ Optional: Quick index only, use DOCUMENTATION_INDEX.md if keeping docs minimal
  
  ❌ PRIORITY_2_ENHANCEMENTS.md
     └─ Future work: Keep only if planning Phase 2 improvements
  
  ❌ DEPLOYMENT_CHECKLIST.md
     └─ Duplicate: Covered in STARTUP_CHECKLIST.md
  
  ❌ IMPLEMENTATION_COMPLETE.md
     └─ Status file: Outdated, not needed for operations
  
  ❌ PROJECT_COMPLETION_SUMMARY.md
     └─ Status file: Summary info in DELIVERY_SUMMARY.md
  
  ❌ FINAL_VERIFICATION.md
     └─ Status file: Content in TRAINED_MODELS_VERIFICATION.md
```

**Total: 10 documentation files** (~80 KB to remove)

---

### 🔧 Shell Scripts (Utilities/Demos)

These are helper/demo scripts not needed for production:

```
REMOVE:
  ❌ auto_phase_a_daily.sh
     └─ Legacy automation script (use systemd instead - see STARTUP_CHECKLIST.md)
  
  ❌ automation_with_sidecar.sh
     └─ Legacy with sidecar (not needed, run_main.py handles it)
  
  ❌ eod_daily.sh
     └─ Legacy end-of-day script (archive data manually if needed)
  
  ❌ todo.txt
     └─ Personal todo file (not part of production)
```

**Total: 4 shell scripts** (~10 KB to remove)

---

### 📁 Directories to Check/Clean

```
CHECK:
  ⚠️ __pycache__/
     └─ Auto-generated Python cache (safe to remove, recreates on run)
     └─ Remove: Yes
  
  ⚠️ old/
     └─ Contains old versions of files (safe to archive or remove)
     └─ Remove: Consider archiving to backup/ instead
  
  ✅ data/
     └─ Needed: Contains intraday_cache for caching
     └─ Keep: Yes
  
  ✅ logs/
     └─ Needed: Runtime logs
     └─ Keep: Yes
  
  ✅ trained_models/
     └─ Needed: Production models
     └─ Keep: Yes
```

---

## FILES TO KEEP (Essential)

### ✅ Python Scripts (Production Core)

```
KEEP:
  ✅ run_main.py                    [Entry point - CRITICAL]
  ✅ main_event_loop.py             [Core orchestration - CRITICAL]
  ✅ core_handler.py                [WebSocket handler - CRITICAL]
  ✅ feature_pipeline.py            [Feature engineering - CRITICAL]
  ✅ model_pipeline.py              [Model inference - CRITICAL]
  ✅ logging_setup.py               [Logging config - NEEDED]
  ✅ calibrator.py                  [Calibration engine - NEEDED]
  ✅ online_trainer.py              [Live training - NEEDED]
  ✅ futures_vwap_cvd_sidecar.py   [Sidecar features - OPTIONAL but present]
  ✅ offline_train_2min.py          [Model training - NEEDED]
  ✅ offline_eval_2min_full.py      [Evaluation - NEEDED]
  ✅ offline_eval.py                [Evaluation variant - OPTIONAL]
  ✅ offline_leakage_sanity_2min.py [Data validation - OPTIONAL]
  ✅ offline_train_q_model_2min.py  [Q-model training - OPTIONAL]
```

**Total: 14 scripts** - All production/training related

---

### ✅ Documentation (Kept)

**Essential (Operations):**
```
KEEP:
  ✅ COMPLETE_OPERATIONAL_SUMMARY.md      [Executive summary]
  ✅ STARTUP_CHECKLIST.md                 [Daily operations]
  ✅ OPERATIONS_AND_TRAINING_GUIDE.md     [How-to guide]
  ✅ README.md                            [Project overview]
```

**Reference (Verification):**
```
KEEP:
  ✅ CODE_REVIEW_REPORT.md                [Code quality analysis]
  ✅ PRIORITY_1_FIXES.md                  [Applied fixes]
  ✅ PRIORITY_1_FIXES_STATUS.md           [Fix verification]
  ✅ TRAINED_MODELS_VERIFICATION.md       [Training capability]
  ✅ CAPABILITY_VERIFICATION.md           [Feature verification]
  ✅ USER_GUIDE.md                        [How to trade]
  ✅ AUTOMATION_EXPLAINED.md              [Architecture]
  ✅ DOCUMENTATION_INDEX.md               [Navigation]
  ✅ DELIVERY_SUMMARY.md                  [What delivered]
```

**Total: 13 documentation files** - All operational or reference

---

## RECOMMENDED CLEANUP (OPTION 1 - Aggressive)

**Remove everything not in KEEP list**

```bash
# Documentation to remove (10 files)
rm -f 00_START_HERE.txt
rm -f README_CODE_REVIEW.md
rm -f REVIEW_INDEX.md
rm -f REVIEW_SUMMARY.md
rm -f README_NEW_DOCUMENTATION.md
rm -f PRIORITY_2_ENHANCEMENTS.md
rm -f DEPLOYMENT_CHECKLIST.md
rm -f IMPLEMENTATION_COMPLETE.md
rm -f PROJECT_COMPLETION_SUMMARY.md
rm -f FINAL_VERIFICATION.md

# Shell scripts to remove (4 files)
rm -f auto_phase_a_daily.sh
rm -f automation_with_sidecar.sh
rm -f eod_daily.sh
rm -f todo.txt

# Cache to remove (auto-regenerates)
rm -rf __pycache__

# Result: 25 files removed, ~90 KB freed
```

**Result: Clean, minimal workspace with only production essentials**

---

## RECOMMENDED CLEANUP (OPTION 2 - Conservative)

**Keep commonly referenced docs, remove only clear duplicates**

```bash
# Only remove clear duplicates/legacy (5 files)
rm -f 00_START_HERE.txt              # Duplicate
rm -f REVIEW_SUMMARY.md              # Duplicate summary
rm -f auto_phase_a_daily.sh          # Legacy shell
rm -f automation_with_sidecar.sh     # Legacy shell
rm -f todo.txt                       # Personal

# Archive for later reference
mkdir -p backups/old_docs
mv PRIORITY_2_ENHANCEMENTS.md backups/old_docs/
mv IMPLEMENTATION_COMPLETE.md backups/old_docs/
mv PROJECT_COMPLETION_SUMMARY.md backups/old_docs/
mv README_CODE_REVIEW.md backups/old_docs/

# Result: Core kept + enhanced docs archived
```

**Result: Clean workspace, historical docs preserved**

---

## FILE SPACE IMPACT

```
Before Cleanup:
  Documentation:  ~240 KB (22 markdown files)
  Scripts:        ~100 KB (14 Python + 4 Shell)
  __pycache__:    ~50 KB
  Other:          ~50 KB
  ──────────────────────
  TOTAL:          ~440 KB

After (Option 1 - Aggressive):
  Documentation:  ~150 KB (13 essential files)
  Scripts:        ~100 KB (14 Python only)
  Other:          ~50 KB
  ──────────────────────
  TOTAL:          ~300 KB
  
  SAVED: ~140 KB (32% reduction)

After (Option 2 - Conservative):
  Documentation:  ~180 KB (13 active + archived docs)
  Scripts:        ~100 KB (14 Python only)
  Other:          ~50 KB
  ──────────────────────
  TOTAL:          ~330 KB
  
  SAVED: ~110 KB (25% reduction)
```

---

## MY RECOMMENDATION: OPTION 2 (Conservative Cleanup)

**Why:**
1. ✅ Keeps all operational essentials
2. ✅ Preserves reference documentation
3. ✅ Archives historical docs (can reference later)
4. ✅ Removes legacy shell scripts (use systemd instead)
5. ✅ Cleans up obvious clutter (todo.txt, duplicate summaries)
6. ✅ Saves space without losing information

**What to keep actively:**
```
Active Documentation (Use Daily):
├── COMPLETE_OPERATIONAL_SUMMARY.md
├── STARTUP_CHECKLIST.md
├── OPERATIONS_AND_TRAINING_GUIDE.md
├── DOCUMENTATION_INDEX.md
└── README.md

Reference Documentation (Lookup as needed):
├── CODE_REVIEW_REPORT.md
├── PRIORITY_1_FIXES.md
├── TRAINED_MODELS_VERIFICATION.md
├── CAPABILITY_VERIFICATION.md
├── USER_GUIDE.md
└── AUTOMATION_EXPLAINED.md
```

**What to archive:**
```
Historical (Reference, not needed daily):
├── IMPLEMENTATION_COMPLETE.md
├── PROJECT_COMPLETION_SUMMARY.md
├── PRIORITY_2_ENHANCEMENTS.md
├── README_CODE_REVIEW.md
└── REVIEW_SUMMARY.md
```

**What to remove:**
```
Legacy/Duplicates (Not needed):
├── 00_START_HERE.txt (use DOCUMENTATION_INDEX.md)
├── auto_phase_a_daily.sh (use systemd)
├── automation_with_sidecar.sh (use run_main.py)
├── eod_daily.sh (manual archive)
└── todo.txt (personal file)
```

---

## CLEAN UP COMMAND (Option 2 - Recommended)

```bash
#!/bin/bash
# Cleanup script - removes unnecessary files

cd /home/hanumanth/Documents/sunflower-group_2/onemin_prediction

# Create backup directory
mkdir -p backups/old_docs

# Archive historical documentation
mv IMPLEMENTATION_COMPLETE.md backups/old_docs/
mv PROJECT_COMPLETION_SUMMARY.md backups/old_docs/
mv PRIORITY_2_ENHANCEMENTS.md backups/old_docs/
mv README_CODE_REVIEW.md backups/old_docs/
mv REVIEW_SUMMARY.md backups/old_docs/

# Remove duplicates
rm -f 00_START_HERE.txt
rm -f REVIEW_INDEX.md
rm -f README_NEW_DOCUMENTATION.md
rm -f DEPLOYMENT_CHECKLIST.md
rm -f FINAL_VERIFICATION.md

# Remove legacy shell scripts
rm -f auto_phase_a_daily.sh
rm -f automation_with_sidecar.sh
rm -f eod_daily.sh

# Remove personal files
rm -f todo.txt

# Clean cache
rm -rf __pycache__

echo "✅ Cleanup complete!"
echo "Removed: 14 files (~80 KB)"
echo "Archived: 5 files (~30 KB)"
echo "Space saved: ~110 KB"
echo "Workspace: Clean and organized"
```

---

## FINAL RESULT (After Cleanup)

```
/onemin_prediction/
├── 📄 README.md                              [Quick start]
├── 📄 COMPLETE_OPERATIONAL_SUMMARY.md       [Executive summary]
├── 📄 STARTUP_CHECKLIST.md                  [Daily operations]
├── 📄 OPERATIONS_AND_TRAINING_GUIDE.md      [How-to guide]
├── 📄 DOCUMENTATION_INDEX.md                [Navigation]
├── 📄 CODE_REVIEW_REPORT.md                 [Code quality]
├── 📄 PRIORITY_1_FIXES.md                   [Applied fixes]
├── 📄 PRIORITY_1_FIXES_STATUS.md            [Fix verification]
├── 📄 TRAINED_MODELS_VERIFICATION.md        [Training ready]
├── 📄 CAPABILITY_VERIFICATION.md            [Features verified]
├── 📄 USER_GUIDE.md                         [How to trade]
├── 📄 AUTOMATION_EXPLAINED.md               [Architecture]
├── 📄 DELIVERY_SUMMARY.md                   [What delivered]
│
├── 🐍 run_main.py                           [Entry point]
├── 🐍 main_event_loop.py                    [Core logic]
├── 🐍 core_handler.py                       [WebSocket]
├── 🐍 feature_pipeline.py                   [Features]
├── 🐍 model_pipeline.py                     [Inference]
├── 🐍 logging_setup.py                      [Logging]
├── 🐍 calibrator.py                         [Calibration]
├── 🐍 online_trainer.py                     [Live training]
├── 🐍 offline_train_2min.py                 [Training]
├── 🐍 offline_eval_2min_full.py            [Evaluation]
├── 🐍 offline_eval.py                       [Eval variant]
├── 🐍 offline_leakage_sanity_2min.py       [Validation]
├── 🐍 offline_train_q_model_2min.py        [Q-training]
├── 🐍 futures_vwap_cvd_sidecar.py          [Sidecar]
│
├── 📁 trained_models/                       [Models]
├── 📁 data/                                 [Data cache]
├── 📁 logs/                                 [Runtime logs]
├── 📁 backups/old_docs/                    [Archived docs]
│
└── ✅ CLEAN & ORGANIZED
```

**Status: Production-ready workspace, easy to navigate**

---

## HOW TO EXECUTE

**Option A: Manual (One file at a time)**
```bash
cd /home/hanumanth/Documents/sunflower-group_2/onemin_prediction
rm -f 00_START_HERE.txt
# ... etc
```

**Option B: Script (All at once)**
```bash
# Save as cleanup.sh
chmod +x cleanup.sh
./cleanup.sh
```

**Option C: Selective (Keep specific files)**
- Pick and choose which files to remove

---

## WHICH OPTION DO YOU PREFER?

1. **Option 1 (Aggressive):** Remove 25 files, save 140 KB, minimal workspace
2. **Option 2 (Conservative):** Remove 14 files, save 110 KB, keep historical reference
3. **Option 3 (Custom):** Tell me which specific files to remove

Let me know and I'll execute the cleanup!

