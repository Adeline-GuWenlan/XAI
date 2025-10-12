# 🗂️ XAI Project Organization Guide

**Last Updated**: 2025-10-08
**Purpose**: Clean organization of git branches, local folders, and project responsibilities

---

## 📌 Git Branch Structure & Responsibilities

### Branch Overview
```
master (main)          → Production/stable baseline with project structure
├── DataGeneration     → Data generation utilities & RoM files
└── 4qubits           → 4-qubit MLP training experiments & higher dimensions
```

### Branch Details

#### 1. **`master`** - Production Baseline
- **Purpose**: Stable project structure and documentation
- **Key Content**:
  - `PROJECT_STRUCTURE.md` - Overall architecture documentation
  - Original CONFIGs/ folder with 2-qubit and 3-qubit trained models
  - Core Model_Archi/ components
  - Probing/ analysis pipeline
  - Rawdata/ with original 2-qubit datasets
- **Status**: ✅ Stable reference point
- **Use When**: Need to reference original project structure or 2-3 qubit experiments

#### 2. **`DataGeneration`** - Data Generation & Utilities
- **Purpose**: Dataset generation tools and RoM (Range of Magic) file processing
- **Key Content**:
  - `Utils/larger_generation.py` - Generate 3-5 qubit datasets with Poisson distribution
  - `Utils/RoM_File/` - Range of Magic data generation notebooks and functions
  - `Utils/Data/3q_100000_states.npy` and `3q_100000_labels.npy` - Generated 3-qubit datasets
  - Grouped_Dataset/ removed (moved to dedicated storage)
- **Key Commits**:
  - `80e9c8d` - RoM data generation from quiz
  - `8e1c9d5` - Larger system data generation
- **Status**: ✅ Active for data generation tasks
- **Use When**:
  - Generating new quantum state datasets
  - Working with RoM label generation
  - Creating 3+ qubit training data

#### 3. **`4qubits`** - 4-Qubit Training & High-Dimensional Experiments
- **Purpose**: MLP training for 4-qubit systems with higher embedding dimensions (2048, 4096)
- **Key Content**:
  - `MixMLP_4q/` - Dedicated 4-qubit training scripts
    - `train_single.py` - Single GPU training
    - `train_parallel.py` - Parallel multi-GPU training
    - `mlp.py` - MLP architecture (d_model=2048/4096)
    - `4qlog.md` - Training logs and results
  - `MixMLP_4q/Decoded_Tokens/` - Preprocessed 4-qubit data
    - `4q_50000_states_real.npy`, `4q_50000_states_imag.npy`
    - `4q_50000_states_labels.npy`
  - `Dataset/` - Generated 4-qubit datasets (50K samples)
- **Key Commits**:
  - `5e666ca` - Parallel running for 2048 dimensions
  - `c6b4f57` - 4qubits with dim_feedforward=4096
- **Status**: ⚠️ Active development (uncommitted changes)
- **Uncommitted Changes**:
  - Updated `HPC_command.sh` for HPC cluster execution
  - Modified `train_parallel.py` and `train_single.py` for better GPU handling
  - Updated `4qlog.md` with latest training results
  - Deleted old mixed-format decoded tokens (consolidated to single format)
- **Use When**:
  - Training 4-qubit magic prediction models
  - Experimenting with higher dimensions (2048+)
  - Scaling to larger quantum systems

---

## 📁 Local Folder Organization & Responsibilities

### Current Status: 🔴 **NEEDS CLEANUP**

### Recommended Folder Structure

#### **Category 1: Core Training & Models** 🎯

##### `MixMLP_4q/` - 4-Qubit Experiments (KEEP)
- **Purpose**: 4-qubit system training
- **Status**: Active on `4qubits` branch
- **Contains**: Training scripts, logs, HPC commands, decoded tokens

##### `MixMLP_Adjusted/` - 2-3 Qubit Baseline (KEEP)
- **Purpose**: 2-3 qubit baseline experiments
- **Status**: Reference implementation
- **Contains**: Original training scripts for smaller systems
- **Note**: Some overlap with MixMLP_4q - consider merging common utilities

**📋 ACTION**:
- Merge common utilities between MixMLP_4q and MixMLP_Adjusted
- Keep architecture-specific configs separate

---

#### **Category 2: Data Generation & Storage** 💾

##### `Utils/` - Data Generation Utilities (KEEP & ORGANIZE)
- **Purpose**: Centralized data generation and processing tools
- **Subcategories**:
  - `Utils/Data/` - Generated datasets
  - `Utils/RoM_File/` - Range of Magic generation (notebooks & functions)
  - `Utils/File/` - Misc utilities
- **Status**: Active on `DataGeneration` branch

##### `Dataset/` - Generated Datasets (KEEP)
- **Purpose**: Processed quantum state datasets ready for training
- **Current**: 4q_50000_states.npy, 4q_50000_labels.npy
- **Note**: Should be branch-specific or use naming conventions

**📋 ACTION**:
- Standardize dataset naming: `{n}q_{samples}_{variant}_states.npy`
- Move large datasets (>100MB) to external storage or .gitignore
- Create Dataset/README.md documenting each dataset

---

#### **Category 3: Analysis & Interpretability** 🔬

##### `Probing/` - Mechanistic Interpretability (KEEP)
- **Purpose**: Layer-wise representation analysis
- **Subdirs**:
  - `Get_Instance_Feature/` - Extract quantum properties per sample
  - `Get_Layer_Represen/` - Hook-based activation extraction
  - `Start_Matching/` - Feature-prediction correlation
  - `probing_analysis_result/` - Saved analysis outputs
- **Status**: ✅ Stable on master

**📋 ACTION**: No changes needed - well organized

---

#### **Category 4: Support Files** 🛠️

##### Root-level Scripts (ORGANIZE)
Current scattered files:
- `analyze_labels.py` ✅ Keep - useful utility
- `check_matrix_layout.py` ✅ Keep - debugging tool
- `count_model_params.py` ✅ Keep - model analysis
- `visualize_labels.py` ✅ Keep - visualization
- `peek_archi.py` ✅ Keep - architecture inspection
- `peek_data.py` ❓ Check if duplicate with Dataset tools
- `HPC_command.sh` ❓ Duplicate with MixMLP_4q version
- `conmand.sh` ❓ Check purpose
- `prompt.md` ❓ Documentation - move to docs/
- `temp_unused.zip` ❌ DELETE or unzip to temp_unused/

**📋 ACTION**:
- Create `scripts/` folder for utilities
- Create `docs/` folder for documentation
- Remove or archive redundant files

---

#### **Category 5: Archives & Cleanup** 🗑️

##### Files to Handle:

**DELETE:**
- `temp_unused.zip` - Already have temp_unused/ directory elsewhere

**MOVE to .gitignore:**
- All `.DS_Store` files (macOS artifacts)
- `__pycache__/` directories

**ARCHIVE (if needed):**
- Old CONFIGs/ from master branch (consider compressing old experiments)

---

## 🎯 Recommended Actions by Priority

### Priority 1: Immediate Cleanup (Do Now)
1. ✅ Commit uncommitted changes on `4qubits` branch
2. ✅ Delete temp_unused.zip
3. ✅ Add .DS_Store to .gitignore
4. ✅ Create scripts/ and docs/ folders

### Priority 2: Organization (This Week)
1. 📁 Merge common utilities from MixMLP_4q and MixMLP_Adjusted
2. 📝 Create Dataset/README.md documenting all datasets
3. 🗂️ Move root-level scripts to scripts/
4. 📚 Move markdown docs to docs/

### Priority 3: Future Improvements
1. 🏗️ Consider creating a unified training interface that works across n-qubits
2. 📊 Standardize experiment logging format
3. 🔄 Set up data versioning for large datasets

---

## 📝 Commit Strategy for Current Uncommitted Changes

### On `4qubits` branch:

**Should Commit:**
```
✅ MixMLP_4q/4qlog.md          - Training logs for 2048 dim experiments
✅ MixMLP_4q/HPC_command.sh     - Updated HPC cluster commands
✅ MixMLP_4q/mlp.py             - Architecture updates for higher dims
✅ MixMLP_4q/train_parallel.py  - Parallel training improvements
✅ MixMLP_4q/train_single.py    - Single GPU training updates
✅ analyze_labels.py            - Label analysis utility update
```

**Should Delete (Already Done):**
```
❌ Old decoded token files (mixed_1 format) - replaced with cleaner format
❌ Utils/RoM_File/label_gen.py - moved to appropriate location
```

**Recommended Commit Message:**
```
Update 4-qubit training for 2048 dim experiments

- Add parallel GPU support for higher dimensional models
- Update HPC cluster commands for new architecture
- Consolidate decoded token format (removed old mixed_1 files)
- Update training logs with 2048 dim results
- Improve GPU memory management in training scripts

🤖 Generated with Claude Code
Co-Authored-By: Claude <noreply@anthropic.com>
```

---

## 🔄 Branch Workflow

### For New Experiments:
1. **New qubit dimension?** → Create branch from `4qubits` (e.g., `5qubits`)
2. **New data generation?** → Work on `DataGeneration` branch
3. **New analysis method?** → Branch from `master`

### Merging Strategy:
- `DataGeneration` → `master` (when datasets are stable)
- `4qubits` → `master` (when results are publication-ready)
- Keep experimental branches separate until validated

---

## 📊 Quick Reference: What Lives Where

| Content Type | Primary Branch | Local Folder |
|-------------|---------------|--------------|
| 2-3 qubit models | `master` | MixMLP_Adjusted/ |
| 4+ qubit models | `4qubits` | MixMLP_4q/ |
| Data generation | `DataGeneration` | Utils/ |
| Datasets | Any | Dataset/ |
| Analysis tools | `master` | Probing/ |
| Documentation | `master` | docs/ (to create) |
| Utilities | Any | scripts/ (to create) |

---

## 🚀 Next Steps

1. **Review and commit** current changes on `4qubits` branch
2. **Clean up** root directory by creating scripts/ and docs/
3. **Document** each dataset in Dataset/README.md
4. **Standardize** experiment naming and logging
5. **Archive** old experiments that are no longer needed

---

*This organization guide should be updated whenever new branches or major folder changes are made.*
