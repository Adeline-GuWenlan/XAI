# 🧠 Quantum Magic Prediction XAI Project Structure

## 📁 **Active Research Components**

### 🏗️ **CONFIGs/** - *Trained Model Repository*
> **Purpose**: Storage hub for trained transformer models with different architectures and configurations
- **Model Variants**: Attention-based, CLS pooling, MoE (Mixture of Experts) architectures
- **Training Artifacts**: Model weights (.pth), training logs (.csv), hyperparameter configs (.json)
- **Performance Records**: Final metrics and best checkpoint preservation

### 🧬 **Model_Archi/** - *Core Architecture Definitions*
> **Purpose**: Foundational neural network components for quantum magic prediction
- **Transformer Components**: Embedding layers, encoder stacks, attention mechanisms
- **MLP Variants**: Physics-aware, standard, and ensemble prediction heads
- **Dataset Handling**: Quantum density matrix preprocessing and tokenization
- **Utility Functions**: Hermitian matrix operations, upper triangular extraction

### 🔬 **Probing/** - *Internal Representation Analysis*
> **Purpose**: Mechanistic interpretability of transformer learned features

#### 🎯 **Get_Instance_Feature/** - *Individual Sample Analysis*
- **Function**: Extract quantum physical properties from specific density matrices
- **Output**: Per-instance feature vectors for correlation analysis

#### 🧠 **Get_Layer_Represen/** - *Layer-wise Representation Extraction*
- **Function**: Hook-based intermediate activation capture across transformer layers
- **GPU Optimization**: Efficient memory management for large-scale probing
- **Attention Probing**: Specialized analysis for attention mechanism patterns

#### 🎲 **Start_Matching/** - *Feature-Prediction Correlation Discovery*
- **Function**: Probe each layer's ability to predict quantum physical quantities
- **Analysis**: Identify which layers best encode specific quantum properties
- **Visualization**: Heatmaps showing layer-property correlation strengths

### 🗄️ **Rawdata/** - *Quantum Dataset Storage*
> **Purpose**: Preprocessed quantum density matrices and magic monotone labels
- **Input Matrices**: 2-qubit density matrices (4×4 complex) in separate real/imaginary components
- **Target Labels**: Quantum magic values for supervised learning
- **Multiple Scales**: 50K and 200K sample datasets for different experimental needs

### 📊 **MLP_MoE/** - *Mixture of Experts Implementation*
> **Purpose**: Advanced multi-expert prediction architecture with competitive routing
- **Competitive Training**: Expert specialization through competitive loss functions
- **Diagnostic Tools**: MoE-specific analysis and performance monitoring

### 🧪 **experiments/** - *Experimental Results Archive*
> **Purpose**: Timestamped experimental runs with extracted features and analysis results
- **Layer Features**: CSV exports of intermediate representations
- **Experiment Tracking**: Organized by timestamp and model configuration

## 📦 **Utility & Support Files**

### 📋 **Standalone Scripts**
- `peek_archi.py` - *Model architecture inspection tool*
- `test_analysis.py` - *Analysis pipeline validation*
- `test_quantum_dataset.py` - *Dataset integrity verification*
- `model_loader.py` - *Universal model loading utility*
- `HPC_command.sh` - *High-performance computing execution scripts*
- `conmand.sh` - *Command automation for common tasks*

### 📤 **upload/** - *External Deployment Package*
> **Purpose**: Clean, portable version of core components for external systems
- **Streamlined Architecture**: Essential model files without experimental artifacts
- **Documentation**: HPC deployment instructions and requirements

## 🗂️ **temp_unused/** - *Temporarily Inactive Components*
> **Purpose**: Archive of analysis tools not currently in active development
- **Attention Analysis**: Tools for transformer attention pattern visualization
- **Neural Analysis**: Neuron-level activation pattern investigation  
- **Saliency Analysis**: Gradient-based input attribution methods
- **General Visualization**: Multi-purpose plotting and analysis utilities
- **Previous XAI Results**: Historical interpretability analysis outputs

---

## 🔍 **Current Research Focus**

The project is currently centered on **mechanistic interpretability** through the **Probing/** pipeline, investigating:
- Which transformer layers learn specific quantum physical properties
- How internal representations correlate with magic monotone predictions  
- Whether the model develops quantum physics-aligned feature hierarchies

The **temp_unused/** directory contains previously developed XAI tools that may be reactivated for future analysis phases, while the active components focus on representation probing and layer-wise analysis of quantum feature emergence.