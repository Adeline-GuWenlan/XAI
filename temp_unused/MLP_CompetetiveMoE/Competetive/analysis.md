
## 1. How to check if competitive experts specialize

Since you switched from **collaborative** (soft sharing, smoother gating) to **competitive** (harder routing, experts specializing), you want to see if each expert develops its own “strengths.” Some established diagnostics:

* **Task- or Feature-conditioned Utilization**

  * Group your samples by some physical property (purity, entanglement, Bloch vector norms, CHSH bound, etc. from your probe set).
  * Show how often each expert is selected for each group.
  * If experts specialize, you’ll see Expert A dominate “low-entanglement,” Expert B on “high-entropy,” etc.

* **Performance per Expert**

  * Temporarily evaluate each expert alone (disable gating, route all data through one expert).
  * Compare their MSE/MAE across subsets. Diverging curves suggest specialization.

* **t-SNE / UMAP of Routed Inputs**

  * Embed the CLS or hidden states, color points by which expert they were routed to.
  * Distinct clusters imply each expert is carving its own domain.

---

## 2. How to visualize collaborative vs. competitive

The difference is mostly about **routing entropy and overlap**:

* **Entropy of Routing Weights**

  * Collaborative: high entropy → samples are shared by multiple experts.
  * Competitive: lower entropy → one expert dominates.
  * You can plot histograms of per-sample entropy across both modes.

* **Expert Utilization Balance**

  * Collaborative: all experts fire softly, utilization looks smooth and overlapping.
  * Competitive: more skewed (but ideally balanced across the dataset overall).
  * A bar chart of average utilization per expert is standard.

* **Routing Heatmap Over Time**

  * Plot routing distribution over training epochs.
  * Competitive training should show sharper assignment (probabilities moving toward 0 or 1) compared to collaborative.

* **Loss Curves & NLL Comparison**

  * Competitive tends to have sharper specialization but may pay in stability (so showing training/validation loss with annotations helps).

---

## 3. Established practices in MoE research

You can borrow directly from literature (Google’s Switch Transformer, GShard, and follow-ups):

* **Routing statistics tables**: mean entropy, expert utilization variance, load balancing metrics.
* **Expert specialization plots**: frequency of assignment vs. task/feature labels.
* **Visualization of gate logits**: to show whether the competitive objective enforces hard routing.
* **Comparative training dynamics**: loss curves, utilization entropy, accuracy/mse side-by-side.

These are common in MoE papers because the central question is always: *did experts actually specialize, or are they redundant?*

# For competetive one:
📊 Loaded 200000 labels
📊 Found 1 file pairs
Loading 4096 samples for MoE analysis...
Loaded 4096 quantum samples for MoE analysis
Processing 4096 samples in batches of 1024...

================= MoE Diagnostics (Frozen) =================
Device: mps | N=4096 | Experts E=3
Overall mixture MSE: 0.142247

-- Routing Entropy --
Mean H_route: 1.0888  (normalized: 0.9911)
Std: 0.0002
Quantiles: {0: '1.0885', 10: '1.0886', 25: '1.0887', 50: '1.0888', 75: '1.0890', 90: '1.0892', 100: '1.0898'}

-- Utilization (Top-1 and Soft) --
H_util_top1: -0.0000  (norm: -0.0000)  N_eff_top1: 1.00  CV_top1: 1.414
H_util_soft: 1.0888  (norm: 0.9911)  N_eff_soft: 2.97  CV_soft: 0.138
u_top1 (share): [1.0000, 0.0000, 0.0000]
u_soft (mean π): [0.3802, 0.3490, 0.2708]

-- Contribution per Expert (avg |π_j * f_j|) --
[0.0876, 0.2208, 0.1371]

-- Per-Expert MSE on Top-1 Assigned Samples --
[0.189227, nan, nan]

# For collaborationn one:
(base) guwenlan@guwenlans-MacBook-Pro XAI % /opt/anaconda3/bin/python /Users/guwenlan/Desktop
/XAI/MLP_MoE/moe_diagnostics.py
Loading model with config: mixture_of_experts
🔧 Model: In Encoder: Msk=False, and Msk Threshold=1            AND In Transformation: Pooling=cls            AND MLP=mixture_of_experts
/Users/guwenlan/Desktop/XAI/MLP_MoE/moe_diagnostics.py:185: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  state_dict = torch.load(checkpoint, map_location=device)
📊 Loaded 200000 labels
📊 Found 1 file pairs
Loading 4096 samples for MoE analysis...
Loaded 4096 quantum samples for MoE analysis
Processing 4096 samples in batches of 1024...

================= MoE Diagnostics (Frozen) =================
Device: mps | N=4096 | Experts E=3
Overall mixture MSE: 0.014279

-- Routing Entropy --
Mean H_route: 0.8285  (normalized: 0.7542)
Std: 0.0491
Quantiles: {0: '0.5651', 10: '0.7619', 25: '0.8072', 50: '0.8378', 75: '0.8604', 90: '0.8809', 100: '0.9586'}

-- Utilization (Top-1 and Soft) --
H_util_top1: 0.6302  (norm: 0.5736)  N_eff_top1: 1.88  CV_top1: 0.828
H_util_soft: 0.8628  (norm: 0.7853)  N_eff_soft: 2.37  CV_soft: 0.611
u_top1 (share): [0.3245, 0.0000, 0.6755]
u_soft (mean π): [0.3986, 0.0578, 0.5436]

-- Contribution per Expert (avg |π_j * f_j|) --
[0.0563, 0.0163, 0.2349]

-- Per-Expert MSE on Top-1 Assigned Samples --
[0.126401, nan, 0.063739]