# CognitiveTwin-V: Vascular-Enhanced Digital Twin for Alzheimer's Disease

CognitiveTwin-V is a multi-modal, temporally coherent deep learning architecture designed for Alzheimer’s Disease (AD) staging and conversion forecasting. By synthesizing cognitive, imaging, genetic, and critically—**vascular and metabolic health data**—it acts as a digital twin to track and predict patient progression. 

Traditional models often regress a single cognitive score or treat MCI-to-AD conversion as a siloed task. CognitiveTwin-V solves this by jointly predicting discrete disease stages, future conversion risks, and benchmark cognitive trajectories, all while bypassing the need for time-consuming clinical interviews (the "CDR Paradox") during inference.

## ✨ Key Innovations
* **The 5th Modality (Vascular/Metabolic):** The first AD digital twin to integrate systemic health (diabetes, hypertension, TG/HDL), mapping metabolic stress directly to neurodegeneration.
* **Three-Head Multi-Task Architecture:**
  * **Head 1:** 4-Class Current Biological Stage (CN / Mild / Moderate / Severe).
  * **Head 2:** 36-Month MCI $\rightarrow$ AD Conversion Probability.
  * **Head 3:** 24-Month MMSE Trajectory Forecasting.
* **Biologically-Grounded Label Engineering:** Prevents data leakage by permanently dropping ground-truth clinical labels (CDRSB, DX) from the feature matrix before training. The model is forced to map proxy biomarkers directly to biological states.
* **Trust & Transparency:** Native integration of XGBoost-surrogate SHAP explainability and comprehensive Fairlearn demographic auditing (bias detection across Sex, APOE4 status, Diabetes, Race, and Ethnicity).

## 🧠 Architecture Overview
The pipeline processes temporal data through five modality-specific Multi-Layer Perceptrons (MLPs), followed by a cross-modality Transformer. Missing data is managed gracefully using Boolean masks rather than artificial imputation.

1. **Feature Stratification:** Z-score normalization and encoding via 256-dim MLPs.
2. **Transformer Fusion:** Multi-head self-attention unifies the 5 modalities.
3. **Deep Markov Model (DMM):** Generates a 64-dimensional temporally coherent latent disease state $z_t$. Missing clinical visits are propagated forward using learned disease dynamics and growing uncertainty.
4. **Joint Optimization:** The network is optimized across all tasks simultaneously using a dynamic joint loss function:
   $$ \mathcal{L}_{total} = \lambda_1 \mathcal{L}_{Focal} + \lambda_2 \mathcal{L}_{BCE} + \lambda_3 \mathcal{L}_{MSE} + \lambda_4 \mathcal{L}_{ELBO} $$

## 📊 Benchmark & Performance Summary
Evaluated on the ADNI dataset (TADPOLE + MEDHIST + VITALS + LABDATA), the model achieves strong predictive alignment with a composite score of **0.7237**.

* **Head 1 (Stage Classification):** Macro F1: 0.6353 | Accuracy: 0.8421 (Tuned F1: 0.6404)
* **Head 2 (MCI Conversion):** AUC: 0.8336 | Calibrated Brier Score: 0.1426
* **Head 3 (MMSE Trajectory):** MAE: 2.1452 | RMSE: 3.1575

**Fairness Audit:** 
CognitiveTwin-V maintains rigorous equity with minimal accuracy disparity across high-risk groups:
* **Diabetes Status:** Max accuracy disparity of just 1.19%.
* **APOE4 Status:** Max accuracy disparity of 8.36%.
