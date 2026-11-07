"""
Peer-Review-Grade ASD 3D Kinematic Gait Classification Pipeline.

Modules:
- data_preprocessing: Traverses the 700 Al-Jubouri Kinect v2 trial folders, maps each trial
  to its ground-truth subject_id (Child_01 to Child_100), and performs root-centering
  (SpineBase at origin) and torso-scaling (normalized by SpineShoulder-to-SpineBase height).
- feature_extraction: Computes biomechanical and kinematic features from normalized 3D joint
  trajectories and caches the tabular dataset to data/processed_features.csv.
- train: Leak-free 5-fold StratifiedGroupKFold cross-validation benchmarking CatBoost against
  Random Forest and XGBoost.
- explainability: TreeSHAP biomarker attribution and publication-ready visualization artifacts.
"""

__version__ = "2.0.0"
