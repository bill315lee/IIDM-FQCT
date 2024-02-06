# IIDM - FQCT
---
## Intro
---
Official implementation of the paper "Unifying Feature-level and Rule-level Anomaly Explanations for Prototype-based Models on Data Streams" submitted to Knowledge and Information Systems.

An explanation framework for prototype-based anomaly detection on data streams is proposed to provide high-quality explanations. To the best of our knowledge, none of existing methods is designed to explain the detected instances from prototype-based models on streams.

1. To provide feature explanations, IIDM optimizes the dual masks on perturbing both the explained anomaly and prototypes, which fully use the data distribution information from prototypes.

2. FQCT derives rule explanations from a quantile corrected tree that is trained on the core normal prototypes and the explained anomaly based on the feature explanations from IIDM, which relaxes the deviation values around the single explained anomaly, leading to better generalization on unseen anomalies.

3. Comprehensive experiments using several real-world data sets demonstrate that IIDM and FQCT achieve significant improvements over previous methods.


## Usage
---
To run IIDM, you need to:
`python main.py --datatype $1 --train_date $2 --exp_method $3 --explain_flag feature`

To run FQCT, you need to:
`python main.py --datatype $1 --train_date $2 --exp_method $3 --explain_flag rule`
