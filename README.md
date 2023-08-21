# IIDM - FQCT
---
## Intro
---
Official implementation of the paper "Data-driven Explanations for Prototype-based
Intrusion Detection on Network Traffic Streams" submitted to Computers & Security.

An explanation framework for prototype-based intrusion detection on network traffic streams is proposed to provide high-quality explanations. To the best of our knowledge, none of existing methods is designed to explain the predicted instances from prototypes.

1. To provide feature explanations, IIDM optimizes the dual masks on perturbing both the explained attacks and prototypes, which fully use the data distribution information from prototypes.

2. FQCT derives rule explanations from a quantile corrected tree that is trained on the core normal prototypes and the explained attack based on the feature explanations from IIDM, which relaxes the deviation values around the single explained attack, leading to better generalization on unseen attacks.

3. Comprehensive experiments using seven setups of two intrusion detection benchmarks demonstrate that IIDM and FQCT achieve significant improvements over previous methods.


## Usage
---
To run IIDM, you need to:
`python main.py --datatype $1 --train_date $2 --exp_method $3 --explain_flag feature`

To run FQCT, you need to:
`python main.py --datatype $1 --train_date $2 --exp_method $3 --explain_flag rule`
