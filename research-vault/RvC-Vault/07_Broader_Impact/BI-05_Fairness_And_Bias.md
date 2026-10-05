# BI-05 Fairness and bias auditing

status: proposed translation · amendment: [[THE_PLAN_AMENDMENT_A2]] · directions: [[D18_Protected_Attribute_Surface]], [[D19_Bias_Geometry]], [[D20_Debias_By_Repair]]

**What the instrument adds.** It treats a protected attribute in a verifiable-gold item as one more answer-irrelevant surface. That lets bias auditing reuse Probe 1 metrics (accuracy gap, retention, flip rate), Probe 2 CCI, and intrusion, and ask whether attribute-swap cost covaries with rename cost and whether both share a representational direction.

**What it does not do.** It does not debias deployed models today. Steering and RLVR arms (D20) are proposed repairs under programme gates; they are not a shipping mitigation.

**Where it is limited.** Verifiable-gold tasks only (GSM, coin_change, and cleared shortest_path cells). Attributes are used only where gold is provably independent of the attribute. No claim about open-ended generation, retrieval-augmented answers, or real persons named in the pools.
