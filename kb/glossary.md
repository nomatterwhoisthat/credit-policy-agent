# Glossary of credit-risk terms

## PD, LGD, EAD
PD is the probability of default over the horizon. LGD is the share of the exposure that is lost when default happens. EAD is the exposure at default.

## ECL
Expected credit loss is the sum over clients of PD x LGD x EAD. IFRS 9 stage 1 uses a 12-month PD.

## CCF
The credit conversion factor is the share of an undrawn limit that is expected to be drawn before default.

## Gini and KS
Gini = 2 x AUC - 1. KS is the maximum distance between the cumulative score distributions of defaulters and non-defaulters.

## WoE and IV
Weight of evidence for a bin = ln(share of good clients / share of bad clients). Information value is the sum over bins of (share good - share bad) x WoE. A feature with IV above 0.3 is usually considered strong.

## PSI
The population stability index compares a reference and a current distribution: PSI = sum of (actual share - expected share) x ln(actual share / expected share) over bins.
