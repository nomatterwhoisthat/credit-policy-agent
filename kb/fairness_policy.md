# Fairness policy (demo)

## Protected attributes
SEX and MARRIAGE are never used as model inputs or in decision rules. Age is used only as a model input in the scorecard and the boosting model, and it is audited.

## Audit
After each model release the approval rate is compared across sex and age bands. In the current audit the approval rate is 43.0 % for sex = 1 and 38.9 % for sex = 2, a disparate-impact ratio of 0.91. Approval rates by age band range from 37.9 % to 42.3 %.

## Interpretation
Observed default rates also differ between groups, so a gap in approval rates is not by itself evidence of unfair treatment. The audit does not prove the absence of proxy effects.
