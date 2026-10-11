# Model card: retail PD model

## Purpose
The model estimates the probability of default (PD) within the next month for a credit-card client. The estimate feeds the approve or decline decision described in the decision policy.

## Data
UCI / OpenML "Default of Credit Card Clients": 30 000 clients, 6 months of repayment status, bills and payments. The default rate is 22.1 %.

## Model
CatBoost gradient boosting on 39 behavioural features, with PD calibrated by isotonic regression on a validation set. A WoE logistic-regression scorecard is kept as an interpretable challenger.

## Performance on the hold-out set
CatBoost reaches AUC 0.792, Gini 0.584 and KS 0.453. The scorecard reaches AUC 0.777 and Gini 0.554. Expected calibration error is about 0.01.

## Limitations
The data contains only clients that were already approved, so there is selection bias. The data has no calendar dates, so there is no out-of-time validation. SEX and MARRIAGE are not used as features.
