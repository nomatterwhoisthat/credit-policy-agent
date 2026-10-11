# Monitoring policy (demo)

## PSI thresholds
PSI below 0.10 means the population is stable. PSI between 0.10 and 0.25 means a moderate shift and requires review. PSI above 0.25 means a significant shift and triggers a model investigation. These are common industry rules of thumb.

## What is monitored
The PSI of every input feature and the PSI of the model score are computed monthly against the training sample. Share of applications flagged as anomalous by an Isolation Forest is tracked as well.

## Simulated shock result
In a synthetic macro-shock scenario with more delays, higher bills and lower payments, 10 features exceeded PSI 0.25 and the score PSI reached 0.14. Mean predicted PD increased from 22.3 % to 25.4 %.

## Actions
On an alert the model owner checks data quality first, then recalibrates PD, and only then considers retraining.
