# Explaining a decision to a client (demo)

## Reason codes
Every decline is explained with the three features that contribute most to the model score, taken from SHAP values. Each reason is stated in plain language, for example "longest payment delay in the last 6 months".

## What must not be said
Reasons never mention protected attributes. A reason must correspond to a feature that actually influenced the score.

## Economic declines
When the PD is below the portfolio average but the client is still declined, the cause is the economic rule. Low utilisation produces low revenue relative to the exposure, so the break-even PD is not reached. The explanation must say this explicitly.
