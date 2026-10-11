"""Scoring service used by the agent: calibrated PD, expected-NPV decision and SHAP reason codes.
The model and calibrator are produced by the sibling project `credit-decision-engine` (see artifacts/)."""
from __future__ import annotations
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from .features import build_features

ART = Path(__file__).resolve().parents[2] / "artifacts"
RAW_COLS = (["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"] + [f"PAY_{i}" for i in range(1, 7)]
            + [f"BILL_AMT{i}" for i in range(1, 7)] + [f"PAY_AMT{i}" for i in range(1, 7)])

# same economic assumptions as credit-decision-engine/src/strategy.py -> Economics()
APR, LGD, CCF, UTIL_FLOOR = 0.22, 0.60, 0.75, 0.10

REASON_TEXT = {
    "PAY_1": "repayment status in the most recent month", "PAY_2": "repayment status two months ago",
    "PAY_3": "repayment status three months ago", "PAY_4": "repayment status four months ago",
    "PAY_5": "repayment status five months ago", "PAY_6": "repayment status six months ago",
    "max_delay": "longest payment delay in the last 6 months", "mean_delay": "average payment delay over 6 months",
    "recent_delay_3m": "average payment delay over the last 3 months", "n_late_months": "number of late months out of 6",
    "longest_late_streak": "longest streak of consecutive late months", "delay_trend": "trend of delinquency (recent vs older months)",
    "n_revolving": "number of months with revolving (minimum) payment", "n_paid_in_full": "number of months paid in full",
    "util_mean": "average credit utilisation", "util_last": "latest credit utilisation", "util_max": "peak credit utilisation",
    "util_trend": "trend of credit utilisation", "over_limit_months": "months above the credit limit",
    "pay_ratio_mean": "average share of the bill that was repaid", "pay_ratio_last": "share of the latest bill that was repaid",
    "paid_total_to_bill_total": "total repaid relative to total billed", "bill_volatility": "volatility of monthly bills",
    "zero_payment_months": "months with no payment", "LIMIT_BAL": "credit limit", "AGE": "age",
}


for _i in range(1, 7):
    REASON_TEXT[f"PAY_AMT{_i}"] = f"amount repaid {_i} month(s) before the snapshot"
    REASON_TEXT[f"BILL_AMT{_i}"] = f"bill amount {_i} month(s) before the snapshot"


@dataclass
class Scorer:
    model: CatBoostClassifier
    iso: object
    features: list

    def score(self, client: dict) -> dict:
        missing = [c for c in RAW_COLS if c not in client]
        if missing:
            raise ValueError(f"missing fields: {missing}")
        df = pd.DataFrame([{c: float(client[c]) for c in RAW_COLS}])
        X = build_features(df)[self.features]
        raw = float(self.model.predict_proba(X)[:, 1][0])
        pdc = float(self.iso.predict([raw])[0])
        util = float(X["util_mean"].iloc[0]); limit = float(client["LIMIT_BAL"])
        revenue = APR * max(util, UTIL_FLOOR) * limit
        loss = LGD * CCF * limit
        npv = (1 - pdc) * revenue - pdc * loss
        breakeven = revenue / (revenue + loss)
        return {"pd": round(pdc, 4), "decision": "approve" if npv > 0 else "decline", "expected_npv": round(npv, 1),
                "breakeven_pd": round(breakeven, 4), "expected_revenue_if_good": round(revenue, 1),
                "expected_loss_if_default": round(loss, 1), "reasons": self.reasons(X)}

    def reasons(self, X: pd.DataFrame, top: int = 3) -> list[dict]:
        sv = self.model.get_feature_importance(data=__import__("catboost").Pool(X), type="ShapValues")[0][:-1]
        order = np.argsort(-np.abs(sv))[:top]
        return [{"feature": self.features[i], "meaning": REASON_TEXT.get(self.features[i], self.features[i]),
                 "value": round(float(X.iloc[0, i]), 3), "effect": "raises PD" if sv[i] > 0 else "lowers PD",
                 "shap": round(float(sv[i]), 3)} for i in order]


@lru_cache(maxsize=1)
def get_scorer() -> Scorer:
    m = CatBoostClassifier(); m.load_model(str(ART / "catboost_pd.cbm"))
    return Scorer(m, joblib.load(ART / "isotonic_catboost.joblib"), json.load(open(ART / "meta.json"))["features"])
