# Feature dictionary

## Repayment status
PAY_1 is the repayment status in the most recent month and PAY_6 the oldest of six months. The value -1 means paid in full, 0 means revolving credit with minimum payment, and a positive value N means N months of delay.

## Delinquency features
max_delay is the maximum delay over six months. n_late_months counts months with a delay. longest_late_streak is the longest run of consecutive late months. delay_trend is the mean delay of the last three months minus the mean delay of the earlier three months.

## Utilisation and repayment
util_mean is the average of bill amount divided by credit limit. pay_ratio_mean is the average of payment divided by bill, capped at 5. paid_total_to_bill_total is total payments divided by total bills.
