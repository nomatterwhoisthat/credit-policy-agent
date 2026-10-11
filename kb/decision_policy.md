# Decision policy (demo)

## Rule
A client is approved when the expected NPV is positive. Expected NPV = (1 - PD) x revenue - PD x loss. Revenue = 22 % x max(average utilisation, 10 %) x credit limit. Loss = LGD 60 % x CCF 75 % x credit limit.

## Break-even PD
The rule is equivalent to approving when PD is below revenue / (revenue + loss). The break-even PD differs per client. Clients with low utilisation bring little revenue but carry the full limit risk, so their break-even PD is lower.

## Consequences
A client with a low PD can still be declined when utilisation is low. A client with a moderate PD can be approved when utilisation is high.

## Parameters
Yield 22 %, LGD 60 %, CCF 75 % and a utilisation floor of 10 % are assumptions of this demo. They are not estimated from data. Changing them changes the approval rate.
