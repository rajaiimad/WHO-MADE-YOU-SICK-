# Pre-registration v7 — Stage 7: from one generator to a generator family
Written 2026-10-06, after the second committee review and before any Stage-7 corpus was
generated or any model was trained on one.

## Why
Round-2 review (6 of 8 reviewers: major revision) traced the remaining problems to one root:
every claim was tested on one hand-written generator whose per-edge form,
(channels x view overlap) / distance summed over peers, matches the MPNN. The max-over-peers
variant (v4) does not break this match, because the MPNN's readout has a max branch.
This stage tests whether the claims survive generators whose per-edge form differs from the
MPNN's assumptions, and across a broad prior over all unvalidated parameters.

## A. Structural variants (each changes the per-edge form; the gain is recalibrated label-free
to the v2 expected prevalence of 1.38% on seeds 10000-10029)
| id | change to c_{j->i} |
|---|---|
| d2 | distance kernel 1/d^2 instead of 1/d |
| cutoff | distance kernel 1[d <= 2.5 m] (no falloff inside, nothing outside) |
| fovfree | angular velocity and flicker also act out of view (FOV factor 1 for them) |
| and | adds a non-additive term 0.3 * x_ang * x_flicker (both channels must be high) |
| sat | saturating per-edge effect: tau * (1 - exp(-c / tau)), tau = 0.15 |

## B. Random generator family
12 generators drawn once from a fixed prior (numpy seed 7000 + k), each with the gain
recalibrated to 1.38%:
* channel weights ~ Dirichlet(1,1,1,1,1);
* distance kernel: 1/d^p, p ~ U(0.5, 2) (prob. 0.5), else cutoff 1[d <= c], c ~ U(1.5, 4) m;
* view-overlap exponent q ~ U(0, 2) (q = 0: any in-view peer counts fully);
* each of the four visual channels acts out of view with prob. 0.2 (audio always does);
* interaction term weight 0 (prob. 0.5) else U(0, 0.3) on a random channel pair;
* saturation: none (prob. 0.5) else tau ~ U(0.1, 0.5);
* aggregation over peers: sum (0.5), max (0.25), log-sum-exp with temperature U(0.02, 0.1) (0.25);
* noise SD ~ U(0.03, 0.08).

## C. Transfer
Models trained once on the v2 main corpus (5 seeds) are scored on every A and B test corpus.

## D. Temporal context (RQ3)
Two dose variants (P(t) = 0.9 P(t-1) + 0.1 sum_j c_{j->i}(t)): `dose_ramp` (the v4 dose) and
`dose_intermittent` (the attacker is on in a random 30% of frames; history now carries
information that the current frame does not). Temporal models: edge-interaction HGB and MPNN
with (i) the causal 8-frame mean (v4) and (ii) an exponential filter bank of each edge's
features with decays {0.5, 0.75, 0.875, 0.95} (0.9 deliberately excluded).

## E. Fairer baseline
`edgefeatraw` HGB: the edge-interaction features plus the same interaction terms computed on the
raw normalised channels (not only the excess over the benign band). It is added to every
variant of A, B, D and to the v2 lock-box comparison (trained on the main corpus, 5 seeds).

## F. Within-band proximity attacker
The attacker keeps its attack channel at the top of the benign band (never above it) and stands
at the position, at least 1.0 m from every user, that maximises sum_i v_i/d_i over the other users
among 64 random candidates. Same test seeds and detectors as Stage 5 (seeds 70000-70059).

## Models and protocol (unchanged)
rule, prototype (original MLP), edge-interaction HGB, edgefeatraw HGB, GAT, MPNN; fixed v2
configurations (edgefeatraw uses the edge-interaction HGB configuration); 5 seeds, 45/15 cohort
split of a 60-cohort development corpus (seeds 60000-60059); one score on a 60-cohort test
corpus (seeds 61000-61059); cohort-bootstrap CIs (B = 1000) with paired differences.

## Decision rules fixed in advance
1. Circularity. The claim "the MPNN's advantage is not explained by matching the generator's
   per-edge form" is kept only if the MPNN has the highest PR-AUC in >= 4 of 5 A-variants AND
   >= 10 of 12 B-generators. Otherwise the claim is restricted to "generators whose per-edge
   form the MPNN matches", and the generators where it loses are reported.
2. Strength. For each A/B generator the paired PR-AUC difference MPNN - best other model is
   reported with its CI; "significantly better" only where the CI excludes 0.
3. RQ3. Temporal context helps only if the filter-bank MPNN minus the per-frame MPNN has a CI
   above 0 in `dose_intermittent`. Results for `dose_ramp` are reported either way.
4. Fairness. If `edgefeatraw` HGB's lock-box PR-AUC is within the CI of, or above, the MPNN's,
   the paper must say that explicit interaction features match the MPNN.
5. Within-band attacker. The adaptive-attacker sentence keeps the v6 rule (MPNN recall at the
   10% budget >= 0.80 when harm >= 25% of the standard attack); otherwise it is removed.
Every generator and model is reported, whatever it shows.
