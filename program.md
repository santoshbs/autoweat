# AutoWEAT Program

The goal is to discover large, significant, socially meaningful WEAT biases in a frozen embedding source.

## Human-owned surface

The human owns:

- the embedding source
- the exact WEAT evaluator
- the target and attribute set sizes
- the significance and effect-size thresholds
- the history format
- the deduplication rules

## LLM-owned surface

The LLM may propose only:

- candidate `X/Y` target contrasts
- candidate `A/B` attribute contrasts
- discipline tags
- short rationales

## Hard rules

1. Never modify the embedding source.
2. Never modify the WEAT math.
3. Final evaluated sets must be equal-sized.
4. Report both effect size and exact permutation `p`.
5. Accept a finding only if it is large and significant.
6. Prefer socially interpretable biases over arbitrary lexical contrasts.
