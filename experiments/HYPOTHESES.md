# Milestone 3 Hypotheses

## Section A (Member A: Pipeline & SHAP)

### Token Budget Experiment
* **H-A1 (Length):** As the expanded prompt grows from 15 to 70 tokens, the style fidelity (measured by `style_margin`) will slightly decrease because the attention mechanism dilutes the trigger's weight.
* **H-A2 (Trigger Position):** Placing the `rrvarma style` trigger at the very beginning of the prompt will yield significantly higher style fidelity than placing it after the scene description.
* **H-A3 (Style Tail):** Appending the style tail will marginally improve style scores, but may crowd out faithfulness details at the 70-token limit.

### SHAP Attribution Experiment
* **H-A4 (Faithfulness):** The `char` (character descriptions) and `action` players will have the highest Shapley values for faithfulness.
* **H-A5 (Style vs. Faithfulness):** The `tail` player will have a positive Shapley value for style, but a near-zero or negative value for faithfulness.
* **H-A6 (Names):** Raw character names will add ~0 to faithfulness once the visual descriptions are already present in the prompt.