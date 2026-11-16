1.0 initial
1.1 base_model -> stable-diffusion-v1-5/stable-diffusion-v1-5 (matches M2 kit); negative -> M2 value; added safety_checker flag (false = evaluation setting)
1.2 config/m3.json: add `llm` block (D), `dev_seeds` / `heldout_seeds` (Overview §6.1), `rrv_v1.sha256` and `source` filled from the M2 hub weights
1.3 negative back to "ugly, blurry, bad anatomy": the value every dev image (31 Oct - 15 Nov) was made with, so the held-out run uses the pipeline that was evaluated. Base model rename and safety_checker flag from 1.1 kept. The M2 negative is a candidate for M4. (B; approval: A or D)
