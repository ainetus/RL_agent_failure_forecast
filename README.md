# Forecast RL Agent Failure

This repository implements a framework to **quantify and predict the reliability of
pre-trained Reinforcement Learning (RL) agents** used for real-time congestion management in
power grids.

Assessing the reliability of AI-assisted decision support systems under unseen operating
conditions is critical. This project anticipates unreliable AI recommendations and provides
early warnings to human operators.

The pipeline integrates **Uncertainty Quantification (UQ)** to support risk-aware
decision-making by separating uncertainty into two components:

- **Aleatoric uncertainty** — the predictive variance of the forecasts. It captures the
  inherent stochastic variability and forecast errors of load and generation, estimated by
  modelling the residuals of the primary forecaster (Histogram Gradient Boosting, HGB).
- **Epistemic uncertainty** — the uncertainty associated with the RL agent's decisions when
  facing out-of-distribution or unseen grid states, computed with an **Evidential Neural
  Network (ENN)**.

These indicators are fed into a **failure prediction model** that estimates the probability
of RL agent failure under future contingencies (line disconnections).

Finally, a **dual-LLM architecture** distils the black-box failure predictor into compact,
human-readable symbolic `if/else` rules (`best_rule.py`), one per contingency. This turns
opaque uncertainty metrics into interpretable operational guidelines, making the AI
assistant's boundaries transparent and auditable.

---

## Supported environment

- **Network 36** (`l2rpn_icaps_2021_small`) — IEEE 118-bus system, 36 substations, 59 lines,
  22 generators, 37 loads.

---

## Project structure

```text
grid_security_project/
│
├── agents/
│   └── network36/                       # Pre-trained Grid2Op agent (CurriculumAgent)
│
├── forecasts/
│   └── HBGB_36.pkl                      # Load / generation forecaster
│
├── models/
│   ├── enn_36.pth                       # Evidential Neural Network (epistemic uncertainty)
│   └── HBGB_36_aleatoric.pkl            # Aleatoric uncertainty model
│
├── results/                             # Dual-LLM output (generated rules + metrics)
│   └── seed_<value>/
│       └── line_<line_id>/
│           └── best_rule.py             # The final interpretable Python rule
│
├── prompts/                             # Exact LLM prompts (see llm_prompts.md)
│   ├── generator_prompt.md
│   ├── critic_prompt.md
│   └── repair_prompt.md
│
├── src/
│   ├── collect_data.py                  # Simulation and dataset generation
│   ├── config.py                        # Central configuration (environment, modes, paths)
│   ├── enn_models.py                    # ENN architectures
│   ├── rule_predictor.py                # Rule inference and natural-language translation
│   ├── test_rule_predictor.py           # Live rule inference over an episode
│   ├── train_classifier.py              # Failure classifier training and inference
│   ├── train_forecast.py                # Forecaster training
│   ├── training_enn.py                  # ENN training pipeline
│   └── utils.py                         # Feature extraction and grid statistics
│
├── dual_llm.py                          # Dual-LLM interpretable rule extraction
├── compute_shap_rankings.py            # Per-line SHAP feature ranking (run before dual_llm.py)
├── llm_prompts.md                       # Documentation of the three LLM prompts
├── run_pipeline.py                      # Main entry point for the UQ / forecasting pipeline
└── requirements.txt                     # Python dependencies
```

---

## Installation

### 1. Python version

This project requires **Python 3.10.13**.

```bash
python --version   # Python 3.10.13
```

### 2. Clone the repository

```bash
git clone <repository_url>
```

### 3. Install dependencies

A Python 3.10 virtual environment is strongly recommended.

```bash
pip install -r requirements.txt
```

### 4. Agent setup

Place your pre-trained agent (CurriculumAgent) in the `agents/` folder. The agent is
available from the repository's GitHub Releases (`v1.0-agents`). The path is configurable in
`src/config.py`.

### 5. Pre-trained models

The forecaster and uncertainty models are too large for the repository and are hosted as
binary attachments in GitHub Releases (`v1.0-models`). Download and place them as follows:

- `HBGB_36.pkl` → `forecasts/`
- `HBGB_36_aleatoric.pkl` → `models/`
- `enn_36.pth` → `models/`

### 6. LLM endpoint

The dual-LLM step queries an OpenAI-compatible chat endpoint. Create
`llm_config_inesctec.json` next to `dual_llm.py`:

```json
{
  "api_url": "https://<your-endpoint>/v1/chat/completions",
  "api_key": "<your-key>",
  "model": "gemma4-31b",
  "max_tokens": 4000,
  "verify_ssl": true,
  "timeout": 180
}
```

> Do **not** commit this file: add it to `.gitignore`. The base model used in the paper is the
> open-source `gemma4-31b`.

---

## Usage

### A. UQ / forecasting pipeline

All settings are controlled via `src/config.py`; you do not need to edit the logic scripts.
Switch behaviour with the mode flags and run:

```bash
python run_pipeline.py
```

- **Training** (`TRAIN_MODE = True`): trains the forecasters, collects simulation data, and
  trains the failure classifier; models are saved in `models/`.
- **Single-episode simulation** (`TEST_SINGLE_EPISODE = True`, `EPISODE_ID_TO_TEST = <seed>`):
  simulates one episode and records how the uncertainty metrics evolve over time.
- **Single-observation inference** (`PREDICT_PROBA_MODE = True`): computes the failure
  probability for one grid state, without running a physical simulation.

### B. Interpretable rule extraction (dual-LLM)

Run the two steps in order:

```bash
python compute_shap_rankings.py     # once -> shap_rankings.json
python dual_llm.py                  # runs all seeds -> results/ and results/summary.json
```

`dual_llm.py` distils the HGB failure predictor into one `if/else` rule per contingency. For
each line it ranks features by SHAP on the HGB, builds data-driven seed rules (single-feature,
AND and OR), and refines them through a Generator–Critic–Repair loop ranked by F2-score. Rules
are selected on the training partition and reported on the held-out test partition; the whole
procedure is repeated over 10 seeds, and `summary.json` reports the mean ± std of the pooled
test metrics (Table III in the paper).

### C. Natural-language explanations

`src/rule_predictor.py` translates each `best_rule.py` into a plain-English sentence and can
evaluate the rules live against an episode:

```bash
python src/test_rule_predictor.py            # normal episode
python src/test_rule_predictor.py --attack   # adversarial HeavyAttack_1 line attacks
```

For each monitored line and step, it prints the binary prediction (`OK` /
`FAILURE PREDICTED`) and the explanation sentence, and at the end reports whether any rule
warned within the 12 steps (one hour) before the actual failure.

---

## Methodology

The framework combines **grid-state forecasting**, **uncertainty quantification**, and
**risk classification** to anticipate failures caused by line disconnections, and then
distils the predictor into interpretable rules. It is structured in the following stages.

### 1. Uncertainty decomposition

**Aleatoric uncertainty (data uncertainty).** A multi-output HGB forecaster predicts active
and reactive power injections one hour ahead (12 steps). The squared residuals of its mean
forecasts approximate the local variance, and a second HGB model is trained on these residuals
(Poisson loss) to estimate the forecast variance, used as a proxy for aleatoric uncertainty.

**Epistemic uncertainty (model uncertainty).** An ENN is trained by **behavior cloning of the
CurriculumAgent Tutor**, mapping each grid state to the Tutor's selected topology action.
Instead of softmax probabilities, the ENN outputs the parameters of a Dirichlet distribution
over the `K = 250` curated topology actions. Model ignorance (vacuity) is computed as

```
u = K / sum(alpha_i)
```

where `alpha_i` are the Dirichlet parameters. Since the ENN only ever observes the states the
agent actually visited, high `u` flags out-of-distribution grid conditions where the agent's
recommendations are least reliable. The ENN does not replace the policy: the agent issues the
control action, while the ENN runs in parallel as a familiarity signal.

### 2. Forecasting future grid states

Load and generation forecasts are injected into the grid model and a power-flow solver
produces the forecasted state one hour ahead. These future states are combined with the
aleatoric uncertainty estimates.

### 3. Contingency analysis

For each candidate critical line, a what-if `N − 1` disconnection is simulated on the
forecasted grid state, and whether the grid reaches a failure condition one hour after the
contingency defines the label. This produces labelled data linking grid conditions,
uncertainties, and line disconnections to observed failures.

### 4. Risk classification

A binary HGB classifier predicts failures before action execution. Inputs: current and
forecasted grid-state indicators (load–generation balance, thermal stress), epistemic and
aleatoric uncertainty, and the disconnected-line identifier. Output: `0` (stable expected) or
`1` (failure predicted — alarm).

### 5. LLM-guided symbolic rule generation (dual-LLM)

To turn the black-box classifier into interpretable guidelines, a dual-LLM procedure distils
it into a compact `if/else` rule per line. The three roles (Generator, Critic, Repair) are
instances of the **same base model** (`gemma4-31b`) queried with role-specific prompts (see
`llm_prompts.md`). The method works as follows:

- **SHAP feature selection.** For each line, the input features are ranked by applying SHAP on
  the HGB, and only the top `k` (`k ∈ {3, …, 8}`, selected per line) are retained
  (`compute_shap_rankings.py`).
- **Data-driven seed rules.** From class-separated statistics, single-feature, conjunctive
  (AND) and disjunctive (OR) candidate rules are built over the selected features; the best by
  F2 bootstraps the loop. OR rules raise recall, the main lever for F2.
- **Generator–Critic–Repair loop.** The Generator proposes an improved rule, the Repair LLM
  fixes any structural violation, and the Critic returns targeted feedback for the next
  iteration.
- **F2-based ranking and honest evaluation.** Candidates are ranked directly by their
  F2-score, with a guard discarding degenerate rules that flag all inputs (FA ≥ 0.90). Rules
  are selected on the training set and the retained rule is evaluated on the held-out test
  set, so the metrics are comparable to the decision-tree and HGB baselines. The procedure is
  repeated over 10 seeds and the mean ± std of the pooled test metrics is reported.

### 6. Rule translation (natural-language explanations)

`src/rule_predictor.py` parses each `best_rule.py` as an Abstract Syntax Tree and maps every
condition to a human-readable description of the corresponding grid feature. Each distinct
failure path becomes an "or if" clause, and AND conditions within a path are joined with
"while … and". For example:

```python
def rule(x):
    if x["load_gen_ratio"] <= 0.986:
        if x["aleatoric_load_q_mean"] >= 0.185:
            return 1
        else:
            if x["epistemic_after"] >= 0.803:
                return 1
            else:
                return 0
    else:
        return 0
```

is translated to:

> *Following a contingency on line 34_35_110, the RL agent is predicted to fail when the
> load-to-generation ratio at t is ≤ 0.986 and either the mean aleatoric reactive-load
> uncertainty is ≥ 0.185 or the epistemic uncertainty at t+12 is ≥ 0.803.*

In live rule-inference mode, the system also computes the t+12 features from the current
observation, applies the rule, and reports the prediction alongside the explanation.

---

## Final objective

The goal of this framework is to provide **real-time confidence levels** that allow validation
of autonomous agent decisions and prevention of unsafe operations in critical power-grid
environments, through transparent, human-readable guidelines.
