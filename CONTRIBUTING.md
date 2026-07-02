# Contributing

Thanks for your interest in improving the Forensic Sketch → Photo GAN.

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q                      # sanity-check the suite
```

Or with conda: `conda env create -f environment.yml && conda activate sketch2photo`.

## Project layout

| Path                | Purpose                                                   |
|---------------------|-----------------------------------------------------------|
| `src/models/`       | Generator, discriminator, losses, building blocks         |
| `src/data/`         | Dataset, alignment, preprocessing                         |
| `src/utils/`        | Logging, EMA, seeding, config validation, checkpoints     |
| `src/train.py`      | Training loop (TTUR / AMP / EMA / best-FID)               |
| `src/eval.py`       | Metric suite + comparison grids + JSON report             |
| `src/metrics.py`    | FID / LPIPS / SSIM / PSNR / identity / rank-k / NIQE       |
| `src/demo.py`       | Gradio Blocks demo                                        |
| `scripts/`          | Data download + synthetic sketch generation               |
| `tests/`            | Unit tests                                                 |

## Guidelines

1. **Keep it modular.** New losses go in `models/losses.py`, new metrics in
   `metrics.py`, new sketch styles register via `@register_style` in
   `scripts/sketch_styles.py`.
2. **Preserve backward compatibility.** Model upgrades should be
   config-gated and identity-initialised so existing checkpoints still load
   (see `UNetGenerator.init_enhancements`).
3. **Validate config changes.** Add new keys to `utils/config.py::DEFAULTS`
   and extend `validate_config`.
4. **Add a test.** Anything with a pure-Python surface (shapes, splits,
   config, metrics) should get a unit test under `tests/`.
5. **Run the suite** with `pytest -q` before opening a PR. Tests that need
   downloadable pretrained weights are skipped automatically when offline.

## Commit / PR style

- Small, focused commits with imperative subject lines.
- Describe *why*, not just *what*, in the PR body.
- Note any new dependency and whether it is optional.
