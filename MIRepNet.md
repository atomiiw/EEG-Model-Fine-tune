# MIRepNet: EEG Foundation Model for Motor Imagery

## Model Specifications

| Aspect                | Description                                                                    |
| :-------------------- | :----------------------------------------------------------------------------- |
| **Architecture Type** | Transformer-based EEG foundation model                                         |
| **Training Strategy** | Hybrid (masked reconstruction + classification)                                |
| **Key Innovation**    | MI-specific pretraining + cross-headset distribution alignment                 |
| **Total Parameters**  | ~5.2 M                                                                         |
| **Performance**       | 65–75% accuracy across 5 public MI datasets (SOTA)                            |
| **Main Purpose**      | Generalizable, paradigm-specific EEG representation for motor imagery decoding |

---

## Architecture

- Transformer-based
- 5.2M parameters
- Handles arbitrary EEG headset configurations

---

## Training Procedure

### Hybrid Pretraining Strategy

1. **Self-supervised:** Masked token reconstruction
2. **Supervised:** Motor imagery classification

### Preprocessing Pipeline

- Subject screening
- Neurophysiologically-informed channel template alignment
- Frequency filtering and temporal resampling
- Distribution alignment across datasets

---

## Datasets

### Pretraining
- 7 public MI datasets from MOABB

### Validation
- 5 public MI datasets
- 47 subjects total

---

## Performance

- **Accuracy:** 65–75% across 5 datasets
- **Fine-tuning:** <30 trials per class required
- **Convergence:** Few epochs
- **Comparison:** Outperforms 9 specialist EEG models and 5 generalized foundation models

---

## References

**Published:** July 29, 2025
**arXiv:** 2507.20254
**Repository:** https://github.com/staraink/MIRepNet/tree/main
**Authors:** Liu et al. (2025)
