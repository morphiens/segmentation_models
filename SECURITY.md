# Security Guide — Segmentation Models

This document describes security considerations for the **segmentation-models** Python library (Keras/TensorFlow image segmentation: U-Net, PSPNet, Linknet, FPN, and ImageNet-pretrained backbones). It is intended for **library maintainers**, **downstream integrators**, and **security reviewers** who train or deploy segmentation systems built on this code.

**Scope:** This repository is a **model-construction and training library**, not a network service. It does not expose HTTP APIs, authenticate users, or persist customer data. Most security risk appears when you **train on sensitive data**, **load weights or checkpoints from untrusted sources**, **publish models to production**, or **operate inference in regulated domains** (healthcare, automotive, robotics). This guide maps industry frameworks to those real-world usage patterns and records risks observed in **this** codebase.

**Last updated:** 2026-11-12  
**Library version tracked:** see `segmentation_models/__version__.py`

---

## Table of contents

1. [Executive summary](#1-executive-summary)
2. [Threat model](#2-threat-model)
3. [Repository security posture](#3-repository-security-posture)
4. [Secure usage for downstream consumers](#4-secure-usage-for-downstream-consumers)
5. [Training and data security](#5-training-and-data-security)
6. [Model artifacts and deserialization](#6-model-artifacts-and-deserialization)
7. [Pre-trained weights and supply chain](#7-pre-trained-weights-and-supply-chain)
8. [Production inference and deployment](#8-production-inference-and-deployment)
9. [Adversarial robustness (segmentation-specific)](#9-adversarial-robustness-segmentation-specific)
10. [Privacy and regulated use cases](#10-privacy-and-regulated-use-cases)
11. [Maintenance, CI/CD, and release integrity](#11-maintenance-cicd-and-release-integrity)
12. [Security testing checklist](#12-security-testing-checklist)
13. [Incident response](#13-incident-response)
14. [Framework mapping](#14-framework-mapping)
15. [References and further reading](#15-references-and-further-reading)

---

## 1. Executive summary

| Area | Risk level (typical) | Notes |
|------|---------------------|-------|
| Untrusted model/checkpoint load | **Critical** | Keras/TensorFlow `.h5` archives can execute code via Lambda layers; `safe_mode` does not fully protect legacy H5 in older Keras versions. |
| Dependency / PyPI supply chain | **High** | Pinned but aging deps (`efficientnet`, `keras-applications`, transitive TensorFlow/Keras). |
| Pre-trained weight download | **Medium** | Backbone weights fetched from GitHub release URLs (HTTPS); integrity depends on host trust, not local verification in this library. |
| Adversarial inputs at inference | **Context-dependent** | Standard architectures are not adversarially robust by default. |
| Library as attack surface | **Low** | No remote listeners; primary exposure is malicious **weights** or **dependencies** at import/train/load time. |
| Sensitive training data | **Context-dependent** | Library does not handle PHI; consumers must govern datasets and checkpoints. |

**Top actions for integrators:**

1. Never load segmentation checkpoints (`.h5`, full `load_model`, pickle/joblib) from untrusted origins.
2. Pin dependencies with lockfiles or hash checking; scan with `pip-audit` and keep TensorFlow/Keras current.
3. Verify weight provenance (SHA-256) for custom `weights=` paths passed to `Unet`, `PSPNet`, etc.
4. Separate dev/staging/prod; sign release artifacts; monitor prediction drift in production.
5. For clinical or safety-critical use, add human review, uncertainty monitoring, and adversarial evaluation.

---

## 2. Threat model

### 2.1 Assets

- **Source code** of `segmentation_models` and its dependencies (`keras-applications`, `image-classifiers`, `efficientnet`, `classification_models`).
- **Trained weights** produced by consumers (`model.load_weights`, Keras `save`/`load_model`).
- **Pre-trained encoder weights** (`encoder_weights='imagenet'` or remote H5 URLs in custom backbones).
- **Training datasets** (images, masks) — outside this repo but primary confidentiality asset.
- **PyPI packages** (`segmentation-models` wheel/sdist).
- **CI credentials** (historically Travis PyPI deploy in `.travis.yml`).

### 2.2 Trust boundaries

```
[Untrusted inputs]          [This library]           [Your infrastructure]
  - Internet PyPI      -->  import sm / build model  -->  Training GPUs
  - Random .h5 weights -->  load_weights()           -->  Inference API
  - User images        -->  (consumer code)          -->  PHI / safety systems
```

### 2.3 Relevant attack classes (OWASP ML Top 10 / MITRE ATLAS)

Align reviews with:

- **Input manipulation / evasion** — adversarial patches or perturbations on images.
- **Data poisoning** — corrupt labels or images in training pipelines.
- **Model theft / inversion / membership inference** — query access to inference endpoints.
- **Supply chain** — malicious or vulnerable dependencies; tampered wheels or weights.
- **Transfer learning attacks** — compromised ImageNet or custom encoder weights used with `encoder_weights=...`.

See [§14 Framework mapping](#14-framework-mapping) and [§15 References](#15-references-and-further-reading).

---

## 3. Repository security posture

This section reflects a **static review** of the codebase at the time of writing. It is not a formal audit.

### 3.1 Architecture

- **Framework selection** at import via `SM_FRAMEWORK` or `set_framework()` (`segmentation_models/__init__.py`). Wrong framework choice is an operational risk (incompatible checkpoints), not remote code execution by itself.
- **Models** are Keras `Model` instances built programmatically (`segmentation_models/models/*.py`).
- **Backbones** resolve through `BackbonesFactory` (`segmentation_models/backbones/backbones_factory.py`), including vendored Inception variants that download H5 files from fixed GitHub URLs.

### 3.2 Sensitive code patterns

| Location | Pattern | Security note |
|----------|---------|----------------|
| `segmentation_models/models/unet.py` (and PSPNet, Linknet, FPN) | `model.load_weights(weights)` when `weights is not None` | Loads **arbitrary filesystem paths** supplied by the caller. Malicious H5 can compromise the host when Keras deserializes unsafe layers. |
| `segmentation_models/backbones/inception_v3.py`, `inception_resnet_v2.py` | `WEIGHTS_PATH` HTTP(S) to GitHub releases | Supply-chain trust on `fchollet/deep-learning-models` and TLS. No embedded hash verification in-library. |
| `segmentation_models/utils.py` | `models.model_from_json(model.to_json())` | Safe when round-tripping **your own** model; **unsafe** if `model_from_json` is ever fed untrusted JSON. |
| `setup.py` | `exec(f.read(), about)` on `__version__.py` | Common setuptools pattern; only trusted repo content should be executed at build time. |
| `setup.py` `UploadCommand` | `os.system(...)` for twine, git tag/push | Maintainer workstation risk; prefer modern CI with OIDC publishing. |
| `segmentation_models/__init__.py` | `print('Segmentation Models: using ...')` on import | Minor information disclosure in logs about framework choice. |
| `.travis.yml` | Encrypted PyPI password, legacy stack (TensorFlow 1.14, Python 3.6) | Historical; modern pipelines should use GitHub Actions, Trusted Publishing, and supported runtimes. |

### 3.3 Dependencies (`requirements.txt`)

```
keras_applications>=1.0.7,<=1.0.8
image-classifiers==1.0.0
efficientnet==1.0.0
```

Runtime also requires **Keras** or **TensorFlow** (consumer-installed). Versions are **bounded but not fully locked**; transitive CVEs require consumer-side `pip-audit` and upgrades.

### 3.4 What this repo does *not* provide

- Authentication, authorization, or encryption.
- Input validation on image tensors (shape/dtype checks are the consumer’s responsibility).
- SBOM, signed releases, or dependency hash files (recommended additions for maintainers).

---

## 4. Secure usage for downstream consumers

### 4.1 Installation

- Install from **pinned versions** in a virtual environment or container image dedicated to ML.
- Prefer **hash-checked** installs where practical:

  ```bash
  pip install --require-hashes -r requirements.lock
  ```

- Verify PyPI **provenance/attestations** when available for `segmentation-models` and dependencies.
- Avoid `pip install git+https://...` from unreviewed forks in production.

### 4.2 Configuration

- Set `SM_FRAMEWORK` explicitly in deployment manifests (`keras` vs `tf.keras`) to match how models were trained and saved.
- Do not mount writable directories into training containers unless required; restrict outbound network if weights are pre-staged.

### 4.3 Custom weights argument

All four architects accept `weights=` for **full model** checkpoints (not only `encoder_weights`):

```python
model = sm.Unet('resnet34', encoder_weights='imagenet', weights='/path/to/segmentation.h5')
```

**Policy:** Treat `weights` like executable code. Only load from:

- Internal artifact registry with access control and audit logs.
- Files verified by **SHA-256** against a signed manifest.
- Formats that do not deserialize arbitrary Python (see [§6](#6-model-artifacts-and-deserialization)).

### 4.4 Version compatibility

README documents breaking changes across `1.*`. Loading old checkpoints after upgrades without validation can cause **silent mis-segmentation** (integrity failure), which may be as serious as a crash in safety-critical systems.

---

## 5. Training and data security

Follow NIST AI RMF **Govern / Map / Measure / Manage** and OWASP Secure AI Model Ops guidance:

1. **Version control** training code, hyperparameters, dataset snapshots (DVC, MLflow, W&B with access controls).
2. **Validate and sanitize** datasets: scan for path traversal in filenames, strip EXIF/metadata if publishing models, detect label poisoning via statistical QC.
3. **Least privilege** on storage buckets and notebook environments; no shared credentials across dev/prod.
4. **Secrets:** use secret managers; never commit API keys or cloud tokens (`.gitignore` already excludes `.env`).
5. **Reproducible environments:** container images with digest pinning for training jobs.
6. **Logging:** avoid logging raw images or patient identifiers; log hashes, model version, and run IDs.

If training on **personal or regulated data**, apply differential privacy or aggregation only when threat model requires it; document residual re-identification risk.

---

## 6. Model artifacts and deserialization

### 6.1 High-risk formats

| Format | Risk | Guidance |
|--------|------|----------|
| Keras `.h5` / `.hdf5` with Lambda layers | **Arbitrary code execution** on load | Do not load untrusted H5. Scan for Lambda layers; prefer weights-only load into a **known architecture** built from this library. |
| `keras.models.load_model` on legacy H5 | `safe_mode=True` **ignored** in Keras 3.0–3.11.2 for H5 (CVE-2025-9905) | Upgrade Keras **≥ 3.11.3**; prefer native `.keras` format with `safe_mode=True`. |
| Pickle / joblib | **Never** in production ML pipelines | Use safetensors, ONNX, or TensorFlow SavedModel with integrity checks. |
| `model_from_json` + weights | JSON from untrusted source is dangerous | Only round-trip trusted models; do not expose JSON upload endpoints without validation. |

**Weights-only pattern (recommended):**

```python
import segmentation_models as sm

model = sm.Unet('resnet34', classes=N, activation='softmax', encoder_weights=None)
model.load_weights('trusted.weights.h5')  # Still verify file hash and provenance
```

Even `load_weights` assumes a **compatible, honestly saved** file; prefer loading into architectures instantiated from **known code**, not full `load_model` from third parties.

### 6.2 Serialization hygiene

- Store **model version**, `segmentation_models` version, TensorFlow/Keras version, and `SM_FRAMEWORK` in metadata alongside weights.
- Sign artifacts (Sigstore/cosign, GPG, or internal CA) and verify before deployment.

---

## 7. Pre-trained weights and supply chain

### 7.1 ImageNet encoders (`encoder_weights='imagenet'`)

Backbones delegate to `keras-applications` / `classification_models` / `efficientnet`. Weights are typically downloaded on first use to `~/.keras/models`.

**Controls:**

- Pre-cache weights in controlled mirrors; block general internet egress from training nodes.
- Record download URLs and checksums in an internal manifest.
- For Inception variants, this repo points at:

  - `https://github.com/fchollet/deep-learning-models/releases/download/v0.5/...`

  Mirror these blobs internally and verify SHA-256 after download.

### 7.2 Python package supply chain

Apply OpenSSF/S2C2F and PyPI guidance:

- Maintain dependency inventory (SBOM: CycloneDX or SPDX).
- Enable Dependabot or equivalent; enforce PR checks for known CVEs.
- Use **cooldown** before auto-upgrading packages without advisories (~3 days) to reduce malicious release risk.
- Require code review on dependency bumps.

### 7.3 This package as a dependency

When vendoring or forking under `robotome/segmentation_models`:

- Track divergence from upstream [qubvel/segmentation_models](https://github.com/qubvel/segmentation_models).
- Document why pins differ; re-run security scans after merges.

---

## 8. Production inference and deployment

If you wrap models built with this library in Flask, FastAPI, TensorFlow Serving, Triton, or batch jobs:

### 8.1 API security (OWASP Secure AI Model Ops)

- **Authentication & authorization** on all inference endpoints (OAuth2, mTLS, API keys with rotation).
- **Input validation:** dtype, shape, channel count, max spatial resolution, batch size caps.
- **Rate limiting** per tenant, per model, and by **compute cost** (large images = higher GPU time).
- **Timeouts and circuit breakers** to mitigate DoS and “denial-of-wallet.”
- **No sensitive data in responses** unless authorized; avoid returning full softmax tensors if logits enable model extraction.

### 8.2 Infrastructure

- Run inference containers as **non-root**; read-only root filesystem; mount model weights read-only.
- Restrict GPU device access (NVIDIA Container Toolkit with least privilege).
- **Network policies:** inference pods should not reach metadata endpoints or arbitrary egress.
- Separate **train / eval / prod** clusters and credentials.

### 8.3 Monitoring

- Log: model ID, version, latency, input hash, output summary statistics — not raw PHI.
- Detect **distribution shift** and **confidence collapse**; alert on anomaly spikes (possible evasion campaign).
- Shadow or canary deployments for model updates with rapid rollback.

---

## 9. Adversarial robustness (segmentation-specific)

Semantic segmentation faces **pixel-level** adversarial examples, **patch attacks**, and **size bias** (small objects may be easier to fool than large regions). Research-backed practices:

1. **Evaluation:** Use diverse attack ensembles; report mIoU distributions, not only average mIoU (see ECCV 2024 and arXiv:2407.09150).
2. **Training:** Adversarial training can help but requires careful tuning; 50/50 clean/adversarial mixing is not always effective.
3. **Certified defenses:** For patch attacks, consider certified methods (e.g., Demasked Smoothing) where threat model matches deployment.
4. **Runtime:** Uncertainty-based detection (entropy of pixel posteriors) plus optional secondary models for flagged inputs.
5. **Membership inference:** Overfitted models leak training data membership; use regularization, knowledge distillation, or DP-SGD when privacy is required (trade-offs documented in segmentation MIA literature).

Treat robustness as a **release gate** for autonomous or clinical segmentation, not an optional benchmark.

---

## 10. Privacy and regulated use cases

For **medical imaging** or other regulated data:

- Execute **BAA** and HIPAA Security Rule controls with cloud vendors when PHI is involved.
- **De-identify** DICOM and strip private tags **before** sending data to external GPUs unless explicitly in scope.
- Encrypt at rest and in transit; maintain audit trails (who ran inference, which model version).
- Follow [ACR AI-LAB Reference Architecture](https://ailab.acr.org/doc/AI-LAB%20Reference%20Architecture%20Framework.pdf) for on-prem vs cloud partitioning.
- Human-in-the-loop review of segmentations used for treatment decisions.

This library does not implement HIPAA controls; consumers own compliance.

---

## 11. Maintenance, CI/CD, and release integrity

### 11.1 Recommendations for maintainers

| Topic | Current state | Recommendation |
|-------|---------------|----------------|
| CI | Travis CI (`.travis.yml`), EOL Python/TF | Migrate to GitHub Actions with matrix on supported Python (3.10+) and TensorFlow 2.x. |
| PyPI publish | Password in Travis secure env | **PyPI Trusted Publishing** (OIDC); rotate legacy credentials. |
| Workflow security | N/A | Pin actions by SHA; `permissions: contents: read`; `persist-credentials: false` on checkout. |
| Static analysis | Tests only | Add `bandit -r segmentation_models/ -ll` and `pip-audit` on locked deps. |
| Release artifacts | Wheel/sdist via setuptools | Publish SBOM; sign tags/releases; target SLSA Build L2+. |
| Branch protection | Unknown | Require PR reviews; block force-push to default branch. |

### 11.2 Contributor security

- MFA on GitHub and PyPI for all maintainers (OpenSSF baseline).
- Security policy: add `SECURITY.md` contact (e.g., `security@...` or GitHub private advisory).
- Responsible disclosure process aligned with OpenSSF guide.

---

## 12. Security testing checklist

Use before production promotion of a segmentation system built on this library:

### Supply chain

- [ ] Locked dependency file with hashes or uv/poetry lock
- [ ] `pip-audit` / OSV scan clean or accepted exceptions documented
- [ ] SBOM generated and stored with release
- [ ] Pre-trained weights checksum-verified

### Model integrity

- [ ] No untrusted `load_model` on H5/pickle
- [ ] Weights loaded only into architecture built from reviewed code
- [ ] Model version metadata present and verified at deploy
- [ ] Keras/TF versions meet CVE patches (incl. H5 `safe_mode` fix)

### Application (if exposed as a service)

- [ ] AuthN/AuthZ on inference API
- [ ] Input size/type limits enforced
- [ ] Rate limits and request timeouts
- [ ] TLS 1.2+ end-to-end

### ML-specific

- [ ] Adversarial / OOD evaluation on representative data
- [ ] Drift monitoring enabled
- [ ] Rollback procedure tested

### Privacy (if applicable)

- [ ] Data classification documented
- [ ] De-identification before cross-trust-boundary transfer
- [ ] Audit logging without leaking PHI in logs

---

## 13. Incident response

1. **Suspected malicious weights or dependency:** isolate hosts, preserve artifacts, rotate secrets, replace models from last known-good signed build.
2. **CVE in TensorFlow/Keras/efficientnet:** assess exploitability in your inference path; patch and redeploy; check for active exploitation in logs.
3. **Data breach via membership inference or model inversion:** notify per legal requirements; consider model retraining with privacy mitigations.
4. **Adversarial incident in production** (systematic mis-segmentation): enable kill switch, route to human review, capture adversarial inputs for forensics.

Align playbooks with NIST CSF / organization SOC procedures; extend tabletop exercises to **ML-specific** scenarios (OWASP GenAI checklist recommends this for AI systems).

---

## 14. Framework mapping

| Framework | How it applies here |
|-----------|---------------------|
| **NIST AI RMF 1.0** | Govern training data; measure security/resilience (Measure 2.7); manage deployment risks. |
| **NIST SSDF / SP 800-218A** | Secure SDLC for forks; dependency review; release signing. |
| **OWASP ML Security Top 10** | Input manipulation, poisoning, model theft, supply chain, transfer learning (ImageNet encoders). |
| **OWASP Secure AI Model Ops** | Training hygiene, artifact storage, inference API controls, isolation. |
| **MITRE ATLAS** | ML technique taxonomy for threat modeling workshops. |
| **Google SAIF** | Map risks across data → infrastructure → model → application. |
| **SLSA / OpenSSF S2C2F** | Build provenance, vulnerability scanning, curated feeds. |
| **Microsoft Zero Trust (DS-124)** | Model registry, signed artifacts, safe serialization formats. |

---

## 15. References and further reading

Industry guidance and research consulted for this document (20+ sources):

### Frameworks and standards

1. OWASP — [Secure AI Model Ops Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Secure_AI_Model_Ops_Cheat_Sheet.html)
2. OWASP — [Top 10 for LLM Applications 2025](https://genai.owasp.org/download/43299/)
3. OWASP — [LLM05 Supply Chain Vulnerabilities](https://genai.owasp.org/llmrisk2023-24/llm05-supply-chain-vulnerabilities/)
4. OWASP — [LLM AI Security and Governance Checklist (PDF)](https://genai.owasp.org/wp-content/uploads/2024/05/LLM_AI_Security_and_Governance_Checklist-v1.1.pdf)
5. OWASP — [Machine Learning Security Top 10 (project)](https://owasp.org/www-project-machine-learning-security-top-ten/)
6. OWASP — [ML Security Top 10 PDF v0.3](https://mltop10.info/OWASP-Machine-Learning-Security-Top-10.pdf)
7. NIST — [AI Risk Management Framework 1.0 (PDF)](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf)
8. NIST — [AI RMF Core](https://airc.nist.gov/airmf-resources/airmf/5-sec-core/)
9. NIST — [SP 800-218A SSDF Profile for GenAI / Foundation Models (PDF)](https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-218A.pdf)
10. MITRE — [ATLAS](https://atlas.mitre.org/)
11. Google — [Secure AI Framework (SAIF)](https://saif.google/secure-ai-framework)
12. SLSA — [Supply-chain Levels for Software Artifacts](https://slsa.dev/)

### Supply chain and Python packaging

13. OpenSSF — [MLSecOps Whitepaper (PDF)](https://openssf.org/wp-content/uploads/2025/08/OpenSSF_MLSecOps_Whitepaper.pdf)
14. OpenSSF — [Concise Guide for Developing More Secure Software](https://best.openssf.org/Concise-Guide-for-Developing-More-Secure-Software.html)
15. OpenSSF — [S2C2F (Secure Supply Chain Consumption Framework)](https://github.com/ossf/s2c2f)
16. Microsoft — [Zero Trust: AI/ML pipeline and model artifact security (DS_124)](https://microsoft.github.io/zerotrustassessment/docs/workshop-guidance/devsecops/DS_124)
17. PyPI Docs — [Common Security Attacks](https://docs.pypi.org/using-pypi/package-security/common-attacks/)
18. PyPI Docs — [Verifying Package Integrity](https://docs.pypi.org/using-pypi/package-security/package-integrity/)
19. PyPI Docs — [Checking Provenance / Attestations](https://docs.pypi.org/using-pypi/package-security/provenance/)
20. pip — [Secure installs / hash-checking mode](https://pip.pypa.io/en/latest/topics/secure-installs/)

### Model deserialization and deployment

21. Keras — [GHSA-36rr-ww3j-vrjv: H5 `safe_mode` and code execution](https://github.com/keras-team/keras/security/advisories/GHSA-36rr-ww3j-vrjv)
22. NVD — [CVE-2025-9905](https://nvd.nist.gov/vuln/detail/cve-2025-9905)
23. JFrog Security Research — [H5-LAMBDA threat](https://research.jfrog.com/model-threats/h5-lambda/)
24. HiddenLayer — [Machine Learning Models are Code](https://www.hiddenlayer.com/research/models-are-code)
25. LockML — [Secure ML Deployment Guide](https://lockml.com/guides/secure-ml-deployment.html)
26. Microsoft Learn — [AI Risk Assessment for ML Engineers](https://learn.microsoft.com/en-us/security/ai-red-team/ai-risk-assessment)

### CI/CD

27. GitHub Docs — [Secure use of GitHub Actions](https://docs.github.com/en/actions/reference/security/secure-use)
28. OWASP — [GitHub Actions Security Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/GitHub_Actions_Security_Cheat_Sheet.html)

### Segmentation adversarial / privacy research

29. Zhang et al. — [Evaluating Adversarial Robustness of Semantic Segmentation (arXiv:2407.09150)](https://arxiv.org/abs/2407.09150)
30. ECCV 2024 — [Towards Reliable Evaluation and Fast Training of Robust Semantic Segmentation Models (PDF)](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/10167.pdf)
31. Schuchardt et al. — [Certified Defences Against Adversarial Patch Attacks (arXiv:2209.05980)](https://doi.org/10.48550/arxiv.2209.05980)
32. ISNCC 2025 — [Post-Processing Strategies for Detecting and Mitigating Adversarial Perturbations in Segmentation Networks](https://doi.org/10.1109/isncc66965.2025.11250486)
33. Arnab et al. — [Membership Inference Attacks Against Semantic Segmentation Models](https://arxiv.org/html/2212.01082)

### Regulated / clinical imaging

34. ACR — [AI-LAB Reference Architecture Framework (PDF)](https://ailab.acr.org/doc/AI-LAB%20Reference%20Architecture%20Framework.pdf)
35. European Radiology — [Cybersecurity considerations for radiology departments involved with AI](https://doi.org/10.1007/s00330-023-09860-1)

### Tooling

36. PyCQA — [Bandit](https://pypi.org/project/bandit/)
37. PyPA — [pip-audit](https://github.com/pypa/pip-audit)

---

## Reporting vulnerabilities

If you discover a security issue **in this repository**:

1. **Do not** open a public issue for exploit details.
2. Contact the repository maintainers through your organization’s security channel or GitHub **Private vulnerability reporting** (if enabled).
3. Include affected version, reproduction steps, and impact assessment.

For issues in **upstream** [qubvel/segmentation_models](https://github.com/qubvel/segmentation_models) or in **TensorFlow/Keras**, report to those projects per their security policies while applying mitigations locally.

---

*This document is informational and does not constitute legal or compliance advice. Align controls with your organizational risk management process and regulatory obligations.*
