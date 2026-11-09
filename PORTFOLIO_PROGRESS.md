# Portfolio progress

## Verified inventory — 2026-11-04

LinkedIn Projects section contains exactly five entries, each linked to its public repository:

- ML Model Lab: https://github.com/erickvera610/ml-model-lab
- Research Search: https://github.com/erickvera610/research-search
- Road Vision Lab: https://github.com/erickvera610/road-vision-lab
- Media Content Lab: https://github.com/erickvera610/media-content-lab
- Consulting Demand Lab: https://github.com/erickvera610/consulting-demand-lab

## 2026-11-04 — ML Model Lab

Added a paired robustness audit with independent corruption seeds, hashes, split IDs, per-class recall and all 360 paired predictions. Added six regression tests. All 20 Python tests passed locally; JavaScript/Python inference parity passed. Published source, raw report and ROBUSTNESS_AUDIT.md. README and LinkedIn project description updated. No model accuracy improvement claimed.

Results: clean 98.06%, Gaussian noise 33.89%, salt/pepper 60.56%, center occlusion 85.28%, brightness 70% 95.83%. Example dataset row 1562: true 9; clean prediction 9; occluded prediction 7.

Verified published English LinkedIn update: https://www.linkedin.com/feed/update/urn:li:share:7523849460976680960/

## 2026-11-09 — Research Search

Repaired the robustness benchmark to use exported-model inference and count out-of-vocabulary abstentions as misses. Evaluated 30,135 official held-out sentences. Macro F1 was 0.7025 clean and 0.6816 under synthetic token dropout; predictions changed on 9.24%. All 11 Python tests passed, plus Python/browser parity. Code and results: https://github.com/erickvera610/research-search/commit/722fe232433b0082b6ae173b3ca2a75845993e50

Published English LinkedIn update: https://www.linkedin.com/feed/update/urn:li:activity:7525622482645745664/

## Next turn

Road Vision Lab, followed by Media Content Lab and Consulting Demand Lab. The Research Search LinkedIn project description update is pending. The separate GenAI project remains pending; the existing Research Search components reviewed are supervised NLP and lexical retrieval.
