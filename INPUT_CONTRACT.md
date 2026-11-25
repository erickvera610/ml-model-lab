# Consistent inference inputs — November 25, 2026

## Problem and fix

The Python inference function converted inputs with `np.asarray(pixels, dtype=float)` before checking them. This accepted 64 numeric strings (`"0"`) and booleans as pixel intensities. The JavaScript implementation already rejected those values with `Number.isFinite`. Identical JSON therefore had different acceptance rules in Python and JavaScript.

The Python boundary now checks the original array dtype and rejects booleans in mixed Python lists/tuples before conversion. Real integer/float values remain valid; existing shape, finite-value and [0,16] checks remain. The HTTP endpoint returns 400 for invalid pixel types and remains available afterward. This intentionally tightens the API: clients sending numeric strings must send JSON numbers instead.

## Reproducible example

```bash
python -m pip install -r requirements-benchmark.txt
python audit_input_contract.py
python -m unittest discover -v
node test_neural.cjs
node test_enhanced.cjs
node test_demo.cjs
```

`{"pixels": ["0", ...]}` with 64 string values previously passed Python validation; it now fails, matching JavaScript. An array of 64 numeric zero values remains accepted. A blank image being structurally valid does not mean it is a recognizable digit.

The audit reconstructs the previous validation expression and invokes current Python and JavaScript inference on the same seven fixtures. It records actual outcomes, model/code hashes and preservation of all 16 archived example predictions in [raw results](artifacts/input-contract-results.json).

| Input | Previous Python | Current Python | JavaScript |
|---|---|---|---|
| 64 numeric zeros | Accept | Accept | Accept |
| 64 numeric strings | Accept | Reject | Reject |
| 64 booleans | Accept | Reject | Reject |
| Boolean mixed with numbers | Accept | Reject | Reject |
| Null values | Reject | Reject | Reject |
| 63 values | Reject | Reject | Reject |
| Values above 16 | Reject | Reject | Reject |

All **23 Python tests** and **three JavaScript suites** passed locally. Three new test methods cover type rejection, archived predictions/probabilities and numeric-array compatibility; existing HTTP tests now include string and mixed-boolean inputs. Current inference retained the predictions of all **16 archived examples**, with their reported winning-class probabilities matching to 10 decimal places in regression tests. This is not a rerun of the full 360-image test benchmark and makes no accuracy improvement claim. Model parameters were not changed. Existing CI discovers the new Python tests automatically; remote CI completion is not asserted.

The contract targets JSON inputs and ordinary real NumPy arrays. If a caller has already converted booleans to a numeric NumPy array, their original type cannot be recovered. This fix does not detect out-of-distribution images or make the local demonstration server production-ready.

Independent AI-assisted portfolio development. No private data or secrets were introduced.
