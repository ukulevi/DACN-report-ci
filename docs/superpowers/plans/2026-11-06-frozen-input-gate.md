# Frozen input gate

Scope: a CPU-only input gate for the future final runner, not a test inference
implementation. No private test access or GPU. Candidate stays unfrozen while
LPIPS hashes and dependency lock remain unavailable.

1. Add regression tests for candidate refusal, incomplete pins, mutable dataset
   revision, non-finite thresholds, changed files and captured-byte reuse.
2. Add `dacn.evaluation.frozen_inputs.capture_frozen_inputs`: schema/status and
   immutable revision validation, exact artifact set, SHA256 verification from
   captured bytes, finite image/pixel thresholds. Return captured bytes for the
   downstream runner; do not silently reload paths. No tokens/identifiers/logging.
3. Add candidate JSON with measured known pins and explicit null pending pins;
   document the API and remaining prerequisites. Run targeted tests and Ruff,
   log evidence, commit on the existing working branch.
