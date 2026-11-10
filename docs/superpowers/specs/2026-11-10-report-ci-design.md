# Automated report build

User requests report CI informed by EE5217 and hw2_introAI. EE5217's workflow
compiles XeLaTeX and uploads PDF/log artifacts; second repo is inaccessible(404).
Use its compile/artifact pattern with our own implementation, without copying
course-specific text/template or enabling public deployment.

Keep docs/REPORT_DRAFT.md as the authored source. Generate an evidence appendix
from accepted aggregate JSON and producer ablation CSV. Pandoc renders Markdown
to Unicode LaTeX; Tectonic compiles PDF. Pin both official tools/download hashes
for Windows/Linux. Portable tools live in ignored runs; no training-env changes.
Build script has prepare-only mode, records input hashes/tool versions/Git SHA,
and labels all output draft/validation. No model/data Hub access or freeze changes.

CI triggers for relevant report changes on push/PR and manual dispatch, builds
PDF, uploads PDF/provenance and diagnostic logs. Contents read permission;
no release/Pages publishing. Existing GitHub account startup blocker is separate.
Validate offline generation, evidence error handling, portable download checksum
guards, actual local PDF compilation and visual/text inspection.
