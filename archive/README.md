# Archive

Code from sessions 01-25, plus one-off scripts. **Nothing here is imported by the active
pipeline** -- verified by grepping every `spec_from_file_location` edge in `26_*` through
`50_*`: not one references a file numbered below 26.

`utils.py` is here for a second reason. It was imported by nothing, and it shares a filename
with HARP-Code's `utils.py`, which `50_harp_reeval_prompt_level.py` and
`audit_split_protocols.py` put on `sys.path`. Keeping an unused shadow of a module we
deliberately import from elsewhere was a latent hazard.

Moving anything back to the repo root is safe; moving an active file *in here* is not.
