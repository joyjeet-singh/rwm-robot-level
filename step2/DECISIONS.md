# Step 2 decisions

## M0
- **ANYdrive URL unverified.** The agent found the file name but not the full URL (Isaac Lab reads the asset root at run time). Conservative choice: `fetch.zsh` tries two guessed NVIDIA public roots and records success or failure; failure is not fatal (M1 uses the PD fallback per the plan). Recommendation: M1 reads `manifest.json` first.
- **Helper package pins** (absl-py, etils, glfw, pyopengl, importlib_resources, zipp) are my knowledge, not checked against PyPI; a failed pin shows in the fetch output and the manifest check. mujoco 3.10.0 is the pin from PyPI.
- **Isaac Lab v3.0.0 flat task uses the Newton backend;** the facts above are used as the task's settings only.
- **Local mean angles** the agent gave are whole-file means, not standing angles; joint order was taken from rwm_repro/src/rwm_data.py instead.
