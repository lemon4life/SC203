# Colab workflow
1. Edit scripts and configs locally; keep notebook cells thin.
2. Run the allowed local checks and review the diff.
3. Commit the changes on the task branch.
4. Push that branch with `git push origin <branch>`.
5. Open `notebooks/colab_runner.ipynb` in Colab and mount Drive.
6. Set `BRANCH` to the branch you pushed, then run the pull cell.
7. Run the setup script; it checks dependencies before installing them.
8. Call the experiment script; outputs go to `/content/outputs` and logs to Drive.
9. Paste the log tail, full traceback, and requested listings back here.

## Kernel and four-person coordination

- Use Colab's managed Python 3 kernel; do not create a venv or Conda environment inside the Colab runtime. `colab_setup.py` runs with the active kernel's `sys.executable` and preserves its installed torch and torchvision versions. The runtime is temporary, so rerun setup after a runtime reset. [Colab runtimes are private virtual machines that can be deleted after idle time or lifetime limits](https://research.google.com/colaboratory/faq.html).
- Before GPU work, choose **Runtime → Change runtime type → Hardware accelerator → GPU**, then run the notebook's GPU/PyTorch report cell. GPU availability and types vary; proceed with inference only when CUDA reports available and `nvidia-smi` shows a GPU. [Colab documents that hardware availability varies over time](https://research.google.com/colaboratory/faq.html).
- Each teammate should use their own Colab runtime and work on a personal `feat/<name>-<task>` branch. Push changes, review/merge them through GitHub, and run shared experiments from the agreed branch/commit. The runner prints the checked-out commit hash.
- Use one shared Drive folder for the approved weights and datasets, with access for all four teammates. Each member should make the shared folder available at the expected `MyDrive/lemmepass` path or pass the actual path through script arguments. Do not keep four separate asset copies unless needed.
- `/content/outputs` is local to each runtime and is temporary; the timestamped logs are written to Drive. Before ending a run, save any result files the team needs to retain into the agreed shared results folder, without committing raw datasets, weights, checkpoints, or bulky outputs.
- Avoid simultaneous edits to one notebook copy. Keep the GitHub notebook as the canonical runner; record the branch, commit hash, seed, and run log for each reported result.
