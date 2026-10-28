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
