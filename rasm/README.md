# Rasm (رسم)

Rasm is our own build of [VectorCraft](https://github.com/storytold/vectorcraft). We use it to
assess the project, tune the vector tools, add Arabic and expand the content. The code keeps
VectorCraft's internal names on purpose, so every upstream release merges in cleanly. Everything
that is ours alone lives in this folder or in files upstream does not have.

## Staying up to date

- **On GitHub:** `.github/workflows/rasm-upstream-sync.yml` checks `storytold/vectorcraft` every
  6 hours. A new release tag is merged into `main` and pushed. A conflict with our changes opens
  an issue instead, and nothing is pushed.
  It needs a repository secret `SYNC_TOKEN`: a fine-grained token for `el3rrab/rasm` with
  Contents, Workflows and Issues set to read and write.
- **On the Mac:** `update-local.sh` fast-forwards the checkout from `origin/main` and rebuilds,
  only when it is on `main` with no uncommitted changes. The launchd agent runs it every 6 hours:

  ```sh
  cp rasm/com.el3rrab.rasm.update.plist ~/Library/LaunchAgents/
  launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.el3rrab.rasm.update.plist
  ```

  Log: `~/Library/Logs/rasm-update.log`.

## Remotes

- `origin`: `el3rrab/rasm` (ours; push here)
- `upstream`: `storytold/vectorcraft` (fetch only)

VectorCraft is MIT OR Apache-2.0; keep `LICENSE-MIT`, `LICENSE-APACHE` and `NOTICE`. Its
clean-room and asset rules (see `CLAUDE.md`) apply to our changes too.
