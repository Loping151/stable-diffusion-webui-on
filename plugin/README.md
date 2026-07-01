# plugin/ — companion extensions (opt-in)

Third-party WebUI extensions that carry a small `-on`-specific patch (usually to work with a
newly added architecture). They are **bundled but not loaded**: the launcher only scans
`extensions/`, so nothing here runs until you install it yourself.

**Install** — copy or symlink the one you want into `extensions/`, then restart:

```bash
# copy
cp -r plugin/sd-webui-regional-prompter extensions/
# or symlink (keeps it updated with the repo)
ln -s "$(pwd)/plugin/sd-webui-regional-prompter" extensions/sd-webui-regional-prompter
```

**Uninstall** — remove it from `extensions/` (the copy under `plugin/` is untouched).

> These are pinned, patched snapshots — they won't auto-update with upstream. Reapply the patch
> if you later replace one with a fresh upstream clone.

---

## sd-webui-regional-prompter

- Upstream: <https://github.com/hako-mikan/sd-webui-regional-prompter> @ `be6234522` (2025-06-23)
- License: AGPL-3.0 (see `sd-webui-regional-prompter/LICENCE`)
- `-on` patch: `scripts/rp.py` — Regional Prompter's token accounting assumes a CLIP
  `text_processing_engine` (SD1.5/SDXL). Architectures with a different text encoder (e.g.
  Anima's Qwen3) have none, so it used to crash with
  `AttributeError: '<engine>' object has no attribute 'text_processing_engine'`. The patch makes
  `tokendealer` skip Regional Prompter cleanly on such models instead of crashing (RP can't apply
  to them anyway).

---

# plugin/ — 附带扩展(默认不启用)

带有 `-on` 专属小补丁(通常是为了兼容新接入架构)的第三方 WebUI 扩展。**随库附带但不加载**:
启动器只扫描 `extensions/`,放在这里的东西在你手动安装前不会运行。

**安装** —— 把要用的复制或软链接到 `extensions/`,然后重启:

```bash
# 复制
cp -r plugin/sd-webui-regional-prompter extensions/
# 或软链接(跟随本仓库更新)
ln -s "$(pwd)/plugin/sd-webui-regional-prompter" extensions/sd-webui-regional-prompter
```

**卸载** —— 从 `extensions/` 删掉即可(`plugin/` 下的副本不受影响)。

> 这些是打过补丁的固定快照,不会随上游自动更新。若日后用上游最新版替换,需要重新打补丁。
