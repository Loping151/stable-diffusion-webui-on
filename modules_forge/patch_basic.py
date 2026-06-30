import torch
import os
import time
import httpx
import warnings
import gradio.networking
import safetensors.torch

from pathlib import Path
from tqdm import tqdm


def gradio_url_ok_fix(url: str) -> bool:
    try:
        for _ in range(5):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                r = httpx.head(url, timeout=999, verify=False)
            if r.status_code in (200, 401, 302):
                return True
            time.sleep(0.500)
    except (ConnectionError, httpx.ConnectError):
        return False
    return False


def build_loaded(module, loader_name):
    original_loader_name = loader_name + '_origin'

    if not hasattr(module, original_loader_name):
        setattr(module, original_loader_name, getattr(module, loader_name))

    original_loader = getattr(module, original_loader_name)

    def loader(*args, **kwargs):
        result = None
        try:
            result = original_loader(*args, **kwargs)
        except Exception as e:
            result = None
            exp = str(e) + '\n'
            for path in list(args) + list(kwargs.values()):
                if isinstance(path, str):
                    if os.path.exists(path):
                        exp += f'File corrupted: {path} \n'
                        corrupted_backup_file = path + '.corrupted'
                        if os.path.exists(corrupted_backup_file):
                            os.remove(corrupted_backup_file)
                        os.replace(path, corrupted_backup_file)
                        if os.path.exists(path):
                            os.remove(path)
                        exp += f'Forge has tried to move the corrupted file to {corrupted_backup_file} \n'
                        exp += f'You may try again now and Forge will download models again. \n'
            raise ValueError(exp)
        return result

    setattr(module, loader_name, loader)
    return


def _sanitize_json_schema(node):
    # Replace boolean sub-schemas (e.g. `additionalProperties: true`, `properties.x: false`)
    # with an empty dict ("any"). gradio_client's walker assumes every node is a dict, and a
    # boolean node makes it raise "argument of type 'bool' is not iterable" / "'bool' has no
    # attribute 'get'". Booleans that are NOT schemas (default/exclusiveMinimum/...) are left
    # untouched because the walker never recurses into those keys.
    if isinstance(node, dict):
        out = {}
        for key, value in node.items():
            if key in ("additionalProperties", "items", "additionalItems") and isinstance(value, bool):
                out[key] = {}
            elif key in ("properties", "$defs", "definitions", "patternProperties") and isinstance(value, dict):
                out[key] = {k: ({} if isinstance(v, bool) else _sanitize_json_schema(v)) for k, v in value.items()}
            else:
                out[key] = _sanitize_json_schema(value)
        return out
    if isinstance(node, list):
        return [_sanitize_json_schema(v) for v in node]
    return node


def patch_gradio_client_schema():
    # xwsdwebui: make gradio's --api `get_api_info()` tolerate boolean JSON schemas across
    # gradio 4.x (4.40 .. 4.44+). The public json_schema_to_python_type() is the single entry
    # gradio calls, so sanitising its input fixes every internal helper (including the local
    # get_desc closure that can't be monkeypatched directly).
    import gradio_client.utils as gc_utils

    if getattr(gc_utils, "_xw_schema_patched", False):
        return

    orig_public = gc_utils.json_schema_to_python_type

    def safe_public(schema):
        try:
            return orig_public(_sanitize_json_schema(schema))
        except Exception:
            return "Any"

    gc_utils.json_schema_to_python_type = safe_public
    gc_utils._xw_schema_patched = True


def always_show_tqdm(*args, **kwargs):
    kwargs['disable'] = False
    if 'name' in kwargs:
        del kwargs['name']
    return tqdm(*args, **kwargs)


def long_path_prefix(path: Path) -> Path:
    if os.name == 'nt' and not str(path).startswith("\\\\?\\") and not path.exists():
        return Path("\\\\?\\" + str(path))
    return path


def patch_all_basics():
    import logging
    from huggingface_hub import file_download
    file_download.tqdm = always_show_tqdm
    from transformers.dynamic_module_utils import logger
    logger.setLevel(logging.ERROR)

    from huggingface_hub.file_download import _download_to_tmp_and_move as original_download_to_tmp_and_move

    def patched_download_to_tmp_and_move(incomplete_path, destination_path, url_to_download, proxies, headers, expected_size, filename, force_download):
        incomplete_path = long_path_prefix(incomplete_path)
        destination_path = long_path_prefix(destination_path)
        return original_download_to_tmp_and_move(incomplete_path, destination_path, url_to_download, proxies, headers, expected_size, filename, force_download)

    file_download._download_to_tmp_and_move = patched_download_to_tmp_and_move

    gradio.networking.url_ok = gradio_url_ok_fix
    patch_gradio_client_schema()
    build_loaded(safetensors.torch, 'load_file')
    build_loaded(torch, 'load')
    return
