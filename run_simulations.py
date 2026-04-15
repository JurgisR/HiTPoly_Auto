from __future__ import annotations

import os
import time
import datetime
import json
import uuid
from sys import stdout
import numpy as np
import sys

htvs_path = '/home/jurgis/htvs'
import django
if not htvs_path in sys.path:
    sys.path.append(htvs_path)
if not f'{htvs_path}/djangochem/' in sys.path:
    sys.path.append(f'{htvs_path}/djangochem/')
os.environ["DJANGO_SETTINGS_MODULE"] = "djangochem.settings.orgel"
django.setup()
os.environ["DJANGO_ALLOW_ASYNC_UNSAFE"] = "true"

from pgmols.models import Species, Geom, Group, Calc, SpeciesSet, GeomSet
from jobs.models import Job
from jobs import workflowmanager

from molgen.utils import put_block
from blocks.block import Block
from rdkit import Chem


def get_smiles_reorder_indices(original_smiles: str) -> list[int]:
    """
    For a multi-component SMILES string, returns a list of indices such that:
      Chem.MolToSmiles(Chem.MolFromSmiles(original_smiles)).split('.') ==
        [original_smiles.split('.')[i] for i in reorder_indices]

    If there is only one component, returns [0].
    """
    components = original_smiles.split('.')
    if len(components) == 1:
        return [0]
    canonicalized_input = [Chem.MolToSmiles(Chem.MolFromSmiles(smi), canonical=True) for smi in components]
    merged_canon = Chem.MolToSmiles(Chem.MolFromSmiles(original_smiles), canonical=True)
    output_components = merged_canon.split('.')
    reorder_indices = []
    for out_smi in output_components:
        try:
            idx = canonicalized_input.index(out_smi)
        except ValueError:
            idx = -1
        reorder_indices.append(idx)
    return reorder_indices


def run_md_simulation(parameters: dict) -> str:
    """Submit MD jobs via the HTVS workflow stack.

    Parameters
    ----------
    parameters : dict
        Must contain:
        - ``parameter_sets`` (list[dict]): one dict per run, each with:
            - ``molecule_smiles`` (str): dot-separated component SMILES.
            - ``salt_type`` (str): e.g. ``"Li.PF6"`` / ``"Na.TFSI"``.
            - ``molality`` (float): salt molality (mol/kg solvent).
            - ``system_type`` (str): ``"liquid"``, ``"polymer"``, or ``"gel"``.
            - ``ratio`` (list[float] | str): component ratio aligned with SMILES order,
              or ``":"``-separated floats (e.g. ``"1:4:2"``).
            - ``ratio_type`` (str): ``"mol_ratio"`` or ``"mass_ratio"``.
            - ``temperature`` (int, optional): default 300.
            - ``simulation_length`` (int, optional): default 100.
            - ``charge_scaling`` (float, optional): default 0.7.
        - ``run_id`` (str, optional): identifier used for the output JSON filename.
          Auto-generated (UUID4) if not provided.
        - ``project_name`` (str, optional): project label written into the JSON record.
          Falls back to the ``PROJECT_NAME`` env var, then ``"lipoly"``.

    Results are saved to ``<MD_RESULTS_DIR>/<run_id>.json``.

    Returns
    -------
    str
        ``"MD simulation submitted for N systems."`` on success, or an error string
        starting with ``"[run_md_error]"`` describing what failed.
    """
    run_id_for_log = parameters.get("run_id") if isinstance(parameters, dict) else "<missing-run-id>"
    print(f"Starting the MD simulation initialization for run_id: {run_id_for_log}")
    try:
        dry_run_enabled = os.environ.get("DRY_RUN_MD_SIMULATION", "false").strip().lower() in {
            "1", "true", "yes", "on",
        }
        if dry_run_enabled:
            print("[DRY_RUN_MD_SIMULATION] run_md_simulation received parameters:")
            try:
                payload_text = json.dumps(parameters, ensure_ascii=False, indent=2)
                print("[DRY_RUN_MD_SIMULATION_PAYLOAD_START]")
                print(payload_text)
                print("[DRY_RUN_MD_SIMULATION_PAYLOAD_END]")
            except Exception:
                print(str(parameters))
            count = (
                len(parameters.get("parameter_sets", []))
                if isinstance(parameters, dict) and isinstance(parameters.get("parameter_sets"), list)
                else 0
            )
            return f"[dry_run] Skipped MD submission. Captured run_md_simulation payload for {count} parameter set(s)."

        if not isinstance(parameters, dict):
            return "[run_md_error] invalid_input: expected a dict with keys: 'parameter_sets' and optional 'run_id'."

        run_id = parameters.get("run_id") or str(uuid.uuid4())
        project_name = (
            parameters.get("project_name")
            or os.environ.get("PROJECT_NAME", "lipoly")
        )

        maybe_list = parameters.get("parameter_sets")
        if not isinstance(maybe_list, list) or not maybe_list or not all(isinstance(x, dict) for x in maybe_list):
            return "[run_md_error] invalid_parameter_sets: expected non-empty list[dict] at parameters['parameter_sets']."
        parameter_sets = maybe_list

        required_keys = {"molecule_smiles", "salt_type", "molality", "system_type", "ratio", "ratio_type"}
        for idx, ps in enumerate(parameter_sets):
            missing = [k for k in required_keys if k not in ps]
            if missing:
                return f"[run_md_error] invalid_param_set[{idx}]: missing keys: {', '.join(missing)}"

        # Build and persist selected_system entries
        selected_entries: list[dict] = []
        for idx, params in enumerate(parameter_sets):
            try:
                smiles_raw = params["molecule_smiles"]
                reorder = get_smiles_reorder_indices(smiles_raw)
                if isinstance(params["ratio"], list):
                    ratio_vals = params["ratio"]
                elif isinstance(params["ratio"], str):
                    ratio_vals = [float(x) for x in params["ratio"].split(":")]
                else:
                    return f"[run_md_error] invalid_param_set[{idx}]: ratio must be list[float] or ':'-separated string."

                ratio_reordered = np.array(ratio_vals).astype(float)[reorder].tolist()
                smiles_canon = Chem.MolToSmiles(Chem.MolFromSmiles(smiles_raw))

                selected_entries.append({
                    "selected_system": {
                        "smiles": smiles_canon,
                        "salt_type": params["salt_type"],
                        "molality": params["molality"],
                        "system_type": params["system_type"],
                        "ratio": ratio_reordered,
                        "ratio_type": params["ratio_type"],
                        "createtime": str(datetime.datetime.now()),
                        "project": project_name,
                    }
                })
            except Exception as exc:
                return f"[run_md_error] build_selected_system_failed[{idx}]: {exc}"

        try:
            data_dir = os.environ.get("MD_RESULTS_DIR")
            print(f"MD results directory: {data_dir}")
            out_path = os.path.join(data_dir, f"{run_id}.json")
            existing: list = []
            if os.path.exists(out_path):
                with open(out_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    if isinstance(loaded, list):
                        existing = loaded
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(existing + selected_entries, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            return f"[run_md_error] persist_failed: {exc}"

        # Submit MD jobs
        cluster_path = os.environ.get("CLUSTERPATH", "/home/jurgis/SUPERCLOUD")
        hitpoly_path = os.environ.get("HITPOLY", "$HOME/HiTPoly")
        htvs_env = os.environ.get("HTVSENV", "htvs")

        wf = workflowmanager.WorkFlow(
            root_dir="/home/jurgis",
            cluster=cluster_path,
            project_name="lipoly",
        )

        submitted = 0
        total = len(parameter_sets)
        for idx, params in enumerate(parameter_sets):
            try:
                smiles = params["molecule_smiles"]
                reorder = get_smiles_reorder_indices(smiles)
                if isinstance(params["ratio"], list):
                    ratio = params["ratio"]
                elif isinstance(params["ratio"], str):
                    ratio = [float(x) for x in params["ratio"].split(":")]
                else:
                    return f"[run_md_error] invalid_param_set[{idx}]: ratio must be list[float] or ':'-separated string."

                ratio = np.array(ratio).astype(float)[reorder].tolist()
                smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))

                put_block(Block(smiles), project="lipoly", tags=["run"])
                species = Species.objects.get(smiles=smiles, group__name='lipoly')

                params.setdefault("temperature", 300)
                params.setdefault("simulation_length", 100)
                params.setdefault("charge_scaling", 0.7)

                details = {
                    "compute_platform": "supercloud",
                    "ratio": ratio,
                    "ratio_type": params["ratio_type"],
                    "system": params["system_type"],
                    "salt_type": params["salt_type"],
                    "molality": params["molality"],
                    "temperature": params["temperature"],
                    "simu_length": params["simulation_length"],
                    "charge_scale": params["charge_scaling"],
                    "run_id": run_id,
                    "project": project_name,
                    "hitpoly": hitpoly_path,
                    "hitpoly_env": htvs_env,
                }

                wf.run(
                    "requestjobs",
                    {
                        "project": "lipoly",
                        "request_config": "openmm_qLPG_ionic_cond_multi_system",
                        "inchikeys": [species.inchikey],
                        "details": details,
                        "settings": "djangochem.settings.orgel",
                        "requester": "jurgis",
                    },
                )
                wf.run(
                    "buildjobs",
                    {
                        "project": "lipoly",
                        "config": "openmm_qLPG_ionic_cond_multi_system",
                        "settings": "djangochem.settings.orgel",
                        "requester": "jurgis",
                    },
                )

                submitted += 1
            except Exception as exc:
                return (
                    f"[run_md_error] submit_failed: run_id={run_id} submitted={submitted}/{total} "
                    f"failed_index={idx} error={exc}"
                )

        return f"MD simulation submitted for {submitted} systems."
    except Exception as exc:
        return f"[run_md_error] unexpected: {exc}"
