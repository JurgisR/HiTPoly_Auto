import os
import time
from datetime import datetime, timezone
import datetime
from sys import stdout
import json

import sys

htvs_path = '/home/jurgis/htvs'
import django

if htvs_path not in sys.path:
    sys.path.append(htvs_path)
if f'{htvs_path}/djangochem/' not in sys.path:
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


def process_candidates(candidates):
    """Given a list of candidate dicts (with `selected_system`), attach results & coordination."""
    data_all = []

    # Gather information into structured dictionary
    finished_jobs = 0
    for candidate in candidates:
        species = Species.objects.get(
            group__name='lipoly',
            smiles=candidate['selected_system']['smiles'],
        )
        job = Job.objects.get(
            group__name='lipoly',
            config__name='openmm_qLPG_ionic_cond_multi_system',
            parentid=species.id,
            details__salt_type=candidate['selected_system']['salt_type'],
            details__molality=candidate['selected_system']['molality'],
            details__ratio=candidate['selected_system']['ratio'],
        )
        if job.status in ['done', 'error']:
            finished_jobs += 1
    
    if finished_jobs != len(candidates):
        print(f"Warning: {len(candidates)-finished_jobs} jobs are not finished")
        return None

    for candidate in candidates:
        species = Species.objects.get(
            group__name='lipoly',
            smiles=candidate['selected_system']['smiles'],
        )
        job = Job.objects.get(
            group__name='lipoly',
            config__name='openmm_qLPG_ionic_cond_multi_system',
            parentid=species.id,
            details__salt_type=candidate['selected_system']['salt_type'],
            details__molality=candidate['selected_system']['molality'],
            details__ratio=candidate['selected_system']['ratio'],
        )
        if job.status == 'error':
            # Keep the original candidate entry, but do not attach results/coordination.
            print(
                "Job created from ",
                species.smiles,
                " with salt ",
                candidate['selected_system']['salt_type'],
                " and molality ",
                candidate['selected_system']['molality'],
                " and ratio ",
                candidate['selected_system']['ratio'],
                " failed",
            )
            # Mark as finished-with-error so closed-loop polling can proceed.
            enriched = dict(candidate)
            enriched["results"] = {"status": "error"}
            enriched["coordination"] = {"max_coordination": None, "details": []}
            data_all.append(enriched)
            continue

        calc = job.childcalcs.all()[0]
        smiles = calc.parentjob.parent.smiles.split('.')
        salt = calc.props['info']['details']['salt_type'].split('.')
        solv_names = ["PL" + str(i + 1) for i in range(len(smiles))]

        data = {
            "selected_system": {
                "smiles": calc.parentjob.parent.smiles,
                "salt_type": calc.props["info"]["details"]["salt_type"],
                "molality": calc.props["info"]["details"]["molality"],
                "system_type": calc.props["info"]["details"]["system"],
                "ratio": calc.props["info"]["details"]["ratio"],
                "ratio_type": calc.props["info"]["details"]["ratio_type"],
                "createtime": calc.parentjob.createtime.isoformat(),
            },
            "results": {
                "ionic_conductivity_S_per_cm": calc.props["cNEcond"],
                "transfer_number": calc.props["cNEtransference"],
                "diffusivities": [],
                "fraction_free_ions": (
                    calc.props['clustercounts'].get('freeanions_carriers', 0)
                    + calc.props['clustercounts'].get('freecations_carriers', 0)
                    + calc.props['clustercounts'].get('negativeclusters_carriers', 0)
                    + calc.props['clustercounts'].get('positiveclusters_carriers', 0)
                ),
            },
            "coordination": {
                "max_coordination": None,
                "details": [],
            },
        }

        # Add diffusivity results
        for name, name_real in zip(
            solv_names + ['CA1', 'AN1'],
            smiles+[salt[0], salt[1]],
        ):
            for key in calc.props['diffusivity'].keys():
                if name in key and 'loglog' not in key:
                    data["results"]["diffusivities"].append(
                        {
                            "species": name_real,
                            "diffusivity_cm2_per_s": calc.props["diffusivity"][key],
                            "loglog_slope": calc.props["diffusivity"].get(
                                key + '_loglog'
                            ),
                        }
                    )

        # Max coordination key and its value
        uniq_coord_dict = calc.props['coordinationnumber']
        max_key = max(
            uniq_coord_dict, key=lambda k: uniq_coord_dict[k]['frequency']
        )
        data["coordination"]["max_coordination"] = {
            "cation": salt[0],
            "atoms": max_key,
        }

        # Coordination contributions of each species to the max_coordination
        for name, name_real in zip(
            solv_names + ['AN1'],
            smiles+[salt[1]],
        ):
            coord_contrib = 0
            for key in calc.props['coordinationnumber'][max_key].keys():
                if name in key:
                    coord_contrib += calc.props['coordinationnumber'][max_key][key]
            data["coordination"]["details"].append(
                {
                    "species": name_real,
                    "atoms_contributed": coord_contrib,
                }
            )

        data_all.append(data)

    return data_all


def main(input_path, output_path=None):
    """Read input JSON, enrich entries with results, and write back JSON."""
    with open(input_path, 'r') as f:
        candidates = json.load(f)

    data_all = process_candidates(candidates)
    # If not all jobs are finished yet, do not overwrite the input JSON.
    if data_all is None:
        print("Not all jobs are finished yet; skipping JSON write.")
        return None

    # Default: overwrite the input file if no explicit output is given
    if output_path is None:
        output_path = input_path
    print("Saving results to ", output_path)

    with open(output_path, 'w') as f:
        json.dump(data_all, f, indent=2)
    return output_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python db_parser.py <input_json_path> [output_json_path]\n"
            "If output_json_path is omitted, the input file will be overwritten."
        )
        sys.exit(1)

    in_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) >= 3 else None
    main(in_path, out_path)
