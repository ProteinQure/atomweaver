#!/usr/bin/env python3
"""AtomWeaver one-shot design wrapper.

Run the frozen production AtomWeaver model on your OWN peptide-backbone + target PDBs and read off
residue identity with the shipped ``b2_balanced`` hybrid discretizer -- in a single command:

    python scripts/joint_diffusion/design.py --pdb-dir INPUTS --out OUTDIR

This is a THIN ORCHESTRATOR. It does not reimplement sampling or read-out; it shells out to the two
already-validated scripts (with every production knob baked in as a default) and then converts their
output into user-friendly files:

  1. ``sample.py`` (side-chain cloud sampling + ``--export-clouds``; JOINT default, optional ``--design-positions``),
     wrapped in the exact production environment block + flags (var0.25 read-out recipe).
  2. ``apply_hybrid_readout.py --preset b2_balanced`` on the exported clouds (CPU), with the Learned
     head selected by ``--vocab`` (default: the general full300 head + the clean-300 eval DB, which
     doubles as the NDM reference DB).
  3. Post-processing of the resulting ``preds.json`` into ``designs.fasta`` + ``designs.csv``.

Everything the model needs is bundled under ``data/`` and ``weights/``; the defaults below
point at the bundled checkpoint, DBs, head, and read-out cache. All are overridable.

Outputs written under OUTDIR:
  * ``clouds/`` -- per-design atom-cloud PDBs (raw model output)
  * ``preds.json`` -- the hybrid read-out distributions (meta / classes / designs[])
  * ``designs.fasta`` -- one record per (input structure, sample); see FASTA convention below
  * ``designs.csv`` -- per-position top-3 identities + probabilities

FASTA convention
----------------
One record per input structure PER design sample. Header = the exported cloud stem, which is the
input structure name plus the ``_s<idx>`` sample suffix
(e.g. ``>9RA5_MK8_s0 len=20``). The sequence is the per-site argmax identity:

  * a CANONICAL residue is written as its single-letter code (e.g. ALA -> ``A``);
  * a NON-CANONICAL residue (NCAA) is written as its CCD/PQ code in square brackets
    (e.g. DAL -> ``[DAL]``), so ``A[DAL]C...`` stays human-readable.

Composition control -- ``--vocab`` (default ``full300``):
  * ``full300`` -- the shipped 300-type NCAA-open vocabulary (the default).
  * ``exp450``  -- the expanded 450-type NCAA-open vocabulary. It is NOT a superset of full300.
  * ``canon20`` -- the 20 canonical amino acids only, so NO NCAA can be called (every argmax is
    canonical).

``--head`` still accepts an arbitrary refit bundle and overrides ``--vocab`` entirely -- library
included, so it must be paired with an explicit ``--eval-db``.
"""

from __future__ import annotations

import csv
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(add_completion=False, help=__doc__)

# --------------------------------------------------------------------------------------------------
# Baked production defaults. All exposed as overridable opts.
# Assets are bundled in this repo (paths relative to the repo root, computed from __file__), EXCEPT
# the frozen checkpoint, which is a GitHub Release asset the user downloads into ``weights/`` (see
# README). Every path below is overridable via its flag.
# --------------------------------------------------------------------------------------------------
# repo root = <checkout>/scripts/joint_diffusion/design.py -> parents[2]
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEF_REPO = str(_REPO_ROOT)

DEF_CHECKPOINT = str(_REPO_ROOT / "weights" / "atomweaver.pt")
DEF_EVAL_DB = str(_REPO_ROOT / "data" / "reference_library.pt")
DEF_SAMPLING_DB = str(_REPO_ROOT / "data" / "sampling_library.pt")
VOCABS = {
    "canon20": {"head": str(_REPO_ROOT / "data" / "readout_head_canon20.joblib"), "ref_db": None},
    "full300": {
        "head": str(_REPO_ROOT / "data" / "readout_head_full300.joblib"),
        "ref_db": str(_REPO_ROOT / "data" / "reference_library.pt"),
    },
    "exp450": {
        "head": str(_REPO_ROOT / "data" / "readout_head_exp450.joblib"),
        "ref_db": str(_REPO_ROOT / "data" / "expanded450_library.pt"),
    },
}
DEF_VOCAB = "full300"
DEF_PRESET = "b2_balanced"

EVAL_SCRIPT = "scripts/joint_diffusion/sample.py"
READOUT_SCRIPT = "scripts/joint_diffusion/apply_hybrid_readout.py"

# 20 canonical CCD -> one-letter (everything else is treated as an NCAA and bracketed in the FASTA).
CANON3TO1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


def _fmt_num(x: float) -> str:
    """Render a float without a trailing ``.0`` when it is integral (byte-match the guide command)."""
    xf = float(x)
    return str(int(xf)) if xf.is_integer() else repr(xf)


def _fasta_token(code: str) -> str:
    """Canonical -> 1-letter; NCAA -> bracketed CCD code."""
    return CANON3TO1.get(code.upper(), f"[{code}]")


def _parse_name(name: str) -> tuple[str, int]:
    """Split an exported cloud stem into (input-structure stem, sample index)."""
    m = re.match(r"^(?P<inp>.+)_s(?P<s>\d+)$", name)
    if m:
        return m.group("inp"), int(m.group("s"))
    return name, 0


# --------------------------------------------------------------------------------------------------
# command builders (return argv lists + an env-override dict; rendered to a shell string for dry-run)
# --------------------------------------------------------------------------------------------------
def build_sampling(
    *,
    pdb_dir: str,
    out: Path,
    gpu: int,
    checkpoint: str,
    sampling_db: str,
    repo: str,
    num_samples: int,
    num_steps: int,
    design_positions: str = "",
) -> tuple[dict, list[str]]:
    env = {
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "ATOMWEAVER_ELEMENT_VOCAB": "5",
        "ATOMWEAVER_SHELL_VAR_SCALE": "0.25",
        "ATOMWEAVER_DISC_REPACK": "1",
        "ATOMWEAVER_SAMPLE_RECYCLES": "1",
        "PYTHONPATH": repo,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "CUDA_VISIBLE_DEVICES": str(gpu),
    }
    argv = [
        sys.executable,
        EVAL_SCRIPT,
        "--pdb-dir",
        pdb_dir,
        "--checkpoint",
        checkpoint,
        "--sampling-db",
        sampling_db,
        "--reserved-slot0",
        "--reserved-slot0-prefix-exempt",
        "--repo",
        repo,
        "--num-steps",
        str(num_steps),
        "--num-samples",
        str(num_samples),
        "--device",
        "cuda",
        "--export-clouds",
        str(out / "clouds"),
    ]
    if design_positions:
        argv += ["--design-positions", design_positions]
    return env, argv


def build_readout(
    *, head: Optional[str], vocab: str, eval_db: str, out: Path, preset: str, repo: str
) -> tuple[dict, list[str]]:
    # apply_hybrid_readout.py is CPU-only (it setdefaults CUDA_VISIBLE_DEVICES=""); we pin it empty.
    # PYTHONPATH=repo so the bundled ``atomweaver`` package resolves from the checkout (matches build_sampling);
    # without it a plain-install run would import a differently-installed atomweaver.
    env = {"CUDA_VISIBLE_DEVICES": "", "ATOMWEAVER_ELEMENT_VOCAB": "5", "PYTHONPATH": repo}
    argv = [
        sys.executable,
        READOUT_SCRIPT,
        *(["--head", head] if head is not None else ["--vocab", vocab]),
        "--ref-db",
        eval_db,
        "--clouds",
        str(out / "clouds"),
        "--preset",
        preset,
        "--out",
        str(out / "preds.json"),
    ]
    return env, argv


def render(env: dict, argv: list[str], repo: str) -> str:
    """Render an env + argv into a copy-pasteable shell string (matches the guide's layout)."""
    envstr = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())
    return f"cd {shlex.quote(repo)} && env {envstr} " + " ".join(shlex.quote(a) for a in argv)


def run(env: dict, argv: list[str], repo: str) -> None:
    full_env = {**os.environ, **env}
    typer.echo(f"[design] $ {render(env, argv, repo)}", err=True)
    subprocess.run(argv, cwd=repo, env=full_env, check=True)


# --------------------------------------------------------------------------------------------------
# post-processing: preds.json -> designs.fasta + designs.csv
# --------------------------------------------------------------------------------------------------
def postprocess(preds_path: Path, out: Path) -> tuple[int, int]:
    """Read the hybrid ``preds.json`` and write designs.fasta + designs.csv. Returns (n_inputs, n_designs)."""
    import json

    payload = json.loads(Path(preds_path).read_text())
    classes = payload["classes"]
    designs = payload["designs"]

    fasta_path = out / "designs.fasta"
    csv_path = out / "designs.csv"
    inputs: set[str] = set()

    with open(fasta_path, "w") as fa, open(csv_path, "w", newline="") as fc:
        cw = csv.writer(fc)
        cw.writerow(
            [
                "input",
                "sample",
                "position",
                "top1_code",
                "top1_prob",
                "top2_code",
                "top2_prob",
                "top3_code",
                "top3_prob",
            ]
        )
        for d in designs:
            name = d["name"]
            inp, samp = _parse_name(name)
            inputs.add(inp)
            argmax = d["argmax"]
            probs = d["probs"]
            L = d.get("length", len(argmax))

            seq = "".join(_fasta_token(c) for c in argmax)
            fa.write(f">{name} len={L}\n{seq}\n")

            for pos, prow in enumerate(probs, start=1):
                # rank the row without numpy (keep the post-processor dependency-light)
                ranked = sorted(range(len(prow)), key=lambda j: prow[j], reverse=True)[:3]
                cells = [inp, samp, pos]
                for t in range(3):
                    if t < len(ranked):
                        j = ranked[t]
                        cells += [classes[j], f"{prow[j]:.6f}"]
                    else:
                        cells += ["", ""]
                cw.writerow(cells)

    return len(inputs), len(designs)


# --------------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------------
@app.command()
def main(
    out: str = typer.Option(..., "--out", help="Output directory (created if missing)."),
    pdb_dir: Optional[str] = typer.Option(
        None, "--pdb-dir", help="Directory of input PDBs (peptide backbone + target). Required unless --from-preds."
    ),
    num_samples: int = typer.Option(5, "--num-samples", help="Designs sampled per input structure."),
    vocab: str = typer.Option(
        DEF_VOCAB,
        "--vocab",
        help=f"Candidate vocabulary -- picks the ship head. One of {list(VOCABS)} (default {DEF_VOCAB}). "
        "canon20 excludes every NCAA; full300 and exp450 are NCAA-open and overlap but are NOT nested.",
    ),
    gpu: int = typer.Option(0, "--gpu", help="GPU index for sampling (paired with CUDA_DEVICE_ORDER=PCI_BUS_ID)."),
    # --- overridable production paths / knobs ---
    checkpoint: str = typer.Option(DEF_CHECKPOINT, "--checkpoint"),
    eval_db: Optional[str] = typer.Option(
        None,
        "--eval-db",
        help="Read-out vocab DB; also the hybrid NDM --ref-db. Defaults to the --vocab library, "
        f"else {Path(DEF_EVAL_DB).name}.",
    ),
    sampling_db: str = typer.Option(DEF_SAMPLING_DB, "--sampling-db", help="Model-load DB (not used for scoring)."),
    head: Optional[str] = typer.Option(
        None, "--head", help="Learned head (.joblib). Overrides --vocab; defaults to the --vocab ship head."
    ),
    repo: str = typer.Option(DEF_REPO, "--repo", help="atomweaver checkout root."),
    preset: str = typer.Option(DEF_PRESET, "--preset", help="Hybrid read-out preset."),
    num_steps: int = typer.Option(250, "--num-steps"),
    design_positions: str = typer.Option(
        "",
        "--design-positions",
        help="Comma-separated 1-based positions to design, holding the rest fixed (e.g. '3,7,10'). "
        "Default empty = JOINT: design ALL peptide side chains together (the primary mode).",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the sampling + read-out commands and exit."),
    from_preds: Optional[str] = typer.Option(
        None, "--from-preds", help="Skip sampling+read-out; only post-process this existing preds.json into OUTDIR."
    ),
):
    """Sample designs for every PDB in --pdb-dir, read off identity, and write designs.fasta + designs.csv."""
    if num_steps < 2:
        raise typer.BadParameter("--num-steps must be >= 2 (single-step sampling is not supported).")
    outp = Path(out)

    # ---- resolve the candidate vocabulary -> (head, eval-db) ----
    if vocab not in VOCABS:
        raise typer.BadParameter(f"--vocab must be one of {list(VOCABS)}")

    if head is None:
        ship_head = VOCABS[vocab]["head"]
        if not Path(ship_head).is_file():
            raise typer.BadParameter(
                f"--vocab {vocab} expects the ship head at {ship_head}, which is not present. "
                "Pass --head explicitly, or install that head under data/."
            )
        if vocab == "canon20":
            typer.echo(f"[design] --vocab canon20: canon-20 ship head ({ship_head}); NCAA candidates excluded.")
        else:
            typer.echo(f"[design] --vocab {vocab}: {ship_head} (NCAA-open).")
    elif vocab != DEF_VOCAB:
        typer.echo(f"[design] NOTE: explicit --head given; --vocab {vocab} head-swap ignored (using {head}).")

    # The NDM reference library must cover the head's classes, and full300/exp450 are NOT nested, so
    # each vocabulary pins the library its head was fit against.
    if head is not None and eval_db is None:
        raise typer.BadParameter(
            "--eval-db is required with --head: a custom head must name the reference library it was "
            "fit against (any class missing from it can never be called). Pass --eval-db, or drop "
            "--head and use --vocab to get a matched head/library pair."
        )
    if eval_db is None:
        eval_db = VOCABS[vocab]["ref_db"] or DEF_EVAL_DB

    # ---- post-process-only path (no GPU) ----
    if from_preds:
        outp.mkdir(parents=True, exist_ok=True)
        n_in, n_des = postprocess(Path(from_preds), outp)
        typer.echo(f"[design] post-processed {from_preds}: {n_in} inputs, {n_des} designs")
        typer.echo(f"[design]   {outp / 'designs.fasta'}")
        typer.echo(f"[design]   {outp / 'designs.csv'}")
        return

    if not pdb_dir:
        raise typer.BadParameter("--pdb-dir is required (unless --from-preds is given)")

    samp_env, samp_argv = build_sampling(
        pdb_dir=pdb_dir,
        out=outp,
        gpu=gpu,
        checkpoint=checkpoint,
        sampling_db=sampling_db,
        repo=repo,
        num_samples=num_samples,
        num_steps=num_steps,
        design_positions=design_positions,
    )
    read_env, read_argv = build_readout(head=head, vocab=vocab, eval_db=eval_db, out=outp, preset=preset, repo=repo)

    if dry_run:
        typer.echo("# --- 1. SAMPLING (GPU) ---")
        typer.echo(render(samp_env, samp_argv, repo))
        typer.echo("")
        typer.echo("# --- 2. HYBRID READ-OUT (CPU) ---")
        typer.echo(render(read_env, read_argv, repo))
        typer.echo("")
        typer.echo(f"# --- 3. POST-PROCESS: {outp / 'preds.json'} -> {outp / 'designs.fasta'} + {outp / 'designs.csv'}")
        return

    outp.mkdir(parents=True, exist_ok=True)

    # 1. sample (GPU)
    run(samp_env, samp_argv, repo)
    clouds = outp / "clouds"
    n_clouds = len(list(clouds.glob("*.pdb"))) if clouds.is_dir() else 0
    if n_clouds == 0:
        typer.secho(f"[design] ERROR: no clouds exported under {clouds}; aborting.", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    # 2. hybrid read-out (CPU)
    run(read_env, read_argv, repo)

    # 3. post-process
    preds = outp / "preds.json"
    n_in, n_des = postprocess(preds, outp)

    typer.echo("")
    typer.echo("[design] done.")
    typer.echo(f"[design]   inputs designed : {n_in}")
    typer.echo(f"[design]   total designs   : {n_des}  ({num_samples} samples/input)")
    typer.echo(f"[design]   clouds          : {clouds}")
    typer.echo(f"[design]   preds JSON      : {preds}")
    typer.echo(f"[design]   FASTA           : {outp / 'designs.fasta'}")
    typer.echo(f"[design]   CSV             : {outp / 'designs.csv'}")


if __name__ == "__main__":
    app()
