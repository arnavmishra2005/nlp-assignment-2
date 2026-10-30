"""Explicit prepare -> check -> run one phase -> inspect -> continue workflow."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys
import traceback
import zipfile

from .artifacts import ArtifactReader, TRAIN_PHASES, inspect_artifact, package_phase, restore_artifact
from .common import digest, file_hash, load_config, read_json, source_hash, write_json
from .preflight import check_environment
from .process import run_streaming

PHASES = ["prepare", "smoke", *TRAIN_PHASES, "evaluate", "finalize"]
PREVIOUS = {phase: PHASES[index-1] for index, phase in enumerate(PHASES) if index > 0 and phase not in ("evaluate", "finalize")}
PROJECT = Path(__file__).resolve().parents[1]


def manifest_at(path):
    reader = ArtifactReader(path)
    try:
        return reader.json("phase_manifest.json")
    finally:
        reader.close()


def locate_phase(phase, inputs, explicit=()):
    matches = [Path(p) for p in explicit if manifest_at(p)["phase"] == phase]
    if not matches:
        matches = sorted(Path(inputs).rglob(f"Q2_{phase}_results.zip")) if inputs and Path(inputs).exists() else []
    if not matches and inputs and Path(inputs).exists():
        matches = [p.parent for p in Path(inputs).rglob("phase_manifest.json")
                   if read_json(p).get("phase") == phase and read_json(p).get("status") == "COMPLETE"]
    matches = sorted(set(p.resolve() for p in matches))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {phase} artifact; found {len(matches)}: {matches}. Set --data/--previous explicitly.")
    return matches[0]


def make_plan(config, phase):
    runs = [{"experiment": f"{model}_seed{seed}", "model": model, "seed": seed,
             "maximum_epochs": config["baseline" if model == "baseline" else "roberta"]["epochs"]}
            for model in ("baseline", "roberta_mean", "roberta_first") for seed in config["seeds"]]
    from .distributed import topology
    return {"phase": phase, "all_full_training_runs": runs, "full_run_count": len(runs),
            "distributed_training": topology(config) if phase.startswith('roberta_') else None,
            "sum_of_maximum_epochs": sum(r["maximum_epochs"] for r in runs),
            "early_stopping": True, "performance_target": "No numerical score target specified by the assignment",
            "completion_meaning": "A verified experiment may legitimately underperform",
            "reserved_test": "Only evaluate accesses test after all six runs are fixed",
            "current_run": next((r for r in runs if r["experiment"] == phase), None)}


def setup_phase(args, config, root, payload):
    if config["seeds"] != [42, 123]:
        raise ValueError("The staged notebook names are planned for seeds [42,123]; edit the phase plan if changing seeds")
    require_gpu = args.phase not in ("prepare", "finalize")
    check_environment(payload / "environment", require_gpu, args.allow_cpu, args.min_free_gib,
                      cpu_only=args.phase in ("prepare", "finalize") and not args.allow_cpu)
    if args.phase.startswith('roberta_') and not args.allow_cpu:
        from .distributed import check_dual_gpu
        check_dual_gpu(config, payload / 'environment/dual_gpu.json')
    plan = make_plan(config, args.phase)
    receipts, sources = {}, {}
    prepare_digest = split_fingerprint = None
    if args.phase != "prepare":
        explicit = [*args.previous, *([args.data] if args.data else [])]
        source = locate_phase("prepare", args.inputs, explicit)
        receipts["prepare"] = inspect_artifact(source, "prepare", config, source_hash())
        sources["prepare"] = source
        header = manifest_at(source)
        prepare_digest, split_fingerprint = header["prepared_manifest_digest"], header["split_fingerprint"]
        prerequisites = TRAIN_PHASES if args.phase == "evaluate" else (["evaluate"] if args.phase == "finalize" else [PREVIOUS[args.phase]])
        for phase in prerequisites:
            if phase == "prepare":
                continue
            source = locate_phase(phase, args.inputs, explicit)
            receipt = inspect_artifact(source, phase, config, source_hash())
            if receipt["prepared_manifest_digest"] != prepare_digest or receipt["split_fingerprint"] != split_fingerprint:
                raise ValueError("Prerequisites used different dataset/split fingerprints")
            receipts[phase], sources[phase] = receipt, source
        if args.phase != "finalize":
            source = sources["prepare"]
            if source.is_dir():
                prepared = source / "prepared"
            else:
                prepared = root / "cache" / "prepared"
                stamp = root / "cache" / "PREPARED_RESTORED.json"
                if not stamp.exists() or read_json(stamp)["digest"] != prepare_digest:
                    restore_artifact(source, root / "cache", prefixes=["prepared/"])
                    write_json(stamp, {"digest": prepare_digest})
            plan["prepared_path"] = str(prepared)
    if args.seed is not None and plan["current_run"] and args.seed != plan["current_run"]["seed"]:
        raise ValueError("--seed disagrees with the selected phase")
    if args.resume_from:
        receipt = inspect_artifact(args.resume_from, args.phase, config, source_hash(), allow_incomplete=True)
        if args.phase != "prepare":
            recovery_header = manifest_at(args.resume_from)
            if recovery_header["prepared_manifest_digest"] != prepare_digest:
                raise ValueError("Recovery artifact uses different prepared data")
        restore_artifact(args.resume_from, payload,
                         prefixes=["runs/", "smoke/", "prepared/", "final/"])
        receipts["recovery"] = receipt
    plan["prerequisites"] = {phase: str(path) for phase, path in sources.items()}
    write_json(payload / "prerequisite_inspection.json", receipts)
    write_json(payload / "execution_plan.json", plan)
    print(json.dumps(plan, indent=2), flush=True)
    print("SETUP_CHECK_PASSED\nPHASE: " + args.phase + "\nPREREQUISITES_VERIFIED", flush=True)
    return plan, sources, prepare_digest, split_fingerprint


def launch(args):
    config = load_config(args.config)
    root = Path(args.output).resolve() / args.phase
    payload = root / "payload"
    payload.mkdir(parents=True, exist_ok=True)
    identity = {"phase": args.phase, "config_sha256": digest(config), "source_hash": source_hash()}
    execution_file = root / "execution.json"
    if execution_file.exists() and read_json(execution_file)["identity"] != identity:
        raise ValueError("Output directory belongs to different code/config; use a new --output root")
    archive = root / f"Q2_{args.phase}_results.zip"
    if archive.exists() and not args.setup_only:
        receipt = inspect_artifact(archive, args.phase, config, source_hash())
        write_json(execution_file, {"identity": identity, "status": "COMPLETE", "archive": str(archive), "inspection": receipt})
        print("VERIFIED_COMPLETED_PHASE_REUSED", archive, flush=True)
        return receipt
    write_json(execution_file, {"identity": identity, "status": "PENDING"})
    write_json(payload / "config.json", config)
    (payload / "logs").mkdir(exist_ok=True)
    prepare_digest = split_fingerprint = None
    experiments = [args.phase] if args.phase in TRAIN_PHASES else []
    if args.phase in ("evaluate", "finalize"):
        experiments = TRAIN_PHASES
    try:
        plan, sources, prepare_digest, split_fingerprint = setup_phase(args, config, root, payload)
        if args.setup_only:
            return plan
        write_json(execution_file, {"identity": identity, "status": "RUNNING", "plan": plan})
        config_path = payload / "config.json"
        print("PHASE_START", args.phase, flush=True)
        def module(name, *arguments):
            if name == 'q2.train' and args.phase.startswith('roberta_') and not args.allow_cpu:
                command = [sys.executable, '-u', '-m', 'torch.distributed.run', '--standalone',
                           '--nnodes=1', '--nproc-per-node=2', '--max-restarts=0',
                           '--module', 'q2.train_distributed', *map(str, arguments)]
            else:
                command = [sys.executable, "-u", "-m", name, *map(str, arguments)]
            run_streaming(command, payload / "logs" / "phase.log", cwd=PROJECT)
        if args.phase == "prepare":
            extra = ["--raw-data", args.data] if args.data else []
            if args.glove_path:
                extra += ["--glove-path", args.glove_path]
            module("q2.data", "--config", config_path, "--output", payload / "prepared", *extra)
            prepared = read_json(payload / "prepared/manifest.json")
            prepare_digest = digest(prepared)
            split_fingerprint = read_json(payload / "prepared/audit/audit.json")["split_fingerprint"]
        elif args.phase == "smoke":
            extra = ["--allow-cpu"] if args.allow_cpu else []
            module("q2.smoke", "--config", config_path, "--prepared", plan["prepared_path"], "--output", payload / "smoke", *extra)
        elif args.phase in TRAIN_PHASES:
            current = plan["current_run"]
            extra = ["--allow-cpu"] if args.allow_cpu else []
            if not args.resume:
                extra += ["--no-resume"]
            module("q2.train", "--config", config_path, "--prepared", plan["prepared_path"], "--output", payload / "runs" / args.phase,
                   "--model", current["model"], "--seed", current["seed"], *extra)
        elif args.phase == "evaluate":
            for phase in TRAIN_PHASES:
                restore_artifact(sources[phase], payload, prefixes=[f"runs/{phase}/"], exclude_basenames=["last.pt"])
            extra = ["--allow-cpu"] if args.allow_cpu else []
            module("q2.evaluate", "--config", config_path, "--prepared", plan["prepared_path"], "--runs", payload / "runs", "--output", payload / "final", *extra)
        elif args.phase == "finalize":
            restore_artifact(sources["evaluate"], payload, prefixes=["final/", "runs/"], exclude_basenames=["best.pt", "last.pt"])
            command = [sys.executable, "-u", str(PROJECT / "tools/package.py"), "--output", str(payload / "question_2_submission.zip"),
                       "--results", str(payload / "final"), "--runs", str(payload / "runs"), "--config", str(config_path)]
            run_streaming(command, payload / "logs/phase.log", cwd=PROJECT)
            # Verify actual extraction into a bounded scratch directory.
            check_dir = root / "cache" / "submission_check"
            check_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(payload / "question_2_submission.zip") as package:
                if package.testzip() is not None:
                    raise ValueError("Final submission ZIP integrity failure")
                for name in package.namelist():
                    target = (check_dir / name).resolve()
                    if not target.is_relative_to(check_dir.resolve()):
                        raise ValueError("Unsafe final submission path")
                package.extractall(check_dir)
            for relative in ("question_2/README.md", "question_2/config.json", "question_2/q2/models.py",
                             "question_2/results/test_summary.csv", "question_2/results/REPORT_DRAFT.md"):
                if not (check_dir / relative).exists():
                    raise ValueError("Final package missing " + relative)
            if read_json(check_dir / "question_2/config.json") != config:
                raise ValueError("Final package does not contain the executed configuration")
            shutil.copytree(check_dir / "question_2", payload / "question_2", dirs_exist_ok=True)
            submission_files = {p.relative_to(payload).as_posix(): file_hash(p)
                                for p in (payload / "question_2").rglob("*") if p.is_file()}
            write_json(payload / "FINAL_PACKAGE_CHECKED.json", {"status": "PASS", "submission_layout": "single_flat_archive",
                       "submission_files_sha256": submission_files, "built_from_zip_sha256": file_hash(payload / "question_2_submission.zip"),
                       "extracted_files_verified": True, "human_report_complete": False,
                       "remaining": "Interpret examples, edit Q2 section to <=6 pages, combine group report and executed notebooks"})
        records = {}
        for source in sources.values():
            reader = ArtifactReader(source)
            try:
                if "timing_evidence.json" in reader.names:
                    records.update(reader.json("timing_evidence.json")["records"])
                from .execution import execution_details
                for name in reader.names:
                    if name.startswith('runs/') and name.endswith('/run.json'):
                        run = reader.json(name)
                        experiment = f"{run['model']}_seed{run['seed']}"
                        if experiment in records:
                            records[experiment]['world_size'] = execution_details(run)['world_size']
            finally:
                reader.close()
        for path in (payload / "runs").glob("*/history.json") if (payload / "runs").exists() else []:
            from .execution import execution_details
            records[path.parent.name] = {"complete": (path.parent / "complete.json").exists(),
                                         "epoch_seconds": [r["epoch_seconds"] for r in read_json(path)],
                                         'world_size': execution_details(read_json(path.parent/'run.json'))['world_size']}
        from .timing import estimate_remaining
        write_json(payload / "timing_evidence.json", {"records": records})
        timing = estimate_remaining(config, records, target_world_sizes={'baseline': 1, 'roberta_mean': 2, 'roberta_first': 2})
        write_json(payload / "remaining_time_estimate.json", timing)
        print("MEASURED_RUNTIME_ESTIMATE", json.dumps(timing), flush=True)
        receipt = package_phase(payload, archive, args.phase, config, prepare_digest, split_fingerprint, source_hash(), experiments)
        write_json(execution_file, {"identity": identity, "status": "COMPLETE", "archive": str(archive), "inspection": receipt})
        if not args.keep_cache:
            cache = (root / "cache").resolve()
            if cache.exists():
                if cache.parent != root or cache.name != "cache":
                    raise ValueError("Unexpected generated cache path")
                shutil.rmtree(cache)
        print("PHASE_COMPLETE", args.phase, "\nARCHIVE:", archive, "\nINSPECTION_PASSED", flush=True)
        print("Download this phase archive and inspect it before proceeding. No claim of a numerical performance target.", flush=True)
        return receipt
    except Exception:
        error = traceback.format_exc()
        write_json(payload / "failure.json", {"phase": args.phase, "traceback": error})
        write_json(execution_file, {"identity": identity, "status": "FAILED", "traceback": error})
        try:
            failed = root / f"Q2_{args.phase}_FAILED.zip"
            package_phase(payload, failed, args.phase, config, prepare_digest, split_fingerprint, source_hash(), experiments, status="FAILED")
            print("FAILED_PHASE_RECOVERY_ARCHIVE", failed, flush=True)
        except Exception as package_error:
            print("Failure archive could not be created:", str(package_error), "; original diagnostics remain at", payload, flush=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--config", default=str(PROJECT / "config.json"))
    parser.add_argument("--data", help="prepare: optional attached raw directory; later phases: prepare results ZIP/unpacked phase directory")
    parser.add_argument("--previous", action="append", default=[], help="Explicit preceding phase ZIP/directory; repeat for evaluate")
    parser.add_argument("--inputs", default="/kaggle/input", help="Search nested attached inputs when explicit paths are omitted")
    parser.add_argument("--output", default="/kaggle/working/q2-artifacts")
    parser.add_argument("--seed", type=int, help="Optional assertion matching the phase's planned seed")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume-from", help="Own failed/successful same-phase archive for recovery")
    parser.add_argument("--glove-path", help="Optional official GloVe zip/txt for prepare")
    parser.add_argument("--setup-only", action="store_true", help="Check environment/prerequisites and print the plan; do not train")
    parser.add_argument("--allow-cpu", action="store_true", help="Developer synthetic checks only")
    parser.add_argument("--min-free-gib", type=float, default=8)
    parser.add_argument("--keep-cache", action="store_true", help="Keep generated extracted prerequisite caches after completion")
    args = parser.parse_args()
    launch(args)


if __name__ == "__main__":
    main()
