# tests/test_phase4_plug_and_play.py
import json
import os
import sys
import tempfile
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))
sys.path.insert(0, str(ROOT_DIR / "tests" / "fixtures"))

from src.schema import TreeSchema, CellSchema, PortSchema
from src.harvester import IntelligentHarvester
from src.cli import cmd_harvest, cmd_compile, cmd_validate
from lattice import LatticeOrchestrator
from router import LatticeRouter
from unification import UnificationGate, ExecutionContext
from gevr_sandbox import GEVRSandbox


def test_single_file_per_domain_structure():
    """Verify that trees/ contains strictly 1 unified JSON file per domain and no scattered seeds."""
    trees_dir = ROOT_DIR / "trees"
    assert trees_dir.exists(), "trees/ directory must exist"

    # Assert no legacy directories or scattered files
    assert not (ROOT_DIR / "seeds").exists(), "seeds/ directory must be deleted in Phase 4"
    assert not (ROOT_DIR / "harvests").exists(), "harvests/ directory must be deleted in Phase 4"

    legacy_tree_files = list(trees_dir.glob("*_tree.json"))
    assert len(legacy_tree_files) == 0, f"Found legacy *_tree.json files in trees/: {legacy_tree_files}"

    domain_files = list(trees_dir.glob("*.json"))
    assert len(domain_files) >= 5, f"Expected at least 5 consolidated domain JSON files, got {len(domain_files)}"

    expected_domains = {"pandas", "cv2", "sklearn", "matplotlib", "numpy", "scipy", "python_core"}
    found_domains = {f.stem for f in domain_files}
    assert expected_domains.issubset(found_domains), f"Missing required domain JSON files: {expected_domains - found_domains}"

    # Validate each domain file against TreeSchema
    total_cells = 0
    total_seeds = 0
    for df in domain_files:
        with open(df, "r", encoding="utf-8") as f:
            data = json.load(f)
        tree = TreeSchema(**data)
        assert tree.domain == df.stem
        assert len(tree.cells) > 0, f"Domain {df.stem} has 0 cells"

        seeds_in_domain = [c for c in tree.cells if c.source_priority == 1]
        total_seeds += len(seeds_in_domain)
        total_cells += len(tree.cells)

    print(f"\n[Phase 4] Single-File Verification PASSED: {len(domain_files)} domains, {total_cells} cells, {total_seeds} curated seeds.")
    assert total_seeds >= 15, f"Expected at least 15 curated seeds across domains, got {total_seeds}"


def test_harvester_introspects_mock_audio_lib():
    """Verify that IntelligentHarvester introspects a new library without code edits."""
    import mock_audio_lib
    harvester = IntelligentHarvester(domain="audio", package_name="mock_audio_lib")
    cells = harvester.harvest_all()

    assert len(cells) == 3, f"Expected 3 cells from mock_audio_lib, got {len(cells)}"
    cell_ids = {c.cell_id for c in cells}
    assert cell_ids == {"AUDIO_LOAD_AUDIO", "AUDIO_RESAMPLE_AUDIO", "AUDIO_SAVE_AUDIO"}

    load_cell = next(c for c in cells if c.cell_id == "AUDIO_LOAD_AUDIO")
    assert load_cell.stage == 1
    assert "filepath" in load_cell.inputs
    assert load_cell.inputs["filepath"].state == "source_identifier"
    assert load_cell.outputs["output_data"].type_name == "ndarray"

    resample_cell = next(c for c in cells if c.cell_id == "AUDIO_RESAMPLE_AUDIO")
    assert resample_cell.stage == 2
    assert resample_cell.outputs["output_data"].state == "resampled"

    save_cell = next(c for c in cells if c.cell_id == "AUDIO_SAVE_AUDIO")
    assert save_cell.stage == 3
    assert "dest_path" in save_cell.inputs
    assert save_cell.outputs["output_data"].state == "filepath_written"


def test_zero_code_edit_audio_pipeline_end_to_end():
    """Verify full plug-and-play toolchain: harvest -> compile -> validate -> route -> sandbox execute."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_trees = Path(tmpdir) / "trees"
        tmp_trees.mkdir(parents=True, exist_ok=True)
        tmp_db = Path(tmpdir) / "test_audio.db"

        # 1. Harvest via CLI logic
        class ArgsHarvest:
            package = "mock_audio_lib"
            domain = "audio"
            trees_dir = str(tmp_trees)

        cmd_harvest(ArgsHarvest())
        audio_json = tmp_trees / "audio.json"
        assert audio_json.exists(), "audio.json was not created"

        # 2. Compile via CLI logic
        class ArgsCompile:
            trees_dir = str(tmp_trees)
            output = str(tmp_db)
            domains = None

        cmd_compile(ArgsCompile())
        assert tmp_db.exists(), "test_audio.db was not compiled"

        # 3. Validate via CLI logic
        class ArgsValidate:
            db = str(tmp_db)

        cmd_validate(ArgsValidate())

        # 4. Route and Execute with Zero Edits to src/
        orchestrator = LatticeOrchestrator()
        orchestrator.load_from_database(str(tmp_db))
        orchestrator.build_topology()

        router = LatticeRouter(orchestrator=orchestrator, internal_rag=None)

        sound_wav = os.path.join(tmpdir, "input_sound.wav")
        output_wav = os.path.join(tmpdir, "output_sound.wav")
        with open(sound_wav, "w", encoding="utf-8") as f:
            f.write("RAW_AUDIO_STREAM_TEST")

        prompt = f"Load {sound_wav}, resample audio, and save to {output_wav}"
        path, _ = router.plan_path(prompt, return_tuple=True)
        path_ids = [c.cell_id for c in path]

        assert path_ids == ["AUDIO_LOAD_AUDIO", "AUDIO_RESAMPLE_AUDIO", "AUDIO_SAVE_AUDIO"], \
            f"Unexpected audio routing path: {path_ids}"

        # 5. Unification & Sandbox Execution
        ctx = ExecutionContext(prompt=prompt)
        lines = [UnificationGate.unify_cell(ctx, c) for c in path]
        code = UnificationGate.resolve_imports("\n".join(lines), ctx)

        sandbox = GEVRSandbox()
        res = sandbox.execute(code, timeout=5)
        assert res["success"], f"GEVR execution failed: {res.get('error')}\nCode:\n{code}"
        assert os.path.exists(output_wav), f"Expected output file {output_wav} was not created"

        content = open(output_wav, "r", encoding="utf-8").read()
        assert content == "AUDIO_WAV_BYTES_MOCK", f"Unexpected output content: {content}"
        print("\n[+] Zero-Code-Edit Audio Pipeline Harvested, Compiled, Routed, and Executed 100% CLEANLY!")


def test_cv2_enum_and_constant_handling():
    """Verify that harvester correctly binds OpenCV enum flags (COLOR_BGR2GRAY) into clean templates or parameter defaults."""
    trees_dir = ROOT_DIR / "trees"
    cv2_json = trees_dir / "cv2.json"
    assert cv2_json.exists()

    with open(cv2_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    tree = TreeSchema(**data)
    cvt_cells = [c for c in tree.cells if "CVTCOLOR" in c.cell_id]
    assert len(cvt_cells) > 0, "No cvtColor cell found in cv2.json"

    # Verify that the cell code template or parameter defaults bind the enum/constant
    cvt_cell = cvt_cells[0]
    has_enum_in_template = "COLOR_" in cvt_cell.code_template
    has_enum_in_defaults = any("COLOR_" in str(p.default_value or "") for p in cvt_cell.inputs.values())
    assert has_enum_in_template or has_enum_in_defaults, \
        f"Expected enum binding in template or input defaults, got: template={cvt_cell.code_template}, inputs={cvt_cell.inputs}"

    # Also test that IntelligentHarvester dynamically introspects cvtColor with enum
    harvester = IntelligentHarvester(domain="cv2")
    assert "COLOR_BGR2GRAY" in harvester.enum_constants
    cell = harvester.harvest_function("cvtColor", getattr(harvester.module, "cvtColor"))
    assert cell is not None
    assert "cv2.COLOR_BGR2GRAY" in cell.code_template
