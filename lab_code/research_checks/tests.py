"""Regression tests for consequential failures, using temporary toy workers only."""
import copy
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from .gate import (REVIEW_TOPICS, compare, contract, create, digest, launch, manifest,
                   preflight, read, sha, snapshot, verify_review)
from .persistence import cached, immutable
from .reachability import SEMANTICS, analyze


def spec():
    return dict(semantics=SEMANTICS, layers=28, heads_per_layer=16, sequence_length=4,
                dynamic_heads=[[19, 12], [24, 1]], dynamic_mlps=[0, 17, 19, 20],
                source=dict(layer=0, phase="attention_output", positions=[1, 2]),
                targets=[dict(id="checkpoint", layer=17, phase="mlp_output", positions=[3])])


class ReachabilityTests(unittest.TestCase):
    def test_e023_is_blocked(self):
        self.assertFalse(analyze(spec())["ok"])

    def test_same_layer_attention_precedes_mlp(self):
        value = spec()
        value["dynamic_heads"] = [[17, 1]]
        self.assertTrue(analyze(value)["ok"])

    def test_source_after_attention_does_not_reenter_same_layer(self):
        value = spec()
        value["dynamic_heads"] = [[0, 1]]
        self.assertFalse(analyze(value)["ok"])

    def test_source_before_attention_can_cross_positions(self):
        value = spec()
        value["dynamic_heads"] = [[0, 1]]
        value["source"]["phase"] = "residual_in"
        self.assertTrue(analyze(value)["ok"])

    def test_future_source_cannot_reach_earlier_position(self):
        value = spec()
        value["source"]["positions"] = [3]
        value["dynamic_heads"] = [[1, 0]]
        value["targets"][0]["positions"] = [0]
        self.assertFalse(analyze(value)["ok"])

    def test_frozen_mlp_output_differs_from_its_input(self):
        value = spec()
        value["dynamic_heads"] = [[1, 0]]
        value["dynamic_mlps"] = []
        self.assertFalse(analyze(value)["ok"])
        value["targets"][0]["phase"] = "mlp_input"
        self.assertTrue(analyze(value)["ok"])

    def test_same_position_needs_no_attention_transport(self):
        value = spec()
        value["dynamic_heads"] = []
        value["targets"][0]["positions"] = [1]
        self.assertTrue(analyze(value)["ok"])

    def test_later_source_cannot_change_past_checkpoint(self):
        value = spec()
        value["source"]["layer"] = 18
        self.assertFalse(analyze(value)["ok"])

    def test_target_requires_every_claimed_position(self):
        value = spec()
        value["targets"][0]["positions"] = [1, 3]
        self.assertFalse(analyze(value)["ok"])

    def test_unknown_semantics_and_invalid_indices_rejected(self):
        for change in ({"semantics": "windowed"}, {"sequence_length": True},
                       {"dynamic_heads": [[28, 0]]}, {"dynamic_mlps": [17, 17]},
                       {"targets": []}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                analyze(dict(spec(), **change))


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="lab_review_test_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ("research_checks", "plans", "worker"):
            (self.root/name).mkdir()
        (self.root/"AGENTS.md").write_text("fixture only")
        (self.root/"plans/RESEARCH_REVIEW_POLICY.md").write_text("fixture only")
        for filename in ("__init__.py", "gate.py", "reachability.py", "_worker.py"):
            shutil.copyfile(Path(__file__).parent/filename, self.root/"research_checks"/filename)
        (self.root/"worker/__init__.py").write_text("")
        (self.root/"worker/main.py").write_text("from pathlib import Path\nPath('executed').write_text('yes')\n")
        create(self.root/"evidence.json", dict(ok=True, effect=.3))
        create(self.root/"schedule.json", dict(operations=[dict(id="pilot", kind="forward")]))
        self.value = dict(schema_version=1, id="TOY", kind="experiment", primary_agent="/root",
                          purpose="Non-scientific test fixture", tracked_files=[],
                          tracked_trees=["worker"], input_manifests=[], gates=[
                              dict(id=k, category=k, evidence="evidence.json", pointer="/ok",
                                   op="eq", expected=True, reason="Synthetic fixture")
                              for k in ("hooks", "data", "pilot", "accounting", "restart")],
                          circuits=[], structural_scope="Toy worker has no model or circuit",
                          entrypoint=dict(module="worker.main", args=[]), schedule="schedule.json",
                          budget=dict(forward=1, backward=0))
        self.save_contract()

    def save_contract(self):
        (self.root/"contract.json").write_text(json.dumps(self.value), encoding="utf-8")

    def approve(self, phase="pre_run"):
        snap = snapshot(self.root, "contract.json", phase)
        create(self.root/"snapshot.json", snap)
        # Test fixture only; production receipts must be written by a real agent.
        review = dict(schema_version=1, snapshot_sha256=snap["snapshot_sha256"],
                      reviewer_id="/test_fixture/reviewer", independent=True,
                      verdict="approved", topics_checked=sorted(REVIEW_TOPICS),
                      findings=[], limitations="Synthetic fixture, not a real review")
        create(self.root/"review.json", review)
        return review

    def test_valid_review_and_launch(self):
        self.approve()
        self.assertEqual(launch(self.root, "snapshot.json", "review.json"), 0)
        self.assertTrue((self.root/"executed").exists())
        self.assertFalse((self.root/"operations/research_worker.lock").exists())

    def test_no_review_never_starts_worker(self):
        with self.assertRaises(FileNotFoundError):
            launch(self.root, "snapshot.json", "review.json")
        self.assertFalse((self.root/"executed").exists())

    def test_changed_or_added_source_invalidates_review(self):
        self.approve()
        for name in ("worker/main.py", "worker/new_import.py"):
            (self.root/name).write_text("# changed")
            with self.subTest(name=name), self.assertRaises(ValueError):
                verify_review(self.root, "snapshot.json", "review.json")

    def test_changed_evidence_invalidates_review(self):
        self.approve()
        (self.root/"evidence.json").write_text('{"ok": false}')
        with self.assertRaises(ValueError):
            launch(self.root, "snapshot.json", "review.json")
        self.assertFalse((self.root/"executed").exists())

    def test_changed_interpreter_invalidates_review_before_launch(self):
        self.approve()
        fake = self.root/"other_python.exe"
        fake.write_bytes(b"different interpreter")
        with patch("research_checks.gate.sys.executable", str(fake)):
            with self.assertRaises(ValueError):
                launch(self.root, "snapshot.json", "review.json")
        self.assertFalse((self.root/"executed").exists())

    def test_changed_package_environment_invalidates_review(self):
        self.approve()
        from .gate import environment_identity
        changed = environment_identity()
        changed["distributions"] = [("torch", "changed")]
        with patch("research_checks.gate.environment_identity", return_value=changed):
            with self.assertRaises(ValueError):
                verify_review(self.root, "snapshot.json", "review.json")

    def test_child_environment_is_checked_before_worker_import(self):
        self.approve()
        with patch.dict(os.environ, {"PYTHONOPTIMIZE": "1"}):
            self.assertEqual(launch(self.root, "snapshot.json", "review.json"), 2)
        self.assertFalse((self.root/"executed").exists())
        self.assertFalse((self.root/"operations/research_worker.lock").exists())

    def test_open_medium_finding_is_not_an_approved_disposition(self):
        review = self.approve()
        review["findings"] = [dict(id="followup", severity="medium", status="open",
                                  evidence="something unchecked", resolution="")]
        (self.root/"review.json").write_text(json.dumps(review))
        with self.assertRaises(ValueError):
            verify_review(self.root, "snapshot.json", "review.json")

    def test_documented_noncritical_limitation_can_be_accepted(self):
        review = self.approve()
        review["findings"] = [dict(id="scope", severity="medium", status="accepted",
                                  evidence="only dense causal attention is supported",
                                  resolution="Unsupported semantics are rejected; documented scope is sufficient for this task.")]
        (self.root/"review.json").write_text(json.dumps(review))
        verify_review(self.root, "snapshot.json", "review.json")

    def test_manifest_members_are_verified_not_only_manifest_bytes(self):
        data = self.root/"historical.json"
        data.write_text('{"value": 1}')
        create(self.root/"inputs.json", {"files": {"historical.json": sha(data)}})
        self.value["input_manifests"] = ["inputs.json"]
        self.save_contract()
        self.approve()
        data.write_text('{"value": 2}')
        with self.assertRaises(ValueError):
            verify_review(self.root, "snapshot.json", "review.json")
        self.assertFalse((self.root/"executed").exists())

    def test_serious_finding_blocks_worker(self):
        review = self.approve()
        review["findings"] = [dict(id="bug", severity="high", status="accepted",
                                  evidence="bad edge", resolution="want to proceed")]
        (self.root/"review.json").write_text(json.dumps(review))
        with self.assertRaises(ValueError):
            launch(self.root, "snapshot.json", "review.json")

    def test_self_review_is_rejected(self):
        review = self.approve()
        review["reviewer_id"] = "/root"
        (self.root/"review.json").write_text(json.dumps(review))
        with self.assertRaises(ValueError):
            verify_review(self.root, "snapshot.json", "review.json")

    def test_completion_receipt_cannot_start_worker(self):
        self.approve("completion")
        with self.assertRaises(ValueError):
            launch(self.root, "snapshot.json", "review.json")

    def test_failed_numeric_pilot_blocks_pre_run(self):
        self.value["gates"][2].update(pointer="/effect", op="ge", expected=.5)
        self.save_contract()
        self.assertFalse(preflight(self.root, "contract.json")["ok"])
        with self.assertRaises(ValueError):
            snapshot(self.root, "contract.json", "pre_run")

    def test_missing_critical_category_is_rejected(self):
        self.value["gates"].pop()
        self.save_contract()
        with self.assertRaises(ValueError):
            preflight(self.root, "contract.json")

    def test_e023_bad_graph_blocks_pre_run(self):
        create(self.root/"circuit.json", spec())
        self.value["circuits"] = [dict(id="reachability", spec="circuit.json")]
        self.save_contract()
        with self.assertRaises(ValueError):
            snapshot(self.root, "contract.json", "pre_run")

    def test_over_budget_and_duplicate_schedule(self):
        self.value["budget"]["forward"] = 0
        self.save_contract()
        self.assertFalse(preflight(self.root, "contract.json")["ok"])
        (self.root/"schedule.json").write_text(json.dumps(dict(operations=[
            dict(id="duplicate", kind="forward"), dict(id="duplicate", kind="forward")])))
        with self.assertRaises(ValueError):
            preflight(self.root, "contract.json")

    def test_untracked_worker_is_rejected(self):
        self.value["tracked_trees"] = []
        self.save_contract()
        with self.assertRaises(ValueError):
            snapshot(self.root, "contract.json", "pre_run")

    def test_concurrent_launch_is_blocked(self):
        self.approve()
        create(self.root/"operations/research_worker.lock", {"token": "other"})
        with self.assertRaises(FileExistsError):
            launch(self.root, "snapshot.json", "review.json")
        self.assertEqual(read(self.root/"operations/research_worker.lock")["token"], "other")

    def test_failed_worker_recorded_and_lock_released(self):
        self.approve()
        with patch("research_checks.gate.subprocess.run") as run:
            run.return_value.returncode = 7
            self.assertEqual(launch(self.root, "snapshot.json", "review.json"), 7)
        self.assertFalse((self.root/"operations/research_worker.lock").exists())
        receipts = list((self.root/"operations").glob("research_launch_*.json"))
        self.assertEqual(read(receipts[0])["returncode"], 7)

    def test_receipt_is_write_once(self):
        self.approve()
        with self.assertRaises(FileExistsError):
            create(self.root/"review.json", {})

    def test_path_escape_and_boolean_numeric_threshold_rejected(self):
        self.value["tracked_files"] = ["../outside.json"]
        self.save_contract()
        with self.assertRaises(ValueError):
            manifest(self.root, "contract.json")
        with self.assertRaises(ValueError):
            compare(True, "ge", 1)
        self.assertFalse(compare(True, "eq", 1))
        with self.assertRaises(ValueError):
            compare(float("nan"), "le", 1)


class PersistenceTests(unittest.TestCase):
    def test_non_native_request_rejected_before_compute(self):
        with tempfile.TemporaryDirectory() as folder:
            req = dict(model_identity="model", code_identity="code", document={"ids": [1]},
                       intervention={"partition": ([0], [1])}, readout=[2, 3])
            with patch.object(self, "fail", side_effect=AssertionError("unexpected compute")):
                with self.assertRaisesRegex(ValueError, "JSON-native"):
                    cached(folder, req, self.fail)
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_hash_streams_without_whole_file_read(self):
        import hashlib
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"weights.bin"
            blob = bytes(range(256))*4096
            path.write_bytes(blob)
            with patch.object(Path, "read_bytes", side_effect=AssertionError("whole file read")):
                self.assertEqual(sha(path), hashlib.sha256(blob).hexdigest())

    def test_failed_compute_is_not_a_completed_cache_entry(self):
        with tempfile.TemporaryDirectory() as folder:
            req = dict(model_identity="model", code_identity="code",
                       document={"ids": [1]}, intervention={}, readout=[2, 3])
            def fail():
                raise RuntimeError("interrupted measurement")
            with self.assertRaises(RuntimeError):
                cached(folder, req, fail)
            self.assertEqual(list(Path(folder).iterdir()), [])
            self.assertEqual(cached(folder, req, lambda: 9), 9)

    def test_resume_preserves_bytes_and_timestamp(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"freeze.json"
            immutable(path, {"schedule": ["a"]})
            original = path.read_bytes()
            immutable(path, {"schedule": ["a"]})
            self.assertEqual(original, path.read_bytes())
            with self.assertRaises(ValueError):
                immutable(path, {"schedule": ["b"]})
            self.assertEqual(original, path.read_bytes())

    def test_cache_binds_request_and_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as folder:
            req = dict(model_identity="revision-and-weight-hash", code_identity="dependency-hash",
                       document={"ids": [1, 2]}, intervention={"alpha": .5}, readout=[3, 4])
            self.assertEqual(cached(folder, req, lambda: 42), 42)
            with patch.object(self, "fail", side_effect=AssertionError("unexpected recompute")):
                self.assertEqual(cached(folder, req, self.fail), 42)
            for field, replacement in (("model_identity", "other"), ("code_identity", "edited"),
                                       ("document", {"ids": [1, 5]}), ("intervention", {"alpha": 1}),
                                       ("readout", [5, 6])):
                changed = dict(req, **{field: replacement})
                self.assertEqual(cached(folder, changed, lambda: 13), 13)
            file = Path(folder)/(digest(req)+".json")
            obj = read(file)
            obj["payload"]["result"] = 100
            file.write_text(json.dumps(obj))
            with self.assertRaises(ValueError):
                cached(folder, req, lambda: 1)


def main():
    import argparse
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument("--out")
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if args.out:
        create(args.out, dict(ok=result.wasSuccessful(), tests=result.testsRun,
                              failures=len(result.failures), errors=len(result.errors),
                              pretrained_forwards=0, paid_api_calls=0))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
