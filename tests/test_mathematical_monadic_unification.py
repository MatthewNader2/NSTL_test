"""
tests/test_mathematical_monadic_unification.py
Formal verification of Type-Monadic Unification, Monad Laws, and Domain-Agnostic Tree Modularity.
"""

import os
import sys
import unittest
import tempfile
import json

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from unification import (
    unify, unit, bind, Success, Failure, Substitution,
    UnificationGate, ExecutionContext, AtomicType, TypeVariable, TypestateTerm, TOP,
    TypeTerm, GenericTypeTerm
)
from lattice import (
    TypeRegistry, AlgebraicSignature, PortSignature,
    Cell, MicroCell, MacroCell, LatticeOrchestrator
)
from router import LatticeRouter
from planner import LatticePlanner


class TestTypeMonadAndUnification(unittest.TestCase):
    def setUp(self):
        TypeRegistry.reset()
        self.reg = TypeRegistry.get_instance()

    def test_monad_left_identity(self):
        """Monad Law 1: bind(unit(x), k) == k(x)"""
        x = 42
        def k(val, sigma):
            return Success(val * 2, sigma)

        res1 = bind(unit(x), k)
        res2 = k(x, Substitution())

        self.assertFalse(res1.is_bottom())
        self.assertFalse(res2.is_bottom())
        self.assertEqual(res1.value, res2.value)

    def test_monad_right_identity(self):
        """Monad Law 2: bind(m, unit) == m"""
        m = unit("test_data")
        res = bind(m, lambda val, sigma: unit(val, sigma))

        self.assertFalse(res.is_bottom())
        self.assertEqual(res.value, m.value)

    def test_monad_associativity(self):
        """Monad Law 3: bind(bind(m, f), g) == bind(m, lambda x, s: bind(f(x, s), g))"""
        m = unit(5)
        f = lambda x, s: Success(x + 10, s)
        g = lambda x, s: Success(x * 3, s)

        lhs = bind(bind(m, f), g)
        rhs = bind(m, lambda x, s: bind(f(x, s), g))

        self.assertFalse(lhs.is_bottom())
        self.assertFalse(rhs.is_bottom())
        self.assertEqual(lhs.value, rhs.value)
        self.assertEqual(lhs.value, 45)

    def test_monad_failure_short_circuit(self):
        """Any step returning bottom short-circuits the rest of the chain."""
        m = unit(10)
        fail_step = lambda x, s: Failure("Type incompatibility at step 1")
        never_called = lambda x, s: Success(x * 100, s)

        res = bind(bind(m, fail_step), never_called)
        self.assertTrue(res.is_bottom())
        self.assertEqual(res.reason, "Type incompatibility at step 1")

    def test_robinson_unification_atoms_and_wildcards(self):
        """Robinson unification: atomic types, wildcards, variables."""
        # Exact match
        s = unify(AtomicType("int"), AtomicType("int"))
        self.assertIsNotNone(s)

        # Wildcard Top unifies with anything
        s_top = unify(TOP, AtomicType("CustomDomainType"))
        self.assertIsNotNone(s_top)

        # Variable binding
        var_alpha = TypeVariable("alpha")
        s_var = unify(var_alpha, AtomicType("float"))
        self.assertIsNotNone(s_var)
        self.assertEqual(s_var.get("alpha").name, "float")

    def test_poset_subtyping_unification(self):
        """Poset subtyping: sub unifies with super iff sub <= super."""
        self.reg.register_type("RobotState")
        self.reg.register_type("ArmPose", super_type="RobotState")

        # ArmPose <= RobotState -> should succeed
        s1 = unify(AlgebraicSignature("ArmPose", "any"), AlgebraicSignature("RobotState", "any"))
        self.assertIsNotNone(s1)

        # RobotState is NOT <= ArmPose -> should fail
        s2 = unify(AlgebraicSignature("RobotState", "any"), AlgebraicSignature("ArmPose", "any"))
        self.assertIsNone(s2)

    def test_typestate_unification(self):
        """Typestate consistency: matching types but conflicting states must produce bottom."""
        t1 = AlgebraicSignature("DataPayload", "raw")
        t2 = AlgebraicSignature("DataPayload", "cleaned")
        t_wildcard = AlgebraicSignature("DataPayload", "any")

        # Conflicting states -> bottom
        self.assertIsNone(unify(t1, t2))

        # Wildcard state -> unifies
        self.assertIsNotNone(unify(t1, t_wildcard))
        self.assertIsNotNone(unify(t2, t_wildcard))

    def test_robinson_unification_occurs_check(self):
        """
        Occurs check: unifying a type variable with a compound term containing
        that same variable (e.g. T = Sequence[T]) must fail (return None / bottom).
        """
        t_var = TypeTerm.from_string("T")
        t_compound = TypeTerm.from_string("Sequence[T]")

        # T = Sequence[T] -> None (occurs check failure)
        s1 = unify(t_var, t_compound)
        self.assertIsNone(s1)

        # Reverse: Sequence[T] = T -> None
        s2 = unify(t_compound, t_var)
        self.assertIsNone(s2)

        # Nested occurs check: T = List[Dict[str, T]] -> None
        t_nested = TypeTerm.from_string("List[Dict[str, T]]")
        s3 = unify(t_var, t_nested)
        self.assertIsNone(s3)

        # Transitive occurs check through existing substitution:
        # sigma = {A: T}, unify(T, Sequence[A]) -> None
        sig = Substitution({"A": TypeVariable("T")})
        s4 = unify(TypeVariable("T"), TypeTerm.from_string("Sequence[A]"), sig)
        self.assertIsNone(s4)

    def test_cycle_safe_substitution_application(self):
        """
        Verifies that cycle-safe apply_substitution prevents RecursionError
        even if a cyclical substitution is manually constructed or encountered.
        """
        # Direct self-cycle: T -> Sequence[T]
        t_var = TypeVariable("T")
        t_seq = GenericTypeTerm("Sequence", (t_var,))
        sub_direct = Substitution({"T": t_seq})
        res_direct = t_var.apply_substitution(sub_direct)
        self.assertEqual(str(res_direct), "Sequence[?T]")

        # Mutual cycle: A -> B, B -> A
        var_a = TypeVariable("A")
        var_b = TypeVariable("B")
        sub_mutual = Substitution({"A": var_b, "B": var_a})
        res_a = var_a.apply_substitution(sub_mutual)
        self.assertEqual(str(res_a), "?A")


class TestDomainAgnosticTreeModularity(unittest.TestCase):
    """
    Verifies that NSTL is 100% domain-agnostic:
    Injects a synthetic Robotics FSM tree into the engine and verifies
    routing, path planning, and monadic code emission without any engine code modifications.
    """
    def test_robotics_domain_tree_injection(self):
        # 1. Create a synthetic Robotics domain tree
        robotics_tree = {
            "domain": "robotics_fsm",
            "version": "1.0.0",
            "cells": [
                {
                    "cell_id": "ROBOT_PERCEIVE_ENVIRONMENT",
                    "stage": 1,
                    "inputs": {
                        "camera_id": {
                            "type_name": "str",
                            "state": "source_identifier",
                            "required": True
                        }
                    },
                    "outputs": {
                        "scene_graph": {
                            "type_name": "SceneGraph",
                            "state": "perceived",
                            "required": True
                        }
                    },
                    "code_template": "{output_var} = robot.perceive(camera={camera_id})",
                    "dependencies": ["import robotics_sdk as robot"],
                    "keywords": ["perceive", "camera", "sensor", "detect", "vision"],
                    "semantic_tags": ["perception", "vision", "camera"],
                    "docstring": "Captures raw camera feed and constructs perceived 3D scene graph.",
                    "source_priority": 1
                },
                {
                    "cell_id": "ROBOT_CALCULATE_GRASP_POSE",
                    "stage": 2,
                    "inputs": {
                        "scene": {
                            "type_name": "SceneGraph",
                            "state": "perceived",
                            "required": True
                        },
                        "target_object": {
                            "type_name": "str",
                            "state": "column_name",
                            "required": False,
                            "default_value": "'can'"
                        }
                    },
                    "outputs": {
                        "grasp_trajectory": {
                            "type_name": "Trajectory",
                            "state": "planned",
                            "required": True
                        }
                    },
                    "code_template": "{output_var} = robot.plan_grasp({scene}, target={target_object})",
                    "dependencies": ["import robotics_sdk as robot"],
                    "keywords": ["grasp", "plan", "trajectory", "target", "pose"],
                    "semantic_tags": ["motion_planning", "grasping"],
                    "docstring": "Calculates 6-DOF grasp trajectory for the identified target object in the scene.",
                    "source_priority": 1
                },
                {
                    "cell_id": "ROBOT_EXECUTE_TRAJECTORY",
                    "stage": 3,
                    "inputs": {
                        "traj": {
                            "type_name": "Trajectory",
                            "state": "planned",
                            "required": True
                        },
                        "log_path": {
                            "type_name": "str",
                            "state": "dest_identifier",
                            "required": True
                        }
                    },
                    "outputs": {
                        "status": {
                            "type_name": "str",
                            "state": "completed",
                            "required": True
                        }
                    },
                    "code_template": "{output_var} = robot.execute({traj}, log_file={log_path})",
                    "dependencies": ["import robotics_sdk as robot"],
                    "keywords": ["execute", "move", "actuate", "robot", "arm"],
                    "semantic_tags": ["actuation", "execution"],
                    "docstring": "Executes the planned arm trajectory and logs execution telemetry.",
                    "source_priority": 1
                }
            ]
        }

        # 2. Write tree to a temporary file
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(robotics_tree, f)
            temp_tree_path = f.name

        try:
            # 3. Load tree into a clean LatticeOrchestrator
            orch = LatticeOrchestrator(trees_directory=tempfile.gettempdir())
            orch.load_tree_file(temp_tree_path)

            self.assertIn("ROBOT_PERCEIVE_ENVIRONMENT", orch.loaded_cells)
            self.assertIn("ROBOT_CALCULATE_GRASP_POSE", orch.loaded_cells)
            self.assertIn("ROBOT_EXECUTE_TRAJECTORY", orch.loaded_cells)

            # 4. Route and Plan a path for a Robotics prompt
            prompt = "Perceive environment using 'cam_0', calculate grasp pose for 'cup', and execute trajectory saving log to 'telemetry.log'"
            router = LatticeRouter(orchestrator=orch, internal_rag=None)

            # In pure symbolic mode (no RAG), router collects all candidate nodes from the loaded tree
            tunnel, rel_map = router.route(prompt)
            self.assertEqual(len(tunnel), 3)

            planner = LatticePlanner(orchestrator=orch)
            path = planner.plan(prompt, tunnel, rel_map)

            # Verify that path strictly follows the valid type flow
            expected_ids = ["ROBOT_PERCEIVE_ENVIRONMENT", "ROBOT_CALCULATE_GRASP_POSE", "ROBOT_EXECUTE_TRAJECTORY"]
            self.assertEqual([c.cell_id for c in path], expected_ids)

            # 5. Unify and Emit code
            gate = UnificationGate()
            code = gate.unify_and_emit(path, prompt)

            print("\n--- Emitted Robotics Pipeline Code ---")
            print(code)
            print("--------------------------------------")

            # Verify generated code structure
            self.assertIn("import robotics_sdk as robot", code)
            self.assertIn("var_1 = robot.perceive(camera=\"cam_0\")", code)
            self.assertIn("var_2 = robot.plan_grasp(var_1, target=\"cup\")", code)
            self.assertIn("var_3 = robot.execute(var_2, log_file=\"telemetry.log\")", code)

        finally:
            if os.path.exists(temp_tree_path):
                os.remove(temp_tree_path)


if __name__ == "__main__":
    unittest.main()
