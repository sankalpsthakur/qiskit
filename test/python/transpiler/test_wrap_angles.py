# This code is part of Qiskit.
#
# (C) Copyright IBM 2025
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Wrap angles pass testing"""

import math
from test import QiskitTestCase
from unittest.mock import patch

from ddt import data, ddt

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import (
    ControlFlowOp,
    ForLoopOp,
    Gate,
    IfElseOp,
    Parameter,
    Qubit,
    SwitchCaseOp,
    WhileLoopOp,
)
from qiskit.circuit.library import RZXGate, RZZGate, XGate
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler import WrapAngleRegistry
from qiskit.transpiler.coupling import CouplingMap
from qiskit.transpiler.passes import WrapAngles
from qiskit.transpiler.target import Target

_OUT_OF_BOUNDS_RZZ = -0.4
_WRAPPED_RZZ = 0.4


def _wrap_negative_rzz(angles, _qubits):
    """Map RZZ(theta) to X(0); RZZ(-theta); X(0), sending negative theta into ``[0, pi/2]``."""
    replacement = QuantumCircuit(2)
    replacement.x(0)
    replacement.rzz(-angles[0], 0, 1)
    replacement.x(0)
    return circuit_to_dag(replacement)


def _rzz_params(circuit):
    """Return every RZZ angle in ``circuit``, including nested control-flow blocks."""
    params = []
    for instruction in circuit.data:
        op = instruction.operation
        if op.name == "rzz":
            params.append(float(op.params[0]))
        elif isinstance(op, ControlFlowOp):
            for block in op.blocks:
                params.extend(_rzz_params(block))
    return params


def _rzz_angle_bound_target():
    """Two-qubit target whose RZZ instruction only accepts angles in ``[0, pi/2]``."""
    target = Target(num_qubits=2)
    target.add_instruction(RZZGate(Parameter("theta")), angle_bounds=[(0, math.pi / 2)])
    target.add_instruction(XGate())
    target.add_instruction(IfElseOp, name="if_else")
    target.add_instruction(ForLoopOp, name="for_loop")
    target.add_instruction(WhileLoopOp, name="while_loop")
    target.add_instruction(SwitchCaseOp, name="switch_case")
    return target


class TestWrapAngles(QiskitTestCase):
    """Tests for WrapAngles pass."""

    def test_empty_dag(self):
        """Invalid callback"""
        circuit = QuantumCircuit(2)
        target = Target.from_configuration(["u"], 2, CouplingMap([[0, 1]]))

        def callback(_, __):
            raise NotADirectoryError("test_works")

        target.add_instruction(RZXGate(Parameter("x")), angle_bounds=[(0, 3.14)])
        registry = WrapAngleRegistry()
        registry.add_wrapper("rzx", callback)

        wrap_pass = WrapAngles(target, registry)
        circuit.rzx(6.28, 0, 1)
        with self.assertRaisesRegex(NotADirectoryError, "test_works"):
            wrap_pass(circuit)

    def test_combine_custom_gates(self):
        """Test custom gates are combined as presscribed."""

        class MyCustomGate(Gate):
            """A custom gate definition."""

            def __init__(self, angle):
                super().__init__("my_custom", 1, [angle])

        param = Parameter("a")
        circuit = QuantumCircuit(1)
        circuit.append(MyCustomGate(6.0), [0])
        target = Target(num_qubits=1)
        target.add_instruction(MyCustomGate(param), angle_bounds=[(0, 0.5)])

        def callback(angles, _qubits):
            angle = angles[0]
            if angle > 0:
                number_of_gates = angle / 0.5
            else:
                number_of_gates = (6.28 - angle) / 0.5
            dag = DAGCircuit()
            dag.add_qubits([Qubit()])
            for _ in range(int(number_of_gates)):
                dag.apply_operation_back(MyCustomGate(0.5), [dag.qubits[0]])
            return dag

        registry = WrapAngleRegistry()
        registry.add_wrapper("my_custom", callback)
        wrap_pass = WrapAngles(target, registry)
        res = wrap_pass(circuit)
        self.assertEqual(res.count_ops()["my_custom"], 12)

    def test_legacy_default_registy(self):
        """Test that the legacy Qiskit 2.2 import path works for the default registry."""
        from qiskit.transpiler.passes.utils.wrap_angles import WRAP_ANGLE_REGISTRY

        self.assertIs(WRAP_ANGLE_REGISTRY, WrapAngles.DEFAULT_REGISTRY)


@ddt
class TestWrapAnglesControlFlow(QiskitTestCase):
    """Test WrapAngles and transpile() apply angle bounds inside control flow."""

    def setUp(self):
        super().setUp()
        self.target = _rzz_angle_bound_target()
        self.registry = WrapAngleRegistry()
        self.registry.add_wrapper("rzz", _wrap_negative_rzz)

    def _wrap_angles(self, circuit):
        return WrapAngles(self.target, self.registry)(circuit)

    def _transpile(self, circuit):
        with patch.object(WrapAngles, "DEFAULT_REGISTRY", self.registry):
            return transpile(circuit, target=self.target, optimization_level=0)

    def _run(self, circuit, how):
        runners = {"pass": self._wrap_angles, "transpile": self._transpile}
        return runners[how](circuit)

    def _assert_all_rzz_wrapped(self, circuit):
        params = _rzz_params(circuit)
        self.assertGreater(len(params), 0)
        for param in params:
            self.assertAlmostEqual(param, _WRAPPED_RZZ)

    def _if_else_circuit(self):
        qc = QuantumCircuit(2, 1)
        qc.rzz(_OUT_OF_BOUNDS_RZZ, 0, 1)
        with qc.if_test((qc.clbits[0], 0)) as else_:
            qc.rzz(_OUT_OF_BOUNDS_RZZ, 0, 1)
        with else_:
            qc.rzz(_OUT_OF_BOUNDS_RZZ, 0, 1)
        return qc

    def _for_loop_circuit(self):
        qc = QuantumCircuit(2, 1)
        with qc.for_loop(range(3)):
            qc.rzz(_OUT_OF_BOUNDS_RZZ, 0, 1)
        return qc

    def _while_loop_circuit(self):
        qc = QuantumCircuit(2, 1)
        with qc.while_loop((qc.clbits[0], True)):
            qc.rzz(_OUT_OF_BOUNDS_RZZ, 0, 1)
        return qc

    def _switch_case_circuit(self):
        qc = QuantumCircuit(2, 1)
        with qc.switch(qc.cregs[0]) as case:
            with case(0):
                qc.rzz(_OUT_OF_BOUNDS_RZZ, 0, 1)
            with case(case.DEFAULT):
                qc.rzz(_OUT_OF_BOUNDS_RZZ, 0, 1)
        return qc

    def _nested_control_flow_circuit(self):
        qc = QuantumCircuit(2, 1)
        with qc.if_test((qc.clbits[0], 0)):
            with qc.for_loop(range(2)):
                with qc.while_loop((qc.clbits[0], True)):
                    qc.rzz(_OUT_OF_BOUNDS_RZZ, 0, 1)
                with qc.switch(qc.cregs[0]) as case:
                    with case(0):
                        qc.rzz(_OUT_OF_BOUNDS_RZZ, 0, 1)
                    with case(case.DEFAULT):
                        qc.rzz(_OUT_OF_BOUNDS_RZZ, 0, 1)
        return qc

    @data("pass", "transpile")
    def test_if_else(self, how):
        """RZZ angle bounds are enforced in both branches of an if-else."""
        result = self._run(self._if_else_circuit(), how)
        self.assertEqual(result.count_ops()["if_else"], 1)
        self._assert_all_rzz_wrapped(result)
        true_body, false_body = result.get_instructions("if_else")[0].operation.blocks
        for body in (true_body, false_body):
            self.assertEqual([inst.name for inst in body.data], ["x", "rzz", "x"])

    @data("pass", "transpile")
    def test_for_loop(self, how):
        """RZZ angle bounds are enforced inside a for-loop body."""
        result = self._run(self._for_loop_circuit(), how)
        self.assertEqual(result.count_ops()["for_loop"], 1)
        self._assert_all_rzz_wrapped(result)
        body = result.get_instructions("for_loop")[0].operation.blocks[0]
        self.assertEqual([inst.name for inst in body.data], ["x", "rzz", "x"])

    @data("pass", "transpile")
    def test_while_loop(self, how):
        """RZZ angle bounds are enforced inside a while-loop body."""
        result = self._run(self._while_loop_circuit(), how)
        self.assertEqual(result.count_ops()["while_loop"], 1)
        self._assert_all_rzz_wrapped(result)
        body = result.get_instructions("while_loop")[0].operation.blocks[0]
        self.assertEqual([inst.name for inst in body.data], ["x", "rzz", "x"])

    @data("pass", "transpile")
    def test_switch_case(self, how):
        """RZZ angle bounds are enforced inside every switch-case body."""
        result = self._run(self._switch_case_circuit(), how)
        self.assertEqual(result.count_ops()["switch_case"], 1)
        self._assert_all_rzz_wrapped(result)
        for body in result.get_instructions("switch_case")[0].operation.blocks:
            self.assertEqual([inst.name for inst in body.data], ["x", "rzz", "x"])

    @data("pass", "transpile")
    def test_nested_control_flow(self, how):
        """RZZ angle bounds are enforced through nested control-flow operations."""
        result = self._run(self._nested_control_flow_circuit(), how)
        self.assertEqual(result.count_ops()["if_else"], 1)
        self._assert_all_rzz_wrapped(result)
        self.assertEqual(len(_rzz_params(result)), 3)

    @data("pass", "transpile")
    def test_in_bounds_rzz_inside_if_else_untouched(self, how):
        """In-bounds RZZ gates inside control flow are left unchanged."""
        qc = QuantumCircuit(2, 1)
        with qc.if_test((qc.clbits[0], 0)):
            qc.rzz(_WRAPPED_RZZ, 0, 1)
        result = self._run(qc, how)
        body = result.get_instructions("if_else")[0].operation.blocks[0]
        self.assertEqual([inst.name for inst in body.data], ["rzz"])
        self.assertAlmostEqual(float(body.data[0].operation.params[0]), _WRAPPED_RZZ)
