"""Optional real MultiVeStA (Java, py4j) integration point.

This mirrors anubhavsrivastavaa/multivesta-qnetworks's ``MV_python_integrator
.py`` wrapper API exactly (setSimulatorForNewSimulation / performOneStep
OfSimulation / performWholeSimulation / getTime / rval), but delegates to
``DPSQKDSimulationModel`` (backend/mvesta_adapter.py) instead of the
reference repo's multi-node SequenceModel.

Requires the optional ``py4j`` package (``pip install py4j``) and an actual
MultiVeStA Java installation -- neither is needed to run the batch
statistical experiments in statistical_validation.py, which talk to
DPSQKDSimulationModel directly in-process. Use this file only if you want
MultiVeStA's own Java-side query language / plotting instead of the
lightweight Python driver.

Not imported by anything else in this app (gui/, backend/adapter.py,
main.py); py4j is not a hard dependency of the simulator.
"""

from __future__ import annotations

import sys

from backend.mvesta_adapter import DPSQKDSimulationModel


class SimulationWrapper:
    """py4j-facing wrapper; method names match the reference integrator so
    an existing MultiVeStA Java-side setup can be pointed at this file with
    no changes on the Java side.
    """

    def __init__(self, model: DPSQKDSimulationModel):
        self.model = model

    def setSimulatorForNewSimulation(self, random_seed):
        self.model.reset(int(random_seed))

    def performOneStepOfSimulation(self):
        self.model.run_step()

    def performWholeSimulation(self):
        self.model.run()

    def getTime(self):
        return float(self.model.get_time())

    def rval(self, observation):
        # MultiVeStA passes ints for the built-in observations (0 = time,
        # 3 = is-complete) and strings for custom directives, matching the
        # reference repo's rval() dispatch.
        if isinstance(observation, int):
            if observation == 3:
                return float(self.model.is_step_complete())
            return self.model.rval(observation)
        observation = str(observation).strip()
        return self.model.rval(observation)

    class Java:
        implements = ["vesta.python.IPythonSimulatorWrapper"]


if __name__ == "__main__":
    from py4j.java_gateway import (
        CallbackServerParameters,
        GatewayParameters,
        JavaGateway,
    )

    print("Entered DPS-QKD MultiVeStA Python integration")
    print(sys.argv)
    porta = int(sys.argv[1])
    callback_porta = int(sys.argv[2])

    gateway = JavaGateway(
        start_callback_server=True,
        gateway_parameters=GatewayParameters(port=porta),
        callback_server_parameters=CallbackServerParameters(port=callback_porta),
    )

    model = DPSQKDSimulationModel()
    # Configure fixed (non-seed) physical parameters for the experiment
    # here before handing control to MultiVeStA, e.g.:
    #   model.configure(distance_km=25.0, eve_enabled=True)
    gateway.entry_point.playWithState(SimulationWrapper(model))
