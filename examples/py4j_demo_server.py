import sys, os, time
sys.path.insert(0, os.path.abspath(
    "" + os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + ""
))
from py4j.clientserver import ClientServer, JavaParameters, PythonParameters
from backend.mvesta_adapter import DPSQKDSimulationModel
from backend.mvesta_integrator import SimulationWrapper

model = DPSQKDSimulationModel()
model.configure(num_pulses=300, distance_km=10.0)
wrapper = SimulationWrapper(model)

server = ClientServer(
    java_parameters=JavaParameters(port=25333),
    python_parameters=PythonParameters(port=25334),
    python_server_entry_point=wrapper,
)
print("READY", flush=True)
time.sleep(60)
