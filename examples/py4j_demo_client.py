import math
from py4j.clientserver import ClientServer, JavaParameters, PythonParameters

gw = ClientServer(
    java_parameters=JavaParameters(port=25334),
    python_parameters=PythonParameters(port=25335),
)
entry = gw.entry_point

def batch_mean_ci(observable, seeds):
    vals = []
    for s in seeds:
        entry.setSimulatorForNewSimulation(s)
        entry.performOneStepOfSimulation()
        vals.append(entry.rval(observable))
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / (n - 1)
    hw = 1.96 * math.sqrt(var) / math.sqrt(n)
    return mean, mean - hw, mean + hw, n

mean, lo, hi, n = batch_mean_ci("QBER", range(1, 51))
print(f"[client, real py4j socket RPC] E[QBER] over {n} seeds: mean={mean:.5f}  95% CI=[{lo:.5f}, {hi:.5f}]")

mean, lo, hi, n = batch_mean_ci("SIFTED_KEY_LEN", range(1001, 1051))
print(f"[client, real py4j socket RPC] E[SIFTED_KEY_LEN] over {n} seeds: mean={mean:.3f}  95% CI=[{lo:.3f}, {hi:.3f}]")

entry.setSimulatorForNewSimulation(777)
entry.performOneStepOfSimulation()
a = entry.rval("QBER")
entry.setSimulatorForNewSimulation(777)
entry.performOneStepOfSimulation()
b = entry.rval("QBER")
print(f"[client] same seed twice over the wire -> QBER {a} == {b}: {a == b}")

entry.setSimulatorForNewSimulation(1)
entry.performOneStepOfSimulation()
c = entry.rval("QBER")
entry.setSimulatorForNewSimulation(2)
entry.performOneStepOfSimulation()
d = entry.rval("QBER")
print(f"[client] different seeds over the wire -> QBER {c} vs {d}: different={c != d}")

gw.shutdown()
