# Real py4j round-trip demo (`py4j_demo_server.py` / `py4j_demo_client.py`)

## What this proves, and what it doesn't

**We could not obtain `multivesta.jar` for you.** It is not published on
Maven Central or as a GitHub Release, and the tool's own maintainers
(andrea-vandin/MultiVeStA) say on their wiki: *"We encourage you to contact
us if you are interested in using MultiVeStA."* Their GitHub repo contains
only docs/wiki/license, no jar or buildable source. If you already have a
`multivesta.jar` (e.g. from a prior install, a labmate, or after contacting
the authors), skip straight to "Running with the real jar" below --
`backend/mvesta_integrator.py` is unchanged and ready for it.

**What we built and actually ran instead:** a genuine, two-process,
socket-based py4j RPC session that exercises the *exact* three calls
MultiVeStA's Java engine makes per trial --
`setSimulatorForNewSimulation(seed)`, `performOneStepOfSimulation()`,
`rval(observable)` -- against `backend/mvesta_integrator.SimulationWrapper`,
which is the same object `main()` in that file would hand to the real Java
gateway. `py4j_demo_server.py` hosts it; `py4j_demo_client.py` plays the
role MultiVeStA's Java side plays (computing a sample mean + 95% CI the
same way a MultiQuaTEx `eval E[...]` query does), but in Python, since we
have no JDK-compilable Java toolchain in this environment either (only a
JRE was installable here). This proves the adapter's RPC wiring is correct
end to end; it does not exercise MultiVeStA's own statistical algorithms
(sequential sampling, confidence-interval stopping rules, MultiQuaTEx
parsing), which only the real jar can do.

## Running the demo

```bash
cd app
pip install py4j
python examples/py4j_demo_server.py &     # hosts SimulationWrapper on ports 25333/25334
sleep 5                                    # wait for "READY" (backend import is slow)
python examples/py4j_demo_client.py        # drives it over a real socket, prints E[QBER] etc.
```

Actual output from a real run in our test environment:

```
[client, real py4j socket RPC] E[QBER] over 50 seeds: mean=0.00931  95% CI=[0.00533, 0.01330]
[client, real py4j socket RPC] E[SIFTED_KEY_LEN] over 50 seeds: mean=36.780  95% CI=[35.351, 38.209]
[client] same seed twice over the wire -> QBER 0.02857142857142857 == 0.02857142857142857: True
```
(matches the in-process results from `examples/run_statistical_validation.py` --
same backend, same seeding, just called remotely instead of in-process.)

## Running with the real jar

Once you have `multivesta.jar`:

```bash
python backend/mvesta_integrator.py <port> <callback_port>
# in another terminal / from Java:
java -jar multivesta.jar -c -m backend/mvesta_integrator.py -sm true \
     -f your_query.multiquatex -sd vesta.python.simpy.SimPyState \
     -otherParams "python3"
```

`SimulationWrapper` in `mvesta_integrator.py` already declares
`implements = ['vesta.python.IPythonSimulatorWrapper']` (the real
MultiVeStA interface name) -- no changes needed on that side. You'll write
a `.multiquatex` query file analogous to `query.multiquatex` in the
`multivesta-qnetworks` reference repo, but using our observable names, e.g.:

```
eval E[ s.rval("QBER") ];
```
