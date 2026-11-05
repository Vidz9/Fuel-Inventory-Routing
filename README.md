# Learning-Enhanced Branch-and-Price Matheuristic for
Multi-Depot Multi-Trip Fuel Inventory Routing

Code and data for the paper of the same title (under review).

The repository contains a three-phase matheuristic for a fuel inventory routing
problem with multiple depots, multiple products, a heterogeneous fleet and
multi-trip routes:

1. **Route-pool generation** by branch-and-price, optionally warm-started with
   heuristic routes.
2. **Route-pool filtering**: a trained classifier scores every generated route
   and only the top share *p* of the pool is kept.
3. **Route-based MILP** solved over the retained routes within the remaining
   time budget.

The arc-based MILP that is used as a benchmark is included as well.

---

## Repository structure

```
.
├── main.py                    # generation experiments: arc-based MILP, CG, B&P, warm-started variants
├── main_pool_filter.py        # phase-3 experiments: filter stored route pools and solve the route-based MILP
├── config.py                  # problem, penalty and solver parameters
├── instance.py                # IRPInstance and route-pool bookkeeping
├── results.py                 # result containers (InstanceResult, MethodResult, ...)
├── plotting.py                # route-level distribution plots written by main.py
├── data/
│   ├── loader.py              # reads Input_data_<network>.xlsx into an IRPData object
│   └── models.py              # Location, Node, Product, Truck
├── solvers/
│   ├── milp.py                # arc-based MILP
│   ├── master.py              # restricted master problem (route-based formulation)
│   ├── subproblem.py          # pricing problem
│   └── column_gen.py          # column generation and integer re-solve
├── branching/
│   └── branch_and_price.py    # branch-and-price
├── heuristics/
│   ├── route_generator.py         # random warm-start routes
│   └── demand_route_generator.py  # demand-guided warm-start routes
├── results/
│   ├── svm_model.pkl                   # NL-SVM   (RBF kernel, all 13 features)
│   ├── svm_route_properties_model.pkl  # NL-SVM-R (RBF kernel, 5 route-structure features)
│   ├── svm_linear_model.pkl            # L-SVM    (linear kernel, all 13 features)
│   └── lr_model.pkl                    # LR       (logistic regression, all 13 features)
├── Input_data_small.xlsx      # network data (see "Data")
├── Input_data_medium.xlsx
├── Input_data_big.xlsx
├── Input_data_company.xlsx
├── instances.pkl              # instances 1-10
└── instances_extended.pkl     # instances 1-30
```

---

## Requirements

* Python 3.11 or newer
* [Gurobi](https://www.gurobi.com) with a valid licence. The size-limited
  licence that comes with `pip install gurobipy` is only enough for the small
  network; the medium and big networks need a full licence (free academic
  licences are available from Gurobi).
* The Python packages in `requirements.txt`:

```bash
pip install -r requirements.txt
```

The classifiers were saved with scikit-learn 1.8.0, which is why that version
is pinned. Other versions may load them with a warning, or not at all.

---

## Data

All names in the data are anonymised:

| Original entity        | Anonymised as                |
|------------------------|------------------------------|
| Gas stations           | `station_1`, `station_2`, ... |
| Depots                 | `depot_1`, `depot_2`, ...     |
| Truck parking location | `parking_1`                  |
| Fuel products          | `product_1`, `product_2`, `product_3` |
| Trucks                 | `truck_1`, `truck_2`, `truck_3` |

The same code refers to the same entity in every file. Geographic coordinates
have been removed (the `Longitude` and `Latitude` columns are kept but empty);
they are not used by the model. All other values are unchanged.

### Network files (`Input_data_<network>.xlsx`)

Each workbook has five sheets:

| Sheet             | Content |
|-------------------|---------|
| `Gas stations`    | one row per station-product tank: capacity, safety stock, target stock, initial level, average daily demand and daily demand |
| `Depots`          | products available at each depot and the waiting time |
| `Locations`       | all locations and their type (`Gas station`, `Depot`, `Parking`) |
| `Duration matrix` | travel times between all locations, in hours |
| `Trucks`          | trucks, their parking location, total capacity and compartment capacities |

| Network   | Stations | Depots | Trucks | Products | Nodes |
|-----------|---------:|-------:|-------:|---------:|------:|
| `small`   | 3        | 2      | 2      | 2        | 9     |
| `medium`  | 8        | 4      | 2      | 2        | 21    |
| `big`     | 12       | 4      | 3      | 3        | 41    |
| `company` | 20       | 4      | 2      | 3        | 65    |

### Instance files

An instance fixes the initial inventory level and the demand over the
planning horizon for every station-product pair:

```python
instances[i] = {
    "Initial levels": {"station_1-product_2": 14.03, ...},
    "Demand":         {"station_1-product_2": array([...]), ...},
}
```

`instances_extended.pkl` contains instances 1-30. Instances 1-10 are identical
to those in `instances.pkl`; instances 11-30 are the held-out instances used in
the route-pool filtering experiments. Keys exist for every station-product pair
of the `company` network, so the same instances can be used on every network.

---

## Running the experiments

All paths are relative to the repository folder, so the scripts can be started
from any directory. Results are written to `results/`.

### 1. Route generation and benchmarks: `main.py`

For every instance, `main.py` runs the arc-based MILP and the
column-generation / branch-and-price variants, with or without a warm start,
and stores the full route pools. Settings are at the top of the file:

| Setting                  | Meaning |
|--------------------------|---------|
| `NETWORK_SIZE`           | `"small"`, `"medium"`, `"big"` or `"company"` |
| `INSTANCE_RANGE`         | inclusive range of instance ids, e.g. `(11, 30)` |
| `TIME_LIMIT`             | total time budget *B* per instance, in minutes |
| `DEMAND_COEFFICIENTS`    | demand scaling factors |
| `PARAMETER_VALUES_VT`    | `(number of trucks, horizon in days)` |
| `MAX_WARMSTART_ROUTES`   | warm-start size *K* (0 = no warm start) |
| `DEMAND_AWARE_WARMSTART` | demand-guided (`True`) or random (`False`) warm-start routes |
| `GEN_FRACTIONS`          | generation fractions α at which a result is recorded |

```bash
python main.py
```

Output: `results/results_<tag>.pkl` (full results, including route pools) and
`results/summary_<tag>.csv`.

### 2. Route-pool filtering: `main_pool_filter.py`

`main_pool_filter.py` loads route pools stored by `main.py` (`SOURCE_PKL`),
filters each pool with every requested classifier and retention level *p*, and
solves the route-based MILP over the retained routes. Every filter sees the
same pool, so differences in outcome come from the filter alone.

```bash
# all filters, all retention levels, held-out instances 11-30
python main_pool_filter.py --filters svm_rbf svm_route_props svm_linear lr random \
                           --p 0.2 0.25 0.35 0.5 0.65 0.8 1.0 --range 11 30
```

| Option        | Meaning |
|---------------|---------|
| `--filters`   | any of `svm_rbf` (NL-SVM), `svm_route_props` (NL-SVM-R), `svm_linear` (L-SVM), `lr` (LR), `random` |
| `--p`         | retention levels; `1.0` solves the full pool once |
| `--instances` | explicit instance ids, e.g. `--instances 11 14 17` |
| `--range`     | inclusive range of instance ids, e.g. `--range 11 20` |

Results are saved after every solve, so an interrupted run resumes where it
stopped. Output: `results/results_<OUTPUT_TAG>.pkl` and
`results/summary_<OUTPUT_TAG>.csv`.

### Classifier features

The filters use 13 route features: five describing the route itself
(`n_depots`, `n_stations`, `duration`, `n_stations_net`, `total_demand`) and
eight collected during branch-and-price (`gen_warmstart`, `gen_heuristic`,
`gen_suboptimal`, `gen_optimal`, `reduced_cost`, `route_order`,
`usage_ratio`, `avg_lp_value`). NL-SVM-R uses only the five route-structure
features. Each model file stores a `(model, scaler)` tuple:

```python
import joblib
model, scaler = joblib.load("results/svm_model.pkl")
```
