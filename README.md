# Siena.jl

A Julia implementation of SIENA (Simulation Investigation for Empirical Network Analysis) for analyzing longitudinal network data using Stochastic Actor-Oriented Models (SAOM).

This is a port of [RSiena](https://github.com/stocnet/rsiena), the R implementation developed by Tom Snijders and colleagues.

## Overview

Stochastic Actor-Oriented Models are statistical models for analyzing:
- **Longitudinal network data**: Repeated observations of network ties over time
- **Network-behavior co-evolution**: How networks and actor behaviors influence each other
- **Multivariate networks**: Multiple network relations analyzed jointly
- **Two-mode networks**: Bipartite/affiliation networks

The models assume that the network evolves through a continuous-time Markov chain of actor-driven "micro-steps" - small changes made by individual actors based on their local network position and attributes.

## Installation

```julia
using Pkg
Pkg.add(path="path/to/Siena")
```

Or within the sna monorepo:

```julia
julia --project=Siena
]instantiate
```

## Quick Start

```julia
using Siena

# Create data container
data = siena_data()
add_nodeset!(data, NodeSet(50))

# Add dependent network (3 observation waves)
# Each element is an adjacency matrix
networks = [rand(0:1, 50, 50) for _ in 1:3]
# Remove self-loops
for net in networks
    for i in 1:50
        net[i, i] = 0
    end
end
add_dependent!(data, DependentNetwork(:friendship, networks))

# Add actor covariate
add_covariate!(data, ConstantCovariate(:gender, rand(0:1, 50)))

# Get effects and specify model
effects = get_effects(data)
include_effects!(effects, :friendship, [:outdegree, :recip, :transTrip])

# Configure algorithm
alg = siena_algorithm(seed=42, verbose=true)

# Estimate model
result = siena07(data, effects; algorithm=alg)

# Assess goodness of fit
gof_indeg = siena_gof_indegree(result, data, :friendship; n_sims=100)
```

## Key Functions

### Data Preparation

| Function | Description | RSiena Equivalent |
|----------|-------------|-------------------|
| `siena_data()` | Create data container | `sienaDataCreate()` |
| `NodeSet(n)` | Define node set | `sienaNodeSet()` |
| `DependentNetwork(name, nets)` | Network dependent variable | `sienaDependent(..., type="oneMode")` |
| `DependentBehavior(name, vals)` | Behavior dependent variable | `sienaDependent(..., type="behavior")` |
| `ConstantCovariate(name, vals)` | Time-constant covariate | `coCovar()` |
| `VaryingCovariate(name, vals)` | Time-varying covariate | `varCovar()` |
| `ConstantDyadCovariate(name, mat)` | Dyadic covariate | `coDyadCovar()` |

### Model Specification

| Function | Description | RSiena Equivalent |
|----------|-------------|-------------------|
| `get_effects(data)` | Create effects object | `getEffects()` |
| `include_effects!(effects, var, names)` | Include effects | `includeEffects()` |
| `effects_table(effects)` | View effects as DataFrame | `print(effects)` |

### Estimation

| Function | Description | RSiena Equivalent |
|----------|-------------|-------------------|
| `siena07(data, effects)` | Estimate model | `siena07()` |
| `siena_algorithm(...)` | Configure algorithm | `sienaAlgorithmCreate()` |

### Goodness of Fit

| Function | Description | RSiena Equivalent |
|----------|-------------|-------------------|
| `siena_gof(result, data, stat)` | GOF assessment | `sienaGOF()` |
| `siena_gof_indegree(...)` | Indegree GOF | `sienaGOF(..., IndegreeDistribution)` |
| `siena_gof_outdegree(...)` | Outdegree GOF | `sienaGOF(..., OutdegreeDistribution)` |
| `siena_gof_triad(...)` | Triad census GOF | `sienaGOF(..., TriadCensus)` |

## Available Effects

### Structural Network Effects

- `outdegree` - Density/outdegree effect
- `recip` - Reciprocity
- `transTrip` - Transitive triplets
- `transTies` - Transitive ties
- `cycle3` - Three-cycles
- `inPop`, `inPopSqrt` - Indegree popularity
- `outAct`, `outActSqrt` - Outdegree activity
- `gwesp` - Geometrically weighted edgewise shared partners

### Covariate Effects on Networks

- `egoX` - Ego effect (sender covariate)
- `altX` - Alter effect (receiver covariate)
- `simX` - Similarity effect
- `sameX` - Same value effect (homophily)
- `diffX` - Difference effect
- `dyadX` - Dyadic covariate effect

### Behavior Effects

- `linear` - Linear shape
- `quad` - Quadratic shape
- `avAlt` - Average alter effect
- `avSim` - Average similarity effect
- `totAlt` - Total alter effect
- `effFrom` - Effect from covariate
- `indeg` - Indegree effect on behavior
- `outdeg` - Outdegree effect on behavior

## Model Theory

SAOMs model network change as a sequence of probabilistic micro-steps:

1. **Rate function**: Determines how often each actor gets an opportunity to make a change
2. **Objective function**: Determines which changes actors prefer (ties to create/dissolve, behavior changes)

The objective function for actor *i* considering tie change to actor *j*:

$$f_i(x, z) = \sum_k \beta_k s_{ik}(x, z)$$

where $s_{ik}$ are network statistics and $\beta_k$ are parameters to estimate.

Estimation uses the Method of Moments with stochastic approximation (Robbins-Monro algorithm).

## Differences from RSiena

- **Julia-native**: Uses Julia's type system and multiple dispatch
- **Simplified effects**: Core effects implemented; some advanced effects pending
- **Algorithm**: Basic Robbins-Monro implementation; some advanced features pending

## References

- Snijders, T.A.B. (2017). Stochastic Actor-Oriented Models for Network Dynamics. *Annual Review of Statistics and Its Application*, 4, 343-363.
- Ripley, R.M., Snijders, T.A.B., Boda, Z., Vörös, A., and Preciado, P. (2023). *Manual for RSiena*. University of Oxford.
- [RSiena on CRAN](https://cran.r-project.org/package=RSiena)
- [RSiena on GitHub](https://github.com/stocnet/rsiena)

## License

MIT License. See LICENSE file for details.
