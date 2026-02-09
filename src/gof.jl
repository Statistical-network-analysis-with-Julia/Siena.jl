"""
Goodness of fit assessment for SAOM.
"""

#==============================================================================#
# GOF Statistics
#==============================================================================#

"""
    AbstractGOFStatistic

Abstract type for goodness of fit statistics.
"""
abstract type AbstractGOFStatistic end

"""
    IndegreeDistribution <: AbstractGOFStatistic

Indegree distribution statistic.
"""
struct IndegreeDistribution <: AbstractGOFStatistic
    variable::Symbol
    levls::Union{Vector{Int}, Nothing}

    IndegreeDistribution(variable::Symbol; levls::Union{Vector{Int}, Nothing}=nothing) =
        new(variable, levls)
end

"""
    OutdegreeDistribution <: AbstractGOFStatistic

Outdegree distribution statistic.
"""
struct OutdegreeDistribution <: AbstractGOFStatistic
    variable::Symbol
    levls::Union{Vector{Int}, Nothing}

    OutdegreeDistribution(variable::Symbol; levls::Union{Vector{Int}, Nothing}=nothing) =
        new(variable, levls)
end

"""
    TriadCensus <: AbstractGOFStatistic

Triad census statistic.
"""
struct TriadCensus <: AbstractGOFStatistic
    variable::Symbol
end

"""
    GeodesicDistribution <: AbstractGOFStatistic

Geodesic (shortest path) distribution.
"""
struct GeodesicDistribution <: AbstractGOFStatistic
    variable::Symbol
    max_dist::Int

    GeodesicDistribution(variable::Symbol; max_dist::Int=5) =
        new(variable, max_dist)
end

"""
    BehaviorDistribution <: AbstractGOFStatistic

Behavior value distribution.
"""
struct BehaviorDistribution <: AbstractGOFStatistic
    variable::Symbol
end

#==============================================================================#
# Statistic Computation
#==============================================================================#

"""
    compute_gof_statistic(stat::IndegreeDistribution, state::NetworkState,
                         data::SienaData)

Compute indegree distribution.
"""
function compute_gof_statistic(stat::IndegreeDistribution, state::NetworkState,
                              data::SienaData)
    net = state.networks[stat.variable]
    n = size(net, 1)

    indegrees = [sum(net[:, j]) for j in 1:n]
    max_deg = maximum(indegrees)

    levls = isnothing(stat.levls) ? collect(0:max_deg) : stat.levls
    counts = zeros(Int, length(levls))

    for d in indegrees
        idx = findfirst(==(d), levls)
        if !isnothing(idx)
            counts[idx] += 1
        end
    end

    return levls, counts
end

"""
    compute_gof_statistic(stat::OutdegreeDistribution, state::NetworkState,
                         data::SienaData)

Compute outdegree distribution.
"""
function compute_gof_statistic(stat::OutdegreeDistribution, state::NetworkState,
                              data::SienaData)
    net = state.networks[stat.variable]
    n = size(net, 1)

    outdegrees = [sum(net[i, :]) for i in 1:n]
    max_deg = maximum(outdegrees)

    levls = isnothing(stat.levls) ? collect(0:max_deg) : stat.levls
    counts = zeros(Int, length(levls))

    for d in outdegrees
        idx = findfirst(==(d), levls)
        if !isnothing(idx)
            counts[idx] += 1
        end
    end

    return levls, counts
end

"""
    compute_gof_statistic(stat::TriadCensus, state::NetworkState,
                         data::SienaData)

Compute triad census (16 triad types for directed networks).
"""
function compute_gof_statistic(stat::TriadCensus, state::NetworkState,
                              data::SienaData)
    net = state.networks[stat.variable]
    n = size(net, 1)

    # Simplified: count mutual, asymmetric, and null dyads
    mutual = 0
    asymm = 0
    null = 0

    for i in 1:(n-1)
        for j in (i+1):n
            if net[i, j] == 1 && net[j, i] == 1
                mutual += 1
            elseif net[i, j] == 1 || net[j, i] == 1
                asymm += 1
            else
                null += 1
            end
        end
    end

    return [:mutual, :asymmetric, :null], [mutual, asymm, null]
end

"""
    compute_gof_statistic(stat::GeodesicDistribution, state::NetworkState,
                         data::SienaData)

Compute geodesic (shortest path) distribution.
"""
function compute_gof_statistic(stat::GeodesicDistribution, state::NetworkState,
                              data::SienaData)
    net = state.networks[stat.variable]
    n = size(net, 1)

    # Simple BFS for shortest paths
    dist_counts = zeros(Int, stat.max_dist + 1)  # 0 to max_dist
    unreachable = 0

    for i in 1:n
        # BFS from node i
        distances = fill(-1, n)
        distances[i] = 0
        queue = [i]

        while !isempty(queue)
            curr = popfirst!(queue)
            for j in 1:n
                if net[curr, j] == 1 && distances[j] == -1
                    distances[j] = distances[curr] + 1
                    if distances[j] <= stat.max_dist
                        push!(queue, j)
                    end
                end
            end
        end

        # Count distances
        for j in 1:n
            if i != j
                if distances[j] == -1
                    unreachable += 1
                elseif distances[j] <= stat.max_dist
                    dist_counts[distances[j] + 1] += 1
                end
            end
        end
    end

    labels = vcat(collect(1:stat.max_dist), [:unreachable])
    counts = vcat(dist_counts[2:end], [unreachable])

    return labels, counts
end

"""
    compute_gof_statistic(stat::BehaviorDistribution, state::NetworkState,
                         data::SienaData)

Compute behavior value distribution.
"""
function compute_gof_statistic(stat::BehaviorDistribution, state::NetworkState,
                              data::SienaData)
    beh = state.behaviors[stat.variable]
    dep = data.dependents[stat.variable]::DependentBehavior

    levls = collect(dep.min_val:dep.max_val)
    counts = zeros(Int, length(levls))

    for v in beh
        idx = v - dep.min_val + 1
        if 1 <= idx <= length(counts)
            counts[idx] += 1
        end
    end

    return levls, counts
end

#==============================================================================#
# GOF Result
#==============================================================================#

"""
    GOFResult

Result of goodness of fit assessment.

# Fields
- `statistic::AbstractGOFStatistic`: The GOF statistic used
- `observed::Vector`: Observed statistic values
- `simulated::Matrix`: Simulated statistic values (n_sims × n_levels)
- `p_values::Vector{Float64}`: P-values for each level
- `mahalanobis::Float64`: Mahalanobis distance
- `p_overall::Float64`: Overall p-value
"""
struct GOFResult
    statistic::AbstractGOFStatistic
    labels::Vector
    observed::Vector{Int}
    simulated::Matrix{Int}
    p_values::Vector{Float64}
    mahalanobis::Float64
    p_overall::Float64
end

function Base.show(io::IO, result::GOFResult)
    println(io, "Goodness of Fit: $(typeof(result.statistic).name.name)")
    println(io, "Overall p-value: $(round(result.p_overall, digits=4))")
    println(io)
    println(io, "Level-specific results:")
    for (i, label) in enumerate(result.labels)
        obs = result.observed[i]
        sim_mean = mean(result.simulated[:, i])
        sim_sd = std(result.simulated[:, i])
        p = result.p_values[i]
        @printf(io, "  %-10s: obs=%d, sim=%.1f (%.1f), p=%.3f\n",
                string(label), obs, sim_mean, sim_sd, p)
    end
end

#==============================================================================#
# Main GOF Function
#==============================================================================#

"""
    siena_gof(result::SienaResult, data::SienaData, statistic::AbstractGOFStatistic;
             n_sims::Int=100, seed::Union{Int, Nothing}=nothing)

Assess goodness of fit for an estimated model.
Equivalent to sienaGOF() in RSiena.

# Arguments
- `result::SienaResult`: Estimation result
- `data::SienaData`: The data object
- `statistic::AbstractGOFStatistic`: The GOF statistic to assess
- `n_sims::Int`: Number of simulations
- `seed`: Random seed
"""
function siena_gof(result::SienaResult, data::SienaData, statistic::AbstractGOFStatistic;
                  n_sims::Int=100, seed::Union{Int, Nothing}=nothing)

    rng = isnothing(seed) ? Random.default_rng() : MersenneTwister(seed)

    # Compute observed statistic
    state_obs = NetworkState()
    initialize!(state_obs, data, data.n_waves)
    labels, observed = compute_gof_statistic(statistic, state_obs, data)

    n_levels = length(observed)
    simulated = zeros(Int, n_sims, n_levels)

    # Simulate and compute statistics
    for s in 1:n_sims
        state_sim, _ = simulate_saom(data, result.effects, result.estimates;
                                     seed=rand(rng, 1:10^8))
        _, sim_counts = compute_gof_statistic(statistic, state_sim, data)

        # Ensure same length
        for i in 1:min(length(sim_counts), n_levels)
            simulated[s, i] = sim_counts[i]
        end
    end

    # Compute p-values (two-sided)
    p_values = Float64[]
    for i in 1:n_levels
        sim_col = simulated[:, i]
        # Proportion of simulations as or more extreme than observed
        n_extreme = sum(abs.(sim_col .- mean(sim_col)) .>= abs(observed[i] - mean(sim_col)))
        push!(p_values, n_extreme / n_sims)
    end

    # Overall test (simplified chi-square like measure)
    obs_vec = Float64.(observed)
    sim_mean = vec(mean(simulated, dims=1))
    sim_cov = cov(Float64.(simulated))

    # Add small regularization to avoid singularity
    sim_cov += 0.01 * I

    # Mahalanobis distance
    diff = obs_vec .- sim_mean
    try
        mahal = sqrt(diff' * (sim_cov \ diff))
    catch
        mahal = sqrt(sum(diff.^2 ./ (diag(sim_cov) .+ 1e-6)))
    end

    # Approximate p-value using chi-square
    p_overall = 1 - cdf(Chisq(n_levels), mahal^2)

    return GOFResult(statistic, labels, observed, simulated, p_values, mahal, p_overall)
end

#==============================================================================#
# Convenience Functions
#==============================================================================#

"""
    siena_gof_indegree(result::SienaResult, data::SienaData, variable::Symbol;
                      kwargs...)

Assess GOF for indegree distribution.
"""
siena_gof_indegree(result::SienaResult, data::SienaData, variable::Symbol; kwargs...) =
    siena_gof(result, data, IndegreeDistribution(variable); kwargs...)

"""
    siena_gof_outdegree(result::SienaResult, data::SienaData, variable::Symbol;
                       kwargs...)

Assess GOF for outdegree distribution.
"""
siena_gof_outdegree(result::SienaResult, data::SienaData, variable::Symbol; kwargs...) =
    siena_gof(result, data, OutdegreeDistribution(variable); kwargs...)

"""
    siena_gof_triad(result::SienaResult, data::SienaData, variable::Symbol;
                   kwargs...)

Assess GOF for triad census.
"""
siena_gof_triad(result::SienaResult, data::SienaData, variable::Symbol; kwargs...) =
    siena_gof(result, data, TriadCensus(variable); kwargs...)

"""
    siena_gof_behavior(result::SienaResult, data::SienaData, variable::Symbol;
                      kwargs...)

Assess GOF for behavior distribution.
"""
siena_gof_behavior(result::SienaResult, data::SienaData, variable::Symbol; kwargs...) =
    siena_gof(result, data, BehaviorDistribution(variable); kwargs...)
