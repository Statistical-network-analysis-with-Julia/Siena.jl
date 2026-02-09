"""
Rate effects for SAOM.
"""

#==============================================================================#
# Basic Rate Effects
#==============================================================================#

"""
    BasicRateEffect <: RateEffect

Basic rate parameter for a period.
"""
struct BasicRateEffect <: RateEffect
    variable::Symbol
    period::Int
end

effect_name(e::BasicRateEffect) = Symbol("rate$(e.variable)$(e.period)")
effect_type(::BasicRateEffect) = :rate
target_variable(e::BasicRateEffect) = e.variable

"""
    compute_rate(e::BasicRateEffect, θ::Float64, state::NetworkState,
                data::SienaData, actor::Int)

Compute the rate for a given actor.
"""
function compute_rate(e::BasicRateEffect, θ::Float64, state::NetworkState,
                     data::SienaData, actor::Int)
    return exp(θ)
end

#==============================================================================#
# Covariate Rate Effects
#==============================================================================#

"""
    CovariateRateEffect <: RateEffect

Rate depends on a covariate value.
"""
struct CovariateRateEffect <: RateEffect
    variable::Symbol
    covariate::Symbol
    period::Int
end

effect_name(e::CovariateRateEffect) = Symbol("rate$(e.covariate)")
effect_type(::CovariateRateEffect) = :rate
target_variable(e::CovariateRateEffect) = e.variable
interaction_with(e::CovariateRateEffect) = e.covariate

function compute_rate(e::CovariateRateEffect, θ::Float64, state::NetworkState,
                     data::SienaData, actor::Int)
    cov = data.covariates[e.covariate]
    cov_val = if cov isa ConstantCovariate
        cov.values[actor]
    elseif cov isa VaryingCovariate
        cov.values[min(e.period, length(cov.values))][actor]
    else
        0.0
    end
    return exp(θ * cov_val)
end

#==============================================================================#
# Degree-Based Rate Effects
#==============================================================================#

"""
    OutdegreeRateEffect <: RateEffect

Rate depends on outdegree.
"""
struct OutdegreeRateEffect <: RateEffect
    variable::Symbol
    network::Symbol
    period::Int
end

effect_name(::OutdegreeRateEffect) = :outRateX
effect_type(::OutdegreeRateEffect) = :rate
target_variable(e::OutdegreeRateEffect) = e.variable
interaction_with(e::OutdegreeRateEffect) = e.network

function compute_rate(e::OutdegreeRateEffect, θ::Float64, state::NetworkState,
                     data::SienaData, actor::Int)
    net = state.networks[e.network]
    outdeg = sum(net[actor, :])
    return exp(θ * outdeg)
end

"""
    IndegreeRateEffect <: RateEffect

Rate depends on indegree.
"""
struct IndegreeRateEffect <: RateEffect
    variable::Symbol
    network::Symbol
    period::Int
end

effect_name(::IndegreeRateEffect) = :inRateX
effect_type(::IndegreeRateEffect) = :rate
target_variable(e::IndegreeRateEffect) = e.variable
interaction_with(e::IndegreeRateEffect) = e.network

function compute_rate(e::IndegreeRateEffect, θ::Float64, state::NetworkState,
                     data::SienaData, actor::Int)
    net = state.networks[e.network]
    indeg = sum(net[:, actor])
    return exp(θ * indeg)
end

#==============================================================================#
# Log/Inverse Degree Rate Effects
#==============================================================================#

"""
    OutdegreeLogRateEffect <: RateEffect

Rate depends on log(outdegree + 1).
"""
struct OutdegreeLogRateEffect <: RateEffect
    variable::Symbol
    network::Symbol
    period::Int
end

effect_name(::OutdegreeLogRateEffect) = :outRateLog
effect_type(::OutdegreeLogRateEffect) = :rate
target_variable(e::OutdegreeLogRateEffect) = e.variable
interaction_with(e::OutdegreeLogRateEffect) = e.network

function compute_rate(e::OutdegreeLogRateEffect, θ::Float64, state::NetworkState,
                     data::SienaData, actor::Int)
    net = state.networks[e.network]
    outdeg = sum(net[actor, :])
    return exp(θ * log(outdeg + 1))
end

"""
    IndegreeLogRateEffect <: RateEffect

Rate depends on log(indegree + 1).
"""
struct IndegreeLogRateEffect <: RateEffect
    variable::Symbol
    network::Symbol
    period::Int
end

effect_name(::IndegreeLogRateEffect) = :inRateLog
effect_type(::IndegreeLogRateEffect) = :rate
target_variable(e::IndegreeLogRateEffect) = e.variable
interaction_with(e::IndegreeLogRateEffect) = e.network

function compute_rate(e::IndegreeLogRateEffect, θ::Float64, state::NetworkState,
                     data::SienaData, actor::Int)
    net = state.networks[e.network]
    indeg = sum(net[:, actor])
    return exp(θ * log(indeg + 1))
end

"""
    OutdegreeInvRateEffect <: RateEffect

Rate depends on 1/(outdegree + 1).
"""
struct OutdegreeInvRateEffect <: RateEffect
    variable::Symbol
    network::Symbol
    period::Int
end

effect_name(::OutdegreeInvRateEffect) = :outRateInv
effect_type(::OutdegreeInvRateEffect) = :rate
target_variable(e::OutdegreeInvRateEffect) = e.variable
interaction_with(e::OutdegreeInvRateEffect) = e.network

function compute_rate(e::OutdegreeInvRateEffect, θ::Float64, state::NetworkState,
                     data::SienaData, actor::Int)
    net = state.networks[e.network]
    outdeg = sum(net[actor, :])
    return exp(θ / (outdeg + 1))
end

"""
    IndegreeInvRateEffect <: RateEffect

Rate depends on 1/(indegree + 1).
"""
struct IndegreeInvRateEffect <: RateEffect
    variable::Symbol
    network::Symbol
    period::Int
end

effect_name(::IndegreeInvRateEffect) = :inRateInv
effect_type(::IndegreeInvRateEffect) = :rate
target_variable(e::IndegreeInvRateEffect) = e.variable
interaction_with(e::IndegreeInvRateEffect) = e.network

function compute_rate(e::IndegreeInvRateEffect, θ::Float64, state::NetworkState,
                     data::SienaData, actor::Int)
    net = state.networks[e.network]
    indeg = sum(net[:, actor])
    return exp(θ / (indeg + 1))
end

#==============================================================================#
# Reciprocity-Based Rate Effects
#==============================================================================#

"""
    RecipDegreeRateEffect <: RateEffect

Rate depends on number of reciprocated ties.
"""
struct RecipDegreeRateEffect <: RateEffect
    variable::Symbol
    network::Symbol
    period::Int
end

effect_name(::RecipDegreeRateEffect) = :recipRateX
effect_type(::RecipDegreeRateEffect) = :rate
target_variable(e::RecipDegreeRateEffect) = e.variable
interaction_with(e::RecipDegreeRateEffect) = e.network

function compute_rate(e::RecipDegreeRateEffect, θ::Float64, state::NetworkState,
                     data::SienaData, actor::Int)
    net = state.networks[e.network]
    n = size(net, 1)
    recip = 0
    for j in 1:n
        if j != actor && net[actor, j] == 1 && net[j, actor] == 1
            recip += 1
        end
    end
    return exp(θ * recip)
end

"""
    OutRecipRateEffect <: RateEffect

Rate depends on outgoing ties that are reciprocated.
Same as RecipDegreeRateEffect but named differently in RSiena.
"""
const OutRecipRateEffect = RecipDegreeRateEffect

#==============================================================================#
# Behavior-Based Rate Effects
#==============================================================================#

"""
    BehaviorRateEffect <: RateEffect

Rate depends on actor's own behavior value.
"""
struct BehaviorRateEffect <: RateEffect
    variable::Symbol
    behavior::Symbol
    period::Int
end

effect_name(e::BehaviorRateEffect) = Symbol("behRate$(e.behavior)")
effect_type(::BehaviorRateEffect) = :rate
target_variable(e::BehaviorRateEffect) = e.variable
interaction_with(e::BehaviorRateEffect) = e.behavior

function compute_rate(e::BehaviorRateEffect, θ::Float64, state::NetworkState,
                     data::SienaData, actor::Int)
    beh_val = haskey(state.behaviors, e.behavior) ? state.behaviors[e.behavior][actor] : 0
    # Center the behavior value
    dep = data.dependents[e.behavior]
    if dep isa DependentBehavior
        centered = beh_val - (dep.min_val + dep.max_val) / 2
    else
        centered = Float64(beh_val)
    end
    return exp(θ * centered)
end

"""
    AverageAlterRateEffect <: RateEffect

Rate depends on average behavior of alters (outgoing ties).
"""
struct AverageAlterRateEffect <: RateEffect
    variable::Symbol
    behavior::Symbol
    network::Symbol
    period::Int
end

effect_name(e::AverageAlterRateEffect) = Symbol("avAltRate$(e.behavior)")
effect_type(::AverageAlterRateEffect) = :rate
target_variable(e::AverageAlterRateEffect) = e.variable
interaction_with(e::AverageAlterRateEffect) = e.behavior

function compute_rate(e::AverageAlterRateEffect, θ::Float64, state::NetworkState,
                     data::SienaData, actor::Int)
    net = state.networks[e.network]
    beh = haskey(state.behaviors, e.behavior) ? state.behaviors[e.behavior] : zeros(Int, size(net, 1))

    n = size(net, 1)
    outdeg = sum(net[actor, :])

    if outdeg == 0
        return 1.0  # exp(0) when no alters
    end

    total_beh = 0.0
    for j in 1:n
        if j != actor && net[actor, j] == 1
            total_beh += beh[j]
        end
    end
    avg_beh = total_beh / outdeg

    # Center
    dep = data.dependents[e.behavior]
    if dep isa DependentBehavior
        avg_beh -= (dep.min_val + dep.max_val) / 2
    end

    return exp(θ * avg_beh)
end

"""
    TotalAlterRateEffect <: RateEffect

Rate depends on total behavior of alters (outgoing ties).
"""
struct TotalAlterRateEffect <: RateEffect
    variable::Symbol
    behavior::Symbol
    network::Symbol
    period::Int
end

effect_name(e::TotalAlterRateEffect) = Symbol("totAltRate$(e.behavior)")
effect_type(::TotalAlterRateEffect) = :rate
target_variable(e::TotalAlterRateEffect) = e.variable
interaction_with(e::TotalAlterRateEffect) = e.behavior

function compute_rate(e::TotalAlterRateEffect, θ::Float64, state::NetworkState,
                     data::SienaData, actor::Int)
    net = state.networks[e.network]
    beh = haskey(state.behaviors, e.behavior) ? state.behaviors[e.behavior] : zeros(Int, size(net, 1))

    n = size(net, 1)
    total_beh = 0.0
    for j in 1:n
        if j != actor && net[actor, j] == 1
            total_beh += beh[j]
        end
    end

    return exp(θ * total_beh)
end

#==============================================================================#
# Similarity-Based Rate Effects
#==============================================================================#

"""
    SimilarityRateEffect <: RateEffect

Rate depends on average similarity with alters on behavior.
"""
struct SimilarityRateEffect <: RateEffect
    variable::Symbol
    behavior::Symbol
    network::Symbol
    period::Int
end

effect_name(e::SimilarityRateEffect) = Symbol("simRate$(e.behavior)")
effect_type(::SimilarityRateEffect) = :rate
target_variable(e::SimilarityRateEffect) = e.variable
interaction_with(e::SimilarityRateEffect) = e.behavior

function compute_rate(e::SimilarityRateEffect, θ::Float64, state::NetworkState,
                     data::SienaData, actor::Int)
    net = state.networks[e.network]
    beh = haskey(state.behaviors, e.behavior) ? state.behaviors[e.behavior] : zeros(Int, size(net, 1))

    dep = data.dependents[e.behavior]
    beh_range = dep isa DependentBehavior ? (dep.max_val - dep.min_val) : 1
    if beh_range == 0
        beh_range = 1
    end

    n = size(net, 1)
    outdeg = sum(net[actor, :])

    if outdeg == 0
        return 1.0
    end

    total_sim = 0.0
    ego_val = beh[actor]
    for j in 1:n
        if j != actor && net[actor, j] == 1
            sim = 1.0 - abs(ego_val - beh[j]) / beh_range
            total_sim += sim
        end
    end
    avg_sim = total_sim / outdeg

    return exp(θ * avg_sim)
end

#==============================================================================#
# Setting/Group Rate Effects
#==============================================================================#

"""
    SettingRateEffect <: RateEffect

Rate effect for settings/groups (primary setting model).
Rate multiplier for actors in a particular setting.
"""
struct SettingRateEffect <: RateEffect
    variable::Symbol
    setting::Symbol  # Covariate indicating setting membership
    setting_value::Int  # The setting value to match
    period::Int
end

effect_name(e::SettingRateEffect) = Symbol("settingRate$(e.setting)")
effect_type(::SettingRateEffect) = :rate
target_variable(e::SettingRateEffect) = e.variable
interaction_with(e::SettingRateEffect) = e.setting

function compute_rate(e::SettingRateEffect, θ::Float64, state::NetworkState,
                     data::SienaData, actor::Int)
    cov = data.covariates[e.setting]
    cov_val = if cov isa ConstantCovariate
        cov.values[actor]
    elseif cov isa VaryingCovariate
        cov.values[min(e.period, length(cov.values))][actor]
    else
        0.0
    end

    # Rate multiplier only applies if actor is in this setting
    if round(Int, cov_val) == e.setting_value
        return exp(θ)
    else
        return 1.0  # No effect
    end
end

#==============================================================================#
# Ego x Alter Interaction Rate Effects
#==============================================================================#

"""
    EgoAlterRateEffect <: RateEffect

Rate depends on product of ego and alter covariate values, summed over ties.
"""
struct EgoAlterRateEffect <: RateEffect
    variable::Symbol
    covariate::Symbol
    network::Symbol
    period::Int
end

effect_name(e::EgoAlterRateEffect) = Symbol("egoAltRate$(e.covariate)")
effect_type(::EgoAlterRateEffect) = :rate
target_variable(e::EgoAlterRateEffect) = e.variable
interaction_with(e::EgoAlterRateEffect) = e.covariate

function compute_rate(e::EgoAlterRateEffect, θ::Float64, state::NetworkState,
                     data::SienaData, actor::Int)
    net = state.networks[e.network]
    cov = data.covariates[e.covariate]

    n = size(net, 1)
    ego_val = if cov isa ConstantCovariate
        cov.values[actor]
    elseif cov isa VaryingCovariate
        cov.values[min(e.period, length(cov.values))][actor]
    else
        0.0
    end

    total = 0.0
    for j in 1:n
        if j != actor && net[actor, j] == 1
            alt_val = if cov isa ConstantCovariate
                cov.values[j]
            elseif cov isa VaryingCovariate
                cov.values[min(e.period, length(cov.values))][j]
            else
                0.0
            end
            total += ego_val * alt_val
        end
    end

    return exp(θ * total)
end

#==============================================================================#
# Squared/Quadratic Rate Effects
#==============================================================================#

"""
    CovariateSqRateEffect <: RateEffect

Rate depends on squared covariate value.
"""
struct CovariateSqRateEffect <: RateEffect
    variable::Symbol
    covariate::Symbol
    period::Int
end

effect_name(e::CovariateSqRateEffect) = Symbol("rateSq$(e.covariate)")
effect_type(::CovariateSqRateEffect) = :rate
target_variable(e::CovariateSqRateEffect) = e.variable
interaction_with(e::CovariateSqRateEffect) = e.covariate

function compute_rate(e::CovariateSqRateEffect, θ::Float64, state::NetworkState,
                     data::SienaData, actor::Int)
    cov = data.covariates[e.covariate]
    cov_val = if cov isa ConstantCovariate
        cov.values[actor]
    elseif cov isa VaryingCovariate
        cov.values[min(e.period, length(cov.values))][actor]
    else
        0.0
    end
    return exp(θ * cov_val^2)
end

"""
    OutdegreeSqRateEffect <: RateEffect

Rate depends on squared outdegree.
"""
struct OutdegreeSqRateEffect <: RateEffect
    variable::Symbol
    network::Symbol
    period::Int
end

effect_name(::OutdegreeSqRateEffect) = :outRateSq
effect_type(::OutdegreeSqRateEffect) = :rate
target_variable(e::OutdegreeSqRateEffect) = e.variable
interaction_with(e::OutdegreeSqRateEffect) = e.network

function compute_rate(e::OutdegreeSqRateEffect, θ::Float64, state::NetworkState,
                     data::SienaData, actor::Int)
    net = state.networks[e.network]
    outdeg = sum(net[actor, :])
    return exp(θ * outdeg^2)
end

#==============================================================================#
# Rate Function Computation
#==============================================================================#

"""
    compute_total_rate(effects::Vector{<:RateEffect}, θ::Vector{Float64},
                      state::NetworkState, data::SienaData, actor::Int)

Compute the total rate for an actor given all rate effects and parameters.
If θ is empty or shorter than effects, uses default rate of 1.0.
"""
function compute_total_rate(effects::Vector{<:RateEffect}, θ::Vector{Float64},
                           state::NetworkState, data::SienaData, actor::Int)
    if isempty(effects) || isempty(θ)
        return 1.0
    end

    total = 0.0
    for (i, eff) in enumerate(effects)
        param = i <= length(θ) ? θ[i] : 0.0  # Default to exp(0) = 1.0 rate
        total += compute_rate(eff, param, state, data, actor)
    end
    return total
end

"""
    sample_waiting_time(rate::Float64, rng::AbstractRNG)

Sample waiting time from exponential distribution with given rate.
"""
function sample_waiting_time(rate::Float64, rng::AbstractRNG)
    return -log(rand(rng)) / rate
end
