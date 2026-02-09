"""
Behavior effects for SAOM evaluation function.
Complete implementation of RSiena behavior effects.
"""

#==============================================================================#
# Helper function for behavior covariate access
#==============================================================================#

function _get_beh_covariate_value(cov::AbstractCovariate, actor::Int, wave::Int)
    if cov isa ConstantCovariate
        return cov.values[actor]
    elseif cov isa VaryingCovariate
        w = min(wave, length(cov.values))
        return cov.values[w][actor]
    end
    return 0.0
end

#==============================================================================#
# Basic Shape Effects
#==============================================================================#

"""
    LinearShapeEffect <: BehaviorEffect

Linear shape effect. RSiena: linear
"""
struct LinearShapeEffect <: BehaviorEffect
    variable::Symbol
end

effect_name(::LinearShapeEffect) = :linear
effect_type(::LinearShapeEffect) = :eval
target_variable(e::LinearShapeEffect) = e.variable

function compute_contribution(e::LinearShapeEffect, state::NetworkState,
                             data::SienaData, actor::Int, direction::Int)
    beh = state.behaviors[e.variable]
    return Float64(beh[actor] + direction)
end

"""
    QuadraticShapeEffect <: BehaviorEffect

Quadratic shape effect. RSiena: quad
"""
struct QuadraticShapeEffect <: BehaviorEffect
    variable::Symbol
end

effect_name(::QuadraticShapeEffect) = :quad
effect_type(::QuadraticShapeEffect) = :eval
target_variable(e::QuadraticShapeEffect) = e.variable

function compute_contribution(e::QuadraticShapeEffect, state::NetworkState,
                             data::SienaData, actor::Int, direction::Int)
    beh = state.behaviors[e.variable]
    dep = data.dependents[e.variable]::DependentBehavior
    mean_val = (dep.min_val + dep.max_val) / 2.0
    new_val = beh[actor] + direction
    return (new_val - mean_val)^2
end

"""
    CubicShapeEffect <: BehaviorEffect

Cubic shape effect. RSiena: cubic
"""
struct CubicShapeEffect <: BehaviorEffect
    variable::Symbol
end

effect_name(::CubicShapeEffect) = :cubic
effect_type(::CubicShapeEffect) = :eval
target_variable(e::CubicShapeEffect) = e.variable

function compute_contribution(e::CubicShapeEffect, state::NetworkState,
                             data::SienaData, actor::Int, direction::Int)
    beh = state.behaviors[e.variable]
    dep = data.dependents[e.variable]::DependentBehavior
    mean_val = (dep.min_val + dep.max_val) / 2.0
    new_val = beh[actor] + direction
    return (new_val - mean_val)^3
end

#==============================================================================#
# Network Influence Effects - Average-based
#==============================================================================#

"""
    AverageAlterEffect <: BehaviorEffect

Average alter's behavior effect. RSiena: avAlt
"""
struct AverageAlterEffect <: BehaviorEffect
    variable::Symbol
    network::Symbol
end

effect_name(::AverageAlterEffect) = :avAlt
effect_type(::AverageAlterEffect) = :eval
target_variable(e::AverageAlterEffect) = e.variable
interaction_with(e::AverageAlterEffect) = e.network

function compute_contribution(e::AverageAlterEffect, state::NetworkState,
                             data::SienaData, actor::Int, direction::Int)
    beh = state.behaviors[e.variable]
    net = state.networks[e.network]
    neighbors = findall(net[actor, :] .== 1)
    isempty(neighbors) && return 0.0
    avg_alter = mean([beh[j] for j in neighbors])
    return Float64(beh[actor] + direction) * avg_alter
end

"""
    AverageSimilarityEffect <: BehaviorEffect

Average similarity effect. RSiena: avSim
"""
struct AverageSimilarityEffect <: BehaviorEffect
    variable::Symbol
    network::Symbol
end

effect_name(::AverageSimilarityEffect) = :avSim
effect_type(::AverageSimilarityEffect) = :eval
target_variable(e::AverageSimilarityEffect) = e.variable
interaction_with(e::AverageSimilarityEffect) = e.network

function compute_contribution(e::AverageSimilarityEffect, state::NetworkState,
                             data::SienaData, actor::Int, direction::Int)
    beh = state.behaviors[e.variable]
    net = state.networks[e.network]
    dep = data.dependents[e.variable]::DependentBehavior
    neighbors = findall(net[actor, :] .== 1)
    isempty(neighbors) && return 0.0

    range_val = Float64(dep.max_val - dep.min_val)
    range_val == 0 && return 0.0
    new_val = beh[actor] + direction

    sim_sum = sum(1.0 - abs(new_val - beh[j]) / range_val for j in neighbors)
    return sim_sum / length(neighbors)
end

"""
    AverageInAlterEffect <: BehaviorEffect

Average in-alter's behavior. RSiena: avInAlt
"""
struct AverageInAlterEffect <: BehaviorEffect
    variable::Symbol
    network::Symbol
end

effect_name(::AverageInAlterEffect) = :avInAlt
effect_type(::AverageInAlterEffect) = :eval
target_variable(e::AverageInAlterEffect) = e.variable
interaction_with(e::AverageInAlterEffect) = e.network

function compute_contribution(e::AverageInAlterEffect, state::NetworkState,
                             data::SienaData, actor::Int, direction::Int)
    beh = state.behaviors[e.variable]
    net = state.networks[e.network]
    # Actors who send ties to ego
    in_neighbors = findall(net[:, actor] .== 1)
    isempty(in_neighbors) && return 0.0
    avg_in_alter = mean([beh[j] for j in in_neighbors])
    return Float64(beh[actor] + direction) * avg_in_alter
end

"""
    AverageRecipAlterEffect <: BehaviorEffect

Average reciprocal alter's behavior. RSiena: avRecAlt
"""
struct AverageRecipAlterEffect <: BehaviorEffect
    variable::Symbol
    network::Symbol
end

effect_name(::AverageRecipAlterEffect) = :avRecAlt
effect_type(::AverageRecipAlterEffect) = :eval
target_variable(e::AverageRecipAlterEffect) = e.variable
interaction_with(e::AverageRecipAlterEffect) = e.network

function compute_contribution(e::AverageRecipAlterEffect, state::NetworkState,
                             data::SienaData, actor::Int, direction::Int)
    beh = state.behaviors[e.variable]
    net = state.networks[e.network]
    # Reciprocal ties
    recip_neighbors = findall((net[actor, :] .== 1) .& (net[:, actor] .== 1))
    isempty(recip_neighbors) && return 0.0
    avg_recip = mean([beh[j] for j in recip_neighbors])
    return Float64(beh[actor] + direction) * avg_recip
end

"""
    AverageAttHigherEffect <: BehaviorEffect

Average attraction to higher. RSiena: avAttHigher
"""
struct AverageAttHigherEffect <: BehaviorEffect
    variable::Symbol
    network::Symbol
end

effect_name(::AverageAttHigherEffect) = :avAttHigher
effect_type(::AverageAttHigherEffect) = :eval
target_variable(e::AverageAttHigherEffect) = e.variable
interaction_with(e::AverageAttHigherEffect) = e.network

function compute_contribution(e::AverageAttHigherEffect, state::NetworkState,
                             data::SienaData, actor::Int, direction::Int)
    beh = state.behaviors[e.variable]
    net = state.networks[e.network]
    neighbors = findall(net[actor, :] .== 1)
    isempty(neighbors) && return 0.0

    new_val = beh[actor] + direction
    # Count alters with higher behavior
    higher_count = sum(beh[j] > new_val ? 1.0 : 0.0 for j in neighbors)
    return higher_count / length(neighbors)
end

"""
    AverageAttLowerEffect <: BehaviorEffect

Average attraction to lower. RSiena: avAttLower
"""
struct AverageAttLowerEffect <: BehaviorEffect
    variable::Symbol
    network::Symbol
end

effect_name(::AverageAttLowerEffect) = :avAttLower
effect_type(::AverageAttLowerEffect) = :eval
target_variable(e::AverageAttLowerEffect) = e.variable
interaction_with(e::AverageAttLowerEffect) = e.network

function compute_contribution(e::AverageAttLowerEffect, state::NetworkState,
                             data::SienaData, actor::Int, direction::Int)
    beh = state.behaviors[e.variable]
    net = state.networks[e.network]
    neighbors = findall(net[actor, :] .== 1)
    isempty(neighbors) && return 0.0

    new_val = beh[actor] + direction
    lower_count = sum(beh[j] < new_val ? 1.0 : 0.0 for j in neighbors)
    return lower_count / length(neighbors)
end

#==============================================================================#
# Network Influence Effects - Total-based
#==============================================================================#

"""
    TotalAlterEffect <: BehaviorEffect

Total alter's behavior. RSiena: totAlt
"""
struct TotalAlterEffect <: BehaviorEffect
    variable::Symbol
    network::Symbol
end

effect_name(::TotalAlterEffect) = :totAlt
effect_type(::TotalAlterEffect) = :eval
target_variable(e::TotalAlterEffect) = e.variable
interaction_with(e::TotalAlterEffect) = e.network

function compute_contribution(e::TotalAlterEffect, state::NetworkState,
                             data::SienaData, actor::Int, direction::Int)
    beh = state.behaviors[e.variable]
    net = state.networks[e.network]
    neighbors = findall(net[actor, :] .== 1)
    isempty(neighbors) && return 0.0
    total_alter = sum(beh[j] for j in neighbors)
    return Float64(beh[actor] + direction) * Float64(total_alter)
end

"""
    TotalSimilarityEffect <: BehaviorEffect

Total similarity effect. RSiena: totSim
"""
struct TotalSimilarityEffect <: BehaviorEffect
    variable::Symbol
    network::Symbol
end

effect_name(::TotalSimilarityEffect) = :totSim
effect_type(::TotalSimilarityEffect) = :eval
target_variable(e::TotalSimilarityEffect) = e.variable
interaction_with(e::TotalSimilarityEffect) = e.network

function compute_contribution(e::TotalSimilarityEffect, state::NetworkState,
                             data::SienaData, actor::Int, direction::Int)
    beh = state.behaviors[e.variable]
    net = state.networks[e.network]
    dep = data.dependents[e.variable]::DependentBehavior
    neighbors = findall(net[actor, :] .== 1)
    isempty(neighbors) && return 0.0

    range_val = Float64(dep.max_val - dep.min_val)
    range_val == 0 && return 0.0
    new_val = beh[actor] + direction

    return sum(1.0 - abs(new_val - beh[j]) / range_val for j in neighbors)
end

"""
    TotalInAlterEffect <: BehaviorEffect

Total in-alter's behavior. RSiena: totInAlt
"""
struct TotalInAlterEffect <: BehaviorEffect
    variable::Symbol
    network::Symbol
end

effect_name(::TotalInAlterEffect) = :totInAlt
effect_type(::TotalInAlterEffect) = :eval
target_variable(e::TotalInAlterEffect) = e.variable
interaction_with(e::TotalInAlterEffect) = e.network

function compute_contribution(e::TotalInAlterEffect, state::NetworkState,
                             data::SienaData, actor::Int, direction::Int)
    beh = state.behaviors[e.variable]
    net = state.networks[e.network]
    in_neighbors = findall(net[:, actor] .== 1)
    isempty(in_neighbors) && return 0.0
    total_in = sum(beh[j] for j in in_neighbors)
    return Float64(beh[actor] + direction) * Float64(total_in)
end

#==============================================================================#
# Distance-2 Influence Effects
#==============================================================================#

"""
    AverageAlterDist2Effect <: BehaviorEffect

Average alter at distance 2. RSiena: avAltDist2
"""
struct AverageAlterDist2Effect <: BehaviorEffect
    variable::Symbol
    network::Symbol
end

effect_name(::AverageAlterDist2Effect) = :avAltDist2
effect_type(::AverageAlterDist2Effect) = :eval
target_variable(e::AverageAlterDist2Effect) = e.variable
interaction_with(e::AverageAlterDist2Effect) = e.network

function compute_contribution(e::AverageAlterDist2Effect, state::NetworkState,
                             data::SienaData, actor::Int, direction::Int)
    beh = state.behaviors[e.variable]
    net = state.networks[e.network]
    n = size(net, 1)

    # Find actors at distance 2 (friends of friends, not direct friends)
    direct = Set(findall(net[actor, :] .== 1))
    dist2 = Set{Int}()
    for j in direct
        for k in 1:n
            if net[j, k] == 1 && k != actor && !(k in direct)
                push!(dist2, k)
            end
        end
    end

    isempty(dist2) && return 0.0
    avg_dist2 = mean([beh[k] for k in dist2])
    return Float64(beh[actor] + direction) * avg_dist2
end

#==============================================================================#
# Degree Effects on Behavior
#==============================================================================#

"""
    IndegreeEffect <: BehaviorEffect

Indegree effect on behavior. RSiena: indeg
"""
struct IndegreeEffect <: BehaviorEffect
    variable::Symbol
    network::Symbol
end

effect_name(::IndegreeEffect) = :indeg
effect_type(::IndegreeEffect) = :eval
target_variable(e::IndegreeEffect) = e.variable
interaction_with(e::IndegreeEffect) = e.network

function compute_contribution(e::IndegreeEffect, state::NetworkState,
                             data::SienaData, actor::Int, direction::Int)
    beh = state.behaviors[e.variable]
    net = state.networks[e.network]
    new_val = beh[actor] + direction
    indeg = sum(net[:, actor])
    return Float64(new_val) * Float64(indeg)
end

"""
    BehaviorOutdegreeEffect <: BehaviorEffect

Outdegree effect on behavior. RSiena: outdeg
"""
struct BehaviorOutdegreeEffect <: BehaviorEffect
    variable::Symbol
    network::Symbol
end

effect_name(::BehaviorOutdegreeEffect) = :outdeg
effect_type(::BehaviorOutdegreeEffect) = :eval
target_variable(e::BehaviorOutdegreeEffect) = e.variable
interaction_with(e::BehaviorOutdegreeEffect) = e.network

function compute_contribution(e::BehaviorOutdegreeEffect, state::NetworkState,
                             data::SienaData, actor::Int, direction::Int)
    beh = state.behaviors[e.variable]
    net = state.networks[e.network]
    new_val = beh[actor] + direction
    outdeg = sum(net[actor, :])
    return Float64(new_val) * Float64(outdeg)
end

"""
    RecipDegreeEffect <: BehaviorEffect

Reciprocal degree effect. RSiena: recipDeg
"""
struct RecipDegreeEffect <: BehaviorEffect
    variable::Symbol
    network::Symbol
end

effect_name(::RecipDegreeEffect) = :recipDeg
effect_type(::RecipDegreeEffect) = :eval
target_variable(e::RecipDegreeEffect) = e.variable
interaction_with(e::RecipDegreeEffect) = e.network

function compute_contribution(e::RecipDegreeEffect, state::NetworkState,
                             data::SienaData, actor::Int, direction::Int)
    beh = state.behaviors[e.variable]
    net = state.networks[e.network]
    new_val = beh[actor] + direction
    recip_deg = sum((net[actor, :] .== 1) .& (net[:, actor] .== 1))
    return Float64(new_val) * Float64(recip_deg)
end

#==============================================================================#
# Covariate Effects on Behavior
#==============================================================================#

"""
    BehaviorCovariateEffect <: BehaviorEffect

Effect from covariate. RSiena: effFrom
"""
struct BehaviorCovariateEffect <: BehaviorEffect
    variable::Symbol
    covariate::Symbol
end

effect_name(::BehaviorCovariateEffect) = :effFrom
effect_type(::BehaviorCovariateEffect) = :eval
target_variable(e::BehaviorCovariateEffect) = e.variable
interaction_with(e::BehaviorCovariateEffect) = e.covariate

function compute_contribution(e::BehaviorCovariateEffect, state::NetworkState,
                             data::SienaData, actor::Int, direction::Int)
    beh = state.behaviors[e.variable]
    cov_val = _get_beh_covariate_value(data.covariates[e.covariate], actor, 1)
    return Float64(beh[actor] + direction) * cov_val
end

"""
    CovariateInteractionEffect <: BehaviorEffect

Covariate × behavior interaction.
"""
struct CovariateInteractionEffect <: BehaviorEffect
    variable::Symbol
    covariate::Symbol
end

effect_name(::CovariateInteractionEffect) = :covInt
effect_type(::CovariateInteractionEffect) = :eval
target_variable(e::CovariateInteractionEffect) = e.variable
interaction_with(e::CovariateInteractionEffect) = e.covariate

function compute_contribution(e::CovariateInteractionEffect, state::NetworkState,
                             data::SienaData, actor::Int, direction::Int)
    beh = state.behaviors[e.variable]
    cov_val = _get_beh_covariate_value(data.covariates[e.covariate], actor, 1)
    new_val = beh[actor] + direction
    return Float64(new_val^2) * cov_val
end

#==============================================================================#
# Behavior-Behavior Effects
#==============================================================================#

"""
    BehaviorInteractionEffect <: BehaviorEffect

Effect of one behavior on another. RSiena: behBeh
"""
struct BehaviorInteractionEffect <: BehaviorEffect
    variable::Symbol
    other_behavior::Symbol
end

effect_name(::BehaviorInteractionEffect) = :behBeh
effect_type(::BehaviorInteractionEffect) = :eval
target_variable(e::BehaviorInteractionEffect) = e.variable
interaction_with(e::BehaviorInteractionEffect) = e.other_behavior

function compute_contribution(e::BehaviorInteractionEffect, state::NetworkState,
                             data::SienaData, actor::Int, direction::Int)
    beh = state.behaviors[e.variable]
    other = state.behaviors[e.other_behavior]
    return Float64(beh[actor] + direction) * Float64(other[actor])
end

"""
    BehaviorSimilarityEffect <: BehaviorEffect

Similarity in other behavior. RSiena: simBeh
"""
struct BehaviorSimilarityEffect <: BehaviorEffect
    variable::Symbol
    other_behavior::Symbol
    network::Symbol
end

effect_name(::BehaviorSimilarityEffect) = :simBeh
effect_type(::BehaviorSimilarityEffect) = :eval
target_variable(e::BehaviorSimilarityEffect) = e.variable
interaction_with(e::BehaviorSimilarityEffect) = e.other_behavior

function compute_contribution(e::BehaviorSimilarityEffect, state::NetworkState,
                             data::SienaData, actor::Int, direction::Int)
    beh = state.behaviors[e.variable]
    other = state.behaviors[e.other_behavior]
    net = state.networks[e.network]

    neighbors = findall(net[actor, :] .== 1)
    isempty(neighbors) && return 0.0

    new_val = beh[actor] + direction
    # Similarity based on other behavior
    other_dep = data.dependents[e.other_behavior]::DependentBehavior
    range_val = Float64(other_dep.max_val - other_dep.min_val)
    range_val == 0 && return 0.0

    sim_sum = sum(1.0 - abs(other[actor] - other[j]) / range_val for j in neighbors)
    return Float64(new_val) * sim_sum / length(neighbors)
end

#==============================================================================#
# Threshold Effects
#==============================================================================#

"""
    ThresholdEffect <: BehaviorEffect

Threshold effect. RSiena: threshold
"""
struct ThresholdEffect <: BehaviorEffect
    variable::Symbol
    threshold::Int
end

effect_name(::ThresholdEffect) = :threshold
effect_type(::ThresholdEffect) = :eval
target_variable(e::ThresholdEffect) = e.variable

function compute_contribution(e::ThresholdEffect, state::NetworkState,
                             data::SienaData, actor::Int, direction::Int)
    beh = state.behaviors[e.variable]
    new_val = beh[actor] + direction
    return new_val >= e.threshold ? 1.0 : 0.0
end

"""
    PropThresholdEffect <: BehaviorEffect

Proportional threshold effect.
"""
struct PropThresholdEffect <: BehaviorEffect
    variable::Symbol
    network::Symbol
    threshold::Float64
end

effect_name(::PropThresholdEffect) = :propThreshold
effect_type(::PropThresholdEffect) = :eval
target_variable(e::PropThresholdEffect) = e.variable
interaction_with(e::PropThresholdEffect) = e.network

function compute_contribution(e::PropThresholdEffect, state::NetworkState,
                             data::SienaData, actor::Int, direction::Int)
    beh = state.behaviors[e.variable]
    net = state.networks[e.network]
    dep = data.dependents[e.variable]::DependentBehavior

    neighbors = findall(net[actor, :] .== 1)
    isempty(neighbors) && return 0.0

    new_val = beh[actor] + direction
    # Proportion of alters above threshold
    n_above = sum(beh[j] >= dep.max_val ? 1.0 : 0.0 for j in neighbors)
    prop = n_above / length(neighbors)
    return prop >= e.threshold ? Float64(new_val) : 0.0
end

#==============================================================================#
# Isolate Effects
#==============================================================================#

"""
    BehaviorIsolateEffect <: BehaviorEffect

Isolate effect on behavior. RSiena: isolate
"""
struct BehaviorIsolateEffect <: BehaviorEffect
    variable::Symbol
    network::Symbol
end

effect_name(::BehaviorIsolateEffect) = :behIsolate
effect_type(::BehaviorIsolateEffect) = :eval
target_variable(e::BehaviorIsolateEffect) = e.variable
interaction_with(e::BehaviorIsolateEffect) = e.network

function compute_contribution(e::BehaviorIsolateEffect, state::NetworkState,
                             data::SienaData, actor::Int, direction::Int)
    beh = state.behaviors[e.variable]
    net = state.networks[e.network]
    new_val = beh[actor] + direction
    outdeg = sum(net[actor, :])
    indeg = sum(net[:, actor])
    return (outdeg == 0 && indeg == 0) ? Float64(new_val) : 0.0
end

#==============================================================================#
# Feedback Effects
#==============================================================================#

"""
    FeedbackEffect <: BehaviorEffect

Feedback from network selection. RSiena: feedback
"""
struct FeedbackEffect <: BehaviorEffect
    variable::Symbol
    network::Symbol
end

effect_name(::FeedbackEffect) = :feedback
effect_type(::FeedbackEffect) = :eval
target_variable(e::FeedbackEffect) = e.variable
interaction_with(e::FeedbackEffect) = e.network

function compute_contribution(e::FeedbackEffect, state::NetworkState,
                             data::SienaData, actor::Int, direction::Int)
    beh = state.behaviors[e.variable]
    net = state.networks[e.network]

    new_val = beh[actor] + direction
    # Product of similarity with all alters
    neighbors = findall(net[actor, :] .== 1)
    isempty(neighbors) && return 0.0

    dep = data.dependents[e.variable]::DependentBehavior
    range_val = Float64(dep.max_val - dep.min_val)
    range_val == 0 && return 0.0

    sim_prod = 1.0
    for j in neighbors
        sim = 1.0 - abs(new_val - beh[j]) / range_val
        sim_prod *= sim
    end
    return sim_prod
end

#==============================================================================#
# Main Effect (for compatibility)
#==============================================================================#

"""
    MainBehaviorEffect <: BehaviorEffect

Main effect (constant tendency).
"""
struct MainBehaviorEffect <: BehaviorEffect
    variable::Symbol
end

effect_name(::MainBehaviorEffect) = :main
effect_type(::MainBehaviorEffect) = :eval
target_variable(e::MainBehaviorEffect) = e.variable

function compute_contribution(e::MainBehaviorEffect, state::NetworkState,
                             data::SienaData, actor::Int, direction::Int)
    return Float64(direction)
end
