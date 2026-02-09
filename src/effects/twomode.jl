"""
Two-mode (bipartite) network effects for SAOM.

Two-mode networks connect actors to events/affiliations rather than to other actors.
These effects are used when the network is bipartite (e.g., actors attending events,
students in classes, employees in projects).
"""

#==============================================================================#
# Abstract Type
#==============================================================================#

"""
    TwoModeEffect <: NetworkEffect

Abstract type for two-mode network effects.
"""
abstract type TwoModeEffect <: NetworkEffect end

#==============================================================================#
# Basic Two-Mode Effects
#==============================================================================#

"""
    TwoModeOutdegreeEffect <: TwoModeEffect

Outdegree effect for two-mode networks (number of events attended).
"""
struct TwoModeOutdegreeEffect <: TwoModeEffect
    variable::Symbol
end

effect_name(::TwoModeOutdegreeEffect) = :outdegree2
effect_type(::TwoModeOutdegreeEffect) = :eval
target_variable(e::TwoModeOutdegreeEffect) = e.variable

function compute_contribution(e::TwoModeOutdegreeEffect, state::NetworkState,
                             data::SienaData, actor::Int, event::Int)
    return 1.0  # Contribution is always 1 for adding a tie
end

"""
    TwoModeIndegreeEffect <: TwoModeEffect

Indegree effect for two-mode networks (popularity of events).
"""
struct TwoModeIndegreeEffect <: TwoModeEffect
    variable::Symbol
    sqrt::Bool
end

TwoModeIndegreeEffect(var::Symbol; sqrt::Bool=false) =
    TwoModeIndegreeEffect(var, sqrt)

effect_name(e::TwoModeIndegreeEffect) = e.sqrt ? :indegreeSqrt2 : :indegree2
effect_type(::TwoModeIndegreeEffect) = :eval
target_variable(e::TwoModeIndegreeEffect) = e.variable

function compute_contribution(e::TwoModeIndegreeEffect, state::NetworkState,
                             data::SienaData, actor::Int, event::Int)
    net = state.networks[e.variable]
    # Count how many actors are connected to this event (excluding current actor)
    indeg = sum(net[:, event]) - net[actor, event]
    if e.sqrt
        return sqrt(indeg + 1) - sqrt(indeg)
    else
        return Float64(indeg)
    end
end

#==============================================================================#
# Shared Partners Effects (Two-Mode)
#==============================================================================#

"""
    FourCyclesEffect <: TwoModeEffect

Four-cycles effect: tendency for actors who share events to share more events.
This is the two-mode analogue of transitivity.
"""
struct FourCyclesEffect <: TwoModeEffect
    variable::Symbol
end

effect_name(::FourCyclesEffect) = :fourCycles
effect_type(::FourCyclesEffect) = :eval
target_variable(e::FourCyclesEffect) = e.variable

function compute_contribution(e::FourCyclesEffect, state::NetworkState,
                             data::SienaData, actor::Int, event::Int)
    net = state.networks[e.variable]
    n_actors = size(net, 1)

    # Count shared events with other actors who also attend this event
    count = 0
    for other in 1:n_actors
        if other != actor && net[other, event] == 1
            # Count shared events between actor and other
            shared = sum(net[actor, :] .* net[other, :])
            count += shared
        end
    end
    return Float64(count)
end

"""
    SharedEventsEffect <: TwoModeEffect

Number of events shared with each alter connected to same events.
"""
struct SharedEventsEffect <: TwoModeEffect
    variable::Symbol
    sqrt::Bool
end

SharedEventsEffect(var::Symbol; sqrt::Bool=false) =
    SharedEventsEffect(var, sqrt)

effect_name(e::SharedEventsEffect) = e.sqrt ? :sharedEventsSqrt : :sharedEvents
effect_type(::SharedEventsEffect) = :eval
target_variable(e::SharedEventsEffect) = e.variable

function compute_contribution(e::SharedEventsEffect, state::NetworkState,
                             data::SienaData, actor::Int, event::Int)
    net = state.networks[e.variable]
    n_actors = size(net, 1)

    total = 0.0
    for other in 1:n_actors
        if other != actor && net[other, event] == 1
            shared = sum(net[actor, :] .* net[other, :])
            if e.sqrt
                total += sqrt(shared + 1) - sqrt(shared)
            else
                total += shared
            end
        end
    end
    return total
end

"""
    GWESPTwoModeEffect <: TwoModeEffect

Geometrically weighted shared partners for two-mode networks.
"""
struct GWESPTwoModeEffect <: TwoModeEffect
    variable::Symbol
    α::Float64  # Decay parameter
end

GWESPTwoModeEffect(var::Symbol; α::Float64=0.69) =
    GWESPTwoModeEffect(var, α)

effect_name(::GWESPTwoModeEffect) = :gwesp2
effect_type(::GWESPTwoModeEffect) = :eval
target_variable(e::GWESPTwoModeEffect) = e.variable

function compute_contribution(e::GWESPTwoModeEffect, state::NetworkState,
                             data::SienaData, actor::Int, event::Int)
    net = state.networks[e.variable]
    n_actors = size(net, 1)

    total = 0.0
    for other in 1:n_actors
        if other != actor && net[other, event] == 1
            shared = sum(net[actor, :] .* net[other, :])
            if shared > 0
                # GWESP-style weighting
                total += 1 - (1 - exp(-e.α))^shared
            end
        end
    end
    return total
end

#==============================================================================#
# Covariate Effects (Two-Mode)
#==============================================================================#

"""
    TwoModeEgoEffect <: TwoModeEffect

Effect of actor covariate on two-mode tie formation.
"""
struct TwoModeEgoEffect <: TwoModeEffect
    variable::Symbol
    covariate::Symbol
end

effect_name(e::TwoModeEgoEffect) = Symbol("ego2$(e.covariate)")
effect_type(::TwoModeEgoEffect) = :eval
target_variable(e::TwoModeEgoEffect) = e.variable
interaction_with(e::TwoModeEgoEffect) = e.covariate

function compute_contribution(e::TwoModeEgoEffect, state::NetworkState,
                             data::SienaData, actor::Int, event::Int)
    cov = data.covariates[e.covariate]
    cov_val = if cov isa ConstantCovariate
        cov.values[actor]
    elseif cov isa VaryingCovariate
        cov.values[min(state.period, length(cov.values))][actor]
    else
        0.0
    end
    return cov_val
end

"""
    TwoModeEventEffect <: TwoModeEffect

Effect of event attribute on two-mode tie formation.
Uses a dyadic covariate where rows are actors and columns are events.
"""
struct TwoModeEventEffect <: TwoModeEffect
    variable::Symbol
    event_covariate::Symbol  # Should be a dyadic covariate
end

effect_name(e::TwoModeEventEffect) = Symbol("event2$(e.event_covariate)")
effect_type(::TwoModeEventEffect) = :eval
target_variable(e::TwoModeEventEffect) = e.variable
interaction_with(e::TwoModeEventEffect) = e.event_covariate

function compute_contribution(e::TwoModeEventEffect, state::NetworkState,
                             data::SienaData, actor::Int, event::Int)
    dcov = data.covariates[e.event_covariate]
    if dcov isa ConstantDyadCovariate
        return dcov.values[actor, event]
    elseif dcov isa VaryingDyadCovariate
        period = min(state.period, length(dcov.values))
        return dcov.values[period][actor, event]
    else
        return 0.0
    end
end

"""
    TwoModeSameEffect <: TwoModeEffect

Same covariate effect for two-mode networks.
Actors with same covariate value tend to share events with similar others.
"""
struct TwoModeSameEffect <: TwoModeEffect
    variable::Symbol
    covariate::Symbol
end

effect_name(e::TwoModeSameEffect) = Symbol("same2$(e.covariate)")
effect_type(::TwoModeSameEffect) = :eval
target_variable(e::TwoModeSameEffect) = e.variable
interaction_with(e::TwoModeSameEffect) = e.covariate

function compute_contribution(e::TwoModeSameEffect, state::NetworkState,
                             data::SienaData, actor::Int, event::Int)
    net = state.networks[e.variable]
    cov = data.covariates[e.covariate]
    n_actors = size(net, 1)

    ego_val = if cov isa ConstantCovariate
        cov.values[actor]
    elseif cov isa VaryingCovariate
        cov.values[min(state.period, length(cov.values))][actor]
    else
        0.0
    end

    # Count alters at this event with same covariate value
    count = 0
    for other in 1:n_actors
        if other != actor && net[other, event] == 1
            other_val = if cov isa ConstantCovariate
                cov.values[other]
            elseif cov isa VaryingCovariate
                cov.values[min(state.period, length(cov.values))][other]
            else
                0.0
            end
            if abs(ego_val - other_val) < 0.5  # Same value
                count += 1
            end
        end
    end
    return Float64(count)
end

"""
    TwoModeSimilarityEffect <: TwoModeEffect

Similarity effect for two-mode networks.
"""
struct TwoModeSimilarityEffect <: TwoModeEffect
    variable::Symbol
    covariate::Symbol
end

effect_name(e::TwoModeSimilarityEffect) = Symbol("sim2$(e.covariate)")
effect_type(::TwoModeSimilarityEffect) = :eval
target_variable(e::TwoModeSimilarityEffect) = e.variable
interaction_with(e::TwoModeSimilarityEffect) = e.covariate

function compute_contribution(e::TwoModeSimilarityEffect, state::NetworkState,
                             data::SienaData, actor::Int, event::Int)
    net = state.networks[e.variable]
    cov = data.covariates[e.covariate]
    n_actors = size(net, 1)

    # Get covariate range for normalization
    cov_range = if cov isa ConstantCovariate
        maximum(cov.values) - minimum(cov.values)
    elseif cov isa VaryingCovariate
        all_vals = vcat(cov.values...)
        maximum(all_vals) - minimum(all_vals)
    else
        1.0
    end
    cov_range = max(cov_range, 1e-10)  # Avoid division by zero

    ego_val = if cov isa ConstantCovariate
        cov.values[actor]
    elseif cov isa VaryingCovariate
        cov.values[min(state.period, length(cov.values))][actor]
    else
        0.0
    end

    # Sum similarity with alters at this event
    total_sim = 0.0
    for other in 1:n_actors
        if other != actor && net[other, event] == 1
            other_val = if cov isa ConstantCovariate
                cov.values[other]
            elseif cov isa VaryingCovariate
                cov.values[min(state.period, length(cov.values))][other]
            else
                0.0
            end
            sim = 1.0 - abs(ego_val - other_val) / cov_range
            total_sim += sim
        end
    end
    return total_sim
end

#==============================================================================#
# Degree-Based Effects (Two-Mode)
#==============================================================================#

"""
    TwoModeActivityEffect <: TwoModeEffect

Activity effect: outdegree of alter at the event.
"""
struct TwoModeActivityEffect <: TwoModeEffect
    variable::Symbol
    sqrt::Bool
end

TwoModeActivityEffect(var::Symbol; sqrt::Bool=false) =
    TwoModeActivityEffect(var, sqrt)

effect_name(e::TwoModeActivityEffect) = e.sqrt ? :activitySqrt2 : :activity2
effect_type(::TwoModeActivityEffect) = :eval
target_variable(e::TwoModeActivityEffect) = e.variable

function compute_contribution(e::TwoModeActivityEffect, state::NetworkState,
                             data::SienaData, actor::Int, event::Int)
    net = state.networks[e.variable]
    n_actors = size(net, 1)

    total = 0.0
    for other in 1:n_actors
        if other != actor && net[other, event] == 1
            outdeg = sum(net[other, :])
            if e.sqrt
                total += sqrt(outdeg)
            else
                total += outdeg
            end
        end
    end
    return total
end

"""
    TwoModePopularityAltEffect <: TwoModeEffect

Actors with high outdegree tend to go to popular events.
"""
struct TwoModePopularityAltEffect <: TwoModeEffect
    variable::Symbol
end

effect_name(::TwoModePopularityAltEffect) = :popAlt2
effect_type(::TwoModePopularityAltEffect) = :eval
target_variable(e::TwoModePopularityAltEffect) = e.variable

function compute_contribution(e::TwoModePopularityAltEffect, state::NetworkState,
                             data::SienaData, actor::Int, event::Int)
    net = state.networks[e.variable]
    # Product of ego outdegree and event indegree
    ego_outdeg = sum(net[actor, :])
    event_indeg = sum(net[:, event]) - net[actor, event]
    return Float64(ego_outdeg * event_indeg)
end

#==============================================================================#
# Closure Effects (Two-Mode)
#==============================================================================#

"""
    TwoModeTransitiveClosureEffect <: TwoModeEffect

Tendency to form ties that close four-cycles.
"""
struct TwoModeTransitiveClosureEffect <: TwoModeEffect
    variable::Symbol
end

effect_name(::TwoModeTransitiveClosureEffect) = :transClosure2
effect_type(::TwoModeTransitiveClosureEffect) = :eval
target_variable(e::TwoModeTransitiveClosureEffect) = e.variable

function compute_contribution(e::TwoModeTransitiveClosureEffect, state::NetworkState,
                             data::SienaData, actor::Int, event::Int)
    net = state.networks[e.variable]
    n_actors, n_events = size(net)

    # For each event ego attends, count if alters at that event also attend target event
    count = 0
    for e1 in 1:n_events
        if e1 != event && net[actor, e1] == 1
            for other in 1:n_actors
                if other != actor && net[other, e1] == 1 && net[other, event] == 1
                    count += 1
                end
            end
        end
    end
    return Float64(count)
end

"""
    TwoModeActorAssortativityEffect <: TwoModeEffect

Tendency for actors with similar outdegree to share events.
"""
struct TwoModeActorAssortativityEffect <: TwoModeEffect
    variable::Symbol
end

effect_name(::TwoModeActorAssortativityEffect) = :actAssort2
effect_type(::TwoModeActorAssortativityEffect) = :eval
target_variable(e::TwoModeActorAssortativityEffect) = e.variable

function compute_contribution(e::TwoModeActorAssortativityEffect, state::NetworkState,
                             data::SienaData, actor::Int, event::Int)
    net = state.networks[e.variable]
    n_actors = size(net, 1)

    ego_outdeg = sum(net[actor, :])

    # Sum of product of ego outdegree with outdegree of alters at this event
    total = 0.0
    for other in 1:n_actors
        if other != actor && net[other, event] == 1
            other_outdeg = sum(net[other, :])
            total += ego_outdeg * other_outdeg
        end
    end
    return total
end

#==============================================================================#
# Between/Within Effects (Two-Mode with Settings)
#==============================================================================#

"""
    TwoModeWithinEffect <: TwoModeEffect

Tendency to form ties within same setting/group (for grouped events).
"""
struct TwoModeWithinEffect <: TwoModeEffect
    variable::Symbol
    setting::Symbol  # Covariate indicating setting for each actor
    event_setting::Symbol  # Dyadic covariate indicating event settings
end

effect_name(e::TwoModeWithinEffect) = Symbol("within2$(e.setting)")
effect_type(::TwoModeWithinEffect) = :eval
target_variable(e::TwoModeWithinEffect) = e.variable
interaction_with(e::TwoModeWithinEffect) = e.setting

function compute_contribution(e::TwoModeWithinEffect, state::NetworkState,
                             data::SienaData, actor::Int, event::Int)
    actor_cov = data.covariates[e.setting]
    event_cov = data.covariates[e.event_setting]

    actor_setting = if actor_cov isa ConstantCovariate
        actor_cov.values[actor]
    elseif actor_cov isa VaryingCovariate
        actor_cov.values[min(state.period, length(actor_cov.values))][actor]
    else
        0.0
    end

    event_setting = if event_cov isa ConstantDyadCovariate
        # Assuming event setting is stored in a way that can be retrieved
        # This is simplified - actual implementation would depend on data structure
        0.0
    else
        0.0
    end

    # Return 1 if same setting, 0 otherwise
    return abs(actor_setting - event_setting) < 0.5 ? 1.0 : 0.0
end
