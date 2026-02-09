"""
Simulation module for SAOM.
Implements the continuous-time Markov chain simulation of network and behavior dynamics.
"""

#==============================================================================#
# Objective Function
#==============================================================================#

"""
    compute_objective(effects::SienaEffects, θ::Vector{Float64},
                     state::NetworkState, data::SienaData,
                     actor::Int, alter::Int, variable::Symbol)

Compute the objective function contribution for a potential change.
For networks: change = toggle tie from actor to alter.
For behavior: alter encodes direction (-1 or +1).
"""
function compute_objective(effects::SienaEffects, θ::Vector{Float64},
                          state::NetworkState, data::SienaData,
                          actor::Int, alter::Int, variable::Symbol)
    obj = 0.0
    param_idx = 1

    for entry in get_objective_effects(effects)
        if !entry.fix && target_variable(entry.effect) == variable
            contrib = if entry.effect isa NetworkEffect
                compute_contribution(entry.effect, state, data, actor, alter)
            else
                compute_contribution(entry.effect, state, data, actor, alter)
            end
            obj += θ[param_idx] * contrib
        end
        if !entry.fix
            param_idx += 1
        end
    end

    return obj
end

#==============================================================================#
# Choice Probabilities
#==============================================================================#

"""
    compute_network_choice_probs(effects::SienaEffects, θ::Vector{Float64},
                                state::NetworkState, data::SienaData,
                                actor::Int, variable::Symbol)

Compute probabilities for all possible tie changes for an actor.
Returns (probabilities, alters) where alters includes 0 for no-change option.
"""
function compute_network_choice_probs(effects::SienaEffects, θ::Vector{Float64},
                                     state::NetworkState, data::SienaData,
                                     actor::Int, variable::Symbol)
    net = state.networks[variable]
    n = size(net, 1)
    dep = data.dependents[variable]::DependentNetwork

    # Compute objective for each possible alter
    objectives = Float64[]
    valid_alters = Int[]

    # No-change option (always available)
    push!(objectives, 0.0)
    push!(valid_alters, 0)

    # Tie changes
    for alter in 1:n
        if alter == actor && !dep.allow_self_loops
            continue
        end
        obj = compute_objective(effects, θ, state, data, actor, alter, variable)
        push!(objectives, obj)
        push!(valid_alters, alter)
    end

    # Convert to probabilities (multinomial logit)
    max_obj = maximum(objectives)
    exp_obj = exp.(objectives .- max_obj)
    probs = exp_obj ./ sum(exp_obj)

    return probs, valid_alters
end

"""
    compute_behavior_choice_probs(effects::SienaEffects, θ::Vector{Float64},
                                 state::NetworkState, data::SienaData,
                                 actor::Int, variable::Symbol)

Compute probabilities for behavior changes (-1, 0, +1) for an actor.
Returns (probabilities, directions).
"""
function compute_behavior_choice_probs(effects::SienaEffects, θ::Vector{Float64},
                                      state::NetworkState, data::SienaData,
                                      actor::Int, variable::Symbol)
    beh = state.behaviors[variable]
    dep = data.dependents[variable]::DependentBehavior
    current = beh[actor]

    objectives = Float64[]
    valid_directions = Int[]

    # Check each possible direction
    for dir in [-1, 0, 1]
        new_val = current + dir
        if new_val < dep.min_val || new_val > dep.max_val
            continue
        end
        obj = if dir == 0
            0.0
        else
            compute_objective(effects, θ, state, data, actor, dir, variable)
        end
        push!(objectives, obj)
        push!(valid_directions, dir)
    end

    if isempty(objectives)
        return [1.0], [0]
    end

    # Convert to probabilities
    max_obj = maximum(objectives)
    exp_obj = exp.(objectives .- max_obj)
    probs = exp_obj ./ sum(exp_obj)

    return probs, valid_directions
end

#==============================================================================#
# Mini-Step Simulation
#==============================================================================#

"""
    sample_actor_network(state::NetworkState, data::SienaData, variable::Symbol,
                        rate_effects::Vector{<:RateEffect}, rate_params::Vector{Float64},
                        rng::AbstractRNG)

Sample an actor for a network mini-step based on rates.
"""
function sample_actor_network(state::NetworkState, data::SienaData, variable::Symbol,
                             rate_effects::Vector{<:RateEffect}, rate_params::Vector{Float64},
                             rng::AbstractRNG)
    net = state.networks[variable]
    n = size(net, 1)

    # Compute rates for each actor
    rates = Float64[]
    for i in 1:n
        if isempty(rate_effects)
            push!(rates, 1.0)
        else
            r = compute_total_rate(rate_effects, rate_params, state, data, i)
            push!(rates, r)
        end
    end

    # Sample actor proportional to rates
    total_rate = sum(rates)
    if total_rate <= 0
        return rand(rng, 1:n)
    end

    probs = rates ./ total_rate
    u = rand(rng)
    cumsum_p = 0.0
    for i in 1:n
        cumsum_p += probs[i]
        if u <= cumsum_p
            return i
        end
    end
    return n
end

"""
    sample_actor_behavior(state::NetworkState, data::SienaData, variable::Symbol,
                         rate_effects::Vector{<:RateEffect}, rate_params::Vector{Float64},
                         rng::AbstractRNG)

Sample an actor for a behavior mini-step based on rates.
"""
function sample_actor_behavior(state::NetworkState, data::SienaData, variable::Symbol,
                              rate_effects::Vector{<:RateEffect}, rate_params::Vector{Float64},
                              rng::AbstractRNG)
    beh = state.behaviors[variable]
    n = length(beh)

    # Compute rates for each actor
    rates = Float64[]
    for i in 1:n
        if isempty(rate_effects)
            push!(rates, 1.0)
        else
            r = compute_total_rate(rate_effects, rate_params, state, data, i)
            push!(rates, r)
        end
    end

    # Sample actor proportional to rates
    total_rate = sum(rates)
    if total_rate <= 0
        return rand(rng, 1:n)
    end

    probs = rates ./ total_rate
    u = rand(rng)
    cumsum_p = 0.0
    for i in 1:n
        cumsum_p += probs[i]
        if u <= cumsum_p
            return i
        end
    end
    return n
end

"""
    execute_network_ministep!(state::NetworkState, effects::SienaEffects,
                             θ::Vector{Float64}, data::SienaData,
                             actor::Int, variable::Symbol, rng::AbstractRNG)

Execute a network mini-step for the given actor.
"""
function execute_network_ministep!(state::NetworkState, effects::SienaEffects,
                                  θ::Vector{Float64}, data::SienaData,
                                  actor::Int, variable::Symbol, rng::AbstractRNG)
    probs, alters = compute_network_choice_probs(effects, θ, state, data, actor, variable)

    # Sample alter
    u = rand(rng)
    cumsum_p = 0.0
    chosen_alter = 0
    for i in eachindex(probs)
        cumsum_p += probs[i]
        if u <= cumsum_p
            chosen_alter = alters[i]
            break
        end
    end

    # Execute change if alter > 0
    if chosen_alter > 0
        net = state.networks[variable]
        net[actor, chosen_alter] = 1 - net[actor, chosen_alter]
    end

    return chosen_alter
end

"""
    execute_behavior_ministep!(state::NetworkState, effects::SienaEffects,
                              θ::Vector{Float64}, data::SienaData,
                              actor::Int, variable::Symbol, rng::AbstractRNG)

Execute a behavior mini-step for the given actor.
"""
function execute_behavior_ministep!(state::NetworkState, effects::SienaEffects,
                                   θ::Vector{Float64}, data::SienaData,
                                   actor::Int, variable::Symbol, rng::AbstractRNG)
    probs, directions = compute_behavior_choice_probs(effects, θ, state, data, actor, variable)

    # Sample direction
    u = rand(rng)
    cumsum_p = 0.0
    chosen_dir = 0
    for i in eachindex(probs)
        cumsum_p += probs[i]
        if u <= cumsum_p
            chosen_dir = directions[i]
            break
        end
    end

    # Execute change if direction != 0
    if chosen_dir != 0
        state.behaviors[variable][actor] += chosen_dir
    end

    return chosen_dir
end

#==============================================================================#
# Period Simulation
#==============================================================================#

"""
    SimulationResult

Result of simulating a period.
"""
struct SimulationResult
    final_state::NetworkState
    n_network_changes::Dict{Symbol, Int}
    n_behavior_changes::Dict{Symbol, Int}
end

"""
    simulate_period!(state::NetworkState, effects::SienaEffects,
                    θ::Vector{Float64}, rate_params::Dict{Symbol, Float64},
                    data::SienaData, rng::AbstractRNG;
                    conditional::Bool=false, target_changes::Union{Nothing, Dict{Symbol, Int}}=nothing)

Simulate one period of network/behavior dynamics.
If conditional=true, simulate until target number of changes is reached.
"""
function simulate_period!(state::NetworkState, effects::SienaEffects,
                         θ::Vector{Float64}, rate_params::Dict{Symbol, Float64},
                         data::SienaData, rng::AbstractRNG;
                         conditional::Bool=false,
                         target_changes::Union{Nothing, Dict{Symbol, Int}}=nothing)

    n_network_changes = Dict{Symbol, Int}()
    n_behavior_changes = Dict{Symbol, Int}()

    # Initialize change counts
    for (name, dep) in data.dependents
        if dep isa DependentNetwork
            n_network_changes[name] = 0
        else
            n_behavior_changes[name] = 0
        end
    end

    # Get rate effects
    rate_effs = get_rate_effects(effects)

    # Simulation loop
    max_steps = 10000  # Safety limit
    step = 0

    while step < max_steps
        step += 1

        # Compute total rates for all variables
        total_rate = 0.0
        var_rates = Dict{Symbol, Float64}()

        for (name, dep) in data.dependents
            base_rate = get(rate_params, name, 1.0)
            if dep isa DependentNetwork
                n = n_actors(dep)
                var_rates[name] = base_rate * n  # Simplified
            else
                n = n_actors(dep)
                var_rates[name] = base_rate * n
            end
            total_rate += var_rates[name]
        end

        if total_rate <= 0
            break
        end

        # Sample waiting time
        dt = sample_waiting_time(total_rate, rng)
        state.time += dt

        # Check if period is complete (non-conditional)
        if !conditional && state.time >= 1.0
            break
        end

        # Sample which variable changes
        u = rand(rng) * total_rate
        cumsum_r = 0.0
        selected_var = first(keys(data.dependents))

        for (name, rate) in var_rates
            cumsum_r += rate
            if u <= cumsum_r
                selected_var = name
                break
            end
        end

        # Execute mini-step
        dep = data.dependents[selected_var]
        if dep isa DependentNetwork
            # Sample actor
            rel_rate_effs = filter(e -> target_variable(e.effect) == selected_var,
                                   rate_effs)
            actor = sample_actor_network(state, data, selected_var,
                                        [e.effect for e in rel_rate_effs],
                                        Float64[], rng)
            change = execute_network_ministep!(state, effects, θ, data, actor, selected_var, rng)
            if change > 0
                n_network_changes[selected_var] += 1
            end
        else
            actor = sample_actor_behavior(state, data, selected_var,
                                         RateEffect[], Float64[], rng)
            change = execute_behavior_ministep!(state, effects, θ, data, actor, selected_var, rng)
            if change != 0
                n_behavior_changes[selected_var] += 1
            end
        end

        # Check conditional termination
        if conditional && !isnothing(target_changes)
            reached = true
            for (name, target) in target_changes
                current = get(n_network_changes, name, get(n_behavior_changes, name, 0))
                if current < target
                    reached = false
                    break
                end
            end
            if reached
                break
            end
        end
    end

    return SimulationResult(state, n_network_changes, n_behavior_changes)
end

#==============================================================================#
# Full Simulation
#==============================================================================#

"""
    simulate_saom(data::SienaData, effects::SienaEffects, θ::Vector{Float64};
                 seed::Union{Int, Nothing}=nothing)

Simulate the full SAOM from first to last observation.
Returns the final state and statistics.
"""
function simulate_saom(data::SienaData, effects::SienaEffects, θ::Vector{Float64};
                      seed::Union{Int, Nothing}=nothing)
    rng = isnothing(seed) ? Random.default_rng() : MersenneTwister(seed)

    state = NetworkState()
    initialize!(state, data, 1)

    # Get rate parameters (simplified - assume basic rates)
    rate_params = Dict{Symbol, Float64}()
    for entry in get_rate_effects(effects)
        if entry.effect isa BasicRateEffect
            rate_params[target_variable(entry.effect)] = exp(entry.initial_value)
        end
    end

    # Simulate each period
    all_results = SimulationResult[]
    for period in 1:(data.n_waves - 1)
        state.time = 0.0
        result = simulate_period!(state, effects, θ, rate_params, data, rng)
        push!(all_results, result)
    end

    return state, all_results
end
