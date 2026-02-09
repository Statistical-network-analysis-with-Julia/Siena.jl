"""
Estimation module for SAOM.
Implements the Method of Moments estimation using stochastic approximation.
"""

#==============================================================================#
# Result Types
#==============================================================================#

"""
    SienaResult

Result of SAOM estimation.

# Fields
- `effects::SienaEffects`: The effects object
- `estimates::Vector{Float64}`: Parameter estimates
- `standard_errors::Vector{Float64}`: Standard errors
- `t_ratios::Vector{Float64}`: t-ratios for convergence
- `covariance::Matrix{Float64}`: Covariance matrix
- `converged::Bool`: Whether estimation converged
- `n_iterations::Int`: Number of iterations used
- `rate_estimates::Dict{Symbol, Vector{Float64}}`: Rate estimates by period
"""
struct SienaResult
    effects::SienaEffects
    estimates::Vector{Float64}
    standard_errors::Vector{Float64}
    t_ratios::Vector{Float64}
    covariance::Matrix{Float64}
    converged::Bool
    n_iterations::Int
    rate_estimates::Dict{Symbol, Vector{Float64}}
end

function Base.show(io::IO, result::SienaResult)
    println(io, "SAOM Estimation Results")
    println(io, "=======================")
    println(io, "Converged: $(result.converged)")
    println(io, "Iterations: $(result.n_iterations)")
    println(io)
    println(io, "Parameter Estimates:")
    println(io, "--------------------")

    obj_effects = get_objective_effects(result.effects)
    for (i, entry) in enumerate(obj_effects)
        if !entry.fix
            est = result.estimates[i]
            se = result.standard_errors[i]
            t = est / se
            sig = abs(t) > 1.96 ? "*" : ""
            @printf(io, "%-20s %8.4f (%6.4f) %s\n",
                    entry.shortname, est, se, sig)
        end
    end

    # Rate estimates
    if !isempty(result.rate_estimates)
        println(io)
        println(io, "Rate Parameters:")
        println(io, "----------------")
        for (var, rates) in result.rate_estimates
            for (p, r) in enumerate(rates)
                @printf(io, "Rate %s period %d: %8.4f\n", var, p, r)
            end
        end
    end
end

#==============================================================================#
# Statistics Computation
#==============================================================================#

"""
    compute_target_statistics(data::SienaData, effects::SienaEffects, wave::Int)

Compute target statistics from observed data at a given wave.
"""
function compute_target_statistics(data::SienaData, effects::SienaEffects, wave::Int)
    state = NetworkState()
    initialize!(state, data, wave)

    stats = Float64[]
    for entry in get_objective_effects(effects)
        if !entry.fix
            s = compute_statistic(entry.effect, state, data)
            push!(stats, s)
        end
    end

    return stats
end

"""
    compute_simulated_statistics(state::NetworkState, effects::SienaEffects,
                                data::SienaData)

Compute statistics from simulated state.
"""
function compute_simulated_statistics(state::NetworkState, effects::SienaEffects,
                                     data::SienaData)
    stats = Float64[]
    for entry in get_objective_effects(effects)
        if !entry.fix
            s = compute_statistic(entry.effect, state, data)
            push!(stats, s)
        end
    end

    return stats
end

#==============================================================================#
# Method of Moments
#==============================================================================#

"""
    compute_score(target_stats::Vector{Float64}, sim_stats::Vector{Float64})

Compute the score (deviation) between target and simulated statistics.
"""
function compute_score(target_stats::Vector{Float64}, sim_stats::Vector{Float64})
    return sim_stats .- target_stats
end

"""
    update_parameters!(θ::Vector{Float64}, score::Vector{Float64},
                      D::Matrix{Float64}, gain::Float64)

Update parameters using Robbins-Monro algorithm.
θ_new = θ - gain * D^{-1} * score
"""
function update_parameters!(θ::Vector{Float64}, score::Vector{Float64},
                           D::Matrix{Float64}, gain::Float64)
    try
        update = D \ score
        θ .-= gain .* update
    catch
        # If D is singular, use diagonal approximation
        for i in eachindex(θ)
            if abs(D[i, i]) > 1e-10
                θ[i] -= gain * score[i] / D[i, i]
            end
        end
    end
    return θ
end

#==============================================================================#
# Derivative Matrix Estimation
#==============================================================================#

"""
    estimate_derivative_matrix(data::SienaData, effects::SienaEffects,
                              θ::Vector{Float64}, n_sims::Int, rng::AbstractRNG)

Estimate the derivative matrix D = ∂E[s]/∂θ using finite differences.
"""
function estimate_derivative_matrix(data::SienaData, effects::SienaEffects,
                                   θ::Vector{Float64}, n_sims::Int, rng::AbstractRNG)
    n_params = length(θ)
    D = zeros(n_params, n_params)
    ε = 0.1  # Perturbation size

    # Base statistics
    base_stats = zeros(n_params)
    for _ in 1:n_sims
        state, _ = simulate_saom(data, effects, θ; seed=rand(rng, 1:10^8))
        base_stats .+= compute_simulated_statistics(state, effects, data)
    end
    base_stats ./= n_sims

    # Perturbed statistics
    for j in 1:n_params
        θ_plus = copy(θ)
        θ_plus[j] += ε

        plus_stats = zeros(n_params)
        for _ in 1:n_sims
            state, _ = simulate_saom(data, effects, θ_plus; seed=rand(rng, 1:10^8))
            plus_stats .+= compute_simulated_statistics(state, effects, data)
        end
        plus_stats ./= n_sims

        D[:, j] = (plus_stats .- base_stats) ./ ε
    end

    return D
end

#==============================================================================#
# Main Estimation Function
#==============================================================================#

"""
    siena07(data::SienaData, effects::SienaEffects;
           algorithm::SienaAlgorithm=SienaAlgorithm())

Estimate SAOM parameters using Method of Moments.
This is the main estimation function, equivalent to siena07() in RSiena.

# Arguments
- `data::SienaData`: The data object
- `effects::SienaEffects`: The effects specification
- `algorithm::SienaAlgorithm`: Algorithm configuration

# Returns
- `SienaResult`: Estimation results
"""
function siena07(data::SienaData, effects::SienaEffects;
                algorithm::SienaAlgorithm=SienaAlgorithm())

    rng = isnothing(algorithm.seed) ? Random.default_rng() : MersenneTwister(algorithm.seed)

    # Initialize parameters
    n_obj = n_objective_parameters(effects)
    θ = zeros(n_obj)

    # Set initial values from effects
    obj_effects = get_objective_effects(effects)
    for (i, entry) in enumerate(obj_effects)
        if !entry.fix
            θ[i] = entry.initial_value
        end
    end

    # Compute target statistics from final observation
    target_stats = compute_target_statistics(data, effects, data.n_waves)

    if algorithm.verbose
        println("Starting SAOM estimation")
        println("Number of parameters: $n_obj")
        println("Target statistics computed")
    end

    # Initialize phase state
    phase_state = PhaseState(algorithm)
    conv_stats = ConvergenceStats(n_obj)

    # Storage for statistics
    sim_stats_history = Vector{Float64}[]
    score_history = Vector{Float64}[]

    # Initialize derivative matrix (identity for start)
    D = Matrix{Float64}(I, n_obj, n_obj)

    total_iterations = 0
    converged = false

    #==========================================================================
    # Phase 1: Initial rough estimation
    ==========================================================================#
    if algorithm.verbose
        println("\n--- Phase 1 ---")
    end

    for iter in 1:algorithm.phase1_iterations
        total_iterations += 1
        gain = next_gain!(phase_state.gain_seq)

        # Simulate
        state, _ = simulate_saom(data, effects, θ; seed=rand(rng, 1:10^8))
        sim_stats = compute_simulated_statistics(state, effects, data)
        push!(sim_stats_history, sim_stats)

        # Compute score
        score = compute_score(target_stats, sim_stats)
        push!(score_history, score)

        # Update parameters
        update_parameters!(θ, score, D, gain)

        if algorithm.verbose && iter % 10 == 0
            println("  Iteration $iter, max deviation: $(maximum(abs.(score)))")
        end
    end

    advance_phase!(phase_state, algorithm)

    #==========================================================================
    # Phase 2: Refinement with subphases
    ==========================================================================#
    if algorithm.verbose
        println("\n--- Phase 2 ---")
    end

    for subphase in 1:algorithm.n_subphases
        if algorithm.verbose
            println("  Subphase $subphase")
        end

        # Estimate derivative matrix periodically
        if subphase == 1 || subphase == algorithm.n_subphases
            D = estimate_derivative_matrix(data, effects, θ, 10, rng)
            # Regularize
            D += 0.01 * I
        end

        n_iter_subphase = algorithm.phase1_iterations ÷ 2
        for iter in 1:n_iter_subphase
            total_iterations += 1
            gain = next_gain!(phase_state.gain_seq)

            # Simulate
            state, _ = simulate_saom(data, effects, θ; seed=rand(rng, 1:10^8))
            sim_stats = compute_simulated_statistics(state, effects, data)
            push!(sim_stats_history, sim_stats)

            # Compute score
            score = compute_score(target_stats, sim_stats)
            push!(score_history, score)

            # Update parameters
            update_parameters!(θ, score, D, gain)
        end

        advance_phase!(phase_state, algorithm)
    end

    #==========================================================================
    # Phase 3: Final estimation and standard errors
    ==========================================================================#
    if algorithm.verbose
        println("\n--- Phase 3 ---")
    end

    # Collect statistics for covariance estimation
    phase3_stats = zeros(algorithm.phase3_iterations, n_obj)

    for iter in 1:algorithm.phase3_iterations
        total_iterations += 1

        # Simulate
        state, _ = simulate_saom(data, effects, θ; seed=rand(rng, 1:10^8))
        sim_stats = compute_simulated_statistics(state, effects, data)
        phase3_stats[iter, :] = sim_stats

        if algorithm.verbose && iter % 100 == 0
            println("  Iteration $iter / $(algorithm.phase3_iterations)")
        end
    end

    # Compute final statistics
    mean_sim_stats = vec(mean(phase3_stats, dims=1))
    final_score = compute_score(target_stats, mean_sim_stats)

    # Estimate covariance of statistics
    Sigma = cov(phase3_stats)

    # Re-estimate derivative matrix
    D_final = estimate_derivative_matrix(data, effects, θ, 50, rng)

    # Compute covariance of parameters: Var(θ) ≈ D^{-1} Σ D^{-T}
    try
        D_inv = inv(D_final)
        param_cov = D_inv * Sigma * D_inv'
    catch
        param_cov = Matrix{Float64}(I, n_obj, n_obj)
    end

    # Standard errors
    se = sqrt.(max.(diag(param_cov), 0.0))

    # Convergence check
    update_convergence!(conv_stats, final_score, se)
    converged = is_converged(conv_stats, algorithm.convergence_threshold)

    if algorithm.verbose
        println("\n--- Results ---")
        println("Converged: $converged")
        println("Max t-ratio: $(conv_stats.max_t_ratio)")
    end

    # Rate estimates (simplified)
    rate_estimates = Dict{Symbol, Vector{Float64}}()
    for entry in get_rate_effects(effects)
        if entry.effect isa BasicRateEffect
            var = target_variable(entry.effect)
            if !haskey(rate_estimates, var)
                rate_estimates[var] = Float64[]
            end
            push!(rate_estimates[var], exp(entry.initial_value))
        end
    end

    return SienaResult(
        effects,
        θ,
        se,
        conv_stats.t_ratios,
        param_cov,
        converged,
        total_iterations,
        rate_estimates
    )
end

#==============================================================================#
# Coefficient Access Functions
#==============================================================================#

"""
    coef(result::SienaResult)

Return parameter estimates.
"""
coef(result::SienaResult) = result.estimates

"""
    stderror(result::SienaResult)

Return standard errors.
"""
stderror(result::SienaResult) = result.standard_errors

"""
    vcov(result::SienaResult)

Return covariance matrix.
"""
vcov(result::SienaResult) = result.covariance

"""
    confint(result::SienaResult; level::Float64=0.95)

Compute confidence intervals.
"""
function confint(result::SienaResult; level::Float64=0.95)
    z = quantile(Normal(), 1 - (1 - level) / 2)
    lower = result.estimates .- z .* result.standard_errors
    upper = result.estimates .+ z .* result.standard_errors
    return hcat(lower, upper)
end
