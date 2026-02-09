"""
Network effects for SAOM evaluation function.
Complete implementation of RSiena network effects.
"""

#==============================================================================#
# Helper Functions
#==============================================================================#

function _get_covariate_value(cov::AbstractCovariate, actor::Int, wave::Int)
    if cov isa ConstantCovariate
        return cov.values[actor]
    elseif cov isa VaryingCovariate
        w = min(wave, length(cov.values))
        return cov.values[w][actor]
    end
    return 0.0
end

function _get_covariate_range(cov::AbstractCovariate)
    if cov isa ConstantCovariate
        vals = filter(!isnan, cov.values)
        return isempty(vals) ? 1.0 : maximum(vals) - minimum(vals)
    elseif cov isa VaryingCovariate
        all_vals = filter(!isnan, vcat(cov.values...))
        return isempty(all_vals) ? 1.0 : maximum(all_vals) - minimum(all_vals)
    end
    return 1.0
end

function _get_dyad_covariate_value(cov::AbstractCovariate, i::Int, j::Int, wave::Int)
    if cov isa ConstantDyadCovariate
        return cov.values[i, j]
    elseif cov isa VaryingDyadCovariate
        w = min(wave, length(cov.values))
        return cov.values[w][i, j]
    end
    return 0.0
end

#==============================================================================#
# Basic Structural Effects
#==============================================================================#

"""
    OutdegreeEffect <: NetworkEffect

Basic outdegree effect (density). Sum of outgoing ties.
RSiena: density
"""
struct OutdegreeEffect <: NetworkEffect
    variable::Symbol
end

effect_name(::OutdegreeEffect) = :density
effect_type(::OutdegreeEffect) = :eval
target_variable(e::OutdegreeEffect) = e.variable

function compute_contribution(e::OutdegreeEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    return 1.0
end

function compute_statistic(e::OutdegreeEffect, state::NetworkState, data::SienaData)
    net = state.networks[e.variable]
    return Float64(sum(net))
end

"""
    ReciprocityEffect <: NetworkEffect

Reciprocity effect. Number of reciprocated ties.
RSiena: recip
"""
struct ReciprocityEffect <: NetworkEffect
    variable::Symbol
end

effect_name(::ReciprocityEffect) = :recip
effect_type(::ReciprocityEffect) = :eval
target_variable(e::ReciprocityEffect) = e.variable

function compute_contribution(e::ReciprocityEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    net = state.networks[e.variable]
    return Float64(net[alter, actor])
end

function compute_statistic(e::ReciprocityEffect, state::NetworkState, data::SienaData)
    net = state.networks[e.variable]
    n = size(net, 1)
    count = 0.0
    for i in 1:n, j in (i+1):n
        if net[i, j] == 1 && net[j, i] == 1
            count += 2.0
        end
    end
    return count
end

#==============================================================================#
# Triadic Effects
#==============================================================================#

"""
    TransitiveTripletsEffect <: NetworkEffect

Transitive triplets effect. RSiena: transTrip
"""
struct TransitiveTripletsEffect <: NetworkEffect
    variable::Symbol
end

effect_name(::TransitiveTripletsEffect) = :transTrip
effect_type(::TransitiveTripletsEffect) = :eval
target_variable(e::TransitiveTripletsEffect) = e.variable

function compute_contribution(e::TransitiveTripletsEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    net = state.networks[e.variable]
    n = size(net, 1)
    count = 0.0
    for h in 1:n
        h == actor || h == alter && continue
        if net[actor, h] == 1 && net[h, alter] == 1
            count += 1.0
        end
        if net[alter, h] == 1 && net[actor, h] == 1
            count += 1.0
        end
    end
    return count
end

"""
    TransitiveTiesEffect <: NetworkEffect

Transitive ties (closure) effect. RSiena: transTies
"""
struct TransitiveTiesEffect <: NetworkEffect
    variable::Symbol
end

effect_name(::TransitiveTiesEffect) = :transTies
effect_type(::TransitiveTiesEffect) = :eval
target_variable(e::TransitiveTiesEffect) = e.variable

function compute_contribution(e::TransitiveTiesEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    net = state.networks[e.variable]
    n = size(net, 1)
    for h in 1:n
        h == actor || h == alter && continue
        net[actor, h] == 1 && net[h, alter] == 1 && return 1.0
    end
    return 0.0
end

# Alias: TransitiveTriadsEffect is the same as TransitiveTiesEffect
const TransitiveTriadsEffect = TransitiveTiesEffect

"""
    TransitiveMediatedTripletsEffect <: NetworkEffect

Transitive mediated triplets. RSiena: transMedTrip
"""
struct TransitiveMediatedTripletsEffect <: NetworkEffect
    variable::Symbol
end

effect_name(::TransitiveMediatedTripletsEffect) = :transMedTrip
effect_type(::TransitiveMediatedTripletsEffect) = :eval
target_variable(e::TransitiveMediatedTripletsEffect) = e.variable

function compute_contribution(e::TransitiveMediatedTripletsEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    net = state.networks[e.variable]
    n = size(net, 1)
    count = 0.0
    for h in 1:n
        h == actor || h == alter && continue
        if net[actor, h] == 1 && net[h, alter] == 1 && net[h, actor] == 1
            count += 1.0
        end
    end
    return count
end

"""
    TransitiveRecipTripletsEffect <: NetworkEffect

Transitive reciprocated triplets. RSiena: transRecTrip
"""
struct TransitiveRecipTripletsEffect <: NetworkEffect
    variable::Symbol
end

effect_name(::TransitiveRecipTripletsEffect) = :transRecTrip
effect_type(::TransitiveRecipTripletsEffect) = :eval
target_variable(e::TransitiveRecipTripletsEffect) = e.variable

function compute_contribution(e::TransitiveRecipTripletsEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    net = state.networks[e.variable]
    net[alter, actor] == 0 && return 0.0
    n = size(net, 1)
    count = 0.0
    for h in 1:n
        h == actor || h == alter && continue
        if net[actor, h] == 1 && net[h, alter] == 1
            count += 1.0
        end
    end
    return count
end

"""
    CyclicTripletsEffect <: NetworkEffect

Three-cycles effect. RSiena: cycle3
"""
struct CyclicTripletsEffect <: NetworkEffect
    variable::Symbol
end

effect_name(::CyclicTripletsEffect) = :cycle3
effect_type(::CyclicTripletsEffect) = :eval
target_variable(e::CyclicTripletsEffect) = e.variable

function compute_contribution(e::CyclicTripletsEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    net = state.networks[e.variable]
    n = size(net, 1)
    count = 0.0
    for h in 1:n
        h == actor || h == alter && continue
        if net[alter, h] == 1 && net[h, actor] == 1
            count += 1.0
        end
    end
    return count
end

"""
    BalanceEffect <: NetworkEffect

Structural balance effect. RSiena: balance
"""
struct BalanceEffect <: NetworkEffect
    variable::Symbol
end

effect_name(::BalanceEffect) = :balance
effect_type(::BalanceEffect) = :eval
target_variable(e::BalanceEffect) = e.variable

function compute_contribution(e::BalanceEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    net = state.networks[e.variable]
    n = size(net, 1)
    count = 0.0
    for h in 1:n
        h == actor || h == alter && continue
        net[actor, h] == net[alter, h] && (count += 1.0)
    end
    return n > 2 ? count / (n - 2) : 0.0
end

"""
    BetweennessEffect <: NetworkEffect

Betweenness effect. RSiena: between
"""
struct BetweennessEffect <: NetworkEffect
    variable::Symbol
end

effect_name(::BetweennessEffect) = :between
effect_type(::BetweennessEffect) = :eval
target_variable(e::BetweennessEffect) = e.variable

function compute_contribution(e::BetweennessEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    net = state.networks[e.variable]
    n = size(net, 1)
    count = 0.0
    for h in 1:n
        h == actor || h == alter && continue
        if net[h, actor] == 1
            count += 1.0
        end
    end
    return count
end

"""
    NbrDist2Effect <: NetworkEffect

Number of actors at distance 2. RSiena: nbrDist2
"""
struct NbrDist2Effect <: NetworkEffect
    variable::Symbol
end

effect_name(::NbrDist2Effect) = :nbrDist2
effect_type(::NbrDist2Effect) = :eval
target_variable(e::NbrDist2Effect) = e.variable

function compute_contribution(e::NbrDist2Effect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    net = state.networks[e.variable]
    n = size(net, 1)
    count = 0.0
    for h in 1:n
        h == actor || h == alter && continue
        if net[alter, h] == 1 && net[actor, h] == 0
            count += 1.0
        end
    end
    return count
end

"""
    DenseTriadsEffect <: NetworkEffect

Dense triads effect (5+ ties). RSiena: denseTriads
"""
struct DenseTriadsEffect <: NetworkEffect
    variable::Symbol
end

effect_name(::DenseTriadsEffect) = :denseTriads
effect_type(::DenseTriadsEffect) = :eval
target_variable(e::DenseTriadsEffect) = e.variable

function compute_contribution(e::DenseTriadsEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    net = state.networks[e.variable]
    n = size(net, 1)
    count = 0.0
    for h in 1:n
        h == actor || h == alter && continue
        ties = 1 + net[alter, actor] + net[actor, h] + net[h, actor] + net[alter, h] + net[h, alter]
        ties >= 5 && (count += 1.0)
    end
    return count
end

"""
    SharedInEffect <: NetworkEffect

Shared incoming ties. RSiena: sharedIn
"""
struct SharedInEffect <: NetworkEffect
    variable::Symbol
end

effect_name(::SharedInEffect) = :sharedIn
effect_type(::SharedInEffect) = :eval
target_variable(e::SharedInEffect) = e.variable

function compute_contribution(e::SharedInEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    net = state.networks[e.variable]
    n = size(net, 1)
    count = 0.0
    for h in 1:n
        h == actor || h == alter && continue
        if net[h, actor] == 1 && net[h, alter] == 1
            count += 1.0
        end
    end
    return count
end

"""
    SharedOutEffect <: NetworkEffect

Shared outgoing ties. RSiena: sharedOut
"""
struct SharedOutEffect <: NetworkEffect
    variable::Symbol
end

effect_name(::SharedOutEffect) = :sharedOut
effect_type(::SharedOutEffect) = :eval
target_variable(e::SharedOutEffect) = e.variable

function compute_contribution(e::SharedOutEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    net = state.networks[e.variable]
    n = size(net, 1)
    count = 0.0
    for h in 1:n
        h == actor || h == alter && continue
        if net[actor, h] == 1 && net[alter, h] == 1
            count += 1.0
        end
    end
    return count
end

#==============================================================================#
# Degree-Based Effects
#==============================================================================#

"""
    IndegreePopularityEffect <: NetworkEffect

Indegree popularity effect. RSiena: inPop, inPopSqrt
"""
struct IndegreePopularityEffect <: NetworkEffect
    variable::Symbol
    sqrt::Bool
    IndegreePopularityEffect(variable::Symbol; sqrt::Bool=false) = new(variable, sqrt)
end

effect_name(e::IndegreePopularityEffect) = e.sqrt ? :inPopSqrt : :inPop
effect_type(::IndegreePopularityEffect) = :eval
target_variable(e::IndegreePopularityEffect) = e.variable

function compute_contribution(e::IndegreePopularityEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    net = state.networks[e.variable]
    indeg = sum(net[:, alter]) - net[actor, alter]
    return e.sqrt ? sqrt(Float64(max(0, indeg))) : Float64(indeg)
end

"""
    OutdegreePopularityEffect <: NetworkEffect

Outdegree popularity effect. RSiena: outPop, outPopSqrt
"""
struct OutdegreePopularityEffect <: NetworkEffect
    variable::Symbol
    sqrt::Bool
    OutdegreePopularityEffect(variable::Symbol; sqrt::Bool=false) = new(variable, sqrt)
end

effect_name(e::OutdegreePopularityEffect) = e.sqrt ? :outPopSqrt : :outPop
effect_type(::OutdegreePopularityEffect) = :eval
target_variable(e::OutdegreePopularityEffect) = e.variable

function compute_contribution(e::OutdegreePopularityEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    net = state.networks[e.variable]
    outdeg = sum(net[alter, :])
    return e.sqrt ? sqrt(Float64(outdeg)) : Float64(outdeg)
end

"""
    IndegreeActivityEffect <: NetworkEffect

Indegree activity effect. RSiena: inAct, inActSqrt
"""
struct IndegreeActivityEffect <: NetworkEffect
    variable::Symbol
    sqrt::Bool
    IndegreeActivityEffect(variable::Symbol; sqrt::Bool=false) = new(variable, sqrt)
end

effect_name(e::IndegreeActivityEffect) = e.sqrt ? :inActSqrt : :inAct
effect_type(::IndegreeActivityEffect) = :eval
target_variable(e::IndegreeActivityEffect) = e.variable

function compute_contribution(e::IndegreeActivityEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    net = state.networks[e.variable]
    indeg = sum(net[:, actor])
    return e.sqrt ? sqrt(Float64(indeg)) : Float64(indeg)
end

"""
    OutdegreeActivityEffect <: NetworkEffect

Outdegree activity effect. RSiena: outAct, outActSqrt
"""
struct OutdegreeActivityEffect <: NetworkEffect
    variable::Symbol
    sqrt::Bool
    OutdegreeActivityEffect(variable::Symbol; sqrt::Bool=false) = new(variable, sqrt)
end

effect_name(e::OutdegreeActivityEffect) = e.sqrt ? :outActSqrt : :outAct
effect_type(::OutdegreeActivityEffect) = :eval
target_variable(e::OutdegreeActivityEffect) = e.variable

function compute_contribution(e::OutdegreeActivityEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    net = state.networks[e.variable]
    outdeg = sum(net[actor, :]) - net[actor, alter]
    return e.sqrt ? sqrt(Float64(max(0, outdeg))) : Float64(outdeg)
end

"""
    OutdegreeTruncEffect <: NetworkEffect

Truncated outdegree effect. RSiena: outTrunc
"""
struct OutdegreeTruncEffect <: NetworkEffect
    variable::Symbol
    c::Int
    OutdegreeTruncEffect(variable::Symbol; c::Int=1) = new(variable, c)
end

effect_name(::OutdegreeTruncEffect) = :outTrunc
effect_type(::OutdegreeTruncEffect) = :eval
target_variable(e::OutdegreeTruncEffect) = e.variable

function compute_contribution(e::OutdegreeTruncEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    net = state.networks[e.variable]
    outdeg = sum(net[actor, :])
    return Float64(min(outdeg, e.c))
end

"""
    IndegreeTruncEffect <: NetworkEffect

Truncated indegree effect. RSiena: inTrunc
"""
struct IndegreeTruncEffect <: NetworkEffect
    variable::Symbol
    c::Int
    IndegreeTruncEffect(variable::Symbol; c::Int=1) = new(variable, c)
end

effect_name(::IndegreeTruncEffect) = :inTrunc
effect_type(::IndegreeTruncEffect) = :eval
target_variable(e::IndegreeTruncEffect) = e.variable

function compute_contribution(e::IndegreeTruncEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    net = state.networks[e.variable]
    indeg = sum(net[:, alter])
    return Float64(min(indeg, e.c))
end

"""
    DegreeAssortativityEffect <: NetworkEffect

Degree assortativity (ego*alter degree product). RSiena: degPlus
"""
struct DegreeAssortativityEffect <: NetworkEffect
    variable::Symbol
end

effect_name(::DegreeAssortativityEffect) = :degPlus
effect_type(::DegreeAssortativityEffect) = :eval
target_variable(e::DegreeAssortativityEffect) = e.variable

function compute_contribution(e::DegreeAssortativityEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    net = state.networks[e.variable]
    ego_deg = sum(net[actor, :]) + sum(net[:, actor])
    alt_deg = sum(net[alter, :]) + sum(net[:, alter])
    return Float64(ego_deg + alt_deg)
end

#==============================================================================#
# Isolate Effects
#==============================================================================#

"""
    IsolateEffect <: NetworkEffect

Isolate effect. RSiena: isolate
"""
struct IsolateEffect <: NetworkEffect
    variable::Symbol
end

effect_name(::IsolateEffect) = :isolate
effect_type(::IsolateEffect) = :eval
target_variable(e::IsolateEffect) = e.variable

function compute_contribution(e::IsolateEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    net = state.networks[e.variable]
    outdeg = sum(net[actor, :])
    indeg = sum(net[:, actor])
    return (outdeg == 0 && indeg == 0) ? -1.0 : 0.0
end

"""
    IsolateNetEffect <: NetworkEffect

Network isolate (no outgoing ties). RSiena: isolateNet
"""
struct IsolateNetEffect <: NetworkEffect
    variable::Symbol
end

effect_name(::IsolateNetEffect) = :isolateNet
effect_type(::IsolateNetEffect) = :eval
target_variable(e::IsolateNetEffect) = e.variable

function compute_contribution(e::IsolateNetEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    net = state.networks[e.variable]
    outdeg = sum(net[actor, :])
    return outdeg == 0 ? -1.0 : 0.0
end

"""
    OutIsolateEffect <: NetworkEffect

Out-isolate effect. RSiena: outIsolate
"""
struct OutIsolateEffect <: NetworkEffect
    variable::Symbol
end

effect_name(::OutIsolateEffect) = :outIsolate
effect_type(::OutIsolateEffect) = :eval
target_variable(e::OutIsolateEffect) = e.variable

function compute_contribution(e::OutIsolateEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    net = state.networks[e.variable]
    outdeg = sum(net[actor, :]) - net[actor, alter]
    return outdeg == 0 ? 1.0 : 0.0
end

"""
    InIsolateEffect <: NetworkEffect

In-isolate effect. RSiena: inIsolate
"""
struct InIsolateEffect <: NetworkEffect
    variable::Symbol
end

effect_name(::InIsolateEffect) = :inIsolate
effect_type(::InIsolateEffect) = :eval
target_variable(e::InIsolateEffect) = e.variable

function compute_contribution(e::InIsolateEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    net = state.networks[e.variable]
    indeg = sum(net[:, alter]) - net[actor, alter]
    return indeg == 0 ? 1.0 : 0.0
end

#==============================================================================#
# GWESP / Shared Partner Effects
#==============================================================================#

"""
    GWESPEffect <: NetworkEffect

Geometrically weighted edgewise shared partners (forward). RSiena: gwespFF
"""
struct GWESPEffect <: NetworkEffect
    variable::Symbol
    alpha::Float64
    GWESPEffect(variable::Symbol; alpha::Float64=log(2.0)) = new(variable, alpha)
end

effect_name(::GWESPEffect) = :gwespFF
effect_type(::GWESPEffect) = :eval
target_variable(e::GWESPEffect) = e.variable

function compute_contribution(e::GWESPEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    net = state.networks[e.variable]
    n = size(net, 1)
    sp = 0
    for h in 1:n
        h == actor || h == alter && continue
        if net[actor, h] == 1 && net[alter, h] == 1
            sp += 1
        end
    end
    sp == 0 && return 0.0
    return exp(e.alpha) * (1.0 - (1.0 - exp(-e.alpha))^sp)
end

"""
    GWESPBackwardEffect <: NetworkEffect

GWESP backward. RSiena: gwespBB
"""
struct GWESPBackwardEffect <: NetworkEffect
    variable::Symbol
    alpha::Float64
    GWESPBackwardEffect(variable::Symbol; alpha::Float64=log(2.0)) = new(variable, alpha)
end

effect_name(::GWESPBackwardEffect) = :gwespBB
effect_type(::GWESPBackwardEffect) = :eval
target_variable(e::GWESPBackwardEffect) = e.variable

function compute_contribution(e::GWESPBackwardEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    net = state.networks[e.variable]
    n = size(net, 1)
    sp = 0
    for h in 1:n
        h == actor || h == alter && continue
        if net[h, actor] == 1 && net[h, alter] == 1
            sp += 1
        end
    end
    sp == 0 && return 0.0
    return exp(e.alpha) * (1.0 - (1.0 - exp(-e.alpha))^sp)
end

"""
    GWESPMixedEffect <: NetworkEffect

GWESP mixed (two-path). RSiena: gwespFB
"""
struct GWESPMixedEffect <: NetworkEffect
    variable::Symbol
    alpha::Float64
    GWESPMixedEffect(variable::Symbol; alpha::Float64=log(2.0)) = new(variable, alpha)
end

effect_name(::GWESPMixedEffect) = :gwespFB
effect_type(::GWESPMixedEffect) = :eval
target_variable(e::GWESPMixedEffect) = e.variable

function compute_contribution(e::GWESPMixedEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    net = state.networks[e.variable]
    n = size(net, 1)
    sp = 0
    for h in 1:n
        h == actor || h == alter && continue
        if net[actor, h] == 1 && net[h, alter] == 1
            sp += 1
        end
    end
    sp == 0 && return 0.0
    return exp(e.alpha) * (1.0 - (1.0 - exp(-e.alpha))^sp)
end

"""
    GWDSPEffect <: NetworkEffect

Geometrically weighted dyadwise shared partners. RSiena: gwdspFF
"""
struct GWDSPEffect <: NetworkEffect
    variable::Symbol
    alpha::Float64
    GWDSPEffect(variable::Symbol; alpha::Float64=log(2.0)) = new(variable, alpha)
end

effect_name(::GWDSPEffect) = :gwdspFF
effect_type(::GWDSPEffect) = :eval
target_variable(e::GWDSPEffect) = e.variable

function compute_contribution(e::GWDSPEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    net = state.networks[e.variable]
    n = size(net, 1)
    sp = 0
    for h in 1:n
        h == actor || h == alter && continue
        has_path = (net[actor, h] == 1 || net[h, actor] == 1) &&
                   (net[alter, h] == 1 || net[h, alter] == 1)
        has_path && (sp += 1)
    end
    sp == 0 && return 0.0
    return exp(e.alpha) * (1.0 - (1.0 - exp(-e.alpha))^sp)
end

#==============================================================================#
# Covariate Effects
#==============================================================================#

"""
    EgoEffect <: NetworkEffect

Ego covariate effect. RSiena: egoX
"""
struct EgoEffect <: NetworkEffect
    variable::Symbol
    covariate::Symbol
end

effect_name(::EgoEffect) = :egoX
effect_type(::EgoEffect) = :eval
target_variable(e::EgoEffect) = e.variable
interaction_with(e::EgoEffect) = e.covariate

function compute_contribution(e::EgoEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    return _get_covariate_value(data.covariates[e.covariate], actor, 1)
end

"""
    EgoSqEffect <: NetworkEffect

Squared ego covariate effect. RSiena: egoSqX
"""
struct EgoSqEffect <: NetworkEffect
    variable::Symbol
    covariate::Symbol
end

effect_name(::EgoSqEffect) = :egoSqX
effect_type(::EgoSqEffect) = :eval
target_variable(e::EgoSqEffect) = e.variable
interaction_with(e::EgoSqEffect) = e.covariate

function compute_contribution(e::EgoSqEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    v = _get_covariate_value(data.covariates[e.covariate], actor, 1)
    return v^2
end

"""
    AlterEffect <: NetworkEffect

Alter covariate effect. RSiena: altX
"""
struct AlterEffect <: NetworkEffect
    variable::Symbol
    covariate::Symbol
end

effect_name(::AlterEffect) = :altX
effect_type(::AlterEffect) = :eval
target_variable(e::AlterEffect) = e.variable
interaction_with(e::AlterEffect) = e.covariate

function compute_contribution(e::AlterEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    return _get_covariate_value(data.covariates[e.covariate], alter, 1)
end

"""
    AlterSqEffect <: NetworkEffect

Squared alter covariate effect. RSiena: altSqX
"""
struct AlterSqEffect <: NetworkEffect
    variable::Symbol
    covariate::Symbol
end

effect_name(::AlterSqEffect) = :altSqX
effect_type(::AlterSqEffect) = :eval
target_variable(e::AlterSqEffect) = e.variable
interaction_with(e::AlterSqEffect) = e.covariate

function compute_contribution(e::AlterSqEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    v = _get_covariate_value(data.covariates[e.covariate], alter, 1)
    return v^2
end

"""
    SimilarityEffect <: NetworkEffect

Similarity effect. RSiena: simX
"""
struct SimilarityEffect <: NetworkEffect
    variable::Symbol
    covariate::Symbol
end

effect_name(::SimilarityEffect) = :simX
effect_type(::SimilarityEffect) = :eval
target_variable(e::SimilarityEffect) = e.variable
interaction_with(e::SimilarityEffect) = e.covariate

function compute_contribution(e::SimilarityEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    cov = data.covariates[e.covariate]
    v1 = _get_covariate_value(cov, actor, 1)
    v2 = _get_covariate_value(cov, alter, 1)
    r = _get_covariate_range(cov)
    return r > 0 ? 1.0 - abs(v1 - v2) / r : 1.0
end

"""
    SameEffect <: NetworkEffect

Same covariate value effect. RSiena: sameX
"""
struct SameEffect <: NetworkEffect
    variable::Symbol
    covariate::Symbol
end

effect_name(::SameEffect) = :sameX
effect_type(::SameEffect) = :eval
target_variable(e::SameEffect) = e.variable
interaction_with(e::SameEffect) = e.covariate

function compute_contribution(e::SameEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    cov = data.covariates[e.covariate]
    v1 = _get_covariate_value(cov, actor, 1)
    v2 = _get_covariate_value(cov, alter, 1)
    return v1 == v2 ? 1.0 : 0.0
end

"""
    DifferenceEffect <: NetworkEffect

Difference effect (ego - alter). RSiena: diffX
"""
struct DifferenceEffect <: NetworkEffect
    variable::Symbol
    covariate::Symbol
end

effect_name(::DifferenceEffect) = :diffX
effect_type(::DifferenceEffect) = :eval
target_variable(e::DifferenceEffect) = e.variable
interaction_with(e::DifferenceEffect) = e.covariate

function compute_contribution(e::DifferenceEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    cov = data.covariates[e.covariate]
    return _get_covariate_value(cov, actor, 1) - _get_covariate_value(cov, alter, 1)
end

"""
    DifferenceSqEffect <: NetworkEffect

Squared difference effect. RSiena: diffSqX
"""
struct DifferenceSqEffect <: NetworkEffect
    variable::Symbol
    covariate::Symbol
end

effect_name(::DifferenceSqEffect) = :diffSqX
effect_type(::DifferenceSqEffect) = :eval
target_variable(e::DifferenceSqEffect) = e.variable
interaction_with(e::DifferenceSqEffect) = e.covariate

function compute_contribution(e::DifferenceSqEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    cov = data.covariates[e.covariate]
    d = _get_covariate_value(cov, actor, 1) - _get_covariate_value(cov, alter, 1)
    return d^2
end

"""
    AbsDifferenceEffect <: NetworkEffect

Absolute difference effect. RSiena: absDiffX
"""
struct AbsDifferenceEffect <: NetworkEffect
    variable::Symbol
    covariate::Symbol
end

effect_name(::AbsDifferenceEffect) = :absDiffX
effect_type(::AbsDifferenceEffect) = :eval
target_variable(e::AbsDifferenceEffect) = e.variable
interaction_with(e::AbsDifferenceEffect) = e.covariate

function compute_contribution(e::AbsDifferenceEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    cov = data.covariates[e.covariate]
    return abs(_get_covariate_value(cov, actor, 1) - _get_covariate_value(cov, alter, 1))
end

"""
    HigherEffect <: NetworkEffect

Ego higher than alter effect. RSiena: higher
"""
struct HigherEffect <: NetworkEffect
    variable::Symbol
    covariate::Symbol
end

effect_name(::HigherEffect) = :higher
effect_type(::HigherEffect) = :eval
target_variable(e::HigherEffect) = e.variable
interaction_with(e::HigherEffect) = e.covariate

function compute_contribution(e::HigherEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    cov = data.covariates[e.covariate]
    return _get_covariate_value(cov, actor, 1) > _get_covariate_value(cov, alter, 1) ? 1.0 : 0.0
end

"""
    EgoTimesAlterEffect <: NetworkEffect

Ego × alter interaction. RSiena: egoXaltX
"""
struct EgoTimesAlterEffect <: NetworkEffect
    variable::Symbol
    covariate::Symbol
end

effect_name(::EgoTimesAlterEffect) = :egoXaltX
effect_type(::EgoTimesAlterEffect) = :eval
target_variable(e::EgoTimesAlterEffect) = e.variable
interaction_with(e::EgoTimesAlterEffect) = e.covariate

function compute_contribution(e::EgoTimesAlterEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    cov = data.covariates[e.covariate]
    return _get_covariate_value(cov, actor, 1) * _get_covariate_value(cov, alter, 1)
end

"""
    EgoPlusAlterEffect <: NetworkEffect

Ego + alter sum effect. RSiena: egoPlusAltX
"""
struct EgoPlusAlterEffect <: NetworkEffect
    variable::Symbol
    covariate::Symbol
end

effect_name(::EgoPlusAlterEffect) = :egoPlusAltX
effect_type(::EgoPlusAlterEffect) = :eval
target_variable(e::EgoPlusAlterEffect) = e.variable
interaction_with(e::EgoPlusAlterEffect) = e.covariate

function compute_contribution(e::EgoPlusAlterEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    cov = data.covariates[e.covariate]
    return _get_covariate_value(cov, actor, 1) + _get_covariate_value(cov, alter, 1)
end

"""
    DyadCovariateEffect <: NetworkEffect

Dyadic covariate effect. RSiena: X
"""
struct DyadCovariateEffect <: NetworkEffect
    variable::Symbol
    covariate::Symbol
end

effect_name(::DyadCovariateEffect) = :X
effect_type(::DyadCovariateEffect) = :eval
target_variable(e::DyadCovariateEffect) = e.variable
interaction_with(e::DyadCovariateEffect) = e.covariate

function compute_contribution(e::DyadCovariateEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    return _get_dyad_covariate_value(data.covariates[e.covariate], actor, alter, 1)
end

"""
    SameXRecipEffect <: NetworkEffect

Same × reciprocity. RSiena: sameXRecip
"""
struct SameXRecipEffect <: NetworkEffect
    variable::Symbol
    covariate::Symbol
end

effect_name(::SameXRecipEffect) = :sameXRecip
effect_type(::SameXRecipEffect) = :eval
target_variable(e::SameXRecipEffect) = e.variable
interaction_with(e::SameXRecipEffect) = e.covariate

function compute_contribution(e::SameXRecipEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    net = state.networks[e.variable]
    cov = data.covariates[e.covariate]
    v1 = _get_covariate_value(cov, actor, 1)
    v2 = _get_covariate_value(cov, alter, 1)
    return (v1 == v2 && net[alter, actor] == 1) ? 1.0 : 0.0
end

"""
    SimXRecipEffect <: NetworkEffect

Similarity × reciprocity. RSiena: simXRecip
"""
struct SimXRecipEffect <: NetworkEffect
    variable::Symbol
    covariate::Symbol
end

effect_name(::SimXRecipEffect) = :simXRecip
effect_type(::SimXRecipEffect) = :eval
target_variable(e::SimXRecipEffect) = e.variable
interaction_with(e::SimXRecipEffect) = e.covariate

function compute_contribution(e::SimXRecipEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    net = state.networks[e.variable]
    net[alter, actor] == 0 && return 0.0
    cov = data.covariates[e.covariate]
    v1 = _get_covariate_value(cov, actor, 1)
    v2 = _get_covariate_value(cov, alter, 1)
    r = _get_covariate_range(cov)
    return r > 0 ? 1.0 - abs(v1 - v2) / r : 1.0
end

"""
    SimXTransTripEffect <: NetworkEffect

Similarity × transitive triplets. RSiena: simXTransTrip
"""
struct SimXTransTripEffect <: NetworkEffect
    variable::Symbol
    covariate::Symbol
end

effect_name(::SimXTransTripEffect) = :simXTransTrip
effect_type(::SimXTransTripEffect) = :eval
target_variable(e::SimXTransTripEffect) = e.variable
interaction_with(e::SimXTransTripEffect) = e.covariate

function compute_contribution(e::SimXTransTripEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    cov = data.covariates[e.covariate]
    v1 = _get_covariate_value(cov, actor, 1)
    v2 = _get_covariate_value(cov, alter, 1)
    r = _get_covariate_range(cov)
    sim = r > 0 ? 1.0 - abs(v1 - v2) / r : 1.0

    net = state.networks[e.variable]
    n = size(net, 1)
    count = 0.0
    for h in 1:n
        h == actor || h == alter && continue
        if net[actor, h] == 1 && net[h, alter] == 1
            count += 1.0
        end
    end
    return sim * count
end

#==============================================================================#
# Endowment/Creation Effects
#==============================================================================#

"""
    EndowmentEffect <: NetworkEffect

Wrapper for endowment effect (tie dissolution).
"""
struct EndowmentEffect{E<:NetworkEffect} <: NetworkEffect
    base_effect::E
end

effect_name(e::EndowmentEffect) = Symbol(string(effect_name(e.base_effect)), "Endow")
effect_type(::EndowmentEffect) = :endow
target_variable(e::EndowmentEffect) = target_variable(e.base_effect)
interaction_with(e::EndowmentEffect) = interaction_with(e.base_effect)

function compute_contribution(e::EndowmentEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    net = state.networks[target_variable(e)]
    return net[actor, alter] == 1 ? compute_contribution(e.base_effect, state, data, actor, alter) : 0.0
end

"""
    CreationEffect <: NetworkEffect

Wrapper for creation effect (tie creation).
"""
struct CreationEffect{E<:NetworkEffect} <: NetworkEffect
    base_effect::E
end

effect_name(e::CreationEffect) = Symbol(string(effect_name(e.base_effect)), "Create")
effect_type(::CreationEffect) = :creation
target_variable(e::CreationEffect) = target_variable(e.base_effect)
interaction_with(e::CreationEffect) = interaction_with(e.base_effect)

function compute_contribution(e::CreationEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    net = state.networks[target_variable(e)]
    return net[actor, alter] == 0 ? compute_contribution(e.base_effect, state, data, actor, alter) : 0.0
end

#==============================================================================#
# Multiplex Effects
#==============================================================================#

"""
    CrossNetworkReciprocityEffect <: NetworkEffect

Reciprocity from another network. RSiena: crprodRecip
"""
struct CrossNetworkReciprocityEffect <: NetworkEffect
    variable::Symbol
    other_network::Symbol
end

effect_name(::CrossNetworkReciprocityEffect) = :crprodRecip
effect_type(::CrossNetworkReciprocityEffect) = :eval
target_variable(e::CrossNetworkReciprocityEffect) = e.variable
interaction_with(e::CrossNetworkReciprocityEffect) = e.other_network

function compute_contribution(e::CrossNetworkReciprocityEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    return Float64(state.networks[e.other_network][alter, actor])
end

"""
    CrossNetworkActivityEffect <: NetworkEffect

Outdegree activity from another network.
"""
struct CrossNetworkActivityEffect <: NetworkEffect
    variable::Symbol
    other_network::Symbol
end

effect_name(::CrossNetworkActivityEffect) = :crprodAct
effect_type(::CrossNetworkActivityEffect) = :eval
target_variable(e::CrossNetworkActivityEffect) = e.variable
interaction_with(e::CrossNetworkActivityEffect) = e.other_network

function compute_contribution(e::CrossNetworkActivityEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    return Float64(sum(state.networks[e.other_network][actor, :]))
end

"""
    CrossNetworkPopularityEffect <: NetworkEffect

Indegree popularity from another network.
"""
struct CrossNetworkPopularityEffect <: NetworkEffect
    variable::Symbol
    other_network::Symbol
end

effect_name(::CrossNetworkPopularityEffect) = :crprodPop
effect_type(::CrossNetworkPopularityEffect) = :eval
target_variable(e::CrossNetworkPopularityEffect) = e.variable
interaction_with(e::CrossNetworkPopularityEffect) = e.other_network

function compute_contribution(e::CrossNetworkPopularityEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    return Float64(sum(state.networks[e.other_network][:, alter]))
end

"""
    CrossNetworkTiesEffect <: NetworkEffect

Tie in another network. RSiena: crprod
"""
struct CrossNetworkTiesEffect <: NetworkEffect
    variable::Symbol
    other_network::Symbol
end

effect_name(::CrossNetworkTiesEffect) = :crprod
effect_type(::CrossNetworkTiesEffect) = :eval
target_variable(e::CrossNetworkTiesEffect) = e.variable
interaction_with(e::CrossNetworkTiesEffect) = e.other_network

function compute_contribution(e::CrossNetworkTiesEffect, state::NetworkState,
                             data::SienaData, actor::Int, alter::Int)
    return Float64(state.networks[e.other_network][actor, alter])
end
