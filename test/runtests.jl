using Siena
using Test
using Random
using Statistics
using DataFrames

@testset "Siena.jl" begin

    @testset "NodeSet" begin
        ns = NodeSet(10)
        @test length(ns) == 10
        @test ns.id == :actors

        ns_named = NodeSet(3; names=["Alice", "Bob", "Carol"], id=:students)
        @test length(ns_named) == 3
        @test ns_named.names == ["Alice", "Bob", "Carol"]
        @test ns_named.id == :students
    end

    @testset "DependentNetwork" begin
        # Create 3 waves of network data
        Random.seed!(42)
        n = 20
        networks = [zeros(Int, n, n) for _ in 1:3]

        # Add some ties
        for w in 1:3
            for i in 1:n
                for j in 1:n
                    if i != j && rand() < 0.1 + 0.05 * w
                        networks[w][i, j] = 1
                    end
                end
            end
        end

        dep = DependentNetwork(:friendship, networks)
        @test n_waves(dep) == 3
        @test n_actors(dep) == 20
        @test dep.type == :onemode
        @test dep.directed == true
    end

    @testset "DependentBehavior" begin
        # Create 3 waves of behavior data
        Random.seed!(42)
        n = 20
        values = [rand(1:5, n) for _ in 1:3]

        beh = DependentBehavior(:drinking, values)
        @test n_waves(beh) == 3
        @test n_actors(beh) == 20
        @test beh.min_val == 1
        @test beh.max_val == 5
    end

    @testset "Covariates" begin
        # Constant covariate
        vals = randn(20)
        cov = ConstantCovariate(:age, vals)
        @test cov.centered == true
        @test abs(mean(cov.values)) < 1e-10  # Should be centered

        # Varying covariate
        vals_v = [randn(20) for _ in 1:3]
        cov_v = VaryingCovariate(:income, vals_v)
        @test length(cov_v.values) == 3

        # Dyad covariate
        mat = randn(20, 20)
        dcov = ConstantDyadCovariate(:distance, mat)
        @test size(dcov.values) == (20, 20)
    end

    @testset "SienaData" begin
        data = siena_data()

        # Add node set
        add_nodeset!(data, NodeSet(20))

        # Add network
        networks = [rand(0:1, 20, 20) for _ in 1:3]
        for net in networks
            for i in 1:20
                net[i, i] = 0  # No self-loops
            end
        end
        add_dependent!(data, DependentNetwork(:friendship, networks))

        # Add covariate
        add_covariate!(data, ConstantCovariate(:gender, rand(0:1, 20)))

        @test data.n_waves == 3
        @test length(data.dependents) == 1
        @test length(data.covariates) == 1
    end

    @testset "NetworkState" begin
        data = siena_data()
        add_nodeset!(data, NodeSet(10))

        networks = [rand(0:1, 10, 10) for _ in 1:3]
        add_dependent!(data, DependentNetwork(:net, networks))

        state = NetworkState()
        initialize!(state, data, 1)

        @test haskey(state.networks, :net)
        @test size(state.networks[:net]) == (10, 10)
        @test state.time == 0.0
    end

    @testset "Effects" begin
        data = siena_data()
        add_nodeset!(data, NodeSet(20))
        networks = [rand(0:1, 20, 20) for _ in 1:3]
        add_dependent!(data, DependentNetwork(:friendship, networks))

        effects = get_effects(data)
        @test length(effects) > 0

        # Include some effects
        include_effects!(effects, :friendship, [:outdegree, :recip, :transTrip])

        included = get_included_effects(effects)
        obj_included = filter(e -> e.effect isa NetworkEffect, included)
        @test length(obj_included) == 3

        # Check rate effects are included by default
        rate_effs = get_rate_effects(effects)
        @test length(rate_effs) == 2  # 2 periods
    end

    @testset "Effect Computations" begin
        # Setup
        data = siena_data()
        add_nodeset!(data, NodeSet(5))

        net1 = [0 1 0 0 0;
                0 0 1 0 0;
                0 0 0 1 0;
                0 0 0 0 0;
                0 0 0 0 0]
        net2 = copy(net1)
        net2[1, 3] = 1  # Add a tie
        networks = [net1, net2]
        add_dependent!(data, DependentNetwork(:net, networks))

        state = NetworkState()
        initialize!(state, data, 1)

        # Test outdegree
        outdeg_eff = OutdegreeEffect(:net)
        @test compute_contribution(outdeg_eff, state, data, 1, 2) == 1.0

        # Test reciprocity
        recip_eff = ReciprocityEffect(:net)
        @test compute_contribution(recip_eff, state, data, 2, 1) == 1.0  # 1→2 exists
        @test compute_contribution(recip_eff, state, data, 1, 3) == 0.0  # 3→1 doesn't exist

        # Test transitive triplets
        # For 1→3: we have 1→2 and 2→3, so one transitive pattern
        trans_eff = TransitiveTripletsEffect(:net)
        contrib = compute_contribution(trans_eff, state, data, 1, 3)
        @test contrib >= 1.0
    end

    @testset "Algorithm Configuration" begin
        alg = SienaAlgorithm()
        @test alg.n_subphases == 4
        @test alg.convergence_threshold == 0.25

        alg2 = siena_algorithm(n_subphases=3, seed=42)
        @test alg2.n_subphases == 3
        @test alg2.seed == 42
    end

    @testset "Gain Sequence" begin
        gs = GainSequence(0.2, 0.001)
        @test gs.current == 0.2

        g1 = next_gain!(gs)
        @test g1 == 0.2  # First iteration

        g2 = next_gain!(gs)
        @test g2 == 0.1  # Second iteration

        reset_gain!(gs)
        @test gs.iteration == 0
        @test gs.current == 0.2
    end

    @testset "Simulation" begin
        Random.seed!(123)

        # Create simple data
        data = siena_data()
        add_nodeset!(data, NodeSet(10))

        net1 = zeros(Int, 10, 10)
        for i in 1:10
            for j in 1:10
                if i != j && rand() < 0.2
                    net1[i, j] = 1
                end
            end
        end
        net2 = copy(net1)  # Just copy for simplicity
        add_dependent!(data, DependentNetwork(:net, [net1, net2]))

        effects = get_effects(data)
        include_effects!(effects, :net, [:outdegree, :recip])

        θ = [0.0, 0.0]  # Zero parameters

        # Run simulation
        state, results = simulate_saom(data, effects, θ; seed=42)

        @test haskey(state.networks, :net)
        @test length(results) == 1  # One period
    end

    @testset "GOF Statistics" begin
        data = siena_data()
        add_nodeset!(data, NodeSet(10))

        net = zeros(Int, 10, 10)
        for i in 1:10
            for j in 1:10
                if i != j && rand() < 0.3
                    net[i, j] = 1
                end
            end
        end
        add_dependent!(data, DependentNetwork(:net, [net, net]))

        state = NetworkState()
        initialize!(state, data, 1)

        # Test indegree distribution
        indeg_stat = IndegreeDistribution(:net)
        levls, counts = compute_gof_statistic(indeg_stat, state, data)
        @test sum(counts) == 10  # Total should equal number of nodes

        # Test outdegree distribution
        outdeg_stat = OutdegreeDistribution(:net)
        levls, counts = compute_gof_statistic(outdeg_stat, state, data)
        @test sum(counts) == 10

        # Test triad census
        triad_stat = TriadCensus(:net)
        labels, counts = compute_gof_statistic(triad_stat, state, data)
        @test length(labels) == 3
        @test sum(counts) == binomial(10, 2)  # Total number of dyads
    end

    @testset "Integration Test" begin
        Random.seed!(456)

        # Create realistic network data
        n = 15
        data = siena_data()
        add_nodeset!(data, NodeSet(n))

        # Create networks with some structure
        networks = Vector{Matrix{Int}}(undef, 3)
        networks[1] = zeros(Int, n, n)

        # Initial random network
        for i in 1:n
            for j in 1:n
                if i != j && rand() < 0.15
                    networks[1][i, j] = 1
                end
            end
        end

        # Evolve network (add some reciprocity and transitivity)
        for w in 2:3
            networks[w] = copy(networks[w-1])
            for i in 1:n
                for j in 1:n
                    if i == j
                        continue
                    end
                    # Reciprocity: if j→i exists, more likely to create i→j
                    if networks[w-1][j, i] == 1 && networks[w][i, j] == 0
                        if rand() < 0.3
                            networks[w][i, j] = 1
                        end
                    end
                    # Random tie changes
                    if rand() < 0.05
                        networks[w][i, j] = 1 - networks[w][i, j]
                    end
                end
            end
        end

        add_dependent!(data, DependentNetwork(:friendship, networks))

        # Add covariate
        gender = rand(0:1, n)
        add_covariate!(data, ConstantCovariate(:gender, Float64.(gender)))

        # Get effects
        effects = get_effects(data)

        # Include effects
        include_effects!(effects, :friendship, [:outdegree, :recip])

        # Check effects table
        eff_table = effects_table(effects)
        @test eff_table isa DataFrame
        @test nrow(eff_table) > 0

        # Test that we can simulate
        obj_effs = get_objective_effects(effects)
        n_params = sum(e -> !e.fix, obj_effs)
        θ = zeros(n_params)

        state, _ = simulate_saom(data, effects, θ; seed=789)
        @test haskey(state.networks, :friendship)
    end

end
