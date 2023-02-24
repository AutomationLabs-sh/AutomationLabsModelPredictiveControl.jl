# Copyright (c) 2022: Pierre Blaud and contributors
########################################################
# This Source Code Form is subject to the terms of the #
# Mozilla Public License, v. 2.0. If a copy of the MPL #
# was not distributed with this file,  				   #
# You can obtain one at https://mozilla.org/MPL/2.0/.  #
########################################################

module TerminalIngredientsModelPredictiveControl

using MLJ
using MLJFlux
using MLJTuning
using MLJBase
using MLJParticleSwarmOptimization
using StableRNGs
using AutomationLabsIdentification
using JuMP
using Test
using AutomationLabsModelPredictiveControl
using MathematicalSystems
using AutomationLabsSystems

#=
@testset "None terminal ingredient and none state constraint computation" begin

    # Parameters                 
    hmin = 0.2
    h1max = 1.36
    h2max = 1.36
    h3max = 1.30
    h4max = 1.30
    qmin = 0
    qamax = 4
    qbmax = 3.26

    # Constraint definition:
    x_cons = [hmin h1max;
              hmin h2max;
              hmin h3max;
              hmin h4max]

    u_cons = [qmin qamax;
              qmin qbmax]

    # Get the fnn model to design the mpc controler
    fnn_machine = machine("./models_saved/densenet_train_result.jls")

    #MPC design parameters
    horizon = 5
    sample_time = 5
    max_time = 5
    x_ref = [0.65, 0.65, 0.65, 0.65] #.* ones(4, horizon + 1)
    u_ref = [1.2, 1.2] #.* ones(2, horizon)

    system = MathematicalSystems.ConstrainedBlackBoxControlDiscreteSystem(
        fnn_machine,
        4,
        2,
        x_cons,
        u_cons,
    )

    # None terminal constraint with terminal cost
    c_none_none =  proceed_controller( fnn_machine, 
        "model_predictive_control",
        "linear",
        u_cons, 
        horizon,
        sample_time, 
        x_ref, 
        u_ref;
    )

    function proceed_controller(
        system,
        mpc_controller_type::String,
        mpc_horizon::Int,
        mpc_sample_time::Int, 
        mpc_state_reference::Vector, 
        mpc_input_reference::Vector;
        kws_...
    )

    cons = JuMP.all_constraints(c_none_none.tuning.modeler; include_variable_in_set_constraints = false)

    @test length(cons) == 74

    initialization = [0.6, 0.6, 0.6, 0.6]
    update_initialization!(c_none_none, initialization)
    calculate!(c_none_none)

    @test c_none_none.computation_results.x != 0
    @test c_none_none.computation_results.e_x != 0
    @test c_none_none.computation_results.u != 0

end
=#


@testset "None terminal ingredient computation" begin

    # Parameters                 
    hmin = 0.2
    h1max = 1.36
    h2max = 1.36
    h3max = 1.30
    h4max = 1.30
    qmin = 0
    qamax = 4
    qbmax = 3.26

    # Constraint definition:
    x_cons = [hmin h1max;
              hmin h2max;
              hmin h3max;
              hmin h4max]

    u_cons = [qmin qamax;
              qmin qbmax]

    # Get the fnn model to design the mpc controler
    densenet_machine = machine("./models_saved/densenet_train_result.jls")

    #MPC design parameters
    horizon = 5
    sample_time = 5
    max_time = 5
    x_ref = [0.65, 0.65, 0.65, 0.65] #.* ones(4, horizon + 1)
    u_ref = [1.2, 1.2] #.* ones(2, horizon)

    system = AutomationLabsSystems.proceed_system(
        "discrete", 
        "identification"; 
        f=densenet_machine, 
        state_constraint = x_cons, 
        input_constraint = u_cons, 
        nbr_state = 4,
        nbr_input = 2)

    # None terminal constraint with terminal cost
    c_none = proceed_controller(
        system,
        "model_predictive_control",
        horizon,
        sample_time, 
        x_ref, 
        u_ref;
        mpc_terminal_ingredient = "none",
        mpc_programming_type = "linear", 
    )

    cons = JuMP.all_constraints(c_none.tuning.modeler; include_variable_in_set_constraints = false)

    @test length(cons) == 74

    initialization = [0.6, 0.6, 0.6, 0.6]
    update_initialization!(c_none, initialization)
    calculate!(c_none)

    @test c_none.computation_results.x != 0
    @test c_none.computation_results.e_x != 0
    @test c_none.computation_results.u != 0

end

@testset "Contractive terminal ingredient computation" begin

    # Parameters                 
    hmin = 0.2
    h1max = 1.36
    h2max = 1.36
    h3max = 1.30
    h4max = 1.30
    qmin = 0
    qamax = 4
    qbmax = 3.26

    # Constraint definition:
    x_cons = [hmin h1max;
              hmin h2max;
              hmin h3max;
              hmin h4max]

    u_cons = [qmin qamax;
              qmin qbmax]

    # Get the fnn model to design the mpc controler
    densenet_machine = machine("./models_saved/densenet_train_result.jls")

    #MPC design parameters
    horizon = 5
    sample_time = 5
    max_time = 5
    x_ref = [0.65, 0.65, 0.65, 0.65] #.* ones(4, horizon + 1)
    u_ref = [1.2, 1.2] #.* ones(2, horizon)

    system = AutomationLabsSystems.proceed_system(
        "discrete", 
        "identification"; 
        f=densenet_machine, 
        state_constraint = x_cons, 
        input_constraint = u_cons, 
        nbr_state = 4,
        nbr_input = 2)

    # Contractive terminal constraint with terminal cost
    c_contractive = proceed_controller(
        system,
        "model_predictive_control",
        horizon,
        sample_time, 
        x_ref, 
        u_ref;
        mpc_terminal_ingredient = "contractive",
        mpc_programming_type = "linear", 
    )
    
    cons = JuMP.all_constraints(c_contractive.tuning.modeler; include_variable_in_set_constraints = false)

    @test length(cons) == 75
    @test string(cons[end]) == "-0.9 e_x[1,1]² - 0.9 e_x[2,1]² - 0.9 e_x[3,1]² - 0.9 e_x[4,1]² + e_x[1,6]² + e_x[2,6]² + e_x[3,6]² + e_x[4,6]² ≤ 0.0"

    initialization = [0.6, 0.6, 0.6, 0.6]
    update_initialization!(c_contractive, initialization)
    calculate!(c_contractive)

    @test c_contractive.computation_results.x != 0
    @test c_contractive.computation_results.e_x != 0
    @test c_contractive.computation_results.u != 0

end

@testset "Equality terminal ingredient computation" begin

    # Parameters                 
    hmin = 0.2
    h1max = 1.36
    h2max = 1.36
    h3max = 1.30
    h4max = 1.30
    qmin = 0
    qamax = 4
    qbmax = 3.26

    # Constraint definition:
    x_cons = [hmin h1max;
              hmin h2max;
              hmin h3max;
              hmin h4max]

    u_cons = [qmin qamax;
              qmin qbmax]

    # Get the fnn model to design the mpc controler
    densenet_machine = machine("./models_saved/densenet_train_result.jls")

    #MPC design parameters
    horizon = 5
    sample_time = 5
    max_time = 5
    x_ref = [0.65, 0.65, 0.65, 0.65] #.* ones(4, horizon + 1)
    u_ref = [1.2, 1.2] #.* ones(2, horizon)

    system = AutomationLabsSystems.proceed_system(
        "discrete", 
        "identification"; 
        f=densenet_machine, 
        state_constraint = x_cons, 
        input_constraint = u_cons, 
        nbr_state = 4,
        nbr_input = 2)

    # Equality terminal constraint with terminal cost
    c_equality = proceed_controller(
        system,
        "model_predictive_control",
        horizon,
        sample_time, 
        x_ref, 
        u_ref;
        mpc_terminal_ingredient = "equality",
        mpc_programming_type = "linear", 
        mpc_solver = "ipopt"
    )

    cons = JuMP.all_constraints(c_equality.tuning.modeler; include_variable_in_set_constraints = false)

    @test length(cons) == 78
    @test string(cons[55]) == "e_x[1,6] = 0.0"
    @test string(cons[56]) == "e_x[2,6] = 0.0"
    @test string(cons[57]) == "e_x[3,6] = 0.0"
    @test string(cons[58]) == "e_x[4,6] = 0.0"

    initialization = [0.6, 0.6, 0.6, 0.6]
    update_initialization!(c_equality, initialization)
    calculate!(c_equality)

    @test c_equality.computation_results.x != 0
    @test c_equality.computation_results.e_x != 0
    @test c_equality.computation_results.u != 0

end

@testset "Neighborhood terminal ingredient computation" begin

    # Parameters                 
    hmin = 0.2
    h1max = 1.36
    h2max = 1.36
    h3max = 1.30
    h4max = 1.30
    qmin = 0
    qamax = 4
    qbmax = 3.26

    # Constraint definition:
    x_cons = [hmin h1max;
              hmin h2max;
              hmin h3max;
              hmin h4max]

    u_cons = [qmin qamax;
              qmin qbmax]

    # Get the fnn model to design the mpc controler
    densenet_machine = machine("./models_saved/densenet_train_result.jls")

    #MPC design parameters
    horizon = 5
    sample_time = 5
    max_time = 5
    x_ref = [0.65, 0.65, 0.65, 0.65] #.* ones(4, horizon + 1)
    u_ref = [1.2, 1.2] #.* ones(2, horizon)

    system = AutomationLabsSystems.proceed_system(
        "discrete", 
        "identification"; 
        f=densenet_machine, 
        state_constraint = x_cons, 
        input_constraint = u_cons, 
        nbr_state = 4,
        nbr_input = 2)

    # Neighborhood terminal constraint with terminal cost
    c_neighborhood = proceed_controller(
        system,
        "model_predictive_control",
        horizon,
        sample_time, 
        x_ref, 
        u_ref;
        mpc_terminal_ingredient = "neighborhood",
        mpc_programming_type = "linear", 
    )

    cons_neighborhood = JuMP.all_constraints(c_neighborhood.tuning.modeler; include_variable_in_set_constraints = false)
    
    @test length(cons_neighborhood) == 95
    @test string(cons_neighborhood[75]) == "1.1821257469929807 e_x[1,6] + 0.32477783075165184 e_x[2,6] + 3.753955653769598 e_x[3,6] - 0.6171907205612299 e_x[4,6] ≤ 1.2000000000000002"
    @test string(cons_neighborhood[76]) == "1.9844516433175883 e_x[1,6] + 1.0527123209042824 e_x[2,6] + 4.957905573404276 e_x[3,6] - 0.15120229651054853 e_x[4,6] ≤ 1.2000000000000002"
    @test string(cons_neighborhood[77]) == "-0.7174463800834686 e_x[1,6] - 0.16731968358133426 e_x[2,6] - 0.14299884029284093 e_x[3,6] + 0.5743028497226739 e_x[4,6] ≤ 0.45000000000000007"
    @test string(cons_neighborhood[78]) == "1.6907988718505544 e_x[1,6] + 3.6956354224478427 e_x[2,6] + 0.8947871695749029 e_x[3,6] + 5.115699102348034 e_x[4,6] ≤ 1.2000000000000002"
    @test string(cons_neighborhood[79]) == "3.2841587345393615 e_x[1,6] + 2.31606313179734 e_x[2,6] + 6.807979550060749 e_x[3,6] + 0.7214809704540696 e_x[4,6] ≤ 1.2000000000000002"
    @test string(cons_neighborhood[80]) == "0.7917175253223642 e_x[1,6] + 0.18629471807030692 e_x[2,6] + 0.19261779399927226 e_x[3,6] - 0.5388689000604783 e_x[4,6] ≤ 0.7100000000000001"
    @test string(cons_neighborhood[81]) == "-0.7917175253223642 e_x[1,6] - 0.18629471807030692 e_x[2,6] - 0.19261779399927226 e_x[3,6] + 0.5388689000604783 e_x[4,6] ≤ 0.45000000000000007"
    @test string(cons_neighborhood[82]) == "0.1031718933941024 e_x[1,6] + 0.1083387497766164 e_x[2,6] - 0.7878420390865811 e_x[3,6] + 0.02814026958189298 e_x[4,6] ≤ 0.45000000000000007"
    @test string(cons_neighborhood[83]) == "-5.355335429620435 e_x[1,6] - 4.56602538130204 e_x[2,6] - 9.82213786152917 e_x[3,6] - 2.0758851736694397 e_x[4,6] ≤ 2.0599999999999996"
    @test string(cons_neighborhood[84]) == "3.426874514007263 e_x[1,6] + 5.553294145977776 e_x[2,6] + 2.5345823089219666 e_x[3,6] + 7.315365801939883 e_x[4,6] ≤ 1.2000000000000002"
    @test string(cons_neighborhood[85]) == "5.355335429620435 e_x[1,6] + 4.56602538130204 e_x[2,6] + 9.82213786152917 e_x[3,6] + 2.0758851736694397 e_x[4,6] ≤ 1.2000000000000002"
    @test string(cons_neighborhood[86]) == "e_x[1,6] ≤ 0.7100000000000001"
    @test string(cons_neighborhood[87]) == "e_x[2,6] ≤ 0.7100000000000001"
    @test string(cons_neighborhood[88]) == "-e_x[1,6] ≤ 0.45000000000000007"
    @test string(cons_neighborhood[89]) == "-e_x[2,6] ≤ 0.45000000000000007"
    @test string(cons_neighborhood[90]) == "-e_x[3,6] ≤ 0.45000000000000007"
    @test string(cons_neighborhood[91]) == "-e_x[4,6] ≤ 0.45000000000000007"
    @test string(cons_neighborhood[92]) == "-7.122630595237694 e_x[1,6] - 6.837395805570274 e_x[2,6] - 3.3426181833913065 e_x[3,6] - 14.372581083482705 e_x[4,6] ≤ 2.8"
    @test string(cons_neighborhood[93]) == "-8.103867109868654 e_x[1,6] - 9.72924467090224 e_x[2,6] - 16.263131538649258 e_x[3,6] - 1.5721879837156578 e_x[4,6] ≤ 2.0599999999999996"
    @test string(cons_neighborhood[94]) == "7.122630595237694 e_x[1,6] + 6.837395805570274 e_x[2,6] + 3.3426181833913065 e_x[3,6] + 14.372581083482705 e_x[4,6] ≤ 1.2000000000000002"
    @test string(cons_neighborhood[95]) == "8.103867109868654 e_x[1,6] + 9.72924467090224 e_x[2,6] + 16.263131538649258 e_x[3,6] + 1.5721879837156578 e_x[4,6] ≤ 1.2000000000000002"

    initialization = [0.6, 0.6, 0.6, 0.6]
    update_initialization!(c_neighborhood, initialization)
    calculate!(c_neighborhood)

    @test c_neighborhood.computation_results.x != 0
    @test c_neighborhood.computation_results.e_x != 0
    @test c_neighborhood.computation_results.u != 0

end
end