# Copyright (c) 2022: Pierre Blaud and contributors
########################################################
# This Source Code Form is subject to the terms of the #
# Mozilla Public License, v. 2.0. If a copy of the MPL #
# was not distributed with this file,  				   #
# You can obtain one at https://mozilla.org/MPL/2.0/.  #
########################################################

module ComputeEconomicModelPredictiveControl

using MLJ
using MLJFlux
using MLJTuning
using MLJBase
using MLJParticleSwarmOptimization
using StableRNGs
using AutomationLabsIdentification
using MathematicalSystems
using LazySets
using JuMP
using Test
using MathOptInterface

using ModelPredictiveControl

import ModelPredictiveControl: auto_solver_def
import ModelPredictiveControl: _economic_model_predictive_control_design


@testset "compute fnn eMPC: linear, non linear and milp" begin

    ###################################
    ### parameters                  ###
    ###################################

    hmin = 0.2
    h1max = 1.36
    h2max = 1.36
    h3max = 1.30
    h4max = 1.30
    qmin = 0
    qamax = 4
    qbmax = 3.26

    #Constraint definition:
    x_cons = LazySets.Hyperrectangle(
        low = [hmin, hmin, hmin, hmin],
        high = [h1max, h2max, h3max, h4max],
    )
    u_cons = LazySets.Hyperrectangle(low = [qmin, qmin], high = [qamax, qbmax])

    #get the fnn model to design the mpc controler
    fnn_machine = machine("./models_saved/fnn_train_result.jls")

    #extract best model from the all trained models
    mlj_fnn = fitted_params(fitted_params(fnn_machine).machine).best_model
    f_fnn = fitted_params(fitted_params(fnn_machine).machine).best_fitted_params[1]
    type_fnn = mlj_fnn.builder

    #system definition with MAthematical systems
    QTP_sys_fnn = MathematicalSystems.ConstrainedBlackBoxControlDiscreteSystem(
        f_fnn,
        4,
        2,
        x_cons,
        u_cons,
    )

    #MPC design parameters
    horizon = 5
    sample_time = 5
    terminal_ingredients = false
    max_time = 5
    x = [0.35, 0.35, 0.35, 0.35] .* ones(4, horizon + 1)
    u = [0, 0] .* ones(2, horizon)
    references = ModelPredictiveControl.ReferencesStateInput(x, u)

    ##############################
    ###           Fnn          ###                    
    ##############################

    ### Fnn L MPC ###
    method = ModelPredictiveControl.LinearProgramming()
    solver = ModelPredictiveControl.OSQP_solver_def()
    C_fnn_linear = _economic_model_predictive_control_design(
        QTP_sys_fnn,
        type_fnn,
        horizon,
        method,
        sample_time,
        references,
        solver = solver,
    )

    initialization = [0.35, 0.35, 0.35, 0.35]

    UpdateInitialization!(C_fnn_linear, initialization)

    Calculate!(C_fnn_linear)

    C_fnn_linear.computation_results.x
    C_fnn_linear.computation_results.u

    ### Fnn N L MPC ###
    method = ModelPredictiveControl.NonLinearProgramming()
    C_fnn_nl = _economic_model_predictive_control_design(
        QTP_sys_fnn,
        type_fnn,
        horizon,
        method,
        sample_time,
        references,
    )

    initialization = [0.35, 0.35, 0.35, 0.35]

    UpdateInitialization!(C_fnn_nl, initialization)

    Calculate!(C_fnn_nl)

    C_fnn_nl.computation_results.x
    C_fnn_nl.computation_results.u

    ### Fnn MILP MPC ###
    method = ModelPredictiveControl.MixedIntegerLinearProgramming()
    solver = ModelPredictiveControl.SCIP_solver_def()
    C_fnn_milp = _economic_model_predictive_control_design(
        QTP_sys_fnn,
        type_fnn,
        horizon,
        method,
        sample_time,
        references,
        solver = solver
    )

    initialization = [0.35, 0.35, 0.35, 0.35]

    UpdateInitialization!(C_fnn_milp, initialization)

    Calculate!(C_fnn_milp)

    C_fnn_milp.computation_results.x
    C_fnn_milp.computation_results.u

    #@test
    @test C_fnn_linear.computation_results.x ≈ C_fnn_nl.computation_results.x atol = 0.5
    @test C_fnn_linear.computation_results.x ≈ C_fnn_milp.computation_results.x atol = 0.5
    @test C_fnn_nl.computation_results.x ≈ C_fnn_milp.computation_results.x atol = 0.5
    @test C_fnn_linear.computation_results.u[:, 1] ≈ C_fnn_nl.computation_results.u[:, 1] atol = 0.1
    @test C_fnn_linear.computation_results.u[:, 1] ≈ C_fnn_milp.computation_results.u[:, 1] atol = 0.1
    @test C_fnn_nl.computation_results.u ≈ C_fnn_milp.computation_results.u atol = 0.01 

end


@testset "compute resnet eMPC: linear, non linear and milp" begin

    ###################################
    ### parameters                  ###
    ###################################

    hmin = 0.2
    h1max = 1.36
    h2max = 1.36
    h3max = 1.30
    h4max = 1.30
    qmin = 0
    qamax = 4
    qbmax = 3.26

    #Constraint definition:
    x_cons = LazySets.Hyperrectangle(
        low = [hmin, hmin, hmin, hmin],
        high = [h1max, h2max, h3max, h4max],
    )
    u_cons = LazySets.Hyperrectangle(low = [qmin, qmin], high = [qamax, qbmax])

    #get the resnet model to design the mpc controler
    resnet_machine = machine("./models_saved/resnet_train_result.jls")

    #extract best model from the all trained models
    mlj_resnet = fitted_params(fitted_params(resnet_machine).machine).best_model
    f_resnet = fitted_params(fitted_params(resnet_machine).machine).best_fitted_params[1]
    type_resnet = mlj_resnet.builder

    #system definition with Mathematical systems
    QTP_sys_resnet = MathematicalSystems.ConstrainedBlackBoxControlDiscreteSystem(
        f_resnet,
        4,
        2,
        x_cons,
        u_cons,
    )

    #MPC design parameters
    horizon = 5
    sample_time = 5
    terminal_ingredients = false
    max_time = 5
    x = [0.3, 0.3, 0.3, 0.3] .* ones(4, horizon + 1)
    u = [0, 0] .* ones(2, horizon)
    references = ModelPredictiveControl.ReferencesStateInput(x, u)


    ##############################
    ###           ResNet       ###                    
    ##############################


    ### ResNet L MPC ###
    solver = ModelPredictiveControl.HiGHS_solver_def()
    method = ModelPredictiveControl.LinearProgramming()
    C_resnet_linear = _economic_model_predictive_control_design(
        QTP_sys_resnet,
        type_resnet,
        horizon,
        method,
        sample_time,
        references,
        solver = solver,
    )


    initialization = [0.35, 0.35, 0.35, 0.35]

    UpdateInitialization!(C_resnet_linear, initialization)

    Calculate!(C_resnet_linear)

    C_resnet_linear.computation_results.x
    C_resnet_linear.computation_results.u

    ### ResNet N L MPC ###
    method = ModelPredictiveControl.NonLinearProgramming()
    C_resnet_nl = _economic_model_predictive_control_design(
        QTP_sys_resnet,
        type_resnet,
        horizon,
        method,
        sample_time,
        references,
    )

    initialization = [0.35, 0.35, 0.35, 0.35]

    UpdateInitialization!(C_resnet_nl, initialization)

    Calculate!(C_resnet_nl)

    C_resnet_nl.computation_results.x
    C_resnet_nl.computation_results.u

    ### ResNet MILP MPC ###
    solver = ModelPredictiveControl.Mosek_solver_def()

    method = ModelPredictiveControl.MixedIntegerLinearProgramming()
    C_resnet_milp = _economic_model_predictive_control_design(
        QTP_sys_resnet,
        type_resnet,
        horizon,
        method,
        sample_time,
        references,
        solver = solver,
    )

    initialization = [0.35, 0.35, 0.35, 0.35]

    UpdateInitialization!(C_resnet_milp, initialization)

    Calculate!(C_resnet_milp)

    C_resnet_milp.computation_results.x
    C_resnet_milp.computation_results.u

    #@test
    @test C_resnet_linear.computation_results.x ≈ C_resnet_nl.computation_results.x atol = 0.5
    @test C_resnet_linear.computation_results.x ≈ C_resnet_milp.computation_results.x atol = 0.5
    @test C_resnet_nl.computation_results.x ≈ C_resnet_milp.computation_results.x atol = 0.5
    @test C_resnet_linear.computation_results.u[:, 1] ≈
          C_resnet_nl.computation_results.u[:, 1] atol = 1
    @test C_resnet_linear.computation_results.u[:, 1] ≈
          C_resnet_milp.computation_results.u[:, 1] atol = 1
    @test C_resnet_nl.computation_results.u ≈ C_resnet_milp.computation_results.u atol = 0.01

end



@testset "compute densenet eMPC: linear, non linear and milp" begin

    ###################################
    ### parameters                  ###
    ###################################

    hmin = 0.2
    h1max = 1.36
    h2max = 1.36
    h3max = 1.30
    h4max = 1.30
    qmin = 0
    qamax = 4
    qbmax = 3.26

    #Constraint definition:
    x_cons = LazySets.Hyperrectangle(
        low = [hmin, hmin, hmin, hmin],
        high = [h1max, h2max, h3max, h4max],
    )
    u_cons = LazySets.Hyperrectangle(low = [qmin, qmin], high = [qamax, qbmax])

    #get the densenet model to design the mpc controler
    densenet_machine = machine("./models_saved/densenet_train_result.jls")

    #extract best model from the all trained models
    mlj_densenet = fitted_params(fitted_params(densenet_machine).machine).best_model
    f_densenet =
        fitted_params(fitted_params(densenet_machine).machine).best_fitted_params[1]
    type_densenet = mlj_densenet.builder

    #system definition with Mathematical systems
    QTP_sys_densenet = MathematicalSystems.ConstrainedBlackBoxControlDiscreteSystem(
        f_densenet,
        4,
        2,
        x_cons,
        u_cons,
    )

    #MPC design parameters
    horizon = 5
    sample_time = 5
    terminal_ingredients = false
    max_time = 5
    x = [0.3, 0.3, 0.3, 0.3] .* ones(4, horizon + 1)
    u = [0, 0] .* ones(2, horizon)
    references = ModelPredictiveControl.ReferencesStateInput(x, u)


    ##############################
    ###         densenet       ###                    
    ##############################


    ### densenet L MPC ###
    method = ModelPredictiveControl.LinearProgramming()
    C_densenet_linear = _economic_model_predictive_control_design(
        QTP_sys_densenet,
        type_densenet,
        horizon,
        method,
        sample_time,
        references,
    )

    initialization = [0.35, 0.35, 0.35, 0.35]

    UpdateInitialization!(C_densenet_linear, initialization)

    Calculate!(C_densenet_linear)

    C_densenet_linear.computation_results.x
    C_densenet_linear.computation_results.u

    ### densenet N L MPC ###
    method = ModelPredictiveControl.NonLinearProgramming()
    C_densenet_nl = _economic_model_predictive_control_design(
        QTP_sys_densenet,
        type_densenet,
        horizon,
        method,
        sample_time,
        references,
    )

    initialization = [0.35, 0.35, 0.35, 0.35]

    UpdateInitialization!(C_densenet_nl, initialization)

    Calculate!(C_densenet_nl)

    C_densenet_nl.computation_results.x
    C_densenet_nl.computation_results.u

    ### densenet MILP MPC ###
    method = ModelPredictiveControl.MixedIntegerLinearProgramming()
    solver = ModelPredictiveControl.Mosek_solver_def()

    C_densenet_milp = _economic_model_predictive_control_design(
        QTP_sys_densenet,
        type_densenet,
        horizon,
        method,
        sample_time,
        references,
        solver = solver,
    )

    initialization = [0.35, 0.35, 0.35, 0.35]

    UpdateInitialization!(C_densenet_milp, initialization)

    Calculate!(C_densenet_milp)

    C_densenet_milp.computation_results.x
    C_densenet_milp.computation_results.u

    #@test
    @test C_densenet_linear.computation_results.x ≈ C_densenet_nl.computation_results.x atol =
        0.5
    @test C_densenet_linear.computation_results.x ≈ C_densenet_milp.computation_results.x atol =
        0.5
    @test C_densenet_nl.computation_results.x ≈ C_densenet_milp.computation_results.x atol =
        0.5
    @test C_densenet_linear.computation_results.u[:, 1] ≈
          C_densenet_nl.computation_results.u[:, 1] atol = 0.1 
    @test C_densenet_linear.computation_results.u[:, 1] ≈
          C_densenet_milp.computation_results.u[:, 1] atol = 0.1 
    @test C_densenet_nl.computation_results.u ≈ C_densenet_milp.computation_results.u atol =
        0.1 

end



@testset "compute polynet eMPC: linear, non linear and milp" begin

    ###################################
    ### parameters                  ###
    ###################################

    hmin = 0.2
    h1max = 1.36
    h2max = 1.36
    h3max = 1.30
    h4max = 1.30
    qmin = 0
    qamax = 4
    qbmax = 3.26

    #Constraint definition:
    x_cons = LazySets.Hyperrectangle(
        low = [hmin, hmin, hmin, hmin],
        high = [h1max, h2max, h3max, h4max],
    )
    u_cons = LazySets.Hyperrectangle(low = [qmin, qmin], high = [qamax, qbmax])

    #get the polynet model to design the mpc controler
    polynet_machine = machine("./models_saved/polynet_train_result.jls")

    #extract best model from the all trained models
    mlj_polynet = fitted_params(fitted_params(polynet_machine).machine).best_model
    f_polynet = fitted_params(fitted_params(polynet_machine).machine).best_fitted_params[1]
    type_polynet = mlj_polynet.builder

    #system definition with Mathematical systems
    QTP_sys_polynet = MathematicalSystems.ConstrainedBlackBoxControlDiscreteSystem(
        f_polynet,
        4,
        2,
        x_cons,
        u_cons,
    )

    #MPC design parameters
    horizon = 4
    sample_time = 5
    terminal_ingredients = false
    max_time = 5
    x = [0.3, 0.3, 0.3, 0.3] .* ones(4, horizon + 1)
    u = [0, 0] .* ones(2, horizon)
    references = ModelPredictiveControl.ReferencesStateInput(x, u)


    ##############################
    ###           polynet       ###                    
    ##############################


    ### polynet L MPC ###
    solver = ModelPredictiveControl.OSQP_solver_def()
    method = ModelPredictiveControl.LinearProgramming()
    C_polynet_linear = _economic_model_predictive_control_design(
        QTP_sys_polynet,
        type_polynet,
        horizon,
        method,
        sample_time,
        references,
        solver = solver,
    )


    initialization = [0.35, 0.35, 0.35, 0.35]

    UpdateInitialization!(C_polynet_linear, initialization)

    Calculate!(C_polynet_linear)

    C_polynet_linear.computation_results.x
    C_polynet_linear.computation_results.u

    ### polynet N L MPC ###
    method = ModelPredictiveControl.NonLinearProgramming()
    solver = ModelPredictiveControl.Ipopt_solver_def()

    C_polynet_nl = _economic_model_predictive_control_design(
        QTP_sys_polynet,
        type_polynet,
        horizon,
        method,
        sample_time,
        references
            )


    initialization = [0.35, 0.35, 0.35, 0.35]

    UpdateInitialization!(C_polynet_nl, initialization)

    Calculate!(C_polynet_nl)

    C_polynet_nl.computation_results.x
    C_polynet_nl.computation_results.u

    ### polynet MILP MPC ###
   #= solver = ModelPredictiveControl.Mosek_solver_def()

    method = ModelPredictiveControl.MixedIntegerLinearProgramming()
    C_polynet_milp = ModelPredictiveControlDesign(
        QTP_sys_polynet,
        type_polynet,
        horizon,
        method,
        sample_time,
        references,
    )


    initialization = [0.5, 0.5, 0.5, 0.5]

    UpdateInitialization!(C_polynet_milp, initialization)

    Calculate!(C_polynet_milp)

    C_polynet_milp.computation_results.x
    C_polynet_milp.computation_results.u
    C_polynet_milp.computation_results.e_x
    C_polynet_milp.computation_results.e_u =#

    #@test
    @test C_polynet_linear.computation_results.x ≈ C_polynet_nl.computation_results.x atol =
        0.5 broken=true
    @test C_polynet_linear.computation_results.x ≈ C_polynet_milp.computation_results.x atol =
        0.5  skip=true
    @test C_polynet_nl.computation_results.x ≈ C_polynet_milp.computation_results.x atol =
        0.01 skip=true
    @test C_polynet_linear.computation_results.u[:, 1] ≈
          C_polynet_nl.computation_results.u[:, 1] atol = 0.1  
    @test C_polynet_linear.computation_results.u[:, 1] ≈
          C_polynet_milp.computation_results.u[:, 1] atol = 0.1 skip=true
    @test C_polynet_nl.computation_results.u ≈ C_polynet_milp.computation_results.u atol =
        0.01 skip=true

end



@testset "compute icnn eMPC: linear, non linear and milp" begin

    ###################################
    ### parameters                  ###
    ###################################

    hmin = 0.2
    h1max = 1.36
    h2max = 1.36
    h3max = 1.30
    h4max = 1.30
    qmin = 0
    qamax = 4
    qbmax = 3.26

    #Constraint definition:
    x_cons = LazySets.Hyperrectangle(
        low = [hmin, hmin, hmin, hmin],
        high = [h1max, h2max, h3max, h4max],
    )
    u_cons = LazySets.Hyperrectangle(low = [qmin, qmin], high = [qamax, qbmax])

    #get the icnn model to design the mpc controler
    icnn_machine = machine("./models_saved/icnn_train_result.jls")

    #extract best model from the all trained models
    mlj_icnn = fitted_params(fitted_params(icnn_machine).machine).best_model
    f_icnn = fitted_params(fitted_params(icnn_machine).machine).best_fitted_params[1]
    type_icnn = mlj_icnn.builder

    #system definition with MAthematical systems
    QTP_sys_icnn = MathematicalSystems.ConstrainedBlackBoxControlDiscreteSystem(
        f_icnn,
        4,
        2,
        x_cons,
        u_cons,
    )

    #MPC design parameters
    horizon = 5
    sample_time = 5
    terminal_ingredients = false
    max_time = 5
    x = [0.3, 0.3, 0.3, 0.3] .* ones(4, horizon + 1)
    u = [0, 0] .* ones(2, horizon)
    references = ModelPredictiveControl.ReferencesStateInput(x, u)

    ##############################
    ###           icnn          ###                    
    ##############################

    ### icnn L MPC ###
    method = ModelPredictiveControl.LinearProgramming()
    solver = ModelPredictiveControl.OSQP_solver_def()
    C_icnn_linear = _economic_model_predictive_control_design(
        QTP_sys_icnn,
        type_icnn,
        horizon,
        method,
        sample_time,
        references,
        solver = solver,
    )

    initialization = [0.35, 0.35, 0.35, 0.35]

    UpdateInitialization!(C_icnn_linear, initialization)

    Calculate!(C_icnn_linear)

    C_icnn_linear.computation_results.x
    C_icnn_linear.computation_results.u

    ### icnn N L MPC ###
    method = ModelPredictiveControl.NonLinearProgramming()
    C_icnn_nl = _economic_model_predictive_control_design(
        QTP_sys_icnn,
        type_icnn,
        horizon,
        method,
        sample_time,
        references,
    )

    initialization = [0.35, 0.35, 0.35, 0.35]

    UpdateInitialization!(C_icnn_nl, initialization)

    Calculate!(C_icnn_nl)

    C_icnn_nl.computation_results.x
    C_icnn_nl.computation_results.u

    ### icnn MILP MPC ###
    method = ModelPredictiveControl.MixedIntegerLinearProgramming()
    solver = ModelPredictiveControl.SCIP_solver_def()
    C_icnn_milp = _economic_model_predictive_control_design(
        QTP_sys_icnn,
        type_icnn,
        horizon,
        method,
        sample_time,
        references,
        solver = solver
    )

    initialization = [0.35, 0.35, 0.35, 0.35]

    UpdateInitialization!(C_icnn_milp, initialization)

    Calculate!(C_icnn_milp)

    C_icnn_milp.computation_results.x
    C_icnn_milp.computation_results.u

    #@test
    @test C_icnn_linear.computation_results.x ≈ C_icnn_nl.computation_results.x atol = 0.5
    @test C_icnn_linear.computation_results.x ≈ C_icnn_milp.computation_results.x atol = 0.5
    @test C_icnn_nl.computation_results.x ≈ C_icnn_milp.computation_results.x atol = 0.5
    @test C_icnn_linear.computation_results.u[:, 1] ≈ C_icnn_nl.computation_results.u[:, 1] atol = 0.1 

    @test C_icnn_linear.computation_results.u[:, 1] ≈ C_icnn_milp.computation_results.u[:, 1] atol = 0.1 

    @test C_icnn_nl.computation_results.u ≈ C_icnn_milp.computation_results.u atol = 0.01

end


@testset "compute rbf eMPC: linear, non linear" begin

    ###################################
    ### parameters                  ###
    ###################################

    hmin = 0.2
    h1max = 1.36
    h2max = 1.36
    h3max = 1.30
    h4max = 1.30
    qmin = 0
    qamax = 4
    qbmax = 3.26

    #Constraint definition:
    x_cons = LazySets.Hyperrectangle(
        low = [hmin, hmin, hmin, hmin],
        high = [h1max, h2max, h3max, h4max],
    )
    u_cons = LazySets.Hyperrectangle(low = [qmin, qmin], high = [qamax, qbmax])

    #get the rbf model to design the mpc controler
    rbf_machine = machine("./models_saved/rbf_train_result.jls")

    #extract best model from the all trained models
    mlj_rbf = fitted_params(fitted_params(rbf_machine).machine).best_model
    f_rbf = fitted_params(fitted_params(rbf_machine).machine).best_fitted_params[1]
    type_rbf = mlj_rbf.builder

    #system definition with MAthematical systems
    QTP_sys_rbf = MathematicalSystems.ConstrainedBlackBoxControlDiscreteSystem(
        f_rbf,
        4,
        2,
        x_cons,
        u_cons,
    )

    #MPC design parameters
    horizon = 5
    sample_time = 5
    terminal_ingredients = false
    max_time = 5
    x = [0.3, 0.3, 0.3, 0.3] .* ones(4, horizon + 1)
    u = [0, 0] .* ones(2, horizon)
    references = ModelPredictiveControl.ReferencesStateInput(x, u)

    ##############################
    ###           rbf          ###                    
    ##############################

    ### rbf L MPC ###
    method = ModelPredictiveControl.LinearProgramming()
    solver = ModelPredictiveControl.OSQP_solver_def()
    C_rbf_linear = _economic_model_predictive_control_design(
        QTP_sys_rbf,
        type_rbf,
        horizon,
        method,
        sample_time,
        references,
        solver = solver,
    )

    initialization = [0.35, 0.35, 0.35, 0.35]

    UpdateInitialization!(C_rbf_linear, initialization)

    Calculate!(C_rbf_linear)

    C_rbf_linear.computation_results.x
    C_rbf_linear.computation_results.u

    ### rbf N L MPC ###
    method = ModelPredictiveControl.NonLinearProgramming()
    C_rbf_nl = _economic_model_predictive_control_design(
        QTP_sys_rbf,
        type_rbf,
        horizon,
        method,
        sample_time,
        references,
    )

    initialization = [0.35, 0.35, 0.35, 0.35]

    UpdateInitialization!(C_rbf_nl, initialization)

    Calculate!(C_rbf_nl)

    C_rbf_nl.computation_results.x
    C_rbf_nl.computation_results.u
   
    #@test
    @test C_rbf_linear.computation_results.x ≈ C_rbf_nl.computation_results.x atol = 0.5  broken=true
    @test C_rbf_linear.computation_results.u[:, 1] ≈ C_rbf_nl.computation_results.u[:, 1] atol = 0.1 broken=true

end



@testset "compute neuralnetODE_type1 eMPC: linear" begin

    ###################################
    ### parameters                  ###
    ###################################

    hmin = 0.2
    h1max = 1.36
    h2max = 1.36
    h3max = 1.30
    h4max = 1.30
    qmin = 0
    qamax = 4
    qbmax = 3.26

    #Constraint definition:
    x_cons = LazySets.Hyperrectangle(
        low = [hmin, hmin, hmin, hmin],
        high = [h1max, h2max, h3max, h4max],
    )
    u_cons = LazySets.Hyperrectangle(low = [qmin, qmin], high = [qamax, qbmax])

    #get the neuralnetODE_type1 model to design the mpc controler
    neuralnetODE_type1_machine = machine("./models_saved/neuralnetODE_type1_train_result.jls")

    #extract best model from the all trained models
    mlj_neuralnetODE_type1 = fitted_params(fitted_params(neuralnetODE_type1_machine).machine).best_model
    f_neuralnetODE_type1 = fitted_params(fitted_params(neuralnetODE_type1_machine).machine).best_fitted_params[1]
    type_neuralnetODE_type1 = mlj_neuralnetODE_type1.builder

    #system definition with Mathematical systems
    QTP_sys_neuralnetODE_type1 = MathematicalSystems.ConstrainedBlackBoxControlDiscreteSystem(
        f_neuralnetODE_type1,
        4,
        2,
        x_cons,
        u_cons,
    )

    #MPC design parameters
    horizon = 5
    sample_time = 5
    terminal_ingredients = false
    max_time = 5
    x = [0.3, 0.3, 0.3, 0.3] .* ones(4, horizon + 1)
    u = [0, 0] .* ones(2, horizon)
    references = ModelPredictiveControl.ReferencesStateInput(x, u)

    ##############################
    ###  neuralnetODE_type1    ###                    
    ##############################

    ### neuralnetODE_type1 L MPC ###
    method = ModelPredictiveControl.LinearProgramming()
    solver = ModelPredictiveControl.OSQP_solver_def()
    C_neuralnetODE_type1_linear = _economic_model_predictive_control_design(
        QTP_sys_neuralnetODE_type1,
        type_neuralnetODE_type1,
        horizon,
        method,
        sample_time,
        references,
        solver = solver,
    )

    initialization = [0.35, 0.35, 0.35, 0.35]

    UpdateInitialization!(C_neuralnetODE_type1_linear, initialization)

    Calculate!(C_neuralnetODE_type1_linear)

    C_neuralnetODE_type1_linear.computation_results.x
    C_neuralnetODE_type1_linear.computation_results.u
       
    #@test
    @test C_neuralnetODE_type1_linear.computation_results.x ≈  x atol = 0.5 
    @test C_neuralnetODE_type1_linear.computation_results.u[:, 1]  ≈ u[:, 1] atol  = 3

end



@testset "compute linear model eMPC: linear" begin

    ###################################
    ### parameters                  ###
    ###################################

    hmin = 0.2
    h1max = 1.36
    h2max = 1.36
    h3max = 1.30
    h4max = 1.30
    qmin = 0
    qamax = 4
    qbmax = 3.26

    #Constraint definition:
    x_cons = LazySets.Hyperrectangle(
        low = [hmin, hmin, hmin, hmin],
        high = [h1max, h2max, h3max, h4max],
    )
    u_cons = LazySets.Hyperrectangle(low = [qmin, qmin], high = [qamax, qbmax])

    #get the linear_regressor model to design the mpc controler
    linear_regressor_machine = machine("./models_saved/linear_regressor_train_result.jls")

    #extract best model from the all trained models
    fitted_params(linear_regressor_machine)

    AB_t = fitted_params(linear_regressor_machine).coefficients
    AB = copy(AB_t')
    A = AB[:, 1:4]
    B = AB[:, 5: end]

    type_linear_regressor = linear_regressor_machine.model

    #system definition with Mathematical systems
    QTP_sys_linear_regressor = MathematicalSystems.ConstrainedLinearControlDiscreteSystem(
        A, 
        B, 
        x_cons,
        u_cons,
    )

    #MPC design parameters
    horizon = 5
    sample_time = 5
    terminal_ingredients = false
    max_time = 5
    x = [0.3, 0.3, 0.3, 0.3] .* ones(4, horizon + 1)
    u = [0, 0] .* ones(2, horizon)
    references = ModelPredictiveControl.ReferencesStateInput(x, u)

    ##############################
    ###  linear_regressor      ###                    
    ##############################

    ### linear_regressor L MPC ###
    method = ModelPredictiveControl.LinearProgramming()
    solver = ModelPredictiveControl.OSQP_solver_def()
    C_linear_regressor_linear = _economic_model_predictive_control_design(
        QTP_sys_linear_regressor,
        type_linear_regressor,
        horizon,
        method,
        sample_time,
        references,
        solver = solver,
    )

    initialization = [0.35, 0.35, 0.35, 0.35]

    UpdateInitialization!(C_linear_regressor_linear, initialization)

    Calculate!(C_linear_regressor_linear)

    C_linear_regressor_linear.computation_results.x
    C_linear_regressor_linear.computation_results.u
       
    #@test
    @test C_linear_regressor_linear.computation_results.x ≈  x atol = 0.5 
    @test C_linear_regressor_linear.computation_results.u[:, 1]  ≈ u[:, 1] atol  = 3

end



end