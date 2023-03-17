# Copyright (c) 2022: Pierre Blaud and contributors
########################################################
# This Source Code Form is subject to the terms of the #
# Mozilla Public License, v. 2.0. If a copy of the MPL #
# was not distributed with this file,  		       #
# You can obtain one at https://mozilla.org/MPL/2.0/.  #
########################################################

module MpcDesignImplementationTest

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

using AutomationLabsModelPredictiveControl

import AutomationLabsModelPredictiveControl: auto_solver_def
import AutomationLabsModelPredictiveControl: _model_predictive_control_design

@testset "design linear Model Predictive Control with fnn model" begin

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
    horizon = 15
    sample_time = 5
    terminal_ingredient = false
    max_time = 5
    x = [0.65, 0.65, 0.65, 0.65] * ones(1, horizon + 1)
    u = [1.2, 1.2] * ones(1, horizon)
    references = AutomationLabsModelPredictiveControl.ReferencesStateInput(x, u)

    ##############################
    ###           Fnn          ###                    
    ##############################

    ### Fnn L MPC ###
    C_fnn_linear = _model_predictive_control_design(
        QTP_sys_fnn,
        horizon,
        sample_time,
        references;
        mpc_programming_type = "linear",
    )

    ### start evaluate FNN L MPC implementation ###
    @test C_fnn_linear.system == QTP_sys_fnn
    @test typeof(C_fnn_linear.tuning.modeler) == JuMP.Model
    @test JuMP.solver_name(C_fnn_linear.tuning.modeler) == "SCIP"
    @test length(JuMP.object_dictionary(C_fnn_linear.tuning.modeler)) == 6
    @test size(JuMP.object_dictionary(C_fnn_linear.tuning.modeler)[:u_reference]) ==
          (QTP_sys_fnn.inputdim, horizon)
    @test size(JuMP.object_dictionary(C_fnn_linear.tuning.modeler)[:u]) ==
          (QTP_sys_fnn.inputdim, horizon)
    @test size(JuMP.object_dictionary(C_fnn_linear.tuning.modeler)[:e_u]) ==
          (QTP_sys_fnn.inputdim, horizon)
    @test size(JuMP.object_dictionary(C_fnn_linear.tuning.modeler)[:x_reference]) ==
          (QTP_sys_fnn.statedim, horizon + 1)
    @test size(JuMP.object_dictionary(C_fnn_linear.tuning.modeler)[:x]) ==
          (QTP_sys_fnn.statedim, horizon + 1)
    @test size(JuMP.object_dictionary(C_fnn_linear.tuning.modeler)[:e_x]) ==
          (QTP_sys_fnn.statedim, horizon + 1)
    @test JuMP.objective_function(C_fnn_linear.tuning.modeler) != 0 #to improve
    @test JuMP.objective_function_type(C_fnn_linear.tuning.modeler) == JuMP.QuadExpr
    @test JuMP.list_of_constraint_types(C_fnn_linear.tuning.modeler) == [
        (AffExpr, MathOptInterface.EqualTo{Float64}),
        (AffExpr, MathOptInterface.LessThan{Float64}),
        (VariableRef, MathOptInterface.EqualTo{Float64}),
    ]
    @test C_fnn_linear.tuning.reference == references
    @test C_fnn_linear.tuning.horizon == horizon
    @test C_fnn_linear.tuning.terminal_ingredient.Xf == "none"
    @test C_fnn_linear.tuning.terminal_ingredient.P != 0
    @test C_fnn_linear.tuning.sample_time == 5.0
    @test C_fnn_linear.tuning.max_time == 30 #to modify
    @test size(C_fnn_linear.initialization) == (QTP_sys_fnn.statedim,)
    @test size(C_fnn_linear.computation_results.x) == (QTP_sys_fnn.statedim, horizon + 1)
    @test size(C_fnn_linear.computation_results.e_x) == (QTP_sys_fnn.statedim, horizon + 1)
    @test size(C_fnn_linear.computation_results.u) == (QTP_sys_fnn.inputdim, horizon)
    @test size(C_fnn_linear.computation_results.e_u) == (QTP_sys_fnn.inputdim, horizon)
    ### end evaluate FNN L MPC implementation ###

end

@testset "design non linear Model Predictive Control with fnn model" begin

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
    horizon = 15
    sample_time = 5
    terminal_ingredient = false
    max_time = 5
    x = [0.65, 0.65, 0.65, 0.65] * ones(1, horizon + 1)
    u = [1.2, 1.2] * ones(1, horizon)
    references = AutomationLabsModelPredictiveControl.ReferencesStateInput(x, u)

    ### Fnn NL MPC ###
    method = AutomationLabsModelPredictiveControl.NonLinearProgramming()
    C_fnn_nl = _model_predictive_control_design(
        QTP_sys_fnn,
        horizon,
        sample_time,
        references;
        mpc_programming_type = "non_linear",
    )

    ### start evaluate FNN NL MPC implementation ###
    @test C_fnn_nl.system == QTP_sys_fnn
    @test typeof(C_fnn_nl.tuning.modeler) == JuMP.Model
    @test JuMP.solver_name(C_fnn_nl.tuning.modeler) == "Ipopt"
    @test length(JuMP.object_dictionary(C_fnn_nl.tuning.modeler)) == 7
    @test size(JuMP.object_dictionary(C_fnn_nl.tuning.modeler)[:u_reference]) ==
          (QTP_sys_fnn.inputdim, horizon)
    @test size(JuMP.object_dictionary(C_fnn_nl.tuning.modeler)[:u]) ==
          (QTP_sys_fnn.inputdim, horizon)
    @test size(JuMP.object_dictionary(C_fnn_nl.tuning.modeler)[:e_u]) ==
          (QTP_sys_fnn.inputdim, horizon)
    @test size(JuMP.object_dictionary(C_fnn_nl.tuning.modeler)[:x_reference]) ==
          (QTP_sys_fnn.statedim, horizon + 1)
    @test size(JuMP.object_dictionary(C_fnn_nl.tuning.modeler)[:x]) ==
          (QTP_sys_fnn.statedim, horizon + 1)
    @test size(JuMP.object_dictionary(C_fnn_nl.tuning.modeler)[:e_x]) ==
          (QTP_sys_fnn.statedim, horizon + 1)
    @test JuMP.objective_function(C_fnn_nl.tuning.modeler) != 0 #to improve
    @test JuMP.objective_function_type(C_fnn_nl.tuning.modeler) == JuMP.QuadExpr
    @test JuMP.list_of_constraint_types(C_fnn_nl.tuning.modeler) == [
        (AffExpr, MathOptInterface.EqualTo{Float64}),
        (AffExpr, MathOptInterface.LessThan{Float64}),
        (VariableRef, MathOptInterface.EqualTo{Float64}),
    ]
    @test C_fnn_nl.tuning.reference == references
    @test C_fnn_nl.tuning.horizon == horizon
    @test C_fnn_nl.tuning.terminal_ingredient.Xf == "none"
    @test C_fnn_nl.tuning.terminal_ingredient.P != 0
    @test C_fnn_nl.tuning.sample_time == 5.0
    @test C_fnn_nl.tuning.max_time == 30 #to modify
    @test size(C_fnn_nl.initialization) == (QTP_sys_fnn.statedim,)
    @test size(C_fnn_nl.computation_results.x) == (QTP_sys_fnn.statedim, horizon + 1)
    @test size(C_fnn_nl.computation_results.e_x) == (QTP_sys_fnn.statedim, horizon + 1)
    @test size(C_fnn_nl.computation_results.u) == (QTP_sys_fnn.inputdim, horizon)
    @test size(C_fnn_nl.computation_results.e_u) == (QTP_sys_fnn.inputdim, horizon)
    ### end evaluate FNN NL MPC implementation ###

end

@testset "design milp Model Predictive Control with fnn model" begin

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
    horizon = 15
    sample_time = 5
    terminal_ingredient = false
    max_time = 5
    x = [0.65, 0.65, 0.65, 0.65] * ones(1, horizon + 1)
    u = [1.2, 1.2] * ones(1, horizon)
    references = AutomationLabsModelPredictiveControl.ReferencesStateInput(x, u)

    ### Fnn MILP MPC ###
    method = AutomationLabsModelPredictiveControl.MixedIntegerLinearProgramming()

    C_fnn_milp = _model_predictive_control_design(
        QTP_sys_fnn,
        horizon,
        sample_time,
        references;
        mpc_programming_type = "mixed_linear",
    )

    ### start evaluate FNN MILP MPC implementation ###
    @test C_fnn_milp.system == QTP_sys_fnn
    @test typeof(C_fnn_milp.tuning.modeler) == JuMP.Model
    @test JuMP.solver_name(C_fnn_milp.tuning.modeler) == "SCIP"
    @test length(JuMP.object_dictionary(C_fnn_milp.tuning.modeler)) == 9
    @test size(JuMP.object_dictionary(C_fnn_milp.tuning.modeler)[:u_reference]) ==
          (QTP_sys_fnn.inputdim, horizon)
    @test size(JuMP.object_dictionary(C_fnn_milp.tuning.modeler)[:u]) ==
          (QTP_sys_fnn.inputdim, horizon)
    @test size(JuMP.object_dictionary(C_fnn_milp.tuning.modeler)[:e_u]) ==
          (QTP_sys_fnn.inputdim, horizon)
    @test size(JuMP.object_dictionary(C_fnn_milp.tuning.modeler)[:x_reference]) ==
          (QTP_sys_fnn.statedim, horizon + 1)
    @test size(JuMP.object_dictionary(C_fnn_milp.tuning.modeler)[:x]) ==
          (QTP_sys_fnn.statedim, horizon + 1)
    @test size(JuMP.object_dictionary(C_fnn_milp.tuning.modeler)[:e_x]) ==
          (QTP_sys_fnn.statedim, horizon + 1)
    #to do @test size(JuMP.object_dictionary(C_fnn_milp.tuning.modeler)[:y])
    #to do @test size(JuMP.object_dictionary(C_fnn_milp.tuning.modeler)[:Bin_1])
    #to do @test size(JuMP.object_dictionary(C_fnn_milp.tuning.modeler)[:Bin_2])
    @test JuMP.objective_function(C_fnn_milp.tuning.modeler) != 0 #to improve
    @test JuMP.objective_function_type(C_fnn_milp.tuning.modeler) == JuMP.QuadExpr
    @test JuMP.list_of_constraint_types(C_fnn_milp.tuning.modeler) == [
        (AffExpr, MathOptInterface.EqualTo{Float64}),
        (AffExpr, MathOptInterface.GreaterThan{Float64}),
        (AffExpr, MathOptInterface.LessThan{Float64}),
        (VariableRef, MathOptInterface.EqualTo{Float64}),
        (VariableRef, MathOptInterface.ZeroOne),
    ]
    @test C_fnn_milp.tuning.reference == references
    @test C_fnn_milp.tuning.horizon == horizon
    @test C_fnn_milp.tuning.terminal_ingredient.Xf == "none"
    @test C_fnn_milp.tuning.terminal_ingredient.P != 0
    @test C_fnn_milp.tuning.sample_time == 5.0
    @test C_fnn_milp.tuning.max_time == 30 #to modify
    @test size(C_fnn_milp.initialization) == (QTP_sys_fnn.statedim,)
    @test size(C_fnn_milp.computation_results.x) == (QTP_sys_fnn.statedim, horizon + 1)
    @test size(C_fnn_milp.computation_results.e_x) == (QTP_sys_fnn.statedim, horizon + 1)
    @test size(C_fnn_milp.computation_results.u) == (QTP_sys_fnn.inputdim, horizon)
    @test size(C_fnn_milp.computation_results.e_u) == (QTP_sys_fnn.inputdim, horizon)

end

@testset "design linear Model Predictive Control with icnn model" begin

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
    horizon = 15
    sample_time = 5
    terminal_ingredient = false
    max_time = 5
    x = [0.65, 0.65, 0.65, 0.65] * ones(1, horizon + 1)
    u = [1.2, 1.2] * ones(1, horizon)
    references = AutomationLabsModelPredictiveControl.ReferencesStateInput(x, u)

    ##############################
    ###           icnn          ###                    
    ##############################

    ### icnn L MPC ###
    method = AutomationLabsModelPredictiveControl.LinearProgramming()

    C_icnn_linear = _model_predictive_control_design(
        QTP_sys_icnn,
        horizon,
        sample_time,
        references;
        mpc_programming_type = "linear",
    )

    ### start evaluate icnn L MPC implementation ###
    @test C_icnn_linear.system == QTP_sys_icnn
    @test typeof(C_icnn_linear.tuning.modeler) == JuMP.Model
    @test JuMP.solver_name(C_icnn_linear.tuning.modeler) == "SCIP"
    @test length(JuMP.object_dictionary(C_icnn_linear.tuning.modeler)) == 6
    @test size(JuMP.object_dictionary(C_icnn_linear.tuning.modeler)[:u_reference]) ==
          (QTP_sys_icnn.inputdim, horizon)
    @test size(JuMP.object_dictionary(C_icnn_linear.tuning.modeler)[:u]) ==
          (QTP_sys_icnn.inputdim, horizon)
    @test size(JuMP.object_dictionary(C_icnn_linear.tuning.modeler)[:e_u]) ==
          (QTP_sys_icnn.inputdim, horizon)
    @test size(JuMP.object_dictionary(C_icnn_linear.tuning.modeler)[:x_reference]) ==
          (QTP_sys_icnn.statedim, horizon + 1)
    @test size(JuMP.object_dictionary(C_icnn_linear.tuning.modeler)[:x]) ==
          (QTP_sys_icnn.statedim, horizon + 1)
    @test size(JuMP.object_dictionary(C_icnn_linear.tuning.modeler)[:e_x]) ==
          (QTP_sys_icnn.statedim, horizon + 1)
    @test JuMP.objective_function(C_icnn_linear.tuning.modeler) != 0 #to improve
    @test JuMP.objective_function_type(C_icnn_linear.tuning.modeler) == JuMP.QuadExpr
    @test JuMP.list_of_constraint_types(C_icnn_linear.tuning.modeler) == [
        (AffExpr, MathOptInterface.EqualTo{Float64}),
        (AffExpr, MathOptInterface.LessThan{Float64}),
        (VariableRef, MathOptInterface.EqualTo{Float64}),
    ]
    @test C_icnn_linear.tuning.reference == references
    @test C_icnn_linear.tuning.horizon == horizon
    @test C_icnn_linear.tuning.terminal_ingredient.Xf == "none"
    @test C_icnn_linear.tuning.terminal_ingredient.P != 0
    @test C_icnn_linear.tuning.sample_time == 5.0
    @test C_icnn_linear.tuning.max_time == 30 #to modify
    @test size(C_icnn_linear.initialization) == (QTP_sys_icnn.statedim,)
    @test size(C_icnn_linear.computation_results.x) == (QTP_sys_icnn.statedim, horizon + 1)
    @test size(C_icnn_linear.computation_results.e_x) ==
          (QTP_sys_icnn.statedim, horizon + 1)
    @test size(C_icnn_linear.computation_results.u) == (QTP_sys_icnn.inputdim, horizon)
    @test size(C_icnn_linear.computation_results.e_u) == (QTP_sys_icnn.inputdim, horizon)
    ### end evaluate icnn L MPC implementation ###

end

@testset "design non linear Model Predictive Control with icnn model" begin

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
    horizon = 15
    sample_time = 5
    terminal_ingredient = false
    max_time = 5
    x = [0.65, 0.65, 0.65, 0.65] * ones(1, horizon + 1)
    u = [1.2, 1.2] * ones(1, horizon)
    references = AutomationLabsModelPredictiveControl.ReferencesStateInput(x, u)

    ### icnn NL MPC ###
    method = AutomationLabsModelPredictiveControl.NonLinearProgramming()

    C_icnn_nl = _model_predictive_control_design(
        QTP_sys_icnn,
        horizon,
        sample_time,
        references;
        mpc_programming_type = "non_linear",
    )

    ### start evaluate icnn NL MPC implementation ###
    @test C_icnn_nl.system == QTP_sys_icnn
    @test typeof(C_icnn_nl.tuning.modeler) == JuMP.Model
    @test JuMP.solver_name(C_icnn_nl.tuning.modeler) == "Ipopt"
    @test length(JuMP.object_dictionary(C_icnn_nl.tuning.modeler)) == 7
    @test size(JuMP.object_dictionary(C_icnn_nl.tuning.modeler)[:u_reference]) ==
          (QTP_sys_icnn.inputdim, horizon)
    @test size(JuMP.object_dictionary(C_icnn_nl.tuning.modeler)[:u]) ==
          (QTP_sys_icnn.inputdim, horizon)
    @test size(JuMP.object_dictionary(C_icnn_nl.tuning.modeler)[:e_u]) ==
          (QTP_sys_icnn.inputdim, horizon)
    @test size(JuMP.object_dictionary(C_icnn_nl.tuning.modeler)[:x_reference]) ==
          (QTP_sys_icnn.statedim, horizon + 1)
    @test size(JuMP.object_dictionary(C_icnn_nl.tuning.modeler)[:x]) ==
          (QTP_sys_icnn.statedim, horizon + 1)
    @test size(JuMP.object_dictionary(C_icnn_nl.tuning.modeler)[:e_x]) ==
          (QTP_sys_icnn.statedim, horizon + 1)
    @test JuMP.objective_function(C_icnn_nl.tuning.modeler) != 0 #to improve
    @test JuMP.objective_function_type(C_icnn_nl.tuning.modeler) == JuMP.QuadExpr
    @test JuMP.list_of_constraint_types(C_icnn_nl.tuning.modeler) == [
        (AffExpr, MathOptInterface.EqualTo{Float64}),
        (AffExpr, MathOptInterface.LessThan{Float64}),
        (VariableRef, MathOptInterface.EqualTo{Float64}),
    ]
    @test C_icnn_nl.tuning.reference == references
    @test C_icnn_nl.tuning.horizon == horizon
    @test C_icnn_nl.tuning.terminal_ingredient.Xf == "none"
    @test C_icnn_nl.tuning.terminal_ingredient.P != 0
    @test C_icnn_nl.tuning.sample_time == 5.0
    @test C_icnn_nl.tuning.max_time == 30 #to modify
    @test size(C_icnn_nl.initialization) == (QTP_sys_icnn.statedim,)
    @test size(C_icnn_nl.computation_results.x) == (QTP_sys_icnn.statedim, horizon + 1)
    @test size(C_icnn_nl.computation_results.e_x) == (QTP_sys_icnn.statedim, horizon + 1)
    @test size(C_icnn_nl.computation_results.u) == (QTP_sys_icnn.inputdim, horizon)
    @test size(C_icnn_nl.computation_results.e_u) == (QTP_sys_icnn.inputdim, horizon)
    ### end evaluate icnn NL MPC implementation ###

end

@testset "design milp Model Predictive Control with icnn model" begin

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
    horizon = 15
    sample_time = 5
    terminal_ingredient = false
    max_time = 5
    x = [0.65, 0.65, 0.65, 0.65] * ones(1, horizon + 1)
    u = [1.2, 1.2] * ones(1, horizon)
    references = AutomationLabsModelPredictiveControl.ReferencesStateInput(x, u)

    ### icnn MILP MPC ###
    method = AutomationLabsModelPredictiveControl.MixedIntegerLinearProgramming()

    C_icnn_milp = _model_predictive_control_design(
        QTP_sys_icnn,
        horizon,
        sample_time,
        references;
        mpc_programming_type = "mixed_linear",
    )

    ### start evaluate icnn MILP MPC implementation ###
    @test C_icnn_milp.system == QTP_sys_icnn
    @test typeof(C_icnn_milp.tuning.modeler) == JuMP.Model
    @test JuMP.solver_name(C_icnn_milp.tuning.modeler) == "SCIP"
    @test length(JuMP.object_dictionary(C_icnn_milp.tuning.modeler)) == 9
    @test size(JuMP.object_dictionary(C_icnn_milp.tuning.modeler)[:u_reference]) ==
          (QTP_sys_icnn.inputdim, horizon)
    @test size(JuMP.object_dictionary(C_icnn_milp.tuning.modeler)[:u]) ==
          (QTP_sys_icnn.inputdim, horizon)
    @test size(JuMP.object_dictionary(C_icnn_milp.tuning.modeler)[:e_u]) ==
          (QTP_sys_icnn.inputdim, horizon)
    @test size(JuMP.object_dictionary(C_icnn_milp.tuning.modeler)[:x_reference]) ==
          (QTP_sys_icnn.statedim, horizon + 1)
    @test size(JuMP.object_dictionary(C_icnn_milp.tuning.modeler)[:x]) ==
          (QTP_sys_icnn.statedim, horizon + 1)
    @test size(JuMP.object_dictionary(C_icnn_milp.tuning.modeler)[:e_x]) ==
          (QTP_sys_icnn.statedim, horizon + 1)
    #to do @test size(JuMP.object_dictionary(C_icnn_milp.tuning.modeler)[:y])
    #to do @test size(JuMP.object_dictionary(C_icnn_milp.tuning.modeler)[:Bin_1])
    #to do @test size(JuMP.object_dictionary(C_icnn_milp.tuning.modeler)[:Bin_2])
    @test JuMP.objective_function(C_icnn_milp.tuning.modeler) != 0 #to improve
    @test JuMP.objective_function_type(C_icnn_milp.tuning.modeler) == JuMP.QuadExpr
    @test JuMP.list_of_constraint_types(C_icnn_milp.tuning.modeler) == [
        (AffExpr, MathOptInterface.EqualTo{Float64}),
        (AffExpr, MathOptInterface.GreaterThan{Float64}),
        (AffExpr, MathOptInterface.LessThan{Float64}),
        (VariableRef, MathOptInterface.EqualTo{Float64}),
        (VariableRef, MathOptInterface.ZeroOne),
    ]
    @test C_icnn_milp.tuning.reference == references
    @test C_icnn_milp.tuning.horizon == horizon
    @test C_icnn_milp.tuning.terminal_ingredient.Xf == "none"
    @test C_icnn_milp.tuning.terminal_ingredient.P != 0
    @test C_icnn_milp.tuning.sample_time == 5.0
    @test C_icnn_milp.tuning.max_time == 30 #to modify
    @test size(C_icnn_milp.initialization) == (QTP_sys_icnn.statedim,)
    @test size(C_icnn_milp.computation_results.x) == (QTP_sys_icnn.statedim, horizon + 1)
    @test size(C_icnn_milp.computation_results.e_x) == (QTP_sys_icnn.statedim, horizon + 1)
    @test size(C_icnn_milp.computation_results.u) == (QTP_sys_icnn.inputdim, horizon)
    @test size(C_icnn_milp.computation_results.e_u) == (QTP_sys_icnn.inputdim, horizon)

end

@testset "design linear Model Predictive Control with ResNet model" begin

    ##############################
    ###           ResNet       ###                    
    ##############################

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
    horizon = 15
    sample_time = 5
    terminal_ingredient = false
    max_time = 5
    x = [0.65, 0.65, 0.65, 0.65] .* ones(4, horizon + 1)
    u = [1.2, 1.2] .* ones(2, horizon)
    references = AutomationLabsModelPredictiveControl.ReferencesStateInput(x, u)

    ### ResNet L MPC ###
    method = AutomationLabsModelPredictiveControl.LinearProgramming()

    C_resnet_linear = _model_predictive_control_design(
        QTP_sys_resnet,
        horizon,
        sample_time,
        references;
        mpc_programming_type = "linear",
    )

    ### start evaluate resnet L MPC implementation ###
    @test C_resnet_linear.system == QTP_sys_resnet
    @test typeof(C_resnet_linear.tuning.modeler) == JuMP.Model
    @test JuMP.solver_name(C_resnet_linear.tuning.modeler) == "SCIP"
    @test length(JuMP.object_dictionary(C_resnet_linear.tuning.modeler)) == 6
    @test size(JuMP.object_dictionary(C_resnet_linear.tuning.modeler)[:u_reference]) ==
          (QTP_sys_resnet.inputdim, horizon)
    @test size(JuMP.object_dictionary(C_resnet_linear.tuning.modeler)[:u]) ==
          (QTP_sys_resnet.inputdim, horizon)
    @test size(JuMP.object_dictionary(C_resnet_linear.tuning.modeler)[:e_u]) ==
          (QTP_sys_resnet.inputdim, horizon)
    @test size(JuMP.object_dictionary(C_resnet_linear.tuning.modeler)[:x_reference]) ==
          (QTP_sys_resnet.statedim, horizon + 1)
    @test size(JuMP.object_dictionary(C_resnet_linear.tuning.modeler)[:x]) ==
          (QTP_sys_resnet.statedim, horizon + 1)
    @test size(JuMP.object_dictionary(C_resnet_linear.tuning.modeler)[:e_x]) ==
          (QTP_sys_resnet.statedim, horizon + 1)
    @test JuMP.objective_function(C_resnet_linear.tuning.modeler) != 0 #to improve
    @test JuMP.objective_function_type(C_resnet_linear.tuning.modeler) == JuMP.QuadExpr
    @test JuMP.list_of_constraint_types(C_resnet_linear.tuning.modeler) == [
        (AffExpr, MathOptInterface.EqualTo{Float64}),
        (AffExpr, MathOptInterface.LessThan{Float64}),
        (VariableRef, MathOptInterface.EqualTo{Float64}),
    ]
    @test C_resnet_linear.tuning.reference == references
    @test C_resnet_linear.tuning.horizon == horizon
    @test C_resnet_linear.tuning.terminal_ingredient.Xf == "none"
    @test C_resnet_linear.tuning.terminal_ingredient.P != 0
    @test C_resnet_linear.tuning.sample_time == 5.0
    @test C_resnet_linear.tuning.max_time == 30 #to modify
    @test size(C_resnet_linear.initialization) == (QTP_sys_resnet.statedim,)
    @test size(C_resnet_linear.computation_results.x) ==
          (QTP_sys_resnet.statedim, horizon + 1)
    @test size(C_resnet_linear.computation_results.e_x) ==
          (QTP_sys_resnet.statedim, horizon + 1)
    @test size(C_resnet_linear.computation_results.u) == (QTP_sys_resnet.inputdim, horizon)
    @test size(C_resnet_linear.computation_results.e_u) ==
          (QTP_sys_resnet.inputdim, horizon)
    ### end evaluate resnet L MPC implementation ###
end

@testset "design non linear Model Predictive Control with ResNet model" begin

    ##############################
    ###           ResNet       ###                    
    ##############################

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
    horizon = 15
    sample_time = 5
    terminal_ingredient = false
    max_time = 5
    x = [0.65, 0.65, 0.65, 0.65] .* ones(4, horizon + 1)
    u = [1.2, 1.2] .* ones(2, horizon)
    references = AutomationLabsModelPredictiveControl.ReferencesStateInput(x, u)

    ### ResNet N L MPC ###
    method = AutomationLabsModelPredictiveControl.NonLinearProgramming()

    C_resnet_nl = _model_predictive_control_design(
        QTP_sys_resnet,
        horizon,
        sample_time,
        references;
        mpc_programming_type = "non_linear",
    )

    ### start evaluate ResNet NL MPC implementation ###
    @test C_resnet_nl.system == QTP_sys_resnet
    @test typeof(C_resnet_nl.tuning.modeler) == JuMP.Model
    @test JuMP.solver_name(C_resnet_nl.tuning.modeler) == "Ipopt"
    @test length(JuMP.object_dictionary(C_resnet_nl.tuning.modeler)) == 7
    @test size(JuMP.object_dictionary(C_resnet_nl.tuning.modeler)[:u_reference]) ==
          (QTP_sys_resnet.inputdim, horizon)
    @test size(JuMP.object_dictionary(C_resnet_nl.tuning.modeler)[:u]) ==
          (QTP_sys_resnet.inputdim, horizon)
    @test size(JuMP.object_dictionary(C_resnet_nl.tuning.modeler)[:e_u]) ==
          (QTP_sys_resnet.inputdim, horizon)
    @test size(JuMP.object_dictionary(C_resnet_nl.tuning.modeler)[:x_reference]) ==
          (QTP_sys_resnet.statedim, horizon + 1)
    @test size(JuMP.object_dictionary(C_resnet_nl.tuning.modeler)[:x]) ==
          (QTP_sys_resnet.statedim, horizon + 1)
    @test size(JuMP.object_dictionary(C_resnet_nl.tuning.modeler)[:e_x]) ==
          (QTP_sys_resnet.statedim, horizon + 1)
    @test JuMP.objective_function(C_resnet_nl.tuning.modeler) != 0 #to improve
    @test JuMP.objective_function_type(C_resnet_nl.tuning.modeler) == JuMP.QuadExpr
    @test JuMP.list_of_constraint_types(C_resnet_nl.tuning.modeler) == [
        (AffExpr, MathOptInterface.EqualTo{Float64}),
        (AffExpr, MathOptInterface.LessThan{Float64}),
        (VariableRef, MathOptInterface.EqualTo{Float64}),
    ]
    @test C_resnet_nl.tuning.reference == references
    @test C_resnet_nl.tuning.horizon == horizon
    @test C_resnet_nl.tuning.terminal_ingredient.Xf == "none"
    @test C_resnet_nl.tuning.terminal_ingredient.P != 0
    @test C_resnet_nl.tuning.sample_time == 5.0
    @test C_resnet_nl.tuning.max_time == 30 #to modify
    @test size(C_resnet_nl.initialization) == (QTP_sys_resnet.statedim,)
    @test size(C_resnet_nl.computation_results.x) == (QTP_sys_resnet.statedim, horizon + 1)
    @test size(C_resnet_nl.computation_results.e_x) ==
          (QTP_sys_resnet.statedim, horizon + 1)
    @test size(C_resnet_nl.computation_results.u) == (QTP_sys_resnet.inputdim, horizon)
    @test size(C_resnet_nl.computation_results.e_u) == (QTP_sys_resnet.inputdim, horizon)
    ### end evaluate resnet NL MPC implementation ###
end

@testset "design milp Model Predictive Control with ResNet model" begin

    ##############################
    ###           ResNet       ###                    
    ##############################

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
    horizon = 15
    sample_time = 5
    terminal_ingredient = false
    max_time = 5
    x = [0.65, 0.65, 0.65, 0.65] .* ones(4, horizon + 1)
    u = [1.2, 1.2] .* ones(2, horizon)
    references = AutomationLabsModelPredictiveControl.ReferencesStateInput(x, u)

    ### ResNet MILP MPC ###
    method = AutomationLabsModelPredictiveControl.MixedIntegerLinearProgramming()

    C_resnet_milp = _model_predictive_control_design(
        QTP_sys_resnet,
        horizon,
        sample_time,
        references;
        mpc_programming_type = "mixed_linear",
    )

    ### start evaluate resnet MILP MPC implementation ###
    @test C_resnet_milp.system == QTP_sys_resnet
    @test typeof(C_resnet_milp.tuning.modeler) == JuMP.Model
    @test JuMP.solver_name(C_resnet_milp.tuning.modeler) == "SCIP"
    @test length(JuMP.object_dictionary(C_resnet_milp.tuning.modeler)) == 10
    @test size(JuMP.object_dictionary(C_resnet_milp.tuning.modeler)[:u_reference]) ==
          (QTP_sys_resnet.inputdim, horizon)
    @test size(JuMP.object_dictionary(C_resnet_milp.tuning.modeler)[:u]) ==
          (QTP_sys_resnet.inputdim, horizon)
    @test size(JuMP.object_dictionary(C_resnet_milp.tuning.modeler)[:e_u]) ==
          (QTP_sys_resnet.inputdim, horizon)
    @test size(JuMP.object_dictionary(C_resnet_milp.tuning.modeler)[:x_reference]) ==
          (QTP_sys_resnet.statedim, horizon + 1)
    @test size(JuMP.object_dictionary(C_resnet_milp.tuning.modeler)[:x]) ==
          (QTP_sys_resnet.statedim, horizon + 1)
    @test size(JuMP.object_dictionary(C_resnet_milp.tuning.modeler)[:e_x]) ==
          (QTP_sys_resnet.statedim, horizon + 1)
    #to do @test size(JuMP.object_dictionary(C_resnet_milp.tuning.modeler)[:y])
    #to do @test size(JuMP.object_dictionary(C_resnet_milp.tuning.modeler)[:Bin_1])
    #to do @test size(JuMP.object_dictionary(C_resnet_milp.tuning.modeler)[:Bin_2])
    @test JuMP.objective_function(C_resnet_milp.tuning.modeler) != 0 #to improve
    @test JuMP.objective_function_type(C_resnet_milp.tuning.modeler) == JuMP.QuadExpr
    @test JuMP.list_of_constraint_types(C_resnet_milp.tuning.modeler) == [
        (AffExpr, MathOptInterface.EqualTo{Float64}),
        (AffExpr, MathOptInterface.GreaterThan{Float64}),
        (AffExpr, MathOptInterface.LessThan{Float64}),
        (VariableRef, MathOptInterface.EqualTo{Float64}),
        (VariableRef, MathOptInterface.ZeroOne),
    ]
    @test C_resnet_milp.tuning.reference == references
    @test C_resnet_milp.tuning.horizon == horizon
    @test C_resnet_milp.tuning.terminal_ingredient.Xf == "none"
    @test C_resnet_milp.tuning.terminal_ingredient.P != 0
    @test C_resnet_milp.tuning.sample_time == 5.0
    @test C_resnet_milp.tuning.max_time == 30 #to modify
    @test size(C_resnet_milp.initialization) == (QTP_sys_resnet.statedim,)
    @test size(C_resnet_milp.computation_results.x) ==
          (QTP_sys_resnet.statedim, horizon + 1)
    @test size(C_resnet_milp.computation_results.e_x) ==
          (QTP_sys_resnet.statedim, horizon + 1)
    @test size(C_resnet_milp.computation_results.u) == (QTP_sys_resnet.inputdim, horizon)
    @test size(C_resnet_milp.computation_results.e_u) == (QTP_sys_resnet.inputdim, horizon)
    ### end evaluate resnet MILP MPC implementation ###

end

@testset "design linear Model Predictive Control with DenseNet model" begin

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
    horizon = 15
    sample_time = 5
    terminal_ingredient = false
    max_time = 5
    x = [0.65, 0.65, 0.65, 0.65] * ones(1, horizon + 1)
    u = [1.2, 1.2] * ones(1, horizon)
    references = AutomationLabsModelPredictiveControl.ReferencesStateInput(x, u)

    #DenseNet design linear
    method = AutomationLabsModelPredictiveControl.LinearProgramming()

    C_densenet_linear = _model_predictive_control_design(
        QTP_sys_densenet,
        horizon,
        sample_time,
        references;
        mpc_programming_type = "linear",
    )

    ### start evaluate densenet L MPC implementation ###
    @test C_densenet_linear.system == QTP_sys_densenet
    @test typeof(C_densenet_linear.tuning.modeler) == JuMP.Model
    @test JuMP.solver_name(C_densenet_linear.tuning.modeler) == "SCIP"
    @test length(JuMP.object_dictionary(C_densenet_linear.tuning.modeler)) == 6
    @test size(JuMP.object_dictionary(C_densenet_linear.tuning.modeler)[:u_reference]) ==
          (QTP_sys_densenet.inputdim, horizon)
    @test size(JuMP.object_dictionary(C_densenet_linear.tuning.modeler)[:u]) ==
          (QTP_sys_densenet.inputdim, horizon)
    @test size(JuMP.object_dictionary(C_densenet_linear.tuning.modeler)[:e_u]) ==
          (QTP_sys_densenet.inputdim, horizon)
    @test size(JuMP.object_dictionary(C_densenet_linear.tuning.modeler)[:x_reference]) ==
          (QTP_sys_densenet.statedim, horizon + 1)
    @test size(JuMP.object_dictionary(C_densenet_linear.tuning.modeler)[:x]) ==
          (QTP_sys_densenet.statedim, horizon + 1)
    @test size(JuMP.object_dictionary(C_densenet_linear.tuning.modeler)[:e_x]) ==
          (QTP_sys_densenet.statedim, horizon + 1)
    @test JuMP.objective_function(C_densenet_linear.tuning.modeler) != 0 #to improve
    @test JuMP.objective_function_type(C_densenet_linear.tuning.modeler) == JuMP.QuadExpr
    @test JuMP.list_of_constraint_types(C_densenet_linear.tuning.modeler) == [
        (AffExpr, MathOptInterface.EqualTo{Float64}),
        (AffExpr, MathOptInterface.LessThan{Float64}),
        (VariableRef, MathOptInterface.EqualTo{Float64}),
    ]
    @test C_densenet_linear.tuning.reference == references
    @test C_densenet_linear.tuning.horizon == horizon
    @test C_densenet_linear.tuning.terminal_ingredient.Xf == "none"
    @test C_densenet_linear.tuning.terminal_ingredient.P != 0
    @test C_densenet_linear.tuning.sample_time == 5.0
    @test C_densenet_linear.tuning.max_time == 30 #to modify
    @test size(C_densenet_linear.initialization) == (QTP_sys_densenet.statedim,)
    @test size(C_densenet_linear.computation_results.x) ==
          (QTP_sys_densenet.statedim, horizon + 1)
    @test size(C_densenet_linear.computation_results.e_x) ==
          (QTP_sys_densenet.statedim, horizon + 1)
    @test size(C_densenet_linear.computation_results.u) ==
          (QTP_sys_densenet.inputdim, horizon)
    @test size(C_densenet_linear.computation_results.e_u) ==
          (QTP_sys_densenet.inputdim, horizon)
    ### end evaluate densenet L MPC implementation ###
end

@testset "design non linear Model Predictive Control with DenseNet model" begin

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
    horizon = 15
    sample_time = 5
    terminal_ingredient = false
    max_time = 5
    x = [0.65, 0.65, 0.65, 0.65] * ones(1, horizon + 1)
    u = [1.2, 1.2] * ones(1, horizon)
    references = AutomationLabsModelPredictiveControl.ReferencesStateInput(x, u)

    #DenseNet design non-linear
    method = AutomationLabsModelPredictiveControl.NonLinearProgramming()

    C_densenet_nl = _model_predictive_control_design(
        QTP_sys_densenet,
        horizon,
        sample_time,
        references;
        mpc_programming_type = "non_linear",
    )

    ### start evaluate densenet NL MPC implementation ###
    @test C_densenet_nl.system == QTP_sys_densenet
    @test typeof(C_densenet_nl.tuning.modeler) == JuMP.Model
    @test JuMP.solver_name(C_densenet_nl.tuning.modeler) == "Ipopt"
    @test length(JuMP.object_dictionary(C_densenet_nl.tuning.modeler)) == 10
    @test size(JuMP.object_dictionary(C_densenet_nl.tuning.modeler)[:u_reference]) ==
          (QTP_sys_densenet.inputdim, horizon)
    @test size(JuMP.object_dictionary(C_densenet_nl.tuning.modeler)[:u]) ==
          (QTP_sys_densenet.inputdim, horizon)
    @test size(JuMP.object_dictionary(C_densenet_nl.tuning.modeler)[:e_u]) ==
          (QTP_sys_densenet.inputdim, horizon)
    @test size(JuMP.object_dictionary(C_densenet_nl.tuning.modeler)[:x_reference]) ==
          (QTP_sys_densenet.statedim, horizon + 1)
    @test size(JuMP.object_dictionary(C_densenet_nl.tuning.modeler)[:x]) ==
          (QTP_sys_densenet.statedim, horizon + 1)
    @test size(JuMP.object_dictionary(C_densenet_nl.tuning.modeler)[:e_x]) ==
          (QTP_sys_densenet.statedim, horizon + 1)
    @test JuMP.objective_function(C_densenet_nl.tuning.modeler) != 0 #to improve
    @test JuMP.objective_function_type(C_densenet_nl.tuning.modeler) == JuMP.QuadExpr
    @test JuMP.list_of_constraint_types(C_densenet_nl.tuning.modeler) == [
        (AffExpr, MathOptInterface.EqualTo{Float64}),
        (AffExpr, MathOptInterface.LessThan{Float64}),
        (VariableRef, MathOptInterface.EqualTo{Float64}),
    ]
    @test C_densenet_nl.tuning.reference == references
    @test C_densenet_nl.tuning.horizon == horizon
    @test C_densenet_nl.tuning.terminal_ingredient.Xf == "none"
    @test C_densenet_nl.tuning.terminal_ingredient.P != 0
    @test C_densenet_nl.tuning.sample_time == 5.0
    @test C_densenet_nl.tuning.max_time == 30 #to modify
    @test size(C_densenet_nl.initialization) == (QTP_sys_densenet.statedim,)
    @test size(C_densenet_nl.computation_results.x) ==
          (QTP_sys_densenet.statedim, horizon + 1)
    @test size(C_densenet_nl.computation_results.e_x) ==
          (QTP_sys_densenet.statedim, horizon + 1)
    @test size(C_densenet_nl.computation_results.u) == (QTP_sys_densenet.inputdim, horizon)
    @test size(C_densenet_nl.computation_results.e_u) ==
          (QTP_sys_densenet.inputdim, horizon)
    ### end evaluate densenet NL MPC implementation ###   
end

@testset "design milp Model Predictive Control with DenseNet model" begin

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
    horizon = 15
    sample_time = 5
    terminal_ingredient = false
    max_time = 5
    x = [0.65, 0.65, 0.65, 0.65] * ones(1, horizon + 1)
    u = [1.2, 1.2] * ones(1, horizon)
    references = AutomationLabsModelPredictiveControl.ReferencesStateInput(x, u)

    #DenseNet design MILP
    method = AutomationLabsModelPredictiveControl.MixedIntegerLinearProgramming()

    C_densenet_milp = _model_predictive_control_design(
        QTP_sys_densenet,
        horizon,
        sample_time,
        references;
        mpc_programming_type = "mixed_linear",
    )

    ### start evaluate densenet milp MPC implementation ###
    @test C_densenet_milp.system == QTP_sys_densenet
    @test typeof(C_densenet_milp.tuning.modeler) == JuMP.Model
    @test JuMP.solver_name(C_densenet_milp.tuning.modeler) == "SCIP"
    @test length(JuMP.object_dictionary(C_densenet_milp.tuning.modeler)) == 13
    @test size(JuMP.object_dictionary(C_densenet_milp.tuning.modeler)[:u_reference]) ==
          (QTP_sys_densenet.inputdim, horizon)
    @test size(JuMP.object_dictionary(C_densenet_milp.tuning.modeler)[:u]) ==
          (QTP_sys_densenet.inputdim, horizon)
    @test size(JuMP.object_dictionary(C_densenet_milp.tuning.modeler)[:e_u]) ==
          (QTP_sys_densenet.inputdim, horizon)
    @test size(JuMP.object_dictionary(C_densenet_milp.tuning.modeler)[:x_reference]) ==
          (QTP_sys_densenet.statedim, horizon + 1)
    @test size(JuMP.object_dictionary(C_densenet_milp.tuning.modeler)[:x]) ==
          (QTP_sys_densenet.statedim, horizon + 1)
    @test size(JuMP.object_dictionary(C_densenet_milp.tuning.modeler)[:e_x]) ==
          (QTP_sys_densenet.statedim, horizon + 1)
    @test JuMP.objective_function(C_densenet_milp.tuning.modeler) != 0 #to improve
    @test JuMP.objective_function_type(C_densenet_milp.tuning.modeler) == JuMP.QuadExpr
    @test JuMP.list_of_constraint_types(C_densenet_milp.tuning.modeler) == [
        (AffExpr, MathOptInterface.EqualTo{Float64}),
        (AffExpr, MathOptInterface.GreaterThan{Float64}),
        (AffExpr, MathOptInterface.LessThan{Float64}),
        (VariableRef, MathOptInterface.EqualTo{Float64}),
        (VariableRef, MathOptInterface.ZeroOne),
    ]
    @test C_densenet_milp.tuning.reference == references
    @test C_densenet_milp.tuning.horizon == horizon
    @test C_densenet_milp.tuning.terminal_ingredient.Xf == "none"
    @test C_densenet_milp.tuning.terminal_ingredient.P != 0
    @test C_densenet_milp.tuning.sample_time == 5.0
    @test C_densenet_milp.tuning.max_time == 30 #to modify
    @test size(C_densenet_milp.initialization) == (QTP_sys_densenet.statedim,)
    @test size(C_densenet_milp.computation_results.x) ==
          (QTP_sys_densenet.statedim, horizon + 1)
    @test size(C_densenet_milp.computation_results.e_x) ==
          (QTP_sys_densenet.statedim, horizon + 1)
    @test size(C_densenet_milp.computation_results.u) ==
          (QTP_sys_densenet.inputdim, horizon)
    @test size(C_densenet_milp.computation_results.e_u) ==
          (QTP_sys_densenet.inputdim, horizon)
    ### end evaluate densenet milp MPC implementation ###   

end



@testset "design linear Model Predictive Control with rbf model" begin

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
    horizon = 15
    sample_time = 5
    terminal_ingredient = false
    max_time = 5
    x = [0.65, 0.65, 0.65, 0.65] * ones(1, horizon + 1)
    u = [1.2, 1.2] * ones(1, horizon)
    references = AutomationLabsModelPredictiveControl.ReferencesStateInput(x, u)

    ##############################
    ###           rbf          ###                    
    ##############################

    ### rbf L MPC ###
    method = AutomationLabsModelPredictiveControl.LinearProgramming()

    C_rbf_linear = _model_predictive_control_design(
        QTP_sys_rbf,
        horizon,
        sample_time,
        references;
        mpc_programming_type = "linear",
    )

    ### start evaluate rbf L MPC implementation ###
    @test C_rbf_linear.system == QTP_sys_rbf
    @test typeof(C_rbf_linear.tuning.modeler) == JuMP.Model
    @test JuMP.solver_name(C_rbf_linear.tuning.modeler) == "SCIP"
    @test length(JuMP.object_dictionary(C_rbf_linear.tuning.modeler)) == 6
    @test size(JuMP.object_dictionary(C_rbf_linear.tuning.modeler)[:u_reference]) ==
          (QTP_sys_rbf.inputdim, horizon)
    @test size(JuMP.object_dictionary(C_rbf_linear.tuning.modeler)[:u]) ==
          (QTP_sys_rbf.inputdim, horizon)
    @test size(JuMP.object_dictionary(C_rbf_linear.tuning.modeler)[:e_u]) ==
          (QTP_sys_rbf.inputdim, horizon)
    @test size(JuMP.object_dictionary(C_rbf_linear.tuning.modeler)[:x_reference]) ==
          (QTP_sys_rbf.statedim, horizon + 1)
    @test size(JuMP.object_dictionary(C_rbf_linear.tuning.modeler)[:x]) ==
          (QTP_sys_rbf.statedim, horizon + 1)
    @test size(JuMP.object_dictionary(C_rbf_linear.tuning.modeler)[:e_x]) ==
          (QTP_sys_rbf.statedim, horizon + 1)
    @test JuMP.objective_function(C_rbf_linear.tuning.modeler) != 0 #to improve
    @test JuMP.objective_function_type(C_rbf_linear.tuning.modeler) == JuMP.QuadExpr
    @test JuMP.list_of_constraint_types(C_rbf_linear.tuning.modeler) == [
        (AffExpr, MathOptInterface.EqualTo{Float64}),
        (AffExpr, MathOptInterface.LessThan{Float64}),
        (VariableRef, MathOptInterface.EqualTo{Float64}),
    ]
    @test C_rbf_linear.tuning.reference == references
    @test C_rbf_linear.tuning.horizon == horizon
    @test C_rbf_linear.tuning.terminal_ingredient.Xf == "none"
    @test C_rbf_linear.tuning.terminal_ingredient.P != 0
    @test C_rbf_linear.tuning.sample_time == 5.0
    @test C_rbf_linear.tuning.max_time == 30 #to modify
    @test size(C_rbf_linear.initialization) == (QTP_sys_rbf.statedim,)
    @test size(C_rbf_linear.computation_results.x) == (QTP_sys_rbf.statedim, horizon + 1)
    @test size(C_rbf_linear.computation_results.e_x) == (QTP_sys_rbf.statedim, horizon + 1)
    @test size(C_rbf_linear.computation_results.u) == (QTP_sys_rbf.inputdim, horizon)
    @test size(C_rbf_linear.computation_results.e_u) == (QTP_sys_rbf.inputdim, horizon)
    ### end evaluate rbf L MPC implementation ###

end

@testset "design non linear Model Predictive Control with rbf model" begin

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
    horizon = 15
    sample_time = 5
    terminal_ingredient = false
    max_time = 5
    x = [0.65, 0.65, 0.65, 0.65] * ones(1, horizon + 1)
    u = [1.2, 1.2] * ones(1, horizon)
    references = AutomationLabsModelPredictiveControl.ReferencesStateInput(x, u)

    ### rbf NL MPC ###
    method = AutomationLabsModelPredictiveControl.NonLinearProgramming()

    C_rbf_nl = _model_predictive_control_design(
        QTP_sys_rbf,
        horizon,
        sample_time,
        references;
        mpc_programming_type = "non_linear",
    )

    ### start evaluate rbf NL MPC implementation ###
    @test C_rbf_nl.system == QTP_sys_rbf
    @test typeof(C_rbf_nl.tuning.modeler) == JuMP.Model
    @test JuMP.solver_name(C_rbf_nl.tuning.modeler) == "Ipopt"
    @test length(JuMP.object_dictionary(C_rbf_nl.tuning.modeler)) == 7
    @test size(JuMP.object_dictionary(C_rbf_nl.tuning.modeler)[:u_reference]) ==
          (QTP_sys_rbf.inputdim, horizon)
    @test size(JuMP.object_dictionary(C_rbf_nl.tuning.modeler)[:u]) ==
          (QTP_sys_rbf.inputdim, horizon)
    @test size(JuMP.object_dictionary(C_rbf_nl.tuning.modeler)[:e_u]) ==
          (QTP_sys_rbf.inputdim, horizon)
    @test size(JuMP.object_dictionary(C_rbf_nl.tuning.modeler)[:x_reference]) ==
          (QTP_sys_rbf.statedim, horizon + 1)
    @test size(JuMP.object_dictionary(C_rbf_nl.tuning.modeler)[:x]) ==
          (QTP_sys_rbf.statedim, horizon + 1)
    @test size(JuMP.object_dictionary(C_rbf_nl.tuning.modeler)[:e_x]) ==
          (QTP_sys_rbf.statedim, horizon + 1)
    @test JuMP.objective_function(C_rbf_nl.tuning.modeler) != 0 #to improve
    @test JuMP.objective_function_type(C_rbf_nl.tuning.modeler) == JuMP.QuadExpr
    @test JuMP.list_of_constraint_types(C_rbf_nl.tuning.modeler) == [
        (AffExpr, MathOptInterface.EqualTo{Float64}),
        (AffExpr, MathOptInterface.LessThan{Float64}),
        (VariableRef, MathOptInterface.EqualTo{Float64}),
    ]
    @test C_rbf_nl.tuning.reference == references
    @test C_rbf_nl.tuning.horizon == horizon
    @test C_rbf_nl.tuning.terminal_ingredient.Xf == "none"
    @test C_rbf_nl.tuning.terminal_ingredient.P != 0
    @test C_rbf_nl.tuning.sample_time == 5.0
    @test C_rbf_nl.tuning.max_time == 30 #to modify
    @test size(C_rbf_nl.initialization) == (QTP_sys_rbf.statedim,)
    @test size(C_rbf_nl.computation_results.x) == (QTP_sys_rbf.statedim, horizon + 1)
    @test size(C_rbf_nl.computation_results.e_x) == (QTP_sys_rbf.statedim, horizon + 1)
    @test size(C_rbf_nl.computation_results.u) == (QTP_sys_rbf.inputdim, horizon)
    @test size(C_rbf_nl.computation_results.e_u) == (QTP_sys_rbf.inputdim, horizon)
    ### end evaluate rbf NL MPC implementation ###

end

@testset "design linear Model Predictive Control with polynet model" begin

    ##############################
    ###           polynet       ###                    
    ##############################

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
    horizon = 15
    sample_time = 5
    terminal_ingredient = false
    max_time = 5
    x = [0.65, 0.65, 0.65, 0.65] .* ones(4, horizon + 1)
    u = [1.2, 1.2] .* ones(2, horizon)
    references = AutomationLabsModelPredictiveControl.ReferencesStateInput(x, u)

    ### polynet L MPC ###
    method = AutomationLabsModelPredictiveControl.LinearProgramming()

    C_polynet_linear = _model_predictive_control_design(
        QTP_sys_polynet,
        horizon,
        sample_time,
        references;
        mpc_programming_type = "linear",
    )

    ### start evaluate polynet L MPC implementation ###
    @test C_polynet_linear.system == QTP_sys_polynet
    @test typeof(C_polynet_linear.tuning.modeler) == JuMP.Model
    @test JuMP.solver_name(C_polynet_linear.tuning.modeler) == "SCIP"
    @test length(JuMP.object_dictionary(C_polynet_linear.tuning.modeler)) == 6
    @test size(JuMP.object_dictionary(C_polynet_linear.tuning.modeler)[:u_reference]) ==
          (QTP_sys_polynet.inputdim, horizon)
    @test size(JuMP.object_dictionary(C_polynet_linear.tuning.modeler)[:u]) ==
          (QTP_sys_polynet.inputdim, horizon)
    @test size(JuMP.object_dictionary(C_polynet_linear.tuning.modeler)[:e_u]) ==
          (QTP_sys_polynet.inputdim, horizon)
    @test size(JuMP.object_dictionary(C_polynet_linear.tuning.modeler)[:x_reference]) ==
          (QTP_sys_polynet.statedim, horizon + 1)
    @test size(JuMP.object_dictionary(C_polynet_linear.tuning.modeler)[:x]) ==
          (QTP_sys_polynet.statedim, horizon + 1)
    @test size(JuMP.object_dictionary(C_polynet_linear.tuning.modeler)[:e_x]) ==
          (QTP_sys_polynet.statedim, horizon + 1)
    @test JuMP.objective_function(C_polynet_linear.tuning.modeler) != 0 #to improve
    @test JuMP.objective_function_type(C_polynet_linear.tuning.modeler) == JuMP.QuadExpr
    @test JuMP.list_of_constraint_types(C_polynet_linear.tuning.modeler) == [
        (AffExpr, MathOptInterface.EqualTo{Float64}),
        (AffExpr, MathOptInterface.LessThan{Float64}),
        (VariableRef, MathOptInterface.EqualTo{Float64}),
    ]
    @test C_polynet_linear.tuning.reference == references
    @test C_polynet_linear.tuning.horizon == horizon
    @test C_polynet_linear.tuning.terminal_ingredient.Xf == "none"
    @test C_polynet_linear.tuning.terminal_ingredient.P != 0
    @test C_polynet_linear.tuning.sample_time == 5.0
    @test C_polynet_linear.tuning.max_time == 30 #to modify
    @test size(C_polynet_linear.initialization) == (QTP_sys_polynet.statedim,)
    @test size(C_polynet_linear.computation_results.x) ==
          (QTP_sys_polynet.statedim, horizon + 1)
    @test size(C_polynet_linear.computation_results.e_x) ==
          (QTP_sys_polynet.statedim, horizon + 1)
    @test size(C_polynet_linear.computation_results.u) ==
          (QTP_sys_polynet.inputdim, horizon)
    @test size(C_polynet_linear.computation_results.e_u) ==
          (QTP_sys_polynet.inputdim, horizon)
    ### end evaluate polynet L MPC implementation ###
end

@testset "design non linear Model Predictive Control with polynet model" begin

    ##############################
    ###           polynet       ###                    
    ##############################

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
    horizon = 15
    sample_time = 5
    terminal_ingredient = false
    max_time = 5
    x = [0.65, 0.65, 0.65, 0.65] .* ones(4, horizon + 1)
    u = [1.2, 1.2] .* ones(2, horizon)
    references = AutomationLabsModelPredictiveControl.ReferencesStateInput(x, u)

    ### polynet N L MPC ###
    method = AutomationLabsModelPredictiveControl.NonLinearProgramming()

    C_polynet_nl = _model_predictive_control_design(
        QTP_sys_polynet,
        horizon,
        sample_time,
        references;
        mpc_programming_type = "non_linear",
    )

    ### start evaluate polynet NL MPC implementation ###
    @test C_polynet_nl.system == QTP_sys_polynet
    @test typeof(C_polynet_nl.tuning.modeler) == JuMP.Model
    @test JuMP.solver_name(C_polynet_nl.tuning.modeler) == "Ipopt"
    @test length(JuMP.object_dictionary(C_polynet_nl.tuning.modeler)) == 8
    @test size(JuMP.object_dictionary(C_polynet_nl.tuning.modeler)[:u_reference]) ==
          (QTP_sys_polynet.inputdim, horizon)
    @test size(JuMP.object_dictionary(C_polynet_nl.tuning.modeler)[:u]) ==
          (QTP_sys_polynet.inputdim, horizon)
    @test size(JuMP.object_dictionary(C_polynet_nl.tuning.modeler)[:e_u]) ==
          (QTP_sys_polynet.inputdim, horizon)
    @test size(JuMP.object_dictionary(C_polynet_nl.tuning.modeler)[:x_reference]) ==
          (QTP_sys_polynet.statedim, horizon + 1)
    @test size(JuMP.object_dictionary(C_polynet_nl.tuning.modeler)[:x]) ==
          (QTP_sys_polynet.statedim, horizon + 1)
    @test size(JuMP.object_dictionary(C_polynet_nl.tuning.modeler)[:e_x]) ==
          (QTP_sys_polynet.statedim, horizon + 1)
    @test JuMP.objective_function(C_polynet_nl.tuning.modeler) != 0 #to improve
    @test JuMP.objective_function_type(C_polynet_nl.tuning.modeler) == JuMP.QuadExpr
    @test JuMP.list_of_constraint_types(C_polynet_nl.tuning.modeler) == [
        (AffExpr, MathOptInterface.EqualTo{Float64}),
        (AffExpr, MathOptInterface.LessThan{Float64}),
        (VariableRef, MathOptInterface.EqualTo{Float64}),
    ]
    @test C_polynet_nl.tuning.reference == references
    @test C_polynet_nl.tuning.horizon == horizon
    @test C_polynet_nl.tuning.terminal_ingredient.Xf == "none"
    @test C_polynet_nl.tuning.terminal_ingredient.P != 0
    @test C_polynet_nl.tuning.sample_time == 5.0
    @test C_polynet_nl.tuning.max_time == 30 #to modify
    @test size(C_polynet_nl.initialization) == (QTP_sys_polynet.statedim,)
    @test size(C_polynet_nl.computation_results.x) ==
          (QTP_sys_polynet.statedim, horizon + 1)
    @test size(C_polynet_nl.computation_results.e_x) ==
          (QTP_sys_polynet.statedim, horizon + 1)
    @test size(C_polynet_nl.computation_results.u) == (QTP_sys_polynet.inputdim, horizon)
    @test size(C_polynet_nl.computation_results.e_u) == (QTP_sys_polynet.inputdim, horizon)
    ### end evaluate polynet NL MPC implementation ###
end
#=
@testset "design milp Model Predictive Control with polynet model" begin

    ##############################
    ###           polynet       ###                    
    ##############################

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
    horizon = 15
    sample_time = 5
    terminal_ingredient = false
    max_time = 5
    x = [0.65, 0.65, 0.65, 0.65] .* ones(4, horizon + 1)
    u = [1.2, 1.2] .* ones(2, horizon)
    references = AutomationLabsModelPredictiveControl.ReferencesStateInput(x, u)

    ### polynet MILP MPC ###
    method = AutomationLabsModelPredictiveControl.MixedIntegerLinearProgramming()
    C_polynet_milp = _model_predictive_control_design(
        QTP_sys_polynet,
        type_polynet,
        horizon,
        method,
        sample_time,
        references,
    )


    ### start evaluate polynet MILP MPC implementation ###
    @test C_polynet_milp.system == QTP_sys_polynet
    @test typeof(C_polynet_milp.tuning.modeler) == JuMP.Model
    @test JuMP.solver_name(C_polynet_milp.tuning.modeler) == "SCIP"
    @test length(JuMP.object_dictionary(C_polynet_milp.tuning.modeler)) == 10
    @test size(JuMP.object_dictionary(C_polynet_milp.tuning.modeler)[:u_reference]) ==
          (QTP_sys_polynet.inputdim, horizon)
    @test size(JuMP.object_dictionary(C_polynet_milp.tuning.modeler)[:u]) ==
          (QTP_sys_polynet.inputdim, horizon)
    @test size(JuMP.object_dictionary(C_polynet_milp.tuning.modeler)[:e_u]) ==
          (QTP_sys_polynet.inputdim, horizon)
    @test size(JuMP.object_dictionary(C_polynet_milp.tuning.modeler)[:x_reference]) ==
          (QTP_sys_polynet.statedim, horizon + 1)
    @test size(JuMP.object_dictionary(C_polynet_milp.tuning.modeler)[:x]) ==
          (QTP_sys_polynet.statedim, horizon + 1)
    @test size(JuMP.object_dictionary(C_polynet_milp.tuning.modeler)[:e_x]) ==
          (QTP_sys_polynet.statedim, horizon + 1)
    #to do @test size(JuMP.object_dictionary(C_polynet_milp.tuning.modeler)[:y])
    #to do @test size(JuMP.object_dictionary(C_polynet_milp.tuning.modeler)[:Bin_1])
    #to do @test size(JuMP.object_dictionary(C_polynet_milp.tuning.modeler)[:Bin_2])
    @test JuMP.objective_function(C_polynet_milp.tuning.modeler) != 0 #to improve
    @test JuMP.objective_function_type(C_polynet_milp.tuning.modeler) == JuMP.QuadExpr
    @test JuMP.list_of_constraint_types(C_polynet_milp.tuning.modeler) == [
        (AffExpr, MathOptInterface.EqualTo{Float64}),
        (AffExpr, MathOptInterface.GreaterThan{Float64}),
        (AffExpr, MathOptInterface.LessThan{Float64}),
        (VariableRef, MathOptInterface.EqualTo{Float64}),
        (VariableRef, MathOptInterface.ZeroOne),       
    ]
    @test C_polynet_milp.tuning.reference == references
    @test C_polynet_milp.tuning.horizon == horizon
    @test C_polynet_milp.tuning.terminal_ingredient.Xf == false
    @test C_polynet_milp.tuning.terminal_ingredient.P != 0
    @test C_polynet_milp.tuning.sample_time == 5.0
    @test C_polynet_milp.tuning.max_time == 30 #to modify
    @test size(C_polynet_milp.initialization) == (QTP_sys_polynet.statedim,)
    @test size(C_polynet_milp.computation_results.x) ==
          (QTP_sys_polynet.statedim, horizon + 1)
    @test size(C_polynet_milp.computation_results.e_x) ==
          (QTP_sys_polynet.statedim, horizon + 1)
    @test size(C_polynet_milp.computation_results.u) == (QTP_sys_polynet.inputdim, horizon)
    @test size(C_polynet_milp.computation_results.e_u) == (QTP_sys_polynet.inputdim, horizon)
    ### end evaluate polynet MILP MPC implementation ###

end
=#

@testset "design linear Model Predictive Control with linear_regressor model" begin

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
    B = AB[:, 5:end]

    type_linear_regressor = linear_regressor_machine.model

    #system definition with Mathematical systems
    QTP_sys_linear_regressor =
        MathematicalSystems.ConstrainedLinearControlDiscreteSystem(A, B, x_cons, u_cons)

    #MPC design parameters
    horizon = 15
    sample_time = 5
    terminal_ingredient = false
    max_time = 5
    x = [0.65, 0.65, 0.65, 0.65] * ones(1, horizon + 1)
    u = [1.2, 1.2] * ones(1, horizon)
    references = AutomationLabsModelPredictiveControl.ReferencesStateInput(x, u)

    ##############################
    ###           linear_regressor          ###                    
    ##############################

    ### linear_regressor L MPC ###
    method = AutomationLabsModelPredictiveControl.LinearProgramming()

    C_linear_regressor_linear = _model_predictive_control_design(
        QTP_sys_linear_regressor,
        horizon,
        sample_time,
        references;
    )

    ### start evaluate linear_regressor L MPC implementation ###
    @test C_linear_regressor_linear.system == QTP_sys_linear_regressor
    @test typeof(C_linear_regressor_linear.tuning.modeler) == JuMP.Model
    @test JuMP.solver_name(C_linear_regressor_linear.tuning.modeler) == "SCIP"
    @test length(JuMP.object_dictionary(C_linear_regressor_linear.tuning.modeler)) == 6
    @test size(
        JuMP.object_dictionary(C_linear_regressor_linear.tuning.modeler)[:u_reference],
    ) == (size(QTP_sys_linear_regressor.B, 2), horizon)
    @test size(JuMP.object_dictionary(C_linear_regressor_linear.tuning.modeler)[:u]) ==
          (size(QTP_sys_linear_regressor.B, 2), horizon)
    @test size(JuMP.object_dictionary(C_linear_regressor_linear.tuning.modeler)[:e_u]) ==
          (size(QTP_sys_linear_regressor.B, 2), horizon)
    @test size(
        JuMP.object_dictionary(C_linear_regressor_linear.tuning.modeler)[:x_reference],
    ) == (size(QTP_sys_linear_regressor.A, 1), horizon + 1)
    @test size(JuMP.object_dictionary(C_linear_regressor_linear.tuning.modeler)[:x]) ==
          (size(QTP_sys_linear_regressor.A, 1), horizon + 1)
    @test size(JuMP.object_dictionary(C_linear_regressor_linear.tuning.modeler)[:e_x]) ==
          (size(QTP_sys_linear_regressor.A, 1), horizon + 1)
    @test JuMP.objective_function(C_linear_regressor_linear.tuning.modeler) != 0 #to improve
    @test JuMP.objective_function_type(C_linear_regressor_linear.tuning.modeler) ==
          JuMP.QuadExpr
    @test JuMP.list_of_constraint_types(C_linear_regressor_linear.tuning.modeler) == [
        (AffExpr, MathOptInterface.EqualTo{Float64}),
        (AffExpr, MathOptInterface.LessThan{Float64}),
        (VariableRef, MathOptInterface.EqualTo{Float64}),
    ]
    @test C_linear_regressor_linear.tuning.reference == references
    @test C_linear_regressor_linear.tuning.horizon == horizon
    @test C_linear_regressor_linear.tuning.terminal_ingredient.Xf == "none"
    @test C_linear_regressor_linear.tuning.terminal_ingredient.P != 0
    @test C_linear_regressor_linear.tuning.sample_time == 5.0
    @test C_linear_regressor_linear.tuning.max_time == 30 #to modify
    @test size(C_linear_regressor_linear.initialization) ==
          (size(QTP_sys_linear_regressor.A, 1),)
    @test size(C_linear_regressor_linear.computation_results.x) ==
          (size(QTP_sys_linear_regressor.A, 1), horizon + 1)
    @test size(C_linear_regressor_linear.computation_results.e_x) ==
          (size(QTP_sys_linear_regressor.A, 1), horizon + 1)
    @test size(C_linear_regressor_linear.computation_results.u) ==
          (size(QTP_sys_linear_regressor.B, 2), horizon)
    @test size(C_linear_regressor_linear.computation_results.e_u) ==
          (size(QTP_sys_linear_regressor.B, 2), horizon)
    ### end evaluate linear_regressor L MPC implementation ###

end

@testset "design linear Model Predictive Control with NeuralODE model" begin

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
    neuralnetODE_type1_machine =
        machine("./models_saved/NeuralODE_train_result.jls")

    #extract best model from the all trained models
    mlj_neuralnetODE_type1 =
        fitted_params(fitted_params(neuralnetODE_type1_machine).machine).best_model
    f_neuralODE =
        fitted_params(fitted_params(neuralnetODE_type1_machine).machine).best_fitted_params[1]
    type_neuralODE = mlj_neuralnetODE_type1.builder

    #system definition with MAthematical systems
    QTP_sys_neuralnetODE_type1 =
        MathematicalSystems.ConstrainedBlackBoxControlDiscreteSystem(
            f_neuralODE,
            4,
            2,
            x_cons,
            u_cons,
        )

    #MPC design parameters
    horizon = 15
    sample_time = 5
    terminal_ingredient = false
    max_time = 5
    x = [0.65, 0.65, 0.65, 0.65] * ones(1, horizon + 1)
    u = [1.2, 1.2] * ones(1, horizon)
    references = AutomationLabsModelPredictiveControl.ReferencesStateInput(x, u)

    ##############################
    ###           neuralnetODE_type1          ###                    
    ##############################

    ### neuralnetODE_type1 L MPC ###
    method = AutomationLabsModelPredictiveControl.LinearProgramming()

    C_neuralnetODE_type1_linear = _model_predictive_control_design(
        QTP_sys_neuralnetODE_type1,
        horizon,
        sample_time,
        references;
        mpc_programming_type = "linear",
    )

    ### start evaluate neuralnetODE_type1 L MPC implementation ###
    @test C_neuralnetODE_type1_linear.system == QTP_sys_neuralnetODE_type1
    @test typeof(C_neuralnetODE_type1_linear.tuning.modeler) == JuMP.Model
    @test JuMP.solver_name(C_neuralnetODE_type1_linear.tuning.modeler) == "SCIP"
    @test length(JuMP.object_dictionary(C_neuralnetODE_type1_linear.tuning.modeler)) == 6
    @test size(
        JuMP.object_dictionary(C_neuralnetODE_type1_linear.tuning.modeler)[:u_reference],
    ) == (QTP_sys_neuralnetODE_type1.inputdim, horizon)
    @test size(JuMP.object_dictionary(C_neuralnetODE_type1_linear.tuning.modeler)[:u]) ==
          (QTP_sys_neuralnetODE_type1.inputdim, horizon)
    @test size(JuMP.object_dictionary(C_neuralnetODE_type1_linear.tuning.modeler)[:e_u]) ==
          (QTP_sys_neuralnetODE_type1.inputdim, horizon)
    @test size(
        JuMP.object_dictionary(C_neuralnetODE_type1_linear.tuning.modeler)[:x_reference],
    ) == (QTP_sys_neuralnetODE_type1.statedim, horizon + 1)
    @test size(JuMP.object_dictionary(C_neuralnetODE_type1_linear.tuning.modeler)[:x]) ==
          (QTP_sys_neuralnetODE_type1.statedim, horizon + 1)
    @test size(JuMP.object_dictionary(C_neuralnetODE_type1_linear.tuning.modeler)[:e_x]) ==
          (QTP_sys_neuralnetODE_type1.statedim, horizon + 1)
    @test JuMP.objective_function(C_neuralnetODE_type1_linear.tuning.modeler) != 0 #to improve
    @test JuMP.objective_function_type(C_neuralnetODE_type1_linear.tuning.modeler) ==
          JuMP.QuadExpr
    @test JuMP.list_of_constraint_types(C_neuralnetODE_type1_linear.tuning.modeler) == [
        (AffExpr, MathOptInterface.EqualTo{Float64}),
        (AffExpr, MathOptInterface.LessThan{Float64}),
        (VariableRef, MathOptInterface.EqualTo{Float64}),
    ]
    @test C_neuralnetODE_type1_linear.tuning.reference == references
    @test C_neuralnetODE_type1_linear.tuning.horizon == horizon
    @test C_neuralnetODE_type1_linear.tuning.terminal_ingredient.Xf == "none"
    @test C_neuralnetODE_type1_linear.tuning.terminal_ingredient.P != 0
    @test C_neuralnetODE_type1_linear.tuning.sample_time == 5.0
    @test C_neuralnetODE_type1_linear.tuning.max_time == 30 #to modify
    @test size(C_neuralnetODE_type1_linear.initialization) ==
          (QTP_sys_neuralnetODE_type1.statedim,)
    @test size(C_neuralnetODE_type1_linear.computation_results.x) ==
          (QTP_sys_neuralnetODE_type1.statedim, horizon + 1)
    @test size(C_neuralnetODE_type1_linear.computation_results.e_x) ==
          (QTP_sys_neuralnetODE_type1.statedim, horizon + 1)
    @test size(C_neuralnetODE_type1_linear.computation_results.u) ==
          (QTP_sys_neuralnetODE_type1.inputdim, horizon)
    @test size(C_neuralnetODE_type1_linear.computation_results.e_u) ==
          (QTP_sys_neuralnetODE_type1.inputdim, horizon)
    ### end evaluate neuralnetODE_type1 L MPC implementation ###

end

@testset "design linear Model Predictive Control with Rknn1 model" begin

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
      rknn1_machine =
          machine("./models_saved/rknn1_train_result.jls")
  
      #extract best model from the all trained models
      mlj_rknn1 =
          fitted_params(fitted_params(rknn1_machine).machine).best_model
      f_rknn1 =
          fitted_params(fitted_params(rknn1_machine).machine).best_fitted_params[1]
      type_rknn1 = mlj_rknn1.builder
  
      #system definition with MAthematical systems
      QTP_sys_rknn1 =
          MathematicalSystems.ConstrainedBlackBoxControlDiscreteSystem(
              f_rknn1,
              4,
              2,
              x_cons,
              u_cons,
          )
  
      #MPC design parameters
      horizon = 15
      sample_time = 5
      terminal_ingredient = false
      max_time = 5
      x = [0.65, 0.65, 0.65, 0.65] * ones(1, horizon + 1)
      u = [1.2, 1.2] * ones(1, horizon)
      references = AutomationLabsModelPredictiveControl.ReferencesStateInput(x, u)
  
      ##############################
      ###           neuralnetODE_type1          ###                    
      ##############################
  
      ### neuralnetODE_type1 L MPC ###
      method = AutomationLabsModelPredictiveControl.LinearProgramming()
  
      C_rknn1_linear = _model_predictive_control_design(
          QTP_sys_rknn1,
          horizon,
          sample_time,
          references;
          mpc_programming_type = "linear",
      )
  
      ### start evaluate neuralnetODE_type1 L MPC implementation ###
      @test C_rknn1_linear.system == QTP_sys_rknn1
      @test typeof(C_rknn1_linear.tuning.modeler) == JuMP.Model
      @test JuMP.solver_name(C_rknn1_linear.tuning.modeler) == "SCIP"
      @test length(JuMP.object_dictionary(C_rknn1_linear.tuning.modeler)) == 6
      @test size(
          JuMP.object_dictionary(C_rknn1_linear.tuning.modeler)[:u_reference],
      ) == (QTP_sys_rknn1.inputdim, horizon)
      @test size(JuMP.object_dictionary(C_rknn1_linear.tuning.modeler)[:u]) ==
            (QTP_sys_rknn1.inputdim, horizon)
      @test size(JuMP.object_dictionary(C_rknn1_linear.tuning.modeler)[:e_u]) ==
            (QTP_sys_rknn1.inputdim, horizon)
      @test size(
          JuMP.object_dictionary(C_rknn1_linear.tuning.modeler)[:x_reference],
      ) == (QTP_sys_rknn1.statedim, horizon + 1)
      @test size(JuMP.object_dictionary(C_rknn1_linear.tuning.modeler)[:x]) ==
            (QTP_sys_rknn1.statedim, horizon + 1)
      @test size(JuMP.object_dictionary(C_rknn1_linear.tuning.modeler)[:e_x]) ==
            (QTP_sys_rknn1.statedim, horizon + 1)
      @test JuMP.objective_function(C_rknn1_linear.tuning.modeler) != 0 #to improve
      @test JuMP.objective_function_type(C_rknn1_linear.tuning.modeler) ==
            JuMP.QuadExpr
      @test JuMP.list_of_constraint_types(C_rknn1_linear.tuning.modeler) == [
          (AffExpr, MathOptInterface.EqualTo{Float64}),
          (AffExpr, MathOptInterface.LessThan{Float64}),
          (VariableRef, MathOptInterface.EqualTo{Float64}),
      ]
      @test C_rknn1_linear.tuning.reference == references
      @test C_rknn1_linear.tuning.horizon == horizon
      @test C_rknn1_linear.tuning.terminal_ingredient.Xf == "none"
      @test C_rknn1_linear.tuning.terminal_ingredient.P != 0
      @test C_rknn1_linear.tuning.sample_time == 5.0
      @test C_rknn1_linear.tuning.max_time == 30 #to modify
      @test size(C_rknn1_linear.initialization) ==
            (QTP_sys_rknn1.statedim,)
      @test size(C_rknn1_linear.computation_results.x) ==
            (QTP_sys_rknn1.statedim, horizon + 1)
      @test size(C_rknn1_linear.computation_results.e_x) ==
            (QTP_sys_rknn1.statedim, horizon + 1)
      @test size(C_rknn1_linear.computation_results.u) ==
            (QTP_sys_rknn1.inputdim, horizon)
      @test size(C_rknn1_linear.computation_results.e_u) ==
            (QTP_sys_rknn1.inputdim, horizon)
      ### end evaluate neuralnetODE_type1 L MPC implementation ###
  
  end

  @testset "design linear Model Predictive Control with rknn2 model" begin

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
      rknn2_machine =
          machine("./models_saved/rknn2_train_result.jls")
  
      #extract best model from the all trained models
      mlj_rknn2 =
          fitted_params(fitted_params(rknn2_machine).machine).best_model
      f_rknn2 =
          fitted_params(fitted_params(rknn2_machine).machine).best_fitted_params[1]
      type_rknn2 = mlj_rknn2.builder
  
      #system definition with MAthematical systems
      QTP_sys_rknn2 =
          MathematicalSystems.ConstrainedBlackBoxControlDiscreteSystem(
              f_rknn2,
              4,
              2,
              x_cons,
              u_cons,
          )
  
      #MPC design parameters
      horizon = 15
      sample_time = 5
      terminal_ingredient = false
      max_time = 5
      x = [0.65, 0.65, 0.65, 0.65] * ones(1, horizon + 1)
      u = [1.2, 1.2] * ones(1, horizon)
      references = AutomationLabsModelPredictiveControl.ReferencesStateInput(x, u)
  
      ##############################
      ###           neuralnetODE_type1          ###                    
      ##############################
  
      ### neuralnetODE_type1 L MPC ###
      method = AutomationLabsModelPredictiveControl.LinearProgramming()
  
      C_rknn2_linear = _model_predictive_control_design(
          QTP_sys_rknn2,
          horizon,
          sample_time,
          references;
          mpc_programming_type = "linear",
      )
  
      ### start evaluate neuralnetODE_type1 L MPC implementation ###
      @test C_rknn2_linear.system == QTP_sys_rknn2
      @test typeof(C_rknn2_linear.tuning.modeler) == JuMP.Model
      @test JuMP.solver_name(C_rknn2_linear.tuning.modeler) == "SCIP"
      @test length(JuMP.object_dictionary(C_rknn2_linear.tuning.modeler)) == 6
      @test size(
          JuMP.object_dictionary(C_rknn2_linear.tuning.modeler)[:u_reference],
      ) == (QTP_sys_rknn2.inputdim, horizon)
      @test size(JuMP.object_dictionary(C_rknn2_linear.tuning.modeler)[:u]) ==
            (QTP_sys_rknn2.inputdim, horizon)
      @test size(JuMP.object_dictionary(C_rknn2_linear.tuning.modeler)[:e_u]) ==
            (QTP_sys_rknn2.inputdim, horizon)
      @test size(
          JuMP.object_dictionary(C_rknn2_linear.tuning.modeler)[:x_reference],
      ) == (QTP_sys_rknn2.statedim, horizon + 1)
      @test size(JuMP.object_dictionary(C_rknn2_linear.tuning.modeler)[:x]) ==
            (QTP_sys_rknn2.statedim, horizon + 1)
      @test size(JuMP.object_dictionary(C_rknn2_linear.tuning.modeler)[:e_x]) ==
            (QTP_sys_rknn2.statedim, horizon + 1)
      @test JuMP.objective_function(C_rknn2_linear.tuning.modeler) != 0 #to improve
      @test JuMP.objective_function_type(C_rknn2_linear.tuning.modeler) ==
            JuMP.QuadExpr
      @test JuMP.list_of_constraint_types(C_rknn2_linear.tuning.modeler) == [
          (AffExpr, MathOptInterface.EqualTo{Float64}),
          (AffExpr, MathOptInterface.LessThan{Float64}),
          (VariableRef, MathOptInterface.EqualTo{Float64}),
      ]
      @test C_rknn2_linear.tuning.reference == references
      @test C_rknn2_linear.tuning.horizon == horizon
      @test C_rknn2_linear.tuning.terminal_ingredient.Xf == "none"
      @test C_rknn2_linear.tuning.terminal_ingredient.P != 0
      @test C_rknn2_linear.tuning.sample_time == 5.0
      @test C_rknn2_linear.tuning.max_time == 30 #to modify
      @test size(C_rknn2_linear.initialization) ==
            (QTP_sys_rknn2.statedim,)
      @test size(C_rknn2_linear.computation_results.x) ==
            (QTP_sys_rknn2.statedim, horizon + 1)
      @test size(C_rknn2_linear.computation_results.e_x) ==
            (QTP_sys_rknn2.statedim, horizon + 1)
      @test size(C_rknn2_linear.computation_results.u) ==
            (QTP_sys_rknn2.inputdim, horizon)
      @test size(C_rknn2_linear.computation_results.e_u) ==
            (QTP_sys_rknn2.inputdim, horizon)
      ### end evaluate neuralnetODE_type1 L MPC implementation ###
  
  end

@testset "design linear Model Predictive Control with rknn4 model" begin

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
      rknn4_machine =
          machine("./models_saved/rknn4_train_result.jls")
  
      #extract best model from the all trained models
      mlj_rknn4 =
          fitted_params(fitted_params(rknn4_machine).machine).best_model
      f_rknn4 =
          fitted_params(fitted_params(rknn4_machine).machine).best_fitted_params[1]
      type_rknn4 = mlj_rknn4.builder
  
      #system definition with MAthematical systems
      QTP_sys_rknn4 =
          MathematicalSystems.ConstrainedBlackBoxControlDiscreteSystem(
              f_rknn4,
              4,
              2,
              x_cons,
              u_cons,
          )
  
      #MPC design parameters
      horizon = 15
      sample_time = 5
      terminal_ingredient = false
      max_time = 5
      x = [0.65, 0.65, 0.65, 0.65] * ones(1, horizon + 1)
      u = [1.2, 1.2] * ones(1, horizon)
      references = AutomationLabsModelPredictiveControl.ReferencesStateInput(x, u)
  
      ##############################
      ###           neuralnetODE_type1          ###                    
      ##############################
  
      ### neuralnetODE_type1 L MPC ###
      method = AutomationLabsModelPredictiveControl.LinearProgramming()
  
      C_rknn4_linear = _model_predictive_control_design(
          QTP_sys_rknn4,
          horizon,
          sample_time,
          references;
          mpc_programming_type = "linear",
      )
  
      ### start evaluate neuralnetODE_type1 L MPC implementation ###
      @test C_rknn4_linear.system == QTP_sys_rknn4
      @test typeof(C_rknn4_linear.tuning.modeler) == JuMP.Model
      @test JuMP.solver_name(C_rknn4_linear.tuning.modeler) == "SCIP"
      @test length(JuMP.object_dictionary(C_rknn4_linear.tuning.modeler)) == 6
      @test size(
          JuMP.object_dictionary(C_rknn4_linear.tuning.modeler)[:u_reference],
      ) == (QTP_sys_rknn4.inputdim, horizon)
      @test size(JuMP.object_dictionary(C_rknn4_linear.tuning.modeler)[:u]) ==
            (QTP_sys_rknn4.inputdim, horizon)
      @test size(JuMP.object_dictionary(C_rknn4_linear.tuning.modeler)[:e_u]) ==
            (QTP_sys_rknn4.inputdim, horizon)
      @test size(
          JuMP.object_dictionary(C_rknn4_linear.tuning.modeler)[:x_reference],
      ) == (QTP_sys_rknn4.statedim, horizon + 1)
      @test size(JuMP.object_dictionary(C_rknn4_linear.tuning.modeler)[:x]) ==
            (QTP_sys_rknn4.statedim, horizon + 1)
      @test size(JuMP.object_dictionary(C_rknn4_linear.tuning.modeler)[:e_x]) ==
            (QTP_sys_rknn4.statedim, horizon + 1)
      @test JuMP.objective_function(C_rknn4_linear.tuning.modeler) != 0 #to improve
      @test JuMP.objective_function_type(C_rknn4_linear.tuning.modeler) ==
            JuMP.QuadExpr
      @test JuMP.list_of_constraint_types(C_rknn4_linear.tuning.modeler) == [
          (AffExpr, MathOptInterface.EqualTo{Float64}),
          (AffExpr, MathOptInterface.LessThan{Float64}),
          (VariableRef, MathOptInterface.EqualTo{Float64}),
      ]
      @test C_rknn4_linear.tuning.reference == references
      @test C_rknn4_linear.tuning.horizon == horizon
      @test C_rknn4_linear.tuning.terminal_ingredient.Xf == "none"
      @test C_rknn4_linear.tuning.terminal_ingredient.P != 0
      @test C_rknn4_linear.tuning.sample_time == 5.0
      @test C_rknn4_linear.tuning.max_time == 30 #to modify
      @test size(C_rknn4_linear.initialization) ==
            (QTP_sys_rknn4.statedim,)
      @test size(C_rknn4_linear.computation_results.x) ==
            (QTP_sys_rknn4.statedim, horizon + 1)
      @test size(C_rknn4_linear.computation_results.e_x) ==
            (QTP_sys_rknn4.statedim, horizon + 1)
      @test size(C_rknn4_linear.computation_results.u) ==
            (QTP_sys_rknn4.inputdim, horizon)
      @test size(C_rknn4_linear.computation_results.e_u) ==
            (QTP_sys_rknn4.inputdim, horizon)
      ### end evaluate neuralnetODE_type1 L MPC implementation ###
  
  end
end
