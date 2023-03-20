# Copyright (c) 2022: Pierre Blaud and contributors
########################################################
# This Source Code Form is subject to the terms of the #
# Mozilla Public License, v. 2.0. If a copy of the MPL #
# was not distributed with this file,  				   #
# You can obtain one at https://mozilla.org/MPL/2.0/.  #
########################################################

module ModelerImplementationTest

using MLJ
using AutomationLabsIdentification
using MathematicalSystems
using LazySets
using JuMP
using Test
using MathOptInterface

using AutomationLabsModelPredictiveControl

import AutomationLabsModelPredictiveControl: auto_solver_def
import AutomationLabsModelPredictiveControl:
    _model_predictive_control_modeler_implementation
import AutomationLabsModelPredictiveControl: _JuMP_model_definition

@testset "Fnn model linear modeler MPC test" begin

    #get the fnn model to design the mpc controler
    fnn_machine = machine("./models_saved/fnn_train_result.jls")

    #extract best model from the all trained models
    mlj_fnn = fitted_params(fitted_params(fnn_machine).machine).best_model
    f_fnn = fitted_params(fitted_params(fnn_machine).machine).best_fitted_params[1]
    model_fnn = mlj_fnn.builder

    method = AutomationLabsModelPredictiveControl.LinearProgramming()

    #system definition with MAthematical systems
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
    terminal_ingredients = false
    max_time = 5
    x = [0.65, 0.65, 0.65, 0.65] * ones(1, horizon + 1)
    u = [1.2, 1.2] * ones(1, horizon)

    references = AutomationLabsModelPredictiveControl.ReferencesStateInput(x, u)

    solver = auto_solver_def()

    modeler_mpc = _model_predictive_control_modeler_implementation(
        method,
        model_fnn,
        QTP_sys_fnn,
        horizon,
        references,
        solver,
    )

    ### start evaluate FNN L MPC implementation ###
    @test typeof(modeler_mpc) == JuMP.Model
    @test JuMP.solver_name(modeler_mpc) == "SCIP"
    @test length(JuMP.object_dictionary(modeler_mpc)) == 6
    @test size(JuMP.object_dictionary(modeler_mpc)[:u_reference]) ==
          (QTP_sys_fnn.inputdim, horizon)
    @test size(JuMP.object_dictionary(modeler_mpc)[:u]) == (QTP_sys_fnn.inputdim, horizon)
    @test size(JuMP.object_dictionary(modeler_mpc)[:e_u]) == (QTP_sys_fnn.inputdim, horizon)
    @test size(JuMP.object_dictionary(modeler_mpc)[:x_reference]) ==
          (QTP_sys_fnn.statedim, horizon + 1)
    @test size(JuMP.object_dictionary(modeler_mpc)[:x]) ==
          (QTP_sys_fnn.statedim, horizon + 1)
    @test size(JuMP.object_dictionary(modeler_mpc)[:e_x]) ==
          (QTP_sys_fnn.statedim, horizon + 1)
    @test JuMP.objective_function(modeler_mpc) == 0
    @test JuMP.objective_function_type(modeler_mpc) == JuMP.AffExpr
    @test JuMP.list_of_constraint_types(modeler_mpc) == [
        (AffExpr, MathOptInterface.EqualTo{Float64}),
        (AffExpr, MathOptInterface.LessThan{Float64}),
        (VariableRef, MathOptInterface.EqualTo{Float64}),
    ]

end

@testset "Fnn model non linear modeler MPC test" begin

    #get the fnn model to design the mpc controler
    fnn_machine = machine("./models_saved/fnn_train_result.jls")

    #extract best model from the all trained models
    mlj_fnn = fitted_params(fitted_params(fnn_machine).machine).best_model
    f_fnn = fitted_params(fitted_params(fnn_machine).machine).best_fitted_params[1]
    model_fnn = mlj_fnn.builder

    method = AutomationLabsModelPredictiveControl.NonLinearProgramming()

    #system definition with MAthematical systems
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
    terminal_ingredients = false
    max_time = 5
    x = [0.65, 0.65, 0.65, 0.65] * ones(1, horizon + 1)
    u = [1.2, 1.2] * ones(1, horizon)

    references = AutomationLabsModelPredictiveControl.ReferencesStateInput(x, u)

    solver = auto_solver_def()

    modeler_mpc = _model_predictive_control_modeler_implementation(
        method,
        model_fnn,
        QTP_sys_fnn,
        horizon,
        references,
        solver,
    )

    ### start evaluate FNN L MPC implementation ###
    @test typeof(modeler_mpc) == JuMP.Model
    @test JuMP.solver_name(modeler_mpc) == "Ipopt"
    @test length(JuMP.object_dictionary(modeler_mpc)) == 7
    @test size(JuMP.object_dictionary(modeler_mpc)[:u_reference]) ==
          (QTP_sys_fnn.inputdim, horizon)
    @test size(JuMP.object_dictionary(modeler_mpc)[:u]) == (QTP_sys_fnn.inputdim, horizon)
    @test size(JuMP.object_dictionary(modeler_mpc)[:e_u]) == (QTP_sys_fnn.inputdim, horizon)
    @test size(JuMP.object_dictionary(modeler_mpc)[:x_reference]) ==
          (QTP_sys_fnn.statedim, horizon + 1)
    @test size(JuMP.object_dictionary(modeler_mpc)[:x]) ==
          (QTP_sys_fnn.statedim, horizon + 1)
    @test size(JuMP.object_dictionary(modeler_mpc)[:e_x]) ==
          (QTP_sys_fnn.statedim, horizon + 1)
    @test JuMP.objective_function(modeler_mpc) == 0
    @test JuMP.objective_function_type(modeler_mpc) == JuMP.AffExpr
    @test JuMP.list_of_constraint_types(modeler_mpc) == [
        (AffExpr, MathOptInterface.EqualTo{Float64}),
        (AffExpr, MathOptInterface.LessThan{Float64}),
        (VariableRef, MathOptInterface.EqualTo{Float64}),
    ]

end

@testset "Fnn model milp modeler MPC test" begin

    #get the fnn model to design the mpc controler
    fnn_machine = machine("./models_saved/fnn_train_result.jls")

    #extract best model from the all trained models
    mlj_fnn = fitted_params(fitted_params(fnn_machine).machine).best_model
    f_fnn = fitted_params(fitted_params(fnn_machine).machine).best_fitted_params[1]
    model_fnn = mlj_fnn.builder

    method = AutomationLabsModelPredictiveControl.MixedIntegerLinearProgramming()

    #system definition with MAthematical systems
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
    terminal_ingredients = false
    max_time = 5
    x = [0.65, 0.65, 0.65, 0.65] * ones(1, horizon + 1)
    u = [1.2, 1.2] * ones(1, horizon)

    references = AutomationLabsModelPredictiveControl.ReferencesStateInput(x, u)

    solver = auto_solver_def()

    modeler_mpc = _model_predictive_control_modeler_implementation(
        method,
        model_fnn,
        QTP_sys_fnn,
        horizon,
        references,
        solver,
    )

    ### start evaluate FNN L MPC implementation ###
    @test typeof(modeler_mpc) == JuMP.Model
    @test JuMP.solver_name(modeler_mpc) == "SCIP"
    @test length(JuMP.object_dictionary(modeler_mpc)) == 9
    @test size(JuMP.object_dictionary(modeler_mpc)[:u_reference]) ==
          (QTP_sys_fnn.inputdim, horizon)
    @test size(JuMP.object_dictionary(modeler_mpc)[:u]) == (QTP_sys_fnn.inputdim, horizon)
    @test size(JuMP.object_dictionary(modeler_mpc)[:e_u]) == (QTP_sys_fnn.inputdim, horizon)
    @test size(JuMP.object_dictionary(modeler_mpc)[:x_reference]) ==
          (QTP_sys_fnn.statedim, horizon + 1)
    @test size(JuMP.object_dictionary(modeler_mpc)[:x]) ==
          (QTP_sys_fnn.statedim, horizon + 1)
    @test size(JuMP.object_dictionary(modeler_mpc)[:e_x]) ==
          (QTP_sys_fnn.statedim, horizon + 1)
    @test JuMP.objective_function(modeler_mpc) == 0
    @test JuMP.objective_function_type(modeler_mpc) == JuMP.AffExpr
    @test JuMP.list_of_constraint_types(modeler_mpc) == [
        (AffExpr, MathOptInterface.EqualTo{Float64}),
        (AffExpr, MathOptInterface.GreaterThan{Float64}),
        (AffExpr, MathOptInterface.LessThan{Float64}),
        (VariableRef, MathOptInterface.EqualTo{Float64}),
        (VariableRef, MathOptInterface.ZeroOne),
    ]
end

@testset "ResNet model linear modeler MPC test" begin

    #get the resnet model to design the mpc controler
    resnet_machine = machine("./models_saved/resnet_train_result.jls")

    #extract best model from the all trained models
    mlj_resnet = fitted_params(fitted_params(resnet_machine).machine).best_model
    f_resnet = fitted_params(fitted_params(resnet_machine).machine).best_fitted_params[1]
    model_resnet = mlj_resnet.builder

    method = AutomationLabsModelPredictiveControl.LinearProgramming()

    #system definition with MAthematical systems
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
    terminal_ingredients = false
    max_time = 5
    x = [0.65, 0.65, 0.65, 0.65] * ones(1, horizon + 1)
    u = [1.2, 1.2] * ones(1, horizon)

    references = AutomationLabsModelPredictiveControl.ReferencesStateInput(x, u)

    solver = auto_solver_def()

    modeler_mpc = _model_predictive_control_modeler_implementation(
        method,
        model_resnet,
        QTP_sys_resnet,
        horizon,
        references,
        solver,
    )

    ### start evaluate resnet L MPC implementation ###
    @test typeof(modeler_mpc) == JuMP.Model
    @test JuMP.solver_name(modeler_mpc) == "SCIP"
    @test length(JuMP.object_dictionary(modeler_mpc)) == 6
    @test size(JuMP.object_dictionary(modeler_mpc)[:u_reference]) ==
          (QTP_sys_resnet.inputdim, horizon)
    @test size(JuMP.object_dictionary(modeler_mpc)[:u]) ==
          (QTP_sys_resnet.inputdim, horizon)
    @test size(JuMP.object_dictionary(modeler_mpc)[:e_u]) ==
          (QTP_sys_resnet.inputdim, horizon)
    @test size(JuMP.object_dictionary(modeler_mpc)[:x_reference]) ==
          (QTP_sys_resnet.statedim, horizon + 1)
    @test size(JuMP.object_dictionary(modeler_mpc)[:x]) ==
          (QTP_sys_resnet.statedim, horizon + 1)
    @test size(JuMP.object_dictionary(modeler_mpc)[:e_x]) ==
          (QTP_sys_resnet.statedim, horizon + 1)
    @test JuMP.objective_function(modeler_mpc) == 0
    @test JuMP.objective_function_type(modeler_mpc) == JuMP.AffExpr
    @test JuMP.list_of_constraint_types(modeler_mpc) == [
        (AffExpr, MathOptInterface.EqualTo{Float64}),
        (AffExpr, MathOptInterface.LessThan{Float64}),
        (VariableRef, MathOptInterface.EqualTo{Float64}),
    ]

end

@testset "ResNet model non linear modeler MPC test" begin

    #get the resnet model to design the mpc controler
    resnet_machine = machine("./models_saved/resnet_train_result.jls")

    #extract best model from the all trained models
    mlj_resnet = fitted_params(fitted_params(resnet_machine).machine).best_model
    f_resnet = fitted_params(fitted_params(resnet_machine).machine).best_fitted_params[1]
    model_resnet = mlj_resnet.builder

    method = AutomationLabsModelPredictiveControl.NonLinearProgramming()

    #system definition with Mathematical systems
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
    terminal_ingredients = false
    max_time = 5
    x = [0.65, 0.65, 0.65, 0.65] * ones(1, horizon + 1)
    u = [1.2, 1.2] * ones(1, horizon)

    references = AutomationLabsModelPredictiveControl.ReferencesStateInput(x, u)

    solver = auto_solver_def()

    modeler_mpc = _model_predictive_control_modeler_implementation(
        method,
        model_resnet,
        QTP_sys_resnet,
        horizon,
        references,
        solver,
    )

    ### start evaluate Resnet L MPC implementation ###
    @test typeof(modeler_mpc) == JuMP.Model
    @test JuMP.solver_name(modeler_mpc) == "Ipopt"
    @test length(JuMP.object_dictionary(modeler_mpc)) == 7
    @test size(JuMP.object_dictionary(modeler_mpc)[:u_reference]) ==
          (QTP_sys_resnet.inputdim, horizon)
    @test size(JuMP.object_dictionary(modeler_mpc)[:u]) ==
          (QTP_sys_resnet.inputdim, horizon)
    @test size(JuMP.object_dictionary(modeler_mpc)[:e_u]) ==
          (QTP_sys_resnet.inputdim, horizon)
    @test size(JuMP.object_dictionary(modeler_mpc)[:x_reference]) ==
          (QTP_sys_resnet.statedim, horizon + 1)
    @test size(JuMP.object_dictionary(modeler_mpc)[:x]) ==
          (QTP_sys_resnet.statedim, horizon + 1)
    @test size(JuMP.object_dictionary(modeler_mpc)[:e_x]) ==
          (QTP_sys_resnet.statedim, horizon + 1)
    @test JuMP.objective_function(modeler_mpc) == 0
    @test JuMP.objective_function_type(modeler_mpc) == JuMP.AffExpr
    @test JuMP.list_of_constraint_types(modeler_mpc) == [
        (AffExpr, MathOptInterface.EqualTo{Float64}),
        (AffExpr, MathOptInterface.LessThan{Float64}),
        (VariableRef, MathOptInterface.EqualTo{Float64}),
    ]

end

@testset "ResNet model milp modeler MPC test" begin

    #get the resnet model to design the mpc controler
    resnet_machine = machine("./models_saved/resnet_train_result.jls")

    #extract best model from the all trained models
    mlj_resnet = fitted_params(fitted_params(resnet_machine).machine).best_model
    f_resnet = fitted_params(fitted_params(resnet_machine).machine).best_fitted_params[1]
    model_resnet = mlj_resnet.builder

    method = AutomationLabsModelPredictiveControl.MixedIntegerLinearProgramming()

    #system definition with Mathematical systems
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
    terminal_ingredients = false
    max_time = 5
    x = [0.65, 0.65, 0.65, 0.65] * ones(1, horizon + 1)
    u = [1.2, 1.2] * ones(1, horizon)

    references = AutomationLabsModelPredictiveControl.ReferencesStateInput(x, u)

    solver = auto_solver_def()

    modeler_mpc = _model_predictive_control_modeler_implementation(
        method,
        model_resnet,
        QTP_sys_resnet,
        horizon,
        references,
        solver,
    )

    ### start evaluate Resnet L MPC implementation ###
    @test typeof(modeler_mpc) == JuMP.Model
    @test JuMP.solver_name(modeler_mpc) == "SCIP"
    @test length(JuMP.object_dictionary(modeler_mpc)) == 10
    @test size(JuMP.object_dictionary(modeler_mpc)[:u_reference]) ==
          (QTP_sys_resnet.inputdim, horizon)
    @test size(JuMP.object_dictionary(modeler_mpc)[:u]) ==
          (QTP_sys_resnet.inputdim, horizon)
    @test size(JuMP.object_dictionary(modeler_mpc)[:e_u]) ==
          (QTP_sys_resnet.inputdim, horizon)
    @test size(JuMP.object_dictionary(modeler_mpc)[:x_reference]) ==
          (QTP_sys_resnet.statedim, horizon + 1)
    @test size(JuMP.object_dictionary(modeler_mpc)[:x]) ==
          (QTP_sys_resnet.statedim, horizon + 1)
    @test size(JuMP.object_dictionary(modeler_mpc)[:e_x]) ==
          (QTP_sys_resnet.statedim, horizon + 1)
    @test JuMP.objective_function(modeler_mpc) == 0
    @test JuMP.objective_function_type(modeler_mpc) == JuMP.AffExpr
    @test JuMP.list_of_constraint_types(modeler_mpc) == [
        (AffExpr, MathOptInterface.EqualTo{Float64}),
        (AffExpr, MathOptInterface.GreaterThan{Float64}),
        (AffExpr, MathOptInterface.LessThan{Float64}),
        (VariableRef, MathOptInterface.EqualTo{Float64}),
        (VariableRef, MathOptInterface.ZeroOne),
    ]

end

@testset "DenseNet model linear modeler MPC test" begin

    #get the densenet model to design the mpc controler
    densenet_machine = machine("./models_saved/densenet_train_result.jls")

    #extract best model from the all trained models
    mlj_densenet = fitted_params(fitted_params(densenet_machine).machine).best_model
    f_densenet =
        fitted_params(fitted_params(densenet_machine).machine).best_fitted_params[1]
    model_densenet = mlj_densenet.builder

    method = AutomationLabsModelPredictiveControl.LinearProgramming()

    #system definition with MAthematical systems
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
    terminal_ingredients = false
    max_time = 5
    x = [0.65, 0.65, 0.65, 0.65] * ones(1, horizon + 1)
    u = [1.2, 1.2] * ones(1, horizon)

    references = AutomationLabsModelPredictiveControl.ReferencesStateInput(x, u)

    solver = auto_solver_def()

    modeler_mpc = _model_predictive_control_modeler_implementation(
        method,
        model_densenet,
        QTP_sys_densenet,
        horizon,
        references,
        solver,
    )

    ### start evaluate densenet L MPC implementation ###
    @test typeof(modeler_mpc) == JuMP.Model
    @test JuMP.solver_name(modeler_mpc) == "SCIP"
    @test length(JuMP.object_dictionary(modeler_mpc)) == 6
    @test size(JuMP.object_dictionary(modeler_mpc)[:u_reference]) ==
          (QTP_sys_densenet.inputdim, horizon)
    @test size(JuMP.object_dictionary(modeler_mpc)[:u]) ==
          (QTP_sys_densenet.inputdim, horizon)
    @test size(JuMP.object_dictionary(modeler_mpc)[:e_u]) ==
          (QTP_sys_densenet.inputdim, horizon)
    @test size(JuMP.object_dictionary(modeler_mpc)[:x_reference]) ==
          (QTP_sys_densenet.statedim, horizon + 1)
    @test size(JuMP.object_dictionary(modeler_mpc)[:x]) ==
          (QTP_sys_densenet.statedim, horizon + 1)
    @test size(JuMP.object_dictionary(modeler_mpc)[:e_x]) ==
          (QTP_sys_densenet.statedim, horizon + 1)
    @test JuMP.objective_function(modeler_mpc) == 0
    @test JuMP.objective_function_type(modeler_mpc) == JuMP.AffExpr
    @test JuMP.list_of_constraint_types(modeler_mpc) == [
        (AffExpr, MathOptInterface.EqualTo{Float64}),
        (AffExpr, MathOptInterface.LessThan{Float64}),
        (VariableRef, MathOptInterface.EqualTo{Float64}),
    ]

end

@testset "DenseNet model non linear modeler MPC test" begin

    #get the densenet model to design the mpc controler
    densenet_machine = machine("./models_saved/densenet_train_result.jls")

    #extract best model from the all trained models
    mlj_densenet = fitted_params(fitted_params(densenet_machine).machine).best_model
    f_densenet =
        fitted_params(fitted_params(densenet_machine).machine).best_fitted_params[1]
    model_densenet = mlj_densenet.builder

    method = AutomationLabsModelPredictiveControl.NonLinearProgramming()

    #system definition with Mathematical systems
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
    terminal_ingredients = false
    max_time = 5
    x = [0.65, 0.65, 0.65, 0.65] * ones(1, horizon + 1)
    u = [1.2, 1.2] * ones(1, horizon)

    references = AutomationLabsModelPredictiveControl.ReferencesStateInput(x, u)

    solver = auto_solver_def()

    modeler_mpc = _model_predictive_control_modeler_implementation(
        method,
        model_densenet,
        QTP_sys_densenet,
        horizon,
        references,
        solver,
    )

    ### start evaluate densenet L MPC implementation ###
    @test typeof(modeler_mpc) == JuMP.Model
    @test JuMP.solver_name(modeler_mpc) == "Ipopt"
    @test length(JuMP.object_dictionary(modeler_mpc)) == 10
    @test size(JuMP.object_dictionary(modeler_mpc)[:u_reference]) ==
          (QTP_sys_densenet.inputdim, horizon)
    @test size(JuMP.object_dictionary(modeler_mpc)[:u]) ==
          (QTP_sys_densenet.inputdim, horizon)
    @test size(JuMP.object_dictionary(modeler_mpc)[:e_u]) ==
          (QTP_sys_densenet.inputdim, horizon)
    @test size(JuMP.object_dictionary(modeler_mpc)[:x_reference]) ==
          (QTP_sys_densenet.statedim, horizon + 1)
    @test size(JuMP.object_dictionary(modeler_mpc)[:x]) ==
          (QTP_sys_densenet.statedim, horizon + 1)
    @test size(JuMP.object_dictionary(modeler_mpc)[:e_x]) ==
          (QTP_sys_densenet.statedim, horizon + 1)
    @test JuMP.objective_function(modeler_mpc) == 0
    @test JuMP.objective_function_type(modeler_mpc) == JuMP.AffExpr
    @test JuMP.list_of_constraint_types(modeler_mpc) == [
        (AffExpr, MathOptInterface.EqualTo{Float64}),
        (AffExpr, MathOptInterface.LessThan{Float64}),
        (VariableRef, MathOptInterface.EqualTo{Float64}),
    ]

end

@testset "DenseNet model milp modeler MPC test" begin

    #get the densenet model to design the mpc controler
    densenet_machine = machine("./models_saved/densenet_train_result.jls")

    #extract best model from the all trained models
    mlj_densenet = fitted_params(fitted_params(densenet_machine).machine).best_model
    f_densenet =
        fitted_params(fitted_params(densenet_machine).machine).best_fitted_params[1]
    model_densenet = mlj_densenet.builder

    method = AutomationLabsModelPredictiveControl.MixedIntegerLinearProgramming()

    #system definition with Mathematical systems
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
    terminal_ingredients = false
    max_time = 5
    x = [0.65, 0.65, 0.65, 0.65] * ones(1, horizon + 1)
    u = [1.2, 1.2] * ones(1, horizon)

    references = AutomationLabsModelPredictiveControl.ReferencesStateInput(x, u)

    solver = auto_solver_def()

    modeler_mpc = _model_predictive_control_modeler_implementation(
        method,
        model_densenet,
        QTP_sys_densenet,
        horizon,
        references,
        solver,
    )

    ### start evaluate densenet L MPC implementation ###
    @test typeof(modeler_mpc) == JuMP.Model
    @test JuMP.solver_name(modeler_mpc) == "SCIP"
    @test length(JuMP.object_dictionary(modeler_mpc)) == 13
    @test size(JuMP.object_dictionary(modeler_mpc)[:u_reference]) ==
          (QTP_sys_densenet.inputdim, horizon)
    @test size(JuMP.object_dictionary(modeler_mpc)[:u]) ==
          (QTP_sys_densenet.inputdim, horizon)
    @test size(JuMP.object_dictionary(modeler_mpc)[:e_u]) ==
          (QTP_sys_densenet.inputdim, horizon)
    @test size(JuMP.object_dictionary(modeler_mpc)[:x_reference]) ==
          (QTP_sys_densenet.statedim, horizon + 1)
    @test size(JuMP.object_dictionary(modeler_mpc)[:x]) ==
          (QTP_sys_densenet.statedim, horizon + 1)
    @test size(JuMP.object_dictionary(modeler_mpc)[:e_x]) ==
          (QTP_sys_densenet.statedim, horizon + 1)
    @test JuMP.objective_function(modeler_mpc) == 0
    @test JuMP.objective_function_type(modeler_mpc) == JuMP.AffExpr
    @test JuMP.list_of_constraint_types(modeler_mpc) == [
        (AffExpr, MathOptInterface.EqualTo{Float64}),
        (AffExpr, MathOptInterface.GreaterThan{Float64}),
        (AffExpr, MathOptInterface.LessThan{Float64}),
        (VariableRef, MathOptInterface.EqualTo{Float64}),
        (VariableRef, MathOptInterface.ZeroOne),
    ]

end


@testset "Icnn model linear modeler MPC test" begin

    #get the icnn model to design the mpc controler
    icnn_machine = machine("./models_saved/icnn_train_result.jls")

    #extract best model from the all trained models
    mlj_icnn = fitted_params(fitted_params(icnn_machine).machine).best_model
    f_icnn = fitted_params(fitted_params(icnn_machine).machine).best_fitted_params[1]
    model_icnn = mlj_icnn.builder

    method = AutomationLabsModelPredictiveControl.LinearProgramming()

    #system definition with MAthematical systems
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
    terminal_ingredients = false
    max_time = 5
    x = [0.65, 0.65, 0.65, 0.65] * ones(1, horizon + 1)
    u = [1.2, 1.2] * ones(1, horizon)

    references = AutomationLabsModelPredictiveControl.ReferencesStateInput(x, u)

    solver = auto_solver_def()

    modeler_mpc = _model_predictive_control_modeler_implementation(
        method,
        model_icnn,
        QTP_sys_icnn,
        horizon,
        references,
        solver,
    )

    ### start evaluate icnn L MPC implementation ###
    @test typeof(modeler_mpc) == JuMP.Model
    @test JuMP.solver_name(modeler_mpc) == "SCIP"
    @test length(JuMP.object_dictionary(modeler_mpc)) == 6
    @test size(JuMP.object_dictionary(modeler_mpc)[:u_reference]) ==
          (QTP_sys_icnn.inputdim, horizon)
    @test size(JuMP.object_dictionary(modeler_mpc)[:u]) == (QTP_sys_icnn.inputdim, horizon)
    @test size(JuMP.object_dictionary(modeler_mpc)[:e_u]) ==
          (QTP_sys_icnn.inputdim, horizon)
    @test size(JuMP.object_dictionary(modeler_mpc)[:x_reference]) ==
          (QTP_sys_icnn.statedim, horizon + 1)
    @test size(JuMP.object_dictionary(modeler_mpc)[:x]) ==
          (QTP_sys_icnn.statedim, horizon + 1)
    @test size(JuMP.object_dictionary(modeler_mpc)[:e_x]) ==
          (QTP_sys_icnn.statedim, horizon + 1)
    @test JuMP.objective_function(modeler_mpc) == 0
    @test JuMP.objective_function_type(modeler_mpc) == JuMP.AffExpr
    @test JuMP.list_of_constraint_types(modeler_mpc) == [
        (AffExpr, MathOptInterface.EqualTo{Float64}),
        (AffExpr, MathOptInterface.LessThan{Float64}),
        (VariableRef, MathOptInterface.EqualTo{Float64}),
    ]

end

@testset "Icnn model non linear modeler MPC test" begin

    #get the icnn model to design the mpc controler
    icnn_machine = machine("./models_saved/icnn_train_result.jls")

    #extract best model from the all trained models
    mlj_icnn = fitted_params(fitted_params(icnn_machine).machine).best_model
    f_icnn = fitted_params(fitted_params(icnn_machine).machine).best_fitted_params[1]
    model_icnn = mlj_icnn.builder

    method = AutomationLabsModelPredictiveControl.NonLinearProgramming()

    #system definition with Mathematical systems
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
    terminal_ingredients = false
    max_time = 5
    x = [0.65, 0.65, 0.65, 0.65] * ones(1, horizon + 1)
    u = [1.2, 1.2] * ones(1, horizon)

    references = AutomationLabsModelPredictiveControl.ReferencesStateInput(x, u)

    solver = auto_solver_def()

    modeler_mpc = _model_predictive_control_modeler_implementation(
        method,
        model_icnn,
        QTP_sys_icnn,
        horizon,
        references,
        solver,
    )

    ### start evaluate icnn L MPC implementation ###
    @test typeof(modeler_mpc) == JuMP.Model
    @test JuMP.solver_name(modeler_mpc) == "Ipopt"
    @test length(JuMP.object_dictionary(modeler_mpc)) == 7
    @test size(JuMP.object_dictionary(modeler_mpc)[:u_reference]) ==
          (QTP_sys_icnn.inputdim, horizon)
    @test size(JuMP.object_dictionary(modeler_mpc)[:u]) == (QTP_sys_icnn.inputdim, horizon)
    @test size(JuMP.object_dictionary(modeler_mpc)[:e_u]) ==
          (QTP_sys_icnn.inputdim, horizon)
    @test size(JuMP.object_dictionary(modeler_mpc)[:x_reference]) ==
          (QTP_sys_icnn.statedim, horizon + 1)
    @test size(JuMP.object_dictionary(modeler_mpc)[:x]) ==
          (QTP_sys_icnn.statedim, horizon + 1)
    @test size(JuMP.object_dictionary(modeler_mpc)[:e_x]) ==
          (QTP_sys_icnn.statedim, horizon + 1)
    @test JuMP.objective_function(modeler_mpc) == 0
    @test JuMP.objective_function_type(modeler_mpc) == JuMP.AffExpr
    @test JuMP.list_of_constraint_types(modeler_mpc) == [
        (AffExpr, MathOptInterface.EqualTo{Float64}),
        (AffExpr, MathOptInterface.LessThan{Float64}),
        (VariableRef, MathOptInterface.EqualTo{Float64}),
    ]

end

@testset "Icnn model milp modeler MPC test" begin

    #get the icnn model to design the mpc controler
    icnn_machine = machine("./models_saved/icnn_train_result.jls")

    #extract best model from the all trained models
    mlj_icnn = fitted_params(fitted_params(icnn_machine).machine).best_model
    f_icnn = fitted_params(fitted_params(icnn_machine).machine).best_fitted_params[1]
    model_icnn = mlj_icnn.builder

    method = AutomationLabsModelPredictiveControl.MixedIntegerLinearProgramming()

    #system definition with Mathematical systems
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
    terminal_ingredients = false
    max_time = 5
    x = [0.65, 0.65, 0.65, 0.65] * ones(1, horizon + 1)
    u = [1.2, 1.2] * ones(1, horizon)

    references = AutomationLabsModelPredictiveControl.ReferencesStateInput(x, u)

    solver = auto_solver_def()

    modeler_mpc = _model_predictive_control_modeler_implementation(
        method,
        model_icnn,
        QTP_sys_icnn,
        horizon,
        references,
        solver,
    )

    ### start evaluate icnn L MPC implementation ###
    @test typeof(modeler_mpc) == JuMP.Model
    @test JuMP.solver_name(modeler_mpc) == "SCIP"
    @test length(JuMP.object_dictionary(modeler_mpc)) == 9
    @test size(JuMP.object_dictionary(modeler_mpc)[:u_reference]) ==
          (QTP_sys_icnn.inputdim, horizon)
    @test size(JuMP.object_dictionary(modeler_mpc)[:u]) == (QTP_sys_icnn.inputdim, horizon)
    @test size(JuMP.object_dictionary(modeler_mpc)[:e_u]) ==
          (QTP_sys_icnn.inputdim, horizon)
    @test size(JuMP.object_dictionary(modeler_mpc)[:x_reference]) ==
          (QTP_sys_icnn.statedim, horizon + 1)
    @test size(JuMP.object_dictionary(modeler_mpc)[:x]) ==
          (QTP_sys_icnn.statedim, horizon + 1)
    @test size(JuMP.object_dictionary(modeler_mpc)[:e_x]) ==
          (QTP_sys_icnn.statedim, horizon + 1)
    @test JuMP.objective_function(modeler_mpc) == 0
    @test JuMP.objective_function_type(modeler_mpc) == JuMP.AffExpr
    @test JuMP.list_of_constraint_types(modeler_mpc) == [
        (AffExpr, MathOptInterface.EqualTo{Float64}),
        (AffExpr, MathOptInterface.GreaterThan{Float64}),
        (AffExpr, MathOptInterface.LessThan{Float64}),
        (VariableRef, MathOptInterface.EqualTo{Float64}),
        (VariableRef, MathOptInterface.ZeroOne),
    ]

end

@testset "Rbf model linear modeler MPC test" begin


    #get the rbf model to design the mpc controler
    rbf_machine = machine("./models_saved/rbf_train_result.jls")

    #extract best model from the all trained models
    mlj_rbf = fitted_params(fitted_params(rbf_machine).machine).best_model
    f_rbf = fitted_params(fitted_params(rbf_machine).machine).best_fitted_params[1]
    model_rbf = mlj_rbf.builder

    method = AutomationLabsModelPredictiveControl.LinearProgramming()

    #system definition with MAthematical systems
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
    terminal_ingredients = false
    max_time = 5
    x = [0.65, 0.65, 0.65, 0.65] * ones(1, horizon + 1)
    u = [1.2, 1.2] * ones(1, horizon)

    references = AutomationLabsModelPredictiveControl.ReferencesStateInput(x, u)

    solver = auto_solver_def()

    modeler_mpc = _model_predictive_control_modeler_implementation(
        method,
        model_rbf,
        QTP_sys_rbf,
        horizon,
        references,
        solver,
    )

    ### start evaluate rbf L MPC implementation ###
    @test typeof(modeler_mpc) == JuMP.Model
    @test JuMP.solver_name(modeler_mpc) == "SCIP"
    @test length(JuMP.object_dictionary(modeler_mpc)) == 6
    @test size(JuMP.object_dictionary(modeler_mpc)[:u_reference]) ==
          (QTP_sys_rbf.inputdim, horizon)
    @test size(JuMP.object_dictionary(modeler_mpc)[:u]) == (QTP_sys_rbf.inputdim, horizon)
    @test size(JuMP.object_dictionary(modeler_mpc)[:e_u]) == (QTP_sys_rbf.inputdim, horizon)
    @test size(JuMP.object_dictionary(modeler_mpc)[:x_reference]) ==
          (QTP_sys_rbf.statedim, horizon + 1)
    @test size(JuMP.object_dictionary(modeler_mpc)[:x]) ==
          (QTP_sys_rbf.statedim, horizon + 1)
    @test size(JuMP.object_dictionary(modeler_mpc)[:e_x]) ==
          (QTP_sys_rbf.statedim, horizon + 1)
    @test JuMP.objective_function(modeler_mpc) == 0
    @test JuMP.objective_function_type(modeler_mpc) == JuMP.AffExpr
    @test JuMP.list_of_constraint_types(modeler_mpc) == [
        (AffExpr, MathOptInterface.EqualTo{Float64}),
        (AffExpr, MathOptInterface.LessThan{Float64}),
        (VariableRef, MathOptInterface.EqualTo{Float64}),
    ]

end

@testset "Rbf model non linear modeler MPC test" begin

    #get the rbf model to design the mpc controler
    rbf_machine = machine("./models_saved/rbf_train_result.jls")

    #extract best model from the all trained models
    mlj_rbf = fitted_params(fitted_params(rbf_machine).machine).best_model
    f_rbf = fitted_params(fitted_params(rbf_machine).machine).best_fitted_params[1]
    model_rbf = mlj_rbf.builder

    method = AutomationLabsModelPredictiveControl.NonLinearProgramming()

    #system definition with Mathematical systems
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
    terminal_ingredients = false
    max_time = 5
    x = [0.65, 0.65, 0.65, 0.65] * ones(1, horizon + 1)
    u = [1.2, 1.2] * ones(1, horizon)

    references = AutomationLabsModelPredictiveControl.ReferencesStateInput(x, u)

    solver = auto_solver_def()

    modeler_mpc = _model_predictive_control_modeler_implementation(
        method,
        model_rbf,
        QTP_sys_rbf,
        horizon,
        references,
        solver,
    )

    ### start evaluate rbf L MPC implementation ###
    @test typeof(modeler_mpc) == JuMP.Model
    @test JuMP.solver_name(modeler_mpc) == "Ipopt"
    @test length(JuMP.object_dictionary(modeler_mpc)) == 7
    @test size(JuMP.object_dictionary(modeler_mpc)[:u_reference]) ==
          (QTP_sys_rbf.inputdim, horizon)
    @test size(JuMP.object_dictionary(modeler_mpc)[:u]) == (QTP_sys_rbf.inputdim, horizon)
    @test size(JuMP.object_dictionary(modeler_mpc)[:e_u]) == (QTP_sys_rbf.inputdim, horizon)
    @test size(JuMP.object_dictionary(modeler_mpc)[:x_reference]) ==
          (QTP_sys_rbf.statedim, horizon + 1)
    @test size(JuMP.object_dictionary(modeler_mpc)[:x]) ==
          (QTP_sys_rbf.statedim, horizon + 1)
    @test size(JuMP.object_dictionary(modeler_mpc)[:e_x]) ==
          (QTP_sys_rbf.statedim, horizon + 1)
    @test JuMP.objective_function(modeler_mpc) == 0
    @test JuMP.objective_function_type(modeler_mpc) == JuMP.AffExpr
    @test JuMP.list_of_constraint_types(modeler_mpc) == [
        (AffExpr, MathOptInterface.EqualTo{Float64}),
        (AffExpr, MathOptInterface.LessThan{Float64}),
        (VariableRef, MathOptInterface.EqualTo{Float64}),
    ]

end

@testset "PolyNet model linear modeler MPC test" begin

    #get the polynet model to design the mpc controler
    polynet_machine = machine("./models_saved/polynet_train_result.jls")

    #extract best model from the all trained models
    mlj_polynet = fitted_params(fitted_params(polynet_machine).machine).best_model
    f_polynet = fitted_params(fitted_params(polynet_machine).machine).best_fitted_params[1]
    model_polynet = mlj_polynet.builder

    method = AutomationLabsModelPredictiveControl.LinearProgramming()

    #system definition with MAthematical systems
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
    terminal_ingredients = false
    max_time = 5
    x = [0.65, 0.65, 0.65, 0.65] * ones(1, horizon + 1)
    u = [1.2, 1.2] * ones(1, horizon)

    references = AutomationLabsModelPredictiveControl.ReferencesStateInput(x, u)

    solver = auto_solver_def()

    modeler_mpc = _model_predictive_control_modeler_implementation(
        method,
        model_polynet,
        QTP_sys_polynet,
        horizon,
        references,
        solver,
    )

    ### start evaluate polynet L MPC implementation ###
    @test typeof(modeler_mpc) == JuMP.Model
    @test JuMP.solver_name(modeler_mpc) == "SCIP"
    @test length(JuMP.object_dictionary(modeler_mpc)) == 6
    @test size(JuMP.object_dictionary(modeler_mpc)[:u_reference]) ==
          (QTP_sys_polynet.inputdim, horizon)
    @test size(JuMP.object_dictionary(modeler_mpc)[:u]) ==
          (QTP_sys_polynet.inputdim, horizon)
    @test size(JuMP.object_dictionary(modeler_mpc)[:e_u]) ==
          (QTP_sys_polynet.inputdim, horizon)
    @test size(JuMP.object_dictionary(modeler_mpc)[:x_reference]) ==
          (QTP_sys_polynet.statedim, horizon + 1)
    @test size(JuMP.object_dictionary(modeler_mpc)[:x]) ==
          (QTP_sys_polynet.statedim, horizon + 1)
    @test size(JuMP.object_dictionary(modeler_mpc)[:e_x]) ==
          (QTP_sys_polynet.statedim, horizon + 1)
    @test JuMP.objective_function(modeler_mpc) == 0
    @test JuMP.objective_function_type(modeler_mpc) == JuMP.AffExpr
    @test JuMP.list_of_constraint_types(modeler_mpc) == [
        (AffExpr, MathOptInterface.EqualTo{Float64}),
        (AffExpr, MathOptInterface.LessThan{Float64}),
        (VariableRef, MathOptInterface.EqualTo{Float64}),
    ]

end

@testset "PolyNet model non linear modeler MPC test" begin

    #get the polynet model to design the mpc controler
    polynet_machine = machine("./models_saved/polynet_train_result.jls")

    #extract best model from the all trained models
    mlj_polynet = fitted_params(fitted_params(polynet_machine).machine).best_model
    f_polynet = fitted_params(fitted_params(polynet_machine).machine).best_fitted_params[1]
    model_polynet = mlj_polynet.builder

    method = AutomationLabsModelPredictiveControl.NonLinearProgramming()

    #system definition with Mathematical systems
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
    terminal_ingredients = false
    max_time = 5
    x = [0.65, 0.65, 0.65, 0.65] * ones(1, horizon + 1)
    u = [1.2, 1.2] * ones(1, horizon)

    references = AutomationLabsModelPredictiveControl.ReferencesStateInput(x, u)

    solver = auto_solver_def()

    modeler_mpc = _model_predictive_control_modeler_implementation(
        method,
        model_polynet,
        QTP_sys_polynet,
        horizon,
        references,
        solver,
    )

    ### start evaluate polynet L MPC implementation ###
    @test typeof(modeler_mpc) == JuMP.Model
    @test JuMP.solver_name(modeler_mpc) == "Ipopt"
    @test length(JuMP.object_dictionary(modeler_mpc)) == 8
    @test size(JuMP.object_dictionary(modeler_mpc)[:u_reference]) ==
          (QTP_sys_polynet.inputdim, horizon)
    @test size(JuMP.object_dictionary(modeler_mpc)[:u]) ==
          (QTP_sys_polynet.inputdim, horizon)
    @test size(JuMP.object_dictionary(modeler_mpc)[:e_u]) ==
          (QTP_sys_polynet.inputdim, horizon)
    @test size(JuMP.object_dictionary(modeler_mpc)[:x_reference]) ==
          (QTP_sys_polynet.statedim, horizon + 1)
    @test size(JuMP.object_dictionary(modeler_mpc)[:x]) ==
          (QTP_sys_polynet.statedim, horizon + 1)
    @test size(JuMP.object_dictionary(modeler_mpc)[:e_x]) ==
          (QTP_sys_polynet.statedim, horizon + 1)
    @test JuMP.objective_function(modeler_mpc) == 0
    @test JuMP.objective_function_type(modeler_mpc) == JuMP.AffExpr
    @test JuMP.list_of_constraint_types(modeler_mpc) == [
        (AffExpr, MathOptInterface.EqualTo{Float64}),
        (AffExpr, MathOptInterface.LessThan{Float64}),
        (VariableRef, MathOptInterface.EqualTo{Float64}),
    ]

end

@testset "linear_regressor model linear modeler MPC test" begin

    #get the linear_regressor model to design the mpc controler
    linear_regressor_machine = machine("./models_saved/linear_regressor_train_result.jls")

    method = AutomationLabsModelPredictiveControl.LinearProgramming()

    #system definition with MAthematical systems
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
    terminal_ingredients = false
    max_time = 5
    x = [0.65, 0.65, 0.65, 0.65] * ones(1, horizon + 1)
    u = [1.2, 1.2] * ones(1, horizon)

    references = AutomationLabsModelPredictiveControl.ReferencesStateInput(x, u)

    solver = auto_solver_def()

    modeler_mpc = _model_predictive_control_modeler_implementation(
        method,
        QTP_sys_linear_regressor,
        horizon,
        references,
        solver,
    )

    ### start evaluate linear_regressor L MPC implementation ###
    @test typeof(modeler_mpc) == JuMP.Model
    @test JuMP.solver_name(modeler_mpc) == "SCIP"
    @test length(JuMP.object_dictionary(modeler_mpc)) == 6
    @test size(JuMP.object_dictionary(modeler_mpc)[:u_reference]) ==
          (size(QTP_sys_linear_regressor.B, 2), horizon)
    @test size(JuMP.object_dictionary(modeler_mpc)[:u]) ==
          (size(QTP_sys_linear_regressor.B, 2), horizon)
    @test size(JuMP.object_dictionary(modeler_mpc)[:e_u]) ==
          (size(QTP_sys_linear_regressor.B, 2), horizon)
    @test size(JuMP.object_dictionary(modeler_mpc)[:x_reference]) ==
          (size(QTP_sys_linear_regressor.A, 1), horizon + 1)
    @test size(JuMP.object_dictionary(modeler_mpc)[:x]) ==
          (size(QTP_sys_linear_regressor.A, 1), horizon + 1)
    @test size(JuMP.object_dictionary(modeler_mpc)[:e_x]) ==
          (size(QTP_sys_linear_regressor.A, 1), horizon + 1)
    @test JuMP.objective_function(modeler_mpc) == 0
    @test JuMP.objective_function_type(modeler_mpc) == JuMP.AffExpr
    @test JuMP.list_of_constraint_types(modeler_mpc) == [
        (AffExpr, MathOptInterface.EqualTo{Float64}),
        (AffExpr, MathOptInterface.LessThan{Float64}),
        (VariableRef, MathOptInterface.EqualTo{Float64}),
    ]

end

@testset "NeuralODE model linear modeler MPC test" begin

    #get the neuralnetODE_type1 model to design the mpc controler
    neuralnetODE_type1_machine =
        machine("./models_saved/NeuralODE_train_result.jls")

    #extract best model from the all trained models
    mlj_neuralnetODE_type1 =
        fitted_params(fitted_params(neuralnetODE_type1_machine).machine).best_model
    f_neuralnetODE_type1 =
        fitted_params(fitted_params(neuralnetODE_type1_machine).machine).best_fitted_params[1]
    model_neuralnetODE_type1 = mlj_neuralnetODE_type1.builder

    method = AutomationLabsModelPredictiveControl.LinearProgramming()

    #system definition with MAthematical systems
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

    QTP_sys_neuralnetODE_type1 =
        MathematicalSystems.ConstrainedBlackBoxControlDiscreteSystem(
            f_neuralnetODE_type1,
            4,
            2,
            x_cons,
            u_cons,
        )

    #MPC design parameters
    horizon = 15
    sample_time = 5
    terminal_ingredients = false
    max_time = 5
    x = [0.65, 0.65, 0.65, 0.65] * ones(1, horizon + 1)
    u = [1.2, 1.2] * ones(1, horizon)

    references = AutomationLabsModelPredictiveControl.ReferencesStateInput(x, u)

    solver = auto_solver_def()

    modeler_mpc = _model_predictive_control_modeler_implementation(
        method,
        model_neuralnetODE_type1,
        QTP_sys_neuralnetODE_type1,
        horizon,
        references,
        solver,
    )

    ### start evaluate neuralnetODE_type1 L MPC implementation ###
    @test typeof(modeler_mpc) == JuMP.Model
    @test JuMP.solver_name(modeler_mpc) == "SCIP"
    @test length(JuMP.object_dictionary(modeler_mpc)) == 6
    @test size(JuMP.object_dictionary(modeler_mpc)[:u_reference]) ==
          (QTP_sys_neuralnetODE_type1.inputdim, horizon)
    @test size(JuMP.object_dictionary(modeler_mpc)[:u]) ==
          (QTP_sys_neuralnetODE_type1.inputdim, horizon)
    @test size(JuMP.object_dictionary(modeler_mpc)[:e_u]) ==
          (QTP_sys_neuralnetODE_type1.inputdim, horizon)
    @test size(JuMP.object_dictionary(modeler_mpc)[:x_reference]) ==
          (QTP_sys_neuralnetODE_type1.statedim, horizon + 1)
    @test size(JuMP.object_dictionary(modeler_mpc)[:x]) ==
          (QTP_sys_neuralnetODE_type1.statedim, horizon + 1)
    @test size(JuMP.object_dictionary(modeler_mpc)[:e_x]) ==
          (QTP_sys_neuralnetODE_type1.statedim, horizon + 1)
    @test JuMP.objective_function(modeler_mpc) == 0
    @test JuMP.objective_function_type(modeler_mpc) == JuMP.AffExpr
    @test JuMP.list_of_constraint_types(modeler_mpc) == [
        (AffExpr, MathOptInterface.EqualTo{Float64}),
        (AffExpr, MathOptInterface.LessThan{Float64}),
        (VariableRef, MathOptInterface.EqualTo{Float64}),
    ]

end

@testset "Rknn1 model linear modeler MPC test" begin

    #get the neuralnetODE_type1 model to design the mpc controler
    rknn1_machine =
        machine("./models_saved/rknn1_train_result.jls")

    #extract best model from the all trained models
    mlj_neuralnetODE_type1 =
        fitted_params(fitted_params(rknn1_machine).machine).best_model
    f_neuralnetODE_type1 =
        fitted_params(fitted_params(rknn1_machine).machine).best_fitted_params[1]
    model_rknn1 = mlj_neuralnetODE_type1.builder

    method = AutomationLabsModelPredictiveControl.LinearProgramming()

    #system definition with MAthematical systems
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

    QTP_sys_neuralnetODE_type1 =
        MathematicalSystems.ConstrainedBlackBoxControlDiscreteSystem(
            f_neuralnetODE_type1,
            4,
            2,
            x_cons,
            u_cons,
        )

    #MPC design parameters
    horizon = 15
    sample_time = 5
    terminal_ingredients = false
    max_time = 5
    x = [0.65, 0.65, 0.65, 0.65] * ones(1, horizon + 1)
    u = [1.2, 1.2] * ones(1, horizon)

    references = AutomationLabsModelPredictiveControl.ReferencesStateInput(x, u)

    solver = auto_solver_def()

    modeler_mpc = _model_predictive_control_modeler_implementation(
        method,
        model_rknn1,
        QTP_sys_neuralnetODE_type1,
        horizon,
        references,
        solver,
    )


    ### start evaluate neuralnetODE_type1 L MPC implementation ###
    @test typeof(modeler_mpc) == JuMP.Model
    @test JuMP.solver_name(modeler_mpc) == "SCIP"
    @test length(JuMP.object_dictionary(modeler_mpc)) == 6
    @test size(JuMP.object_dictionary(modeler_mpc)[:u_reference]) ==
          (QTP_sys_neuralnetODE_type1.inputdim, horizon)
    @test size(JuMP.object_dictionary(modeler_mpc)[:u]) ==
          (QTP_sys_neuralnetODE_type1.inputdim, horizon)
    @test size(JuMP.object_dictionary(modeler_mpc)[:e_u]) ==
          (QTP_sys_neuralnetODE_type1.inputdim, horizon)
    @test size(JuMP.object_dictionary(modeler_mpc)[:x_reference]) ==
          (QTP_sys_neuralnetODE_type1.statedim, horizon + 1)
    @test size(JuMP.object_dictionary(modeler_mpc)[:x]) ==
          (QTP_sys_neuralnetODE_type1.statedim, horizon + 1)
    @test size(JuMP.object_dictionary(modeler_mpc)[:e_x]) ==
          (QTP_sys_neuralnetODE_type1.statedim, horizon + 1)
    @test JuMP.objective_function(modeler_mpc) == 0
    @test JuMP.objective_function_type(modeler_mpc) == JuMP.AffExpr
    @test JuMP.list_of_constraint_types(modeler_mpc) == [
        (AffExpr, MathOptInterface.EqualTo{Float64}),
        (AffExpr, MathOptInterface.LessThan{Float64}),
        (VariableRef, MathOptInterface.EqualTo{Float64}),
    ]

end

@testset "Rknn2 model linear modeler MPC test" begin

    #get the neuralnetODE_type1 model to design the mpc controler
    rknn2_machine =
        machine("./models_saved/rknn2_train_result.jls")

    #extract best model from the all trained models
    mlj_neuralnetODE_type1 =
        fitted_params(fitted_params(rknn2_machine).machine).best_model
    f_neuralnetODE_type1 =
        fitted_params(fitted_params(rknn2_machine).machine).best_fitted_params[1]
    model_rknn2 = mlj_neuralnetODE_type1.builder

    method = AutomationLabsModelPredictiveControl.LinearProgramming()

    #system definition with MAthematical systems
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

    QTP_sys_neuralnetODE_type1 =
        MathematicalSystems.ConstrainedBlackBoxControlDiscreteSystem(
            f_neuralnetODE_type1,
            4,
            2,
            x_cons,
            u_cons,
        )

    #MPC design parameters
    horizon = 15
    sample_time = 5
    terminal_ingredients = false
    max_time = 5
    x = [0.65, 0.65, 0.65, 0.65] * ones(1, horizon + 1)
    u = [1.2, 1.2] * ones(1, horizon)

    references = AutomationLabsModelPredictiveControl.ReferencesStateInput(x, u)

    solver = auto_solver_def()

    modeler_mpc = _model_predictive_control_modeler_implementation(
        method,
        model_rknn2,
        QTP_sys_neuralnetODE_type1,
        horizon,
        references,
        solver,
    )

    ### start evaluate neuralnetODE_type1 L MPC implementation ###
    @test typeof(modeler_mpc) == JuMP.Model
    @test JuMP.solver_name(modeler_mpc) == "SCIP"
    @test length(JuMP.object_dictionary(modeler_mpc)) == 6
    @test size(JuMP.object_dictionary(modeler_mpc)[:u_reference]) ==
          (QTP_sys_neuralnetODE_type1.inputdim, horizon)
    @test size(JuMP.object_dictionary(modeler_mpc)[:u]) ==
          (QTP_sys_neuralnetODE_type1.inputdim, horizon)
    @test size(JuMP.object_dictionary(modeler_mpc)[:e_u]) ==
          (QTP_sys_neuralnetODE_type1.inputdim, horizon)
    @test size(JuMP.object_dictionary(modeler_mpc)[:x_reference]) ==
          (QTP_sys_neuralnetODE_type1.statedim, horizon + 1)
    @test size(JuMP.object_dictionary(modeler_mpc)[:x]) ==
          (QTP_sys_neuralnetODE_type1.statedim, horizon + 1)
    @test size(JuMP.object_dictionary(modeler_mpc)[:e_x]) ==
          (QTP_sys_neuralnetODE_type1.statedim, horizon + 1)
    @test JuMP.objective_function(modeler_mpc) == 0
    @test JuMP.objective_function_type(modeler_mpc) == JuMP.AffExpr
    @test JuMP.list_of_constraint_types(modeler_mpc) == [
        (AffExpr, MathOptInterface.EqualTo{Float64}),
        (AffExpr, MathOptInterface.LessThan{Float64}),
        (VariableRef, MathOptInterface.EqualTo{Float64}),
    ]

end

@testset "Rknn4 model linear modeler MPC test" begin

    #get the neuralnetODE_type1 model to design the mpc controler
    rknn4_machine =
        machine("./models_saved/rknn4_train_result.jls")

    #extract best model from the all trained models
    mlj_neuralnetODE_type1 =
        fitted_params(fitted_params(rknn4_machine).machine).best_model
    f_neuralnetODE_type1 =
        fitted_params(fitted_params(rknn4_machine).machine).best_fitted_params[1]
    model_rknn4 = mlj_neuralnetODE_type1.builder

    method = AutomationLabsModelPredictiveControl.LinearProgramming()

    #system definition with MAthematical systems
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

    QTP_sys_neuralnetODE_type1 =
        MathematicalSystems.ConstrainedBlackBoxControlDiscreteSystem(
            f_neuralnetODE_type1,
            4,
            2,
            x_cons,
            u_cons,
        )

    #MPC design parameters
    horizon = 15
    sample_time = 5
    terminal_ingredients = false
    max_time = 5
    x = [0.65, 0.65, 0.65, 0.65] * ones(1, horizon + 1)
    u = [1.2, 1.2] * ones(1, horizon)

    references = AutomationLabsModelPredictiveControl.ReferencesStateInput(x, u)

    solver = auto_solver_def()

    modeler_mpc = _model_predictive_control_modeler_implementation(
        method,
        model_rknn4,
        QTP_sys_neuralnetODE_type1,
        horizon,
        references,
        solver,
    )

    ### start evaluate neuralnetODE_type1 L MPC implementation ###
    @test typeof(modeler_mpc) == JuMP.Model
    @test JuMP.solver_name(modeler_mpc) == "SCIP"
    @test length(JuMP.object_dictionary(modeler_mpc)) == 6
    @test size(JuMP.object_dictionary(modeler_mpc)[:u_reference]) ==
          (QTP_sys_neuralnetODE_type1.inputdim, horizon)
    @test size(JuMP.object_dictionary(modeler_mpc)[:u]) ==
          (QTP_sys_neuralnetODE_type1.inputdim, horizon)
    @test size(JuMP.object_dictionary(modeler_mpc)[:e_u]) ==
          (QTP_sys_neuralnetODE_type1.inputdim, horizon)
    @test size(JuMP.object_dictionary(modeler_mpc)[:x_reference]) ==
          (QTP_sys_neuralnetODE_type1.statedim, horizon + 1)
    @test size(JuMP.object_dictionary(modeler_mpc)[:x]) ==
          (QTP_sys_neuralnetODE_type1.statedim, horizon + 1)
    @test size(JuMP.object_dictionary(modeler_mpc)[:e_x]) ==
          (QTP_sys_neuralnetODE_type1.statedim, horizon + 1)
    @test JuMP.objective_function(modeler_mpc) == 0
    @test JuMP.objective_function_type(modeler_mpc) == JuMP.AffExpr
    @test JuMP.list_of_constraint_types(modeler_mpc) == [
        (AffExpr, MathOptInterface.EqualTo{Float64}),
        (AffExpr, MathOptInterface.LessThan{Float64}),
        (VariableRef, MathOptInterface.EqualTo{Float64}),
    ]

end

#=

@testset "Non linear linear MPC test" begin

    # QTP 

    function QTP(du, u, p, t)
        #no trainable parameters
        #constant parameters
        S = 0.06
        gamma_a = 0.3
        gamma_b = 0.4
        g = 9.81

        a1 = 1.34e-4
        a2 = 1.51e-4
        a3 = 9.27e-5
        a4 = 8.82e-5

        #states 
        x1 = u[1]
        x2 = u[2]
        x3 = u[3]
        x4 = u[4]
        qa = u[5]
        qb = u[6]

        du[1] =
            -a1 / S * sqrt(2 * g * x1) +
            a3 / S * sqrt(2 * g * x3) +
            gamma_a / (S * 3600) * qa
        du[2] =
            -a2 / S * sqrt(2 * g * x2) +
            a4 / S * sqrt(2 * g * x4) +
            gamma_b / (S * 3600) * qb
        du[3] = -a3 / S * sqrt(2 * g * x3) + (1 - gamma_b) / (S * 3600) * qb
        du[4] = -a4 / S * sqrt(2 * g * x4) + (1 - gamma_a) / (S * 3600) * qa
    end

    u0 = 

    tspan = 

    init_t_p = [1.1e-4, 1.2e-4, 9e-5, 9e-5]
    #init_t_p [1.34e-4, 1.51e-4, 9.27e-5, 8.82e-5]
    nbr_states = 4
    nbr_inputs = 2
    sample_time = 5.0
    maximum_time = Dates.Minute(15)

    #get the neuralnetODE_type1 model to design the mpc controler
    neuralnetODE_type1_machine = machine("./models_saved/neuralnetODE_type1_train_result.jls")

    #extract best model from the all trained models
    mlj_neuralnetODE_type1 = fitted_params(fitted_params(neuralnetODE_type1_machine).machine).best_model
    f_neuralnetODE_type1 = fitted_params(fitted_params(neuralnetODE_type1_machine).machine).best_fitted_params[1]
    model_neuralnetODE_type1 = mlj_neuralnetODE_type1.builder

    method = AutomationLabsModelPredictiveControl.LinearProgramming()

    #system definition with MAthematical systems
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

    QTP_sys_neuralnetODE_type1 = MathematicalSystems.ConstrainedBlackBoxControlDiscreteSystem(
        f,
        4,
        2,
        x_cons,
        u_cons,
    )

    #MPC design parameters
    horizon = 15
    sample_time = 5
    terminal_ingredients = false
    max_time = 5
    x = [0.65, 0.65, 0.65, 0.65] * ones(1, horizon + 1)
    u = [1.2, 1.2] * ones(1, horizon)

    references = AutomationLabsModelPredictiveControl.ReferencesStateInput(x, u)

    solver = auto_solver_def()

    modeler_mpc = _model_predictive_control_modeler_implementation(
        method,
        model_neuralnetODE_type1,
        QTP_sys_neuralnetODE_type1,
        horizon,
        references,
        solver,
    )

    ### start evaluate neuralnetODE_type1 L MPC implementation ###
    @test typeof(modeler_mpc) == JuMP.Model
    @test JuMP.solver_name(modeler_mpc) == "SCIP"
    @test length(JuMP.object_dictionary(modeler_mpc)) == 6
    @test size(JuMP.object_dictionary(modeler_mpc)[:u_reference]) ==
          (QTP_sys_neuralnetODE_type1.inputdim, horizon)
    @test size(JuMP.object_dictionary(modeler_mpc)[:u]) == (QTP_sys_neuralnetODE_type1.inputdim, horizon)
    @test size(JuMP.object_dictionary(modeler_mpc)[:e_u]) == (QTP_sys_neuralnetODE_type1.inputdim, horizon)
    @test size(JuMP.object_dictionary(modeler_mpc)[:x_reference]) ==
          (QTP_sys_neuralnetODE_type1.statedim, horizon + 1)
    @test size(JuMP.object_dictionary(modeler_mpc)[:x]) ==
          (QTP_sys_neuralnetODE_type1.statedim, horizon + 1)
    @test size(JuMP.object_dictionary(modeler_mpc)[:e_x]) ==
          (QTP_sys_neuralnetODE_type1.statedim, horizon + 1)
    @test JuMP.objective_function(modeler_mpc) == 0
    @test JuMP.objective_function_type(modeler_mpc) == JuMP.AffExpr
    @test JuMP.list_of_constraint_types(modeler_mpc) == [
        (AffExpr, MathOptInterface.EqualTo{Float64}),
        (AffExpr, MathOptInterface.LessThan{Float64}),
        (VariableRef, MathOptInterface.EqualTo{Float64}),
    ]

end

=#
end
