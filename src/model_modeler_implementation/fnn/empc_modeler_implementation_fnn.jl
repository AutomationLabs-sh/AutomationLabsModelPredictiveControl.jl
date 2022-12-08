# Copyright (c) 2022: Pierre Blaud and contributors
########################################################
# This Source Code Form is subject to the terms of the #
# Mozilla Public License, v. 2.0. If a copy of the MPL #
# was not distributed with this file,  				   #
# You can obtain one at https://mozilla.org/MPL/2.0/.  #
########################################################

### Linear eMPC implementation method with fnn ###

"""
    _economic_model_predictive_control_modeler_implementation
Modeler implementation of a Model Predictive Control.

** Required fields **
* `method`: method which is multipled dispatched type. Type are LinearProgramming, TakagiSugeno, NonLinearProgramming and MixedIntegerLinearProgramming.
* `model_mlj`: model of the dynamical system from AutomationLabsIdentification package.
* `system`: system from MethematicalSystems.
* `horizon` : model predictive control horizon parameter.
* `references` : model predictive control references.
* `solver`: model predictive control solver selection.
"""
function _economic_model_predictive_control_modeler_implementation(
    method::LinearProgramming,
    model_mlj::AutomationLabsIdentification.Fnn,
    system::MathematicalSystems.ConstrainedBlackBoxControlDiscreteSystem,
    horizon::Int,
    reference::ReferencesStateInput,
    solver::AbstractSolvers,
)

    #get constraints of the dynamical system
    x_hyperrectangle = LazySets.vertices_list(system.X)
    u_hyperrectangle = LazySets.vertices_list(system.U)

    x_constraints = hcat(x_hyperrectangle[end], x_hyperrectangle[begin])
    u_constraints = hcat(u_hyperrectangle[end], u_hyperrectangle[begin])

    #get A and B matrices from state space
    state_reference = reference.x[:, begin] #state at first reference to compute the jacobian
    input_reference = reference.u[:, begin] #input at first reference to compute the jacobian
    A, B = system_linearization(system, state_reference, input_reference)

    #the jump model is designed       
    model_empc = _JuMP_model_definition(method, solver)

    #decision variables are set in JuMP optimisation model
    JuMP.@variables(model_empc, begin
        x[1:system.statedim, 1:horizon+1]                  #state variable
        u[1:system.inputdim, 1:horizon]             #input variable
    end)

    #Linear model into JuMP    
    for k = 1:1:horizon
        JuMP.@constraint(model_empc, x[:, k+1] .== A * x[:, k] + B * u[:, k]) #output layer
    end

    #States constraints
    for k = 1:1:horizon+1
        for i = 1:1:system.statedim
            JuMP.@constraint(model_empc, x[i, k] <= x_constraints[i, 2])
            JuMP.@constraint(model_empc, x_constraints[i, 1] <= x[i, k])
        end
    end

    #Inputs constraints
    for k = 1:1:horizon
        for i = 1:1:system.inputdim
            JuMP.@constraint(model_empc, u[i, k] <= u_constraints[i, 2])
            JuMP.@constraint(model_empc, u_constraints[i, 1] <= u[i, k])
        end
    end

    return model_empc

end

### Non linear eMPC implementation method with Fnn ###
function _economic_model_predictive_control_modeler_implementation(
    method::NonLinearProgramming,
    model_mlj::AutomationLabsIdentification.Fnn,
    system::MathematicalSystems.ConstrainedBlackBoxControlDiscreteSystem,
    horizon::Int,
    reference::ReferencesStateInput,
    solver::AbstractSolvers,
)

    #get constraints
    x_hyperrectangle = LazySets.vertices_list(system.X)
    u_hyperrectangle = LazySets.vertices_list(system.U)

    x_constraints = hcat(x_hyperrectangle[end], x_hyperrectangle[begin])
    u_constraints = hcat(u_hyperrectangle[end], u_hyperrectangle[begin])

    #the model is designed
    model_empc = _JuMP_model_definition(method, solver)

    #get information from the state, input and neural network and extract neural weughts 
    nn_weights = Flux.params(system.f)
    nbr_inputs = size(nn_weights[1], 2)
    nbr_neurons = size(nn_weights[1], 1)
    nbr_outputs = size(nn_weights[length(nn_weights)], 1)
    nbr_hidden = trunc(Int, (length(nn_weights) - 2) / 2)
    nbr_inputs_control = nbr_inputs - nbr_outputs
    nbr_states = nbr_outputs

    #get neural weights
    W_layer = [] #memory allocation need 3D array multi dimention
    B_layer = [] #memory allocation, need 3D array multi dimention

    push!(W_layer, nn_weights[1])

    for i = 2:2:length(nn_weights)-1
        push!(W_layer, nn_weights[i])
        push!(B_layer, nn_weights[i+1])
    end

    push!(W_layer, nn_weights[length(nn_weights)])

    #decision variables are set in JuMP optimisation model
    JuMP.@variables(model_empc, begin
        x[1:nbr_states, 1:horizon+1]                  #state variable
        u[1:nbr_inputs_control, 1:horizon]             #input variable
        y[1:nbr_neurons, 1:nbr_hidden+1, 1:horizon]   #output hidden neuron network
    end)

    #get the activation function of the Fnn through NNlib
    f_activation = get_activation_function(model_mlj, system)
    JuMP.register(model_empc, :f_activation, 1, f_activation, autodiff = true)

    #Fnn into JuMP    
    for k = 1:1:horizon
        for i = 1:1:nbr_neurons
            JuMP.@constraint(
                model_empc,
                y[i, 1, k] == W_layer[1][i, :]' * [x[:, k]; u[:, k]]
            ) #input layer
        end

        for j = 2:1:nbr_hidden+1
            for i = 1:1:nbr_neurons
                JuMP.@NLconstraint(
                    model_empc,
                    y[i, j, k] ==
                    f_activation([W_layer[j][i, :]' * y[:, j-1, k] + B_layer[j-1][i]][1])
                ) #hidden layer [1]for scalar expression
            end
        end

        JuMP.@constraint(model_empc, x[:, k+1] .== W_layer[:][end] * y[:, end, k]) #output layer
    end

    #States constraints
    for k = 1:1:horizon+1
        for i = 1:1:nbr_states
            JuMP.@constraint(model_empc, x[i, k] <= x_constraints[i, 2])
            JuMP.@constraint(model_empc, x_constraints[i, 1] <= x[i, k])
        end
    end

    #Inputs constraints
    for k = 1:1:horizon
        for i = 1:1:nbr_inputs_control
            JuMP.@constraint(model_empc, u[i, k] <= u_constraints[i, 2])
            JuMP.@constraint(model_empc, u_constraints[i, 1] <= u[i, k])
        end
    end

    return model_empc

end

### Mixed integer linear eMPC implementation method with Fnn ###

function _economic_model_predictive_control_modeler_implementation(
    method::MixedIntegerLinearProgramming,
    model_mlj::AutomationLabsIdentification.Fnn,
    system::MathematicalSystems.ConstrainedBlackBoxControlDiscreteSystem,
    horizon::Int,
    reference::ReferencesStateInput,
    solver::AbstractSolvers,
)

    #get constraints
    x_hyperrectangle = LazySets.vertices_list(system.X)
    u_hyperrectangle = LazySets.vertices_list(system.U)

    x_constraints = hcat(x_hyperrectangle[end], x_hyperrectangle[begin])
    u_constraints = hcat(u_hyperrectangle[end], u_hyperrectangle[begin])

    #the model is designed       
    model_empc = _JuMP_model_definition(method, solver)

    #get information from the state, input and neural network and extract neural weughts 
    weights = Flux.params(system.f)
    nbr_inputs = size(weights[1], 2)
    nbr_neurons = size(weights[1], 1)
    nbr_outputs = size(weights[length(weights)], 1)
    nbr_hidden = trunc(Int, (length(weights) - 2) / 2)
    nbr_inputs_control = nbr_inputs - nbr_outputs
    nbr_states = nbr_outputs

    #get neural weights
    W_layer = [] #memory allocation need 3D array multi dimention
    B_layer = [] #memory allocation, need 3D array multi dimention

    push!(W_layer, weights[1])

    for i = 2:2:length(weights)-1
        push!(W_layer, weights[i])
        push!(B_layer, weights[i+1])
    end

    push!(W_layer, weights[length(weights)])

    #decision variables are set in JuMP optimisation model
    JuMP.@variables(model_empc, begin
        x[1:nbr_states, 1:horizon+1]                 #state variable
        u[1:nbr_inputs_control, 1:horizon]           #input variable
        y[1:nbr_neurons, 1:nbr_hidden+1, 1:horizon]   #output hidden neuron network
        Bin_1[1:nbr_neurons, 1:nbr_hidden, 1:horizon], Bin
        Bin_2[1:nbr_neurons, 1:nbr_hidden, 1:horizon], Bin
    end)

    #BigM variable
    BigM = 1000

    #Modeling the Fnn into JuMP with relu
    for k = 1:1:horizon
        for i = 1:1:nbr_neurons
            JuMP.@constraint(
                model_empc,
                y[i, 1, k] == W_layer[1][i, :]' * [x[:, k]; u[:, k]]
            ) #input layer
        end

        for j = 2:1:nbr_hidden+1
            for i = 1:1:nbr_neurons
                JuMP.@constraint(model_empc, y[i, j, k] >= 0)
                JuMP.@constraint(
                    model_empc,
                    y[i, j, k] >= [W_layer[j][i, :]' * y[:, j-1, k] + B_layer[j-1][i]][1]
                )
                JuMP.@constraint(
                    model_empc,
                    y[i, j, k] .<=
                    [W_layer[j][i, :]' * y[:, j-1, k] + B_layer[j-1][i]][1] .+
                    BigM * (1 .- Bin_1[i, j-1, k])
                )
                JuMP.@constraint(
                    model_empc,
                    y[i, j, k] .<= 0 .+ BigM * (1 .- Bin_2[i, j-1, k])
                )
                JuMP.@constraint(model_empc, Bin_1[i, j-1, k] + Bin_2[i, j-1, k] == 1)
            end
        end

        JuMP.@constraint(model_empc, x[:, k+1] .== W_layer[:][end] * y[:, end, k]) #output layer
    end

    #States constraints
    for k = 1:1:horizon+1
        for i = 1:1:nbr_states
            JuMP.@constraint(model_empc, x[i, k] <= x_constraints[i, 2])
            JuMP.@constraint(model_empc, x_constraints[i, 1] <= x[i, k])
        end
    end

    #Inputs constraints
    for k = 1:1:horizon
        for i = 1:1:nbr_inputs_control
            JuMP.@constraint(model_empc, u[i, k] <= u_constraints[i, 2])
            JuMP.@constraint(model_empc, u_constraints[i, 1] <= u[i, k])
        end
    end

    return model_empc

end