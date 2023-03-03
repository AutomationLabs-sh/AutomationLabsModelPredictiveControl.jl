# Copyright (c) 2022: Pierre Blaud and contributors
########################################################
# This Source Code Form is subject to the terms of the #
# Mozilla Public License, v. 2.0. If a copy of the MPL #
# was not distributed with this file,  				   #
# You can obtain one at https://mozilla.org/MPL/2.0/.  #
########################################################

# economic model predictive control computation (the controller)
"""
    update_and_compute!(C::ModelPredictiveControlController, initialization::Vector; references::ReferencesStateInput)
Model predictive control main computation function. The controller is updated with the new initialization and references (if needed)
and then the controller is computed and new inputs command are displayed.

** Required fields **
* `C`: the model predictive control controller.
* `initialization`: initialization of the model predictive control befre computation (also now as the state measures).

** Optional fields **
* `references`: new states and inputs references is needed.
"""
function update_and_compute!(
    C::EconomicModelPredictiveControlController,
    initialization::Vector;
)

    if C.References != references
        #recompute terminal ingredient, update modeler, intialization, references
        UpdateController!(C)

    else
        #only initialization is updated
        update_initialization!(C, initialization)

    end

    #Compute MPC
    Calculate!(C)

end

"""
    update_initialization!(C::EconomicModelPredictiveControlController, initialization::Vector)
Update initialization (also known as state measure) of Model predictive control controller.

** Required fields **
* `C`: the design model predictive control controller.
* `initialization`: initialization of the model predictive control befre computation (also now as the state measures).
"""
function update_initialization!(
    C::EconomicModelPredictiveControlController,
    initialization::Vector,
)

    #add initialisation
    C.initialization = initialization

    #force within the modeller
    x = C.tuning.modeler[:x]
    #force within JuMP
    for i = 1:size(x, 1)
        JuMP.fix(x[i, 1], initialization[i, 1]; force = true)
    end

end

"""
    Update!(C::ModelPredictiveControlController, initialization::Vector, references::ReferencesStateInput)
Model predictive control main computation function. The controller is updated with the new initialization and references (if needed)
and then the controller is computed and new inputs command are displayed.

** Required fields **
* `C`: the model predictive control controller.
* `initialization`: initialization of the model predictive control befre computation (also now as the state measures).

** Optional fields **
* `references`: new states and inputs references is needed.
"""
function update!(C::EconomicModelPredictiveControlController, initialization::Vector)

    #get the modeler
    model_mpc = C.tuning.modeler

    #get initialization
    x_measured = C.initialization

    #decision vectors are extracted from MyModel_MPC
    x = model_mpc[:x]
    u = model_mpc[:u]

    #First vector size are defined
    SizeVectorX = size(x)
    SizeVectorU = size(u)

    #number of inputs and states and disturbance are extracted
    NumberOfState = SizeVectorX[1]
    NumberOfInputControl = SizeVectorU[1]
    Horizon = SizeVectorU[2]


    ### end update references ###

    ### Update terminal ingredients ###

    # evaluate if the user choses the terminal constraints
    if ingredients == true

        #Terminal ingredients from H. Chen and F. Allgower, A Quasi-Infinite Horizon Nonlinear Model Predictive
        #Control Scheme with Guaranteed Stability, Automatica 1998

        #Linearization to get A and B at references
        A_sys, B_sys = f_linearization(system.f, references)

        #Calculate P matrix with ControlSystems and LQR
        P = ControlSystems.are(Discrete, A_sys, B_sys, weights.Q, weights.R)

        #Calculate Xf
        A = A_sys - B_sys * K1

        #InvariantSets.jl
        # """
        #     THEORY:
        #   Invariance: Region in which an autonomous system
        #   satisifies the constraints for all time.
        #
        #   Control Invariance: Region in which there exists a controller
        #   so that the system satisfies the constraints for all time.
        #
        #   A set 𝒪 is positive invariant if and only if 𝒪 ⊆ pre(𝒪)!
        #   """

        #Set new cost function with terminal cost P
        if weights.R != 0 && weights.S != 0
            JuMP.@objective(
                model_mpc,
                Min,
                sum(
                    e_x[:, i]' * weights.Q * e_x[:, i] +
                    e_u[:, i]' * weights.R * e_u[:, i] +
                    d_u[:, i]' * weights.S * d_[:, i] for i = 1:Horizon
                ) + e_x[:, i]' * P * e_x[:, i]
            )

        elseif weights.R != 0
            JuMP.@objective(
                model_mpc,
                Min,
                sum(
                    e_x[:, i]' * weights.Q * e_x[:, i] + e_u[:, i]' * weights.R * e_u[:, i]
                    for i = 1:Horizon
                ) + e_x[:, i]' * P * e_x[:, i]
            )

        else
            JuMP.@objective(
                model_mpc,
                Min,
                sum(e_x[:, i]' * weights.Q * e_x[:, i] for i = 1:Horizon) +
                e_x[:, i]' * P * e_x[:, i]
            )

        end

        #Add terminal constraint
        x = model_mpc[:x]
        for i = 1:nbr_states
            JuMP.@constraint(model_mpc, (x_ter[i, 1] <= x[i, end] <= x_ter[i, 2]))
        end

        #const TerminalIngredients = TerminalIngredients2(true, P, Xf)
        TerminalIngredients = TerminalIngredients2(true, P, Xf)

        return TerminalIngredients, model_mpc

    else
        #no terminal constraints is choosen

        #Linearization to get A and B at references
        A_sys, B_sys = f_linearization(system.f, references)

        #Calculate P matrix with ControlSystems and LQR
        P = ControlSystems.are(Discrete, A_sys, B_sys, weights.Q, weights.R)

        #set the struct from the package model predictive control
        #const TerminalIngredients = TerminalIngredients1(false::Bool, P)
        TerminalIngredients = TerminalIngredients1(false::Bool, P)

        return TerminalIngredients, model_mpc

    end

    ### end update terminal ingredients ###

    ### update initialization ###
    update_initialization!(C, initialization)
    ### end update intialization

    ### update the cost function ###
    if C.tuning.weights.R != 0 && C.tuning.weights.S != 0
        JuMP.@objective(
            model_mpc,
            Min,
            e_x[:, end]' * terminal_ingredients.P * e_x[:, end] +
            sum(
                e_x[:, i]' * C.tuning.weights.Q * e_x[:, i] +
                e_u[:, i]' * C.tuning.weights.R * e_u[:, i] for i = 1:horizon
            ) +
            sum(delta_u[:, i]' * C.tuning.weights.S * delta_u[:, i] for i = 1:horizon-1)
        )

    elseif C.tuning.weights.R != 0
        JuMP.@objective(
            model_mpc,
            Min,
            e_x[:, end]' * terminal_ingredients.P * e_x[:, end] + sum(
                e_x[:, i]' * C.tuning.weights.Q * e_x[:, i] +
                e_u[:, i]' * C.tuning.weights.R * e_u[:, i] for i = 1:horizon
            )
        )

    else
        JuMP.@objective(
            model_mpc,
            Min,
            e_x[:, end]' * terminal_ingredients.P * e_x[:, end] +
            sum(e_x[:, i]' * C.tuning.weights.Q * e_x[:, i] for i = 1:horizon)
        )
    end
    ### end update cost function ###


end

"""
    calculate!(C::ModelPredictiveControlController)
Model predictive control computation function. The controller is computed and optimization results values are written on mutable struct controller.

** Required fields **
* `C`: the model predictive control controller.
"""
function calculate!(C::EconomicModelPredictiveControlController)

    #vérification que la MPC peut être calculée correctement #to do
    #try ...

    # try
    #optimisation is done here
    JuMP.optimize!(C.tuning.modeler)

    #variables are extracted from MyModel_MPC
    u = C.tuning.modeler[:u]
    x = C.tuning.modeler[:x]

    #get optimisation variables
    C.computation_results.u[:, :] = JuMP.value.(u[:, :])
    C.computation_results.x[:, :] = JuMP.value.(x[:, :])

    #catch err
    #    @error "The controler cannot be computed, something missing."
    #end
end
