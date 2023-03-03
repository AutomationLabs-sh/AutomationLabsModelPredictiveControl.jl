# Copyright (c) 2022: Pierre Blaud and contributors
########################################################
# This Source Code Form is subject to the terms of the #
# Mozilla Public License, v. 2.0. If a copy of the MPL #
# was not distributed with this file,  				   #
# You can obtain one at https://mozilla.org/MPL/2.0/.  #
########################################################

const _IMPLEMENTATION_SOLVER_LIST = (
    osqp = osqp_solver_def(),
    scip = scip_solver_def(),
    mosek = mosek_solver_def(),
    ipopt = ipopt_solver_def(),
    highs = highs_solver_def(),
    auto = auto_solver_def(),
)

#file for solver selection according to MPC implementation, linear, non linear or MILP

function _JuMP_model_definition(
    method::LinearProgramming,
    solver_selection::Union{
        osqp_solver_def,
        scip_solver_def,
        mosek_solver_def,
        highs_solver_def,
        ipopt_solver_def,
    },
)

    model = _selection_solver_JuMP_model(solver_selection)

    return model

end

function _JuMP_model_definition(
    method::NonLinearProgramming,
    solver_selection::Union{ipopt_solver_def,scip_solver_def},
)

    model = _selection_solver_JuMP_model(solver_selection)

    return model

end

function _JuMP_model_definition(
    method::MixedIntegerLinearProgramming,
    solver_selection::Union{mosek_solver_def,scip_solver_def},
)

    model = _selection_solver_JuMP_model(solver_selection)

    return model

end

#auto selection scip solver
function _JuMP_model_definition(
    method::LinearProgramming,
    solver_selection::auto_solver_def,
)

    model = _JuMP_model_definition(method, scip_solver_def())

    return model

end

function _JuMP_model_definition(
    method::NonLinearProgramming,
    solver_selection::auto_solver_def,
)

    model = _JuMP_model_definition(method, ipopt_solver_def())

    return model

end

function _JuMP_model_definition(
    method::MixedIntegerLinearProgramming,
    solver_selection::auto_solver_def,
)

    model = _JuMP_model_definition(method, scip_solver_def())

    return model

end

###############################################
### selection solver according to selection ###
###############################################
function _selection_solver_JuMP_model(solver::osqp_solver_def)

    model = JuMP.Model(JuMP.optimizer_with_attributes(OSQP.Optimizer))
    #to do the right options

    return model
end

function _selection_solver_JuMP_model(solver::highs_solver_def)

    model = JuMP.Model(JuMP.optimizer_with_attributes(HiGHS.Optimizer))
    #to do the right options

    return model
end

function _selection_solver_JuMP_model(solver::ipopt_solver_def)

    model = JuMP.Model(JuMP.optimizer_with_attributes(Ipopt.Optimizer))
    #to do the right options

    return model
end

function _selection_solver_JuMP_model(solver::mosek_solver_def)

    model = JuMP.Model(
        JuMP.optimizer_with_attributes(
            MosekTools.Mosek.Optimizer,
            "QUIET" => false,
            "OPTIMIZER_MAX_TIME" => 300.0,
            "INTPNT_CO_TOL_DFEAS" => 1e-7,
        ),
    )
    # "OPTIMIZER_MAX_TIME" => 300.0,
    #to do the rights option

    return model
end

function _selection_solver_JuMP_model(solver::scip_solver_def)

    model = JuMP.Model(JuMP.optimizer_with_attributes(SCIP.Optimizer))
    #to do the right options

    return model
end
