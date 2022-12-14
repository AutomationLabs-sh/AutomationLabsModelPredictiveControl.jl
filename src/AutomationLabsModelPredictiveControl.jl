# Copyright (c) 2022: Pierre Blaud and contributors
########################################################
# This Source Code Form is subject to the terms of the #
# Mozilla Public License, v. 2.0. If a copy of the MPL #
# was not distributed with this file,  				   #
# You can obtain one at https://mozilla.org/MPL/2.0/.  #
########################################################
module AutomationLabsModelPredictiveControl

import ControlSystems
import Flux
import ForwardDiff
import HiGHS
import AutomationLabsIdentification
import Ipopt
import JuMP
import LazySets
import MathematicalSystems
import MosekTools
import MLJ
import OSQP
import SCIP
import LinearAlgebra
import MLJMultivariateStatsInterface

# export 
export proceed_controller

# export computation and update functions
export update_and_compute!
export update_initialization!
export update!
export calculate!

# include sub files
include("subfunctions/types.jl")
include("subfunctions/design_mpc.jl")
include("subfunctions/design_empc.jl")
include("subfunctions/solver_selection.jl")

# modeler implementation of models
# Fnn
include("model_modeler_implementation/fnn/mpc_modeler_implementation_fnn.jl")
include("model_modeler_implementation/fnn/empc_modeler_implementation_fnn.jl")

#icnn
include("model_modeler_implementation/icnn/mpc_modeler_implementation_icnn.jl")
include("model_modeler_implementation/icnn/empc_modeler_implementation_icnn.jl")

# resnet 
include("model_modeler_implementation/resnet/mpc_modeler_implementation_resnet.jl")
include("model_modeler_implementation/resnet/empc_modeler_implementation_resnet.jl")

# densenet
include("model_modeler_implementation/densenet/mpc_modeler_implementation_densenet.jl")
include("model_modeler_implementation/densenet/empc_modeler_implementation_densenet.jl")

# rbf
include("model_modeler_implementation/rbf/mpc_modeler_implementation_rbf.jl")
include("model_modeler_implementation/rbf/empc_modeler_implementation_rbf.jl")

# polynet
include("model_modeler_implementation/polynet/mpc_modeler_implementation_polynet.jl")
include("model_modeler_implementation/polynet/empc_modeler_implementation_polynet.jl")

# neuralnetode type 1
include("model_modeler_implementation/neuralnetode_type1/mpc_modeler_implementation_neuralnetode_type1.jl")
include("model_modeler_implementation/neuralnetode_type1/empc_modeler_implementation_neuralnetode_type1.jl")

# linear model 
include("model_modeler_implementation/linear/mpc_modeler_implementation_linear.jl")
include("model_modeler_implementation/linear/empc_modeler_implementation_linear.jl")


# include main files
include("mainfunctions/main_mpc.jl")
include("mainfunctions/computation_mpc.jl")
include("mainfunctions/computation_empc.jl")

end
