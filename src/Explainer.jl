using RobustClassifiersBase
const RCB = RobustClassifiersBase
using ResumableFunctions
using MLJBase

abstract type AbstractExplainer end

mutable struct Explainer{M<: AbstractExplainer}
    model::M
    class_fitresult
    x
    fitresult
    function Explainer(model::M, class_fitresult, x) where {M}
        new{M}(model, class_fitresult, x) 
    end

end

function explainer(model::E, machine::MLJBase.Machine{M,OM,C}, x) where {E,M,OM,C}
    Explainer(model, machine.fitresult, x) 
end

"""
    objectif: explainer créé avec NCC, x
    fit!(explainer)
"""
function fit end

function fit!(E::Explainer)
    E.fitresult =
        try
            fit(E.model, E.class_fitresult, E.x)
        catch exception
            @error "Problem fitting the explainer $(typeof(E.model)). $exception"
        end
    @assert !isnothing(E.fitresult) && typeof(E.fitresult[2]) <: RCB.Prediction
    return E
end

function fitted_params(E::Explainer)
    if isdefined(E, :fitresult)
        return fitted_params(E.model, E.fitresult)
    else
        @error "The explainer $(typeof(E.model)) has not been trained. Call `fit!` on the explainer."
    end
end

function fitted_decisions(E::Explainer)
    if isdefined(E, :fitresult)
        return fitted_decisions(E.model, E.fitresult)
    else
        @error "The explainer $(typeof(E.model)) has not been trained. Call `fit!` on the explainer."
    end
end

"""
    objectif: explain(explainer, Decision)
"""
function explain end

@resumable function explain(E::Explainer, decision::D) where {D<:RCB.AbstractDecision}
    for explanation in explain(E.model, E.fitresult, decision)
        @yield explanation
    end
end


"""
    objectif: explain(explainer, Decision)
"""
function explain_all end


function explain_all(E::Explainer, decision::D) where {D}
    explanations = 
        try
            explain_all(E.model, E.fitresult, decision)
        catch exception
            @error """Problem with the explainer $(typeof(E)). 
                Unable to find all explanations for decision $decision.
                $exception"""
        end
    return explanations
end