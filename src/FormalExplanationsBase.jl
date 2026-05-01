module FormalExplanationsBase

export 
    AbstractExplainer,
    explainer,
    fit!,
    fit,
    explain,
    explain_all,
    fitted_decisions,
    fitted_params

include("Explainer.jl")

end
