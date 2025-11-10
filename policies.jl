using Printf
using Serialization
using Distributions
using StatsBase

function save_policy(filename::String, policy_func::Function, num_states::Int)
    open(filename, "w") do f
        for si in 1:num_states
            write(f, @sprintf("%d\n", policy_func(si)))
        end
    end
end

function save_action_value_function(filename::String, model)
    serialize(filename, model)
end

function load_action_value_function(filename::String)
    return deserialize(filename)
end

function get_lines(filename::String)
    lines = Vector{Vector{Int}}()
    open(filename, "r") do input
        header = readline(input) # Header is ignored

        # Parse lines
        for line in eachline(input)
            sample = parse.(Int, split(line, ','))
            push!(lines, sample)
        end
    end
    return lines
end

##################################################
#   Q Learning
##################################################

mutable struct QLearning
    𝒮 # state space (assumes 1:nstates)
    𝒜 # action space (assumes 1:nactions)
    γ # discount
    Q # action value function
    α # learning rate
end

lookahead(model::QLearning, s, a) = model.Q[s,a]

function update!(model::QLearning, s, a, r, s′)
    γ, Q, α = model.γ, model.Q, model.α
    Q[s,a] += α*(r + γ*maximum(Q[s′,:]) - Q[s,a])
    return model
end

##################################################
#   MaximumLikelihoodMDP
##################################################

mutable struct MaximumLikelihoodMDP
    𝒮 # state space (assumes 1:nstates)
    𝒜 # action space (assumes 1:nactions)
    N # transition count N(s,a,s′) - dictionary based
    ρ # reward sum ρ(s, a) - dictionary based
    γ # discount
    U # value function

end

function lookahead(model::MaximumLikelihoodMDP, s, a)
    𝒮, U, γ = model.𝒮, model.U, model.γ
    key = (s, a)

    state_dict = get(model.N, key, Dict())
    if isempty(state_dict)
        return 0.0
    end
    
    n = sum(values(state_dict))
    r = model.ρ[key] / n
    
    return r + γ * sum((count/n) * U[s′] for (s′, count) in state_dict)
end

function backup(model::MaximumLikelihoodMDP, U, s)
    vals = [lookahead(model, s, a) for a in model.𝒜]
    return isempty(vals) ? 0.0 : maximum(vals)
end

function update!(model::MaximumLikelihoodMDP, s, a, r, s′)
    key = (s, a)

    state_dict = get(model.N, key, Dict{Int,Int}())
    state_dict[s′] = get(state_dict, s′, 0) + 1

    model.N[key] = state_dict
    model.ρ[key] = get(model.ρ, key, 0.0) + r

    return model
end

##################################################
#   ValueFunctionPolicy
##################################################

struct ValueFunctionPolicy
    𝒫 # problem
    U # utility function
end

function greedy(𝒫::MaximumLikelihoodMDP, U::Vector{Float64}, s::Int)
    u, a = findmax(a -> lookahead(𝒫, s, a), 𝒫.𝒜)
    return (a=a, u=u)
end

(π::ValueFunctionPolicy)(s) = greedy(π.𝒫, π.U, s).a

##################################################
#   ValueIteration
##################################################

struct ValueIteration
    k_max # maximum number of iterations
end

function solve(M::ValueIteration, 𝒫::MaximumLikelihoodMDP)
    U = [0.0 for s in 𝒫.𝒮]
    for k = 1:M.k_max
        U = [backup(𝒫, U, s) for s in 𝒫.𝒮]
    end
    # Update the model's value function
    𝒫.U = U
    return ValueFunctionPolicy(𝒫, U)
end

##################################################
#   Softmax
##################################################

mutable struct SoftmaxExploration
    λ # precision parameter
    α # precision factor
end

function normalize(weights)
    total = sum(weights)
    return weights ./ total
end

function (π::SoftmaxExploration)(model::MaximumLikelihoodMDP, s)
    # Q-values for actions
    Q_values = [lookahead(model, s, a) for a in model.𝒜]
    
    Q_shift = Q_values .- maximum(Q_values)
    weights = exp.(π.λ * Q_shift)
    π.λ *= π.α
    
    # return s sampled action
    return rand(Categorical(normalize(weights)))
end