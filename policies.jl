using Printf

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

function save_policy(filename::String, policy_func::Function, num_states::Int)
    open(filename, "w") do f
        for si in 1:num_states
            write(f, @sprintf("%d\n", policy_func(si)))
        end
    end
end