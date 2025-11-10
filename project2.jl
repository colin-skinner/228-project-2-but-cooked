
include("policies.jl")

train = false
save_outputs = true
# save_outputs = false


##################################################
#   Small CSV - 100 states
##################################################

learning_rate = .1 # dumb hyperparameter
if isfile("small_cache")
    small_q_learning::QLearning = load_action_value_function("small_cache")
else
    small_q_learning = QLearning(1:100, 1:4, 0.95, zeros(100, 4), learning_rate)
end

small_lines = get_lines("small.csv")

println("Training Small")
for line in small_lines
    update!(small_q_learning, line[1], line[2], line[3], line[4])
end
println("Done")

small_map = [argmax(small_q_learning.Q[row, :]) for row in 1:100]
small_policy(state::Int) = small_map[state]

##################################################
#   Medium CSV - 50000 states
##################################################

learning_rate = .1 # dumb hyperparameter
if isfile("medium_cache")
    medium_q_learning::QLearning = load_action_value_function("medium_cache")
else
    medium_q_learning = QLearning(1:50000, 1:7, 0.95, zeros(50000, 7), learning_rate)
end

medium_lines = get_lines("medium.csv")

println("Training Medium")
for line in medium_lines
    update!(medium_q_learning, line[1], line[2], line[3], line[4])
end
println("Done")

medium_map = [argmax(medium_q_learning.Q[row, :]) for row in 1:50000]
medium_policy(state::Int) = medium_map[state]

##################################################
#   Large CSV - 302020 states
##################################################

##################################################
#   Saving
##################################################

if save_outputs
    save_policy("small.policy", small_policy, 100)
    println("Saved Small")
    save_policy("medium.policy", medium_policy, 50000)
    println("Saved Medium")
    # save_policy("large.policy", bad_policy, 302020)

    save_action_value_function("small_cache", small_q_learning)
    save_action_value_function("medium_cache", medium_q_learning)
    # print(l)
end


# Medium - 50000 states
# Large - 302020 states
