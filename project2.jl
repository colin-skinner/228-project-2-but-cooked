
include("policies.jl")

train = false
save_outputs = true
# save_outputs = false


##################################################
#   Small CSV - 100 states
##################################################

learning_rate = 0.2 # dumb hyperparameter
if isfile("small_cache")
    small_q_learning::QLearning = load_action_value_function("small_cache")
else
    small_q_learning = QLearning(1:100, 1:4, 0.95, zeros(100, 4), learning_rate)
end

small_lines = get_lines("small.csv")

println("Training Small")
for i in 1:1000
    for line in small_lines
        update!(small_q_learning, line[1], line[2], line[3], line[4])
    end
    println(i)
end
println("Done")

small_map = [argmax(small_q_learning.Q[row, :]) for row in 1:100]
small_policy(state::Int) = small_map[state]

save_policy("small.policy", small_policy, 100)
println("Saved Small")

##################################################
#   Medium CSV - 50000 states
##################################################

learning_rate = .2 # dumb hyperparameter
if isfile("medium_cache")
    medium_q_learning::QLearning = load_action_value_function("medium_cache")
else
    medium_q_learning = QLearning(1:50000, 1:7, 0.95, zeros(50000, 7), learning_rate)
end

medium_lines = get_lines("medium.csv")

println("Training Medium")
for i in 1:1000
    for line in medium_lines
        update!(medium_q_learning, line[1], line[2], line[3], line[4])
    end
    println(i)
end
println("Done")

medium_map = [argmax(medium_q_learning.Q[row, :]) for row in 1:50000]
medium_policy(state::Int) = medium_map[state]

save_policy("medium.policy", medium_policy, 50000)
println("Saved Medium")

##################################################
#   Large CSV - 302020 states
##################################################

learning_rate = .2 # dumb hyperparameter
if isfile("large_cache")
    large_q_learning::QLearning = load_action_value_function("large_cache")
else
    large_q_learning = QLearning(1:302020, 1:9, 0.95, zeros(302020, 9), learning_rate)
end

large_lines = get_lines("large.csv")

println("Training large")
for i in 1:100
    for line in large_lines
        update!(large_q_learning, line[1], line[2], line[3], line[4])
    end
    println(i)
end
println("Done")

large_map = [argmax(large_q_learning.Q[row, :]) for row in 1:302020]
large_policy(state::Int) = large_map[state]

save_policy("large.policy", large_policy, 302020)
println("Saved Large")

##################################################
#   Saving
##################################################

if save_outputs
    save_action_value_function("small_cache", small_q_learning)
    save_action_value_function("medium_cache", medium_q_learning)
    save_action_value_function("large_cache", large_q_learning)
end


# Medium - 50000 states
# Large - 302020 states
