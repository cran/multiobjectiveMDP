## ----include = FALSE----------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>"
)

## ----setup--------------------------------------------------------------------
library(multiobjectiveMDP)

## -----------------------------------------------------------------------------
set.seed(1234)
no_states <- 2
action_sets <- list(c(1, 2), c(1, 2))
no_objectives <- 2
# Generate a two-state bi-objective MDP having three epochs and two actions per state
finite_horizon_MMDP <- generate_rand_MMDP(no_states, action_sets, horizon = 3, no_objectives)
# Inspect the transition probabilities
P <- finite_horizon_MMDP$P
P

# Inspect the rewards
R <- finite_horizon_MMDP$R
R

## -----------------------------------------------------------------------------
set.seed(1234)
no_states <- 2
action_sets <- list(c(1, 2), c(1, 2))
no_objectives <- 2
# Generate an infinite-horizon two-state bi-objective MDP having two actions per state
stationary_MMDP <- generate_rand_MMDP(no_states, action_sets, horizon = Inf, no_objectives)

# Inspect the transition probabilities
P <- stationary_MMDP$P
P

# Inspect the rewards
R <- stationary_MMDP$R
R

## -----------------------------------------------------------------------------
set.seed(1234)
no_states <- 2
# The action set for state 1 is {1, 2}; for state 2 it is {1, 2, 3}
action_sets <- list(c(1, 2), c(1, 2, 3))
horizon <- 5
policy <- matrix(data = c(sample(action_sets[[1]], size = horizon-1, replace = T), sample(action_sets[[2]], size = horizon-1, replace = T)), nrow = no_states, ncol = horizon-1)
policy

## -----------------------------------------------------------------------------
set.seed(1234)
no_states <- 2
# The action set for state 1 is {1, 2}; for state 2 it is {1, 2, 3}
action_sets <- list(c(1, 2), c(1, 2, 3))
policy <- c(sample(action_sets[[1]], size = 1), sample(action_sets[[2]], size = 1))
policy

## -----------------------------------------------------------------------------
set.seed(1234)
no_states <- 2
action_sets <- list(c(1, 2), c(1, 2, 3))
horizon <- 5
no_objectives <- 2
# Generate a two-state bi-objective MDP with the specified action sets and horizon
MMDP <- generate_rand_MMDP(no_states, action_sets, horizon, no_objectives)
transition_probabilities <- MMDP$P
rewards <- MMDP$R

policy <- matrix(data = c(sample(action_sets[[1]], size = horizon-1, replace = T),
                          sample(action_sets[[2]], size = horizon-1, replace = T)),
                 nrow = no_states, ncol = horizon-1)
policy

# Evaluate the expected total reward of policy over the five epochs
evaluate_finite_horizon_MMDP_markov_policy(transition_probabilities, rewards, policy)

# What if a discount factor of 70% is applied at each epoch? 
rho <- .7
evaluate_finite_horizon_MMDP_markov_policy(transition_probabilities, rewards, policy, rho)


## -----------------------------------------------------------------------------
set.seed(1234)
no_states <- 2
action_sets <- list(c(1, 2), c(1, 2))
no_objectives <- 2
# Generate an infinite-horizon two-state bi-objective MDP providing two actions per state
stationary_MMDP <- generate_rand_MMDP(no_states, action_sets, horizon = Inf, no_objectives)
# Consider the pure policy that recommends action 2 for state 1 and action 1 for state 2
policy <- c(2, 1)
# Evaluate the policy in the infinite-horizon model generated above for rho = .7
evaluate_discounted_MMDP_pure_policy(stationary_MMDP$P, stationary_MMDP$R, policy, rho = .7)



## -----------------------------------------------------------------------------
set.seed(1234)
# Set up a bi-objective infinite-horizon MMDP
no_states <- 2
action_sets <- list(c(1, 2), c(1, 2, 3))
no_objectives <- 2
stationary_MMDP <- generate_rand_MMDP(no_states, action_sets, horizon = Inf, no_objectives)
stationary_transition_probabilities <- stationary_MMDP$P
stationary_rewards <- stationary_MMDP$R
rho <- .7

# Use policy iteration to locate the efficient pure policies
solution <- solve_discounted_MMDP_policy_iteration(stationary_transition_probabilities, stationary_rewards, rho)
solution$policies

# Inspect their expected discounted total rewards
solution$value_functions

