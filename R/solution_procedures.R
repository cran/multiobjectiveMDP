library(lintools)
library(prodlim)
library(nsga2R)
library(pracma)
library(linprog)
library(dplyr)

#' Generate a random instance of a multi-objective Markov decision process
#'
#' `generate_rand_MMDP()` returns a randomly generated finite- or
#' infinite-horizon multi-objective Markov decision process having the requested
#' states and action sets.
#'
#' @param no_states An integer greater than or equal to 2. This is the number of
#'   states in the model to be generated.
#' @param action_sets A list of `no_states` integer vectors specifying each
#'   state's action set. Each vector must range from 1 to the number of actions
#'   permissible in the corresponding state.
#' @param horizon
#' * `Inf` (default): Leave at this value if the model to be generated is a stationary infinite-horizon one.
#' * Otherwise, this is the number of decision epochs plus the terminal epoch. Has to be greater than or equal to 2.
#' @param no_objectives An integer greater than or equal to 1. This is the
#'   number of reward functions (objectives) in the model to be generated.
#'
#' @return A list summarizing the components of the multi-objective Markov
#'   decision process that was generated in the body of the function. These
#'   components are:
#' * The state set `S`, an integer vector, as determined by `no_states`,
#' * The action sets `action_sets`, an integer list,
#' * The decision-making horizon `horizon`,
#' * A randomly generated transition probability list `P` of length one if the horizon is infinite and of length `horizon - 1` otherwise. There are three levels in `P`. In the finite-horizon case, a typical third-level entry, `P[[t]][[s]][[a]]`, is a numeric vector that gives the probability distribution of the states at epoch `t+1` if the state at epoch `t` was `s` and the action selected was `a`. For an infinite-horizon model, `P[[1]][[s]][[a]]` is to be interpreted as the stationary (i.e., time-homogeneous) probability distribution of future states if the current state is `s` and the action taken was `a`.
#' * A randomly generated reward list `R` of length one if the horizon is infinite and of length `horizon` otherwise. There are likewise three levels in `R`. In the finite-horizon case, a typical third-level entry, `R[[t]][[s]][[a]]`, is a numeric `no_objectives`-dimensional vector that gives the reward for selecting action `a` in state `s` at epoch `t`. A special case is the entry `R[[horizon]][[s]]`, which is a `no_objectives`-dimensional vector that gives the reward for the terminating in state `s`. In the infinite-horizon case, `R[[1]][[s]][[a]]` is the stationary reward for selecting action `a` in state `s`. There are no terminal rewards in infinite-horizon problems.
#' @export
#'
#' @examples
#' # Generate a two-state finite-horizon bi-objective Markov decision process
#' no_states <- 2
#' # Two actions are available in state 1 whereas three are available in state 2
#' action_sets <- list(c(1, 2), c(1, 2, 3))
#' # Let there be four decision epochs and two objectives
#' horizon <- 5
#' no_objectives <- 2
#' MMDP <- generate_rand_MMDP(no_states, action_sets, horizon, no_objectives)
#' # Inspect the randomly generated transition probabilities and rewards
#' MMDP$P
#' MMDP$R
generate_rand_MMDP <- function(no_states, action_sets, horizon = Inf, no_objectives) {
  # Testing the arguments' validity and internal consistency
  if ((no_states%%1 != 0) || no_states <= 1) {
    stop("The number of states must be an integer greater than or equal to 2.")
  }
  else if (!is.list(action_sets) || length(action_sets) != no_states) {
    stop("The action sets must be provided as a single list of length no_states. The components of this list are expected to be integer vectors ranging from 1 to the number of actions available in a state.")
  }
  else if (horizon < 2) {
    stop("The decision-making horizon is either an integer greater than or equal to 2 or infinity (default).")
  }
  else if ((no_objectives%%1 != 0) || no_objectives < 1) {
    stop("The number of objective / reward functions must be an integer greater than or equal to 1.")
  }

  states <- 1:no_states
  action_sets <- lapply(action_sets, function(n) 1:length(n))
  transition_probabilities <- list()
  rewards <- list()
  if (horizon < Inf) {
    for (t in 1:(horizon-1)) {
      transition_probabilities[[t]] <- list()
      rewards[[t]] <- list()
      for (s in 1:no_states) {
        transition_probabilities[[t]][[s]] <- list()
        rewards[[t]][[s]] <- list()
        for (a in 1:length(action_sets[[s]])) {
          transition_probabilities[[t]][[s]][[a]] <- rexp(n = no_states, rate = 1)
          transition_probabilities[[t]][[s]][[a]] <- transition_probabilities[[t]][[s]][[a]] / sum(transition_probabilities[[t]][[s]][[a]])
          rewards[[t]][[s]][[a]] <- rexp(n = no_objectives, rate = 1)
        }
      }
    }
    rewards[[horizon]] <- list()
    for (s in 1:no_states) {
      rewards[[horizon]][[s]] <- rexp(n = no_objectives, rate = 1)
    }
  }
  else {
    rewards[[1]] <- list()
    transition_probabilities[[1]] <- list()
    for (s in 1:no_states) {
      transition_probabilities[[1]][[s]] <- list()
      rewards[[1]][[s]] <- list()
      for (a in 1:length(action_sets[[s]])) {
        transition_probabilities[[1]][[s]][[a]] <- rexp(n = no_states, rate = 1)
        transition_probabilities[[1]][[s]][[a]] <- transition_probabilities[[1]][[s]][[a]] / sum(transition_probabilities[[1]][[s]][[a]])
        rewards[[1]][[s]][[a]] <- rexp(n = no_objectives, rate = 1)
      }
    }

  }



  return(list(S = 1:no_states, A = action_sets, horizon = horizon, P = transition_probabilities, R = rewards))
}

#' Evaluate a Markov deterministic policy for a finite-horizon multi-objective
#' Markov decision process
#'
#' `evaluate_finite_horizon_MMDP_markov_policy()` recursively calculates the
#' value function of a Markov deterministic policy for a finite-horizon
#' multi-objective Markov decision process.
#'
#' Let \eqn{\pi} denote the Markov deterministic policy represented by `policy`.
#' For each state \eqn{s}, define \deqn{ u_t^{\pi}(s) = \mathbb{E}_s^{\pi} ( \sum_{i=t}^{H-1} \rho^{i-t} R_i(X_i, d_i(X_i)) + \rho^{H-t} R_H(X_H) ) }
#' to be the expected reward that accrues from using \eqn{\pi} between
#' a decision epoch \eqn{t} and terminal epoch \eqn{H} (`horizon`), with
#' \eqn{d_i(X_i)} denoting the action prescribed at time \eqn{i} by \eqn{\pi}
#' for (random) state \eqn{X_i}, and \eqn{R_i(X_i, d_i(X_i))} denoting the
#' attendant reward vector. We assume no action is taken at termination, and so
#' the reward \eqn{R_H(X_H)} will merely be a function of the terminal state
#' \eqn{X_H}.
#'
#' This function finds \eqn{u_1^{\pi}}, \eqn{\pi}'s value function, through the
#' recursion \deqn{u_t^{\pi}(s) = R_t(s, d_t(s)) + \rho \sum_{j} p_t(j | s,
#' d_t(s))u_{t+1}^{\pi}(j),} where the index \eqn{j} runs over the states, \eqn{p_t(j | s, d_t(s))} denotes the probability of transition from \eqn{s} to \eqn{j} under action \eqn{d_t(s)}, and
#' where we set \eqn{u_H^{\pi}(s) = R_H(s)} for each state \eqn{s}.
#'
#'
#'
#'
#' @param transition_probabilities A numeric list containing the transition
#'   probabilities of the finite-horizon model in which `policy` is being
#'   evaluated. The length of the list is the number of decision epochs
#'   (excluding the terminal epoch). There should be three levels in this list:
#' * For each decision epoch `t`, `transition_probabilities[[t]]` is expected to be a numeric list whose length equals the number of states in the model.
#' * For each state `s`, `transition_probabilities[[t]][[s]]` should be a numeric list whose length equals the number of actions permissible in `s`.
#' * Each third-level entry `transition_probabilities[[t]][[s]][[a]]` should be a numeric vector that gives the probability distribution of the states at epoch `t+1` if the state at epoch `t` was `s` and the action selected was `a`.
#'
#' @param rewards A numeric list that defines the rewards of the finite-horizon
#'   model in which `policy` is being evaluated. The length of the list is the
#'   number of epochs in the model (including the terminal epoch). There should
#'   be three levels in this list:
#' * For all epochs `t`, `rewards[[t]]` is expected to be a numeric list whose length equals the number of states in the model.
#' * If `t` is a decision epoch, then for each state `s` we expect `rewards[[t]][[s]]` to be a numeric list whose length equals the number of actions permissible in `s`. If `t` is terminal, then `rewards[[t]][[s]]` should be a numeric vector that gives the reward for the MMDP terminating in state `s`.
#' * Given a decision epoch `t` and a state `s`, each third-level entry `rewards[[t]][[s]][[a]]` should be a numeric vector that gives the reward for selecting action `a` in state `s` at epoch `t`.
#'
#' @param policy An integer matrix in which the `(s,t)`-th entry represents the
#'   action prescribed at decision epoch `t` by the policy under consideration
#'   for state `s`. The size and entries of this matrix should be consistent
#'   with the horizon length, states and action sets stipulated by
#'   `transition_probabilities` and `rewards`.
#' @param rho A number between 0 and 1 representing the reward discount factor.
#'   The default value `1` corresponds to an undiscounted model.
#'
#' @return A numeric matrix whose `s`-th column gives the value achieved by
#'   `policy` from initial state `s`.
#' @export
#'
#' @seealso [are_valid_finite_horizon_transition_probabilities()],
#'   [are_valid_finite_horizon_rewards()]
#' @references Puterman, M. L. (2014).
#' *Markov decision processes: discrete stochastic dynamic programming*.
#'   John Wiley & Sons.
#'
#' @examples
#' set.seed(1234)
#' no_states <- 2
#' action_sets <- list(c(1, 2), c(1, 2, 3))
#' horizon <- 5
#' no_objectives <- 2
#' MMDP <- generate_rand_MMDP(no_states, action_sets, horizon, no_objectives)
#' transition_probabilities <- MMDP$P
#' rewards <- MMDP$R
#' rho <- .97
#'
#' # Consider this policy: for state 1, always take action 1;
#' # for state 2, take action 1 at odd decision epochs and action 2 at even decision epochs.
#' policy <- matrix(nrow = no_states, ncol = (horizon - 1), data = c(1, 1, 1, 1,
#'                                                                    1, 3, 1, 3), byrow = TRUE)
#' evaluate_finite_horizon_MMDP_markov_policy(transition_probabilities, rewards, policy, rho)
evaluate_finite_horizon_MMDP_markov_policy <- function(transition_probabilities, rewards, policy, rho = 1) {

  # Testing the fitness and internal consistency of the arguments provided
  if (!are_valid_finite_horizon_transition_probabilities(transition_probabilities) || !are_valid_finite_horizon_rewards(rewards)) {
    stop("The rewards and/or transition probabilities specified are invalid. See the message(s) returned by are_valid_finite_horizon_transition_probabilities() and/or are_valid_finite_horizon_rewards().")
  }
  else if (length(transition_probabilities) != (length(rewards) - 1)) {
    stop("The reward and transition probability lists do not cover the same decision-making horizon and thus cannot represent the same problem.")
  }
  else if (length(transition_probabilities[[1]]) != length(rewards[[1]])) {
    stop("The rewards and transition probabilities you have provided do not specify the same state set.")
  }
  else if (!all.equal(lengths(transition_probabilities[[1]]), lengths(rewards[[1]]))) {
    stop("The rewards and transition probabilities you have provided do not specify the same action sets.")
  }
  # Ensure the policy is compatible with the model specified by rewards and transition_probabilities
  else if (!is_valid_finite_horizon_policy(policy, rewards)) {
    stop("The policy is incompatible with the model as defined by the rewards and transition probabilities. Please ensure that (a) the columns of the policy matrix span the decision-making horizon implied by the rewards and transition probabilities and (b) that the policy does not prescribe actions which lie outside a state's action set (as specified by the rewards and transition probabilities.")
  }
  else if (!is.numeric(rho) || (rho > 1) || (rho < 0)) {
    stop("The discount factor rho must be a number between (and including) 0 and 1.")
  }

  no_objectives <- length(rewards[[1]][[1]][[1]])
  no_states <- length(rewards[[1]])
  horizon <- length(rewards)
  value_function_t_onward <- array(0, c(horizon, no_objectives, no_states))
  value_function_t_onward[horizon, ,] <- Reduce("cbind", rewards[[horizon]])
  P_t <- matrix(nrow = no_states, ncol = no_states)
  immediate_reward <- matrix(nrow = no_objectives, ncol = no_states)
  for (i in 1:(horizon-1)) {
    for (s in 1:no_states) {
      P_t[s, ] <- transition_probabilities[[horizon-i]][[s]][[as.numeric(policy[s, horizon-i])]]
      immediate_reward[, s] <- rewards[[horizon-i]][[s]][[as.numeric(policy[s, horizon-i])]]
    }
    future_reward <- value_function_t_onward[horizon-i+1, ,] %*% t(P_t)
    value_function_t_onward[horizon-i, ,] <- immediate_reward + rho*future_reward
  }

  temp <- as.data.frame(value_function_t_onward[1, ,])
  rownames(temp) <- paste("Objective", seq_len(no_objectives))
  colnames(temp) <- paste("State", seq_len(no_states))

  return(temp)
}

#' Solve a standard finite-horizon Markov decision process through dynamic
#' programming
#'
#' `solve_finite_horizon_MDP_backward_induction()` implements the standard
#' backward induction algorithm in order to determine the optimal value function and an optimal policy
#' for a single-objective finite-horizon Markov decision
#' process.
#'
#' @section On backward induction:
#'
#'   In keeping with the notation of
#'   [evaluate_finite_horizon_MMDP_markov_policy()], let \eqn{u_{t}^{*}(s)}
#'   denote the maximum expected (possibly discounted) reward accrued from time
#'   \eqn{t} onward if the state at that time is \eqn{s}. It is shown in
#'   Puterman (2014) and elsewhere that the optimal value function
#'   \eqn{u_{1}^{*}} can be determined from the inductive relationship
#' \deqn{u_{t}^{*}(s) = \max_{a} (r_t(s, a) +
#' \rho \sum_{j} p_t(j | s, a)u_{t+1}^{*}(j)),} where the index \eqn{a} runs over the action set of state \eqn{s}, and where
#'   \eqn{u_H^{*}(s) = r_H(s)}. In a finite-horizon MDP with finite state and
#'   action sets, there always exists an optimal policy which is Markov
#'   deterministic. In fact, if for each decision epoch \eqn{t} and each state
#'   \eqn{s} we record those actions \eqn{a} that cause the above maximum to be
#'   attained, then we have constructed all optimal Markov deterministic
#'   policies. The present function calls `MDP_update()` to carry out the
#'   maximization for each epoch and constructs an optimal policy in the iterative
#'   manner described.
#'
#'
#' @section On rewards: Inasmuch as this function is intended for
#'   single-objective problems, a reward `scalar_rewards[[t]][[s]][[a]]` (when
#'   `t` is a decision epoch) or `scalar_rewards[[horizon]][[s]]` (`horizon`
#'   being the length of `scalar_rewards`) should be a number.
#'
#' @section On the optimal policy: Standard Markov decision theory assures us
#'   that backward induction in finite-horizon MDPs having finite states and
#'   actions constructs policies that are optimal at every stage of decision
#'   making. In particular, `optimal_policy` achieves the largest expected
#'   reward not only between time `1` and termination, but between any time `t`
#'   and termination.
#'
#' @param transition_probabilities A numeric list defining the transition
#'   probabilities of the model to be solved. See the documentation of
#'   [evaluate_finite_horizon_MMDP_markov_policy()] for the structure expected
#'   of a finite-horizon transition probability list.
#' @param scalar_rewards A numeric list defining the rewards. See the
#'   documentation of [evaluate_finite_horizon_MMDP_markov_policy()] for the
#'   structure expected of a finite-horizon reward list.
#' @param rho A number between (and including) 0 and 1 to use as the reward
#'   discount factor. The default value is `1` (no discount).
#'
#' @return A list of one numeric matrix and one integer matrix:
#' * `optimal_values` has one row per state and one column per epoch. The (`s, t`)-th entry records the largest expected total reward that can be achieved between time `t` and process termination if the initial state is `s`.
#' * `optimal_policy` represents an optimal policy in the manner of [evaluate_finite_horizon_MMDP_markov_policy()].
#' @export
#'
#' @seealso To ensure that the transition probability and reward lists provided are acceptable, use [are_valid_finite_horizon_transition_probabilities()] and [are_valid_finite_horizon_rewards()].
#'
#' @examples
#' set.seed(1234)
#' MDP <- generate_rand_MMDP(2, list(c(1, 2), c(1, 2, 3)), 5, no_objectives = 1)
#' transition_probabilities <- MDP$P
#' scalar_rewards <- MDP$R
#' rho <- .97
#' solve_finite_horizon_MDP_backward_induction(transition_probabilities, scalar_rewards, rho)
solve_finite_horizon_MDP_backward_induction <- function(transition_probabilities, scalar_rewards, rho = 1) {

  # Testing the fitness and internal consistency of the arguments provided
  if (!are_valid_finite_horizon_transition_probabilities(transition_probabilities) || !are_valid_finite_horizon_rewards(scalar_rewards)) {
    stop("The rewards and/or transition probabilities specified are invalid. See the message(s) returned by are_valid_finite_horizon_transition_probabilities() and/or are_valid_finite_horizon_rewards().")
  }
  else if (length(transition_probabilities) != (length(scalar_rewards) - 1)) {
    stop("The rewards and transition probabilities must not span different decision-making horizons.")
  }
  else if (length(scalar_rewards[[1]][[1]][[1]]) > 1) {
    stop("This function is intended for single-objective finite-horizon problems.")
  }

  no_states <- length(scalar_rewards[[1]])
  horizon <- length(scalar_rewards)

  value_functions_t_onward <- matrix(0, nrow = no_states, ncol = (horizon))
  optimal_subpolicies_t_onward <- matrix(nrow = no_states, ncol = (horizon-1))
  value_functions_t_onward[, horizon] <- unlist(scalar_rewards[[horizon]])
  for (i in 0:(horizon-2)) {
    temp <- MDP_update(transition_probabilities, scalar_rewards, horizon-1-i, rho, value_functions_t_onward[, horizon-i])
    value_functions_t_onward[, horizon-1-i] <- temp[[1]]
    optimal_subpolicies_t_onward[, horizon-1-i] <- temp[[2]]
  }

  rownames(value_functions_t_onward) <- paste("State", seq_len(no_states))
  colnames(value_functions_t_onward) <- paste("t = ", seq_len(horizon))
  rownames(optimal_subpolicies_t_onward) <- rownames(value_functions_t_onward)
  colnames(optimal_subpolicies_t_onward) <- paste("t = ", seq_len(horizon-1))

  return(list(optimal_values = value_functions_t_onward, optimal_policy = optimal_subpolicies_t_onward))
}

#' Solve dynamic programming equations for a single-objective finite-horizon
#' Markov decision process
#'
#' `MDP_update()` is an adjunct to
#' `solve_finite_horizon_MDP_backward_induction()` that solves the standard,
#' single-objective dynamic programming equations at time `t` assuming the
#' optimal value function for the period between `t+1` and termination is
#' `V_next`.
#'
#' Let \eqn{v_{t+1}(j)} denote the value of state \eqn{j} as given by `V_next`.
#' Then, for each state \eqn{s}, this function finds an action \eqn{a_s} that
#' causes the quantity \deqn{r_{t}(s, a_s) + \rho \sum_{j}v_{t+1}(j)} to be
#' maximized. When \eqn{v_{t+1}} is optimal for the period between \eqn{t+1} and
#' termination, this means solving the optimality equations at time \eqn{t}.
#'
#' @param transition_probabilities A numeric list describing the finite-horizon
#'   transition probabilities of the model under study. See the documentation of
#'   [evaluate_finite_horizon_MMDP_markov_policy()] for the precise form of such
#'   a list.
#' @param scalar_rewards A numeric list describing the model's rewards. See the
#'   documentation of [evaluate_finite_horizon_MMDP_markov_policy()] for the
#'   precise form of a finite-horizon reward list.
#' @param t An integer greater than or equal to 1 and less than or equal to the
#'   length of `scalar_rewards`.
#' @param rho A number between (and including) 0 and 1 expressing the reward
#'   discount factor. The default value is `1` (no discount).
#' @param V_next A numeric vector whose length equals the number of states as
#'   determined by `transition_probabilities` and `scalar_rewards`.
#'
#' @return A list comprising one numeric vector and one integer vector:
#' * `V_current` is of the length of `V_next`. If `V_next` is optimal from time `t+1` onward, then the `s`-th entry in `V_current` yields the optimal value for the period between `t` and termination if the state at time `t` is `s`.
#' * `optimal_actions`, also of the length of `V_next`, records the actions which cause the maximums in the dynamic programming equations for time `t` to be attained.
#' @noRd
#'
#' @examples
MDP_update <- function(transition_probabilities, scalar_rewards, t, rho, V_next) {

  # Testing the fitness and internal consistency of the arguments passed
  if (!are_valid_finite_horizon_transition_probabilities(transition_probabilities) || !are_valid_finite_horizon_rewards(scalar_rewards)) {
    stop("The rewards and/or transition probabilities specified are invalid. See the message(s) returned by are_valid_finite_horizon_transition_probabilities() and/or are_valid_finite_horizon_rewards() for an explanation.")
  }
  else if (length(transition_probabilities) != (length(scalar_rewards) - 1)) {
    stop("The rewards and transition probabilities must not span different decision-making horizons.")
  }
  else if (length(transition_probabilities[[1]]) != length(scalar_rewards[[1]])) {
    stop("The rewards and transition probabilities you have provided do not specify the same state set.")
  }
  else if (!all.equal(lengths(transition_probabilities[[1]]), lengths(scalar_rewards[[1]]))) {
    stop("The rewards and transition probabilities you have provided do not specify the same action sets.")
  }
  else if (t > length(scalar_rewards)) {
    stop("t must not be greater than the horizon allowed by the model's rewards and transition probabilities.")
  }
  else if (length(scalar_rewards[[1]][[1]][[1]]) > 1) {
    stop("This function is intended for single-objective problems only.")
  }
  #else if (!is.list(V_next)) {
    #stop("prev_value_function does not bear the form required of a solution to the finite-horizon efficiency equations. See the accompanying documentation for more information.")
  #}
  else if (!is.vector(V_next)) {
    stop("V_next must be a vector with the same number of components as there are states in the model.")
  }
  else if (length(V_next) != length(scalar_rewards[[1]])) {
    stop("The dimensions of V_next and scalar_rewards as provided are incompatible. The number of rows in V_next (that is, the number of states in the model) must equal the length of an entry in scalar_rewards.")
  }

  no_states <- length(scalar_rewards[[1]])
  action_sets <- lapply(lengths(scalar_rewards[[1]]), function(n) 1:n)
  V_current <- numeric(no_states)
  optimal_actions <- numeric(no_states)
  for (s in 1:no_states) {
    actions <- action_sets[[s]]
    action_values <- numeric(length(actions))
    for (i in seq_along(actions)) {
      a <- actions[i]
      immediate_reward <- scalar_rewards[[t]][[s]][[a]]
      P_t <- transition_probabilities[[t]][[s]][[a]]
      action_values[i] <- immediate_reward + rho * P_t %*% V_next
    }
    V_current[s] <- max(action_values)
    optimal_actions[s] <- actions[which.max(action_values)]
  }

  return(list(V_current, optimal_actions))
}

#' Optimize a finite-horizon multi-objective Markov decision process through
#' vector-valued dynamic programming
#'
#' `solve_finite_horizon_MMDP_backward_induction()` finds the Pareto efficient
#' values of a finite-horizon multi-objective Markov decision process by solving
#' a vector-valued extension of the dynamic programming equations.
#'
#' This function solves a multi-objective counterpart to the optimality
#' equations involved in [solve_finite_horizon_MDP_backward_induction()]. In
#' order that we define the new equations and the manner in which they characterize the
#' efficient values of a finite-horizon MMDP, it is necessary to expand the policy space to
#' allow for a special kind of policy which is not Markov. If \eqn{h_t =
#' (s_1, \dots, s_t)} denotes a sequence of states observed prior to and
#' including time \eqn{t}, then we shall be interested in policies that map each
#' such sequence to some action permissible in \eqn{s_t}. Notice that this policy space subsumes Markov deterministic policies, which depend on state
#' histories only through the current state. For any policy \eqn{\pi}, let \eqn{u_t^{\pi}(s)} denote the expected total reward vector it achieves from time \eqn{t} onward if the state at that time is \eqn{s}. The set of all such expected reward vectors is denoted by \eqn{U_t(s)}. Then, if \eqn{U^{*}_t(s)} represents the efficient subset of \eqn{U_t(s)}, Mifrani (2024) shows that
#' \eqn{U^{*}_t(s)} can be determined from the recursion
#' \deqn{U^{*}_t(s) = e\left(\bigcup_{a}\left(\{R_t(s, a)\} \bigoplus
#' \sum_{j}p_t(j | s, a)U^{*}_{t + 1}(j)\right)\right),}
#'  where \eqn{e(X)} denotes the efficient subset of some vector set \eqn{X}, and \eqn{\bigoplus} denotes a sum set.
#'
#' This function finds \eqn{U^{*}_1(s)} for each initial state \eqn{s}
#' by carrying out the above recursion backward in time, starting from the boundary condition that the efficient value sets at termination, \eqn{U^{*}_H(s)}, reduce to the terminal rewards \eqn{R_H(s)}.
#'
#' To compute the efficient subsets involved at each epoch, the function calls either `exact_MMDP_update()` or `heuristic_MMDP_update()`, depending on whether the `method` sought is enumerative or heuristic. The former relies on [efficient_subset_sort_prune()], whereas the latter
#'  uses [nsga2R::nsga2R()], an implementation of the popular Non-dominated
#' Sorting Genetic Algorithm II (more commonly known as NSGA-II).
#'
#'
#' @param transition_probabilities A numeric list specifying the finite-horizon
#'   transition probabilities of the model under study. See the documentation of
#'   [evaluate_finite_horizon_MMDP_markov_policy()] for a description of what
#'   such a list should look like.
#' @param rewards A numeric list defining the model's reward structure. See the
#'   documentation of [evaluate_finite_horizon_MMDP_markov_policy()] for a
#'   description of what such a list should look like.
#' @param rho A number between (and including) 0 and 1 to use as the reward
#'   discount factor. The default value is `1` (no discount).
#' @param method A string indicating the method by which the efficiency
#'   equations are solved: `"exact"` for enumeration or `"heuristic"` for
#'   approximate solution using a genetic algorithm (default).
#'
#' @return A list with three components:
#' * `efficient_values`, a numeric list the length of `rewards[[t]]` / `transition_probabilities[[t]]` (i.e., the cardinality of the state set) in which each component is a numeric matrix representing the efficient value set
#' \eqn{U^{*}_1(s)} relative to some starting state \eqn{s} (see the Details section). The column vectors of each matrix give the efficient values that can be achieved from the corresponding state, or an approximation of these values in case the solution procedure is `heuristic`. The length of these column vectors is the number of objectives.
#' * `compromise_solution`, a numeric list the length of `efficient_values` that stores, for each initial state, the "compromise" value obtained by application of [compromise_solution()] to the columns of the matrix in `efficient_values` corresponding to that state. See the documentation of [compromise_solution()] for more information.
#' * `time`, the CPU time in seconds.
#' @export
#'
#' @references Mifrani, A. (2025). A counterexample and a corrective to the
#'   vector extension of the Bellman equations of a Markov decision process.
#' *Annals of Operations Research, 345*(1), 351-369.
#'
#' @seealso To ensure that the transition probability and reward lists provided are acceptable, use [are_valid_finite_horizon_transition_probabilities()] and [are_valid_finite_horizon_rewards()].
#' @examples
#' set.seed(1234)
#' MMDP <- generate_rand_MMDP(no_states = 2, action_sets = list(c(1, 2), c(1, 2, 3)),
#'                            horizon = 5, no_objectives = 2)
#' transition_probabilities <- MMDP$P
#' rewards <- MMDP$R
#' solve_finite_horizon_MMDP_backward_induction(transition_probabilities, rewards)
solve_finite_horizon_MMDP_backward_induction <- function(transition_probabilities, rewards, rho = 1, method = "heuristic") {

  # Testing the fitness and internal consistency of the arguments provided
  if (!are_valid_finite_horizon_transition_probabilities(transition_probabilities) || !are_valid_finite_horizon_rewards(rewards)) {
    stop("The rewards and/or transition probabilities specified are invalid. See the message(s) returned by are_valid_finite_horizon_transition_probabilities() and/or are_valid_finite_horizon_rewards().")
  }
  else if (length(transition_probabilities) != (length(rewards) - 1)) {
    stop("The rewards and transition probabilities must not span different decision-making horizons.")
  }
  else if (length(transition_probabilities[[1]]) != length(rewards[[1]])) {
    stop("The rewards and transition probabilities you have provided do not specify the same state set.")
  }
  else if (!all.equal(lengths(transition_probabilities[[1]]), lengths(rewards[[1]]))) {
    stop("The rewards and transition probabilities you have provided do not specify the same action sets.")
  }
  else if (rho > 1) {
    stop("The discount factor rho should be less than or equal to one.")
  }
  else if (rho < 0) {
    stop("The discount factor rho should be nonnegative.")
  }
  else if (method != "exact" && method != "heuristic") {
    stop("The method argument admits one of two options: heuristic and exact.")
  }

  start <- as.POSIXlt(Sys.time())
  horizon <- length(rewards)
  # Recall that the value functions of an MMDP are set-valued
  value_functions_t_onward <- list()
  value_functions_t_onward[[1]] <- lapply(rewards[[horizon]], as.matrix)
  if (method == "heuristic") {
    for (i in 2:(horizon)) {
      value_functions_t_onward[[i]] <- heuristic_MMDP_update(transition_probabilities, rewards, horizon-i+1, rho, value_functions_t_onward[[i-1]])
      #message(i)
    }
  }
  else if (method == "exact") {
    for (i in 2:(horizon)) {
      value_functions_t_onward[[i]] <- exact_MMDP_update(transition_probabilities, rewards, horizon-i+1, rho, value_functions_t_onward[[i-1]])
      #message(i)
    }
  }

  end <- as.POSIXlt(Sys.time())

  efficient_values <- value_functions_t_onward[[horizon]]
  compromise_solution <- lapply(value_functions_t_onward[[horizon]], compromise_solution)
  time <- end - start

  names(efficient_values) <- paste("Starting state: State", seq_len(length(efficient_values)))
  names(compromise_solution) <- paste("Starting state: State", seq_len(length(efficient_values)))
  for (s in 1:length(efficient_values)) {
    colnames(efficient_values[[s]]) <- paste("Policy", seq_len(ncol(efficient_values[[s]])))
    rownames(efficient_values[[s]]) <- paste("Objective", seq_len(nrow(efficient_values[[s]])))
    rownames(compromise_solution[[s]]) <- paste("Objective", seq_len(nrow(compromise_solution[[s]])))
  }

  return(list(efficient_values = efficient_values, compromise_solution = compromise_solution, time = end - start))
}

#' Approximately solve a vector-valued dynamic programming equation for a
#' finite-horizon multi-objective Markov decision process
#'
#' `heuristic_MMDP_update()` is an adjunct to
#' `solve_finite_horizon_MMDP_backward_induction()` that uses a genetic
#' algorithm (NSGA-II) to heuristically solve a finite-horizon multi-objective
#' Markov decision process's efficiency equations for time `t` assuming the efficient value functions for the period beginning in time `t+1`
#' are tabulated in `V_next`.
#'
#' This function calls [nsga2R::nsga2R()], an implementation of Non-dominated
#' Sorting Genetic Algorithm II, to approximate the efficient sets that define
#' the efficiency equations for time `t`.
#'
#' @param transition_probabilities A numeric list describing the transition
#'   probabilities of the finite-horizon model under study. See the
#'   documentation of [evaluate_finite_horizon_MMDP_markov_policy()] for the
#'   detailed structure of such a list.
#' @param rewards A numeric list describing the rewards. See the documentation
#'   of [evaluate_finite_horizon_MMDP_markov_policy()] for the detailed
#'   structure of such a list.
#' @param t An integer greater than or equal to 1 and less than or equal to the
#'   decision-making horizon determined by the length of `rewards`.
#' @param rho A number between (and including) 0 and 1 to use as the reward
#'   discount factor. The default value is `1` (no discount).
#' @param V_next A list with one numeric matrix per state in which each matrix
#'   represents a value set relative to some initial state. The elements of each value set
#'   are given by the columns of the relevant matrix.
#'
#' @return A numeric list with the same length and number of constituent
#'   matrices as `V_next`.
#' @noRd
#'
#' @references Mifrani, A. (2025). A counterexample and a corrective to the
#' vector extension of the Bellman equations of a Markov decision process.
#' *Annals of Operations Research, 345*(1), 351-369.
#'
#' Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. A. M. T. (2002). A fast and
#' elitist multiobjective genetic algorithm: NSGA-II.
#' *IEEE transactions on evolutionary computation, 6*(2), 182-197.
#'
#' @examples
heuristic_MMDP_update <- function(transition_probabilities, rewards, t, rho, V_next) {
  # Testing the fitness and internal consistency of the arguments passed
  if (!are_valid_finite_horizon_transition_probabilities(transition_probabilities) || !are_valid_finite_horizon_rewards(rewards)) {
    stop("The rewards and/or transition probabilities specified are invalid. See the message(s) returned by are_valid_finite_horizon_transition_probabilities() and/or are_valid_finite_horizon_rewards() for an explanation.")
  }
  else if (length(transition_probabilities) != (length(rewards) - 1)) {
    stop("The rewards and transition probabilities must not span different decision-making horizons.")
  }
  else if (length(transition_probabilities[[1]]) != length(rewards[[1]])) {
    stop("The rewards and transition probabilities you have provided do not specify the same state set.")
  }
  else if (!all.equal(lengths(transition_probabilities[[1]]), lengths(rewards[[1]]))) {
    stop("The rewards and transition probabilities you have provided do not specify the same action sets.")
  }
  else if (t > length(rewards)) {
    stop("t must not be greater than the horizon allowed by the model's rewards and transition probabilities.")
  }
  else if (!is.list(V_next)) {
    stop("V_next does not bear the form required of a solution to the finite-horizon efficiency equations. See the accompanying documentation for more information.")
  }
  else if ((length(V_next) != length(rewards[[1]])) || (!all(unlist(lapply(V_next, function(v) (nrow(v) ==  length(rewards[[1]][[1]][[1]]))))))) {
    stop("V_next does not bear the form required of a solution to the finite-horizon efficiency equations. See the accompanying documentation for more information.")
  }

  no_objectives <- length(rewards[[1]][[1]][[1]])
  no_states <- length(rewards[[1]])
  action_sets <- lapply(lengths(rewards[[1]]), function(n) 1:n)

  V_current <- list()
  for (s in 1:no_states) {
    expected_total_rewards <- list()
    action_indices <- seq_along(action_sets[[s]])
    for (a_index in action_indices) {
      P_t <- transition_probabilities[[t]][[s]][[a_index]]
      expected_total_rewards[[a_index]] <- rewards[[t]][[s]][[a_index]] + rho * sum_set(transition_probabilities_times_sets(P_t, V_next))
    }
    expected_total_rewards <- Reduce("cbind", expected_total_rewards)
    V_current[[s]] <- -t(nsga2R(fn = function(x) {-expected_total_rewards[, sample(ncol(expected_total_rewards), size = 1)]}, varNo = 2, objDim = no_objectives, lowerBounds = c(1,1), upperBounds = c(1, 1), popSize = 20)$objectives)
    V_current[[s]] <- as.matrix(V_current[[s]][, !duplicated(V_current[[s]], MARGIN = 2)])
  }
  return(V_current)
}

#' Solve a vector-valued dynamic programming equation for a finite-horizon
#' multi-objective Markov decision process
#'
#' `exact_MMDP_update()` is an adjunct to
#' `solve_finite_horizon_MMDP_backward_induction()` that uses exhaustive search
#' to solve a finite-horizon multi-objective Markov decision process's
#' efficiency equations for time `t` assuming the efficient value
#' functions for the period beginning in time `t+1` are tabulated in `V_next`.
#'
#' This function calls [efficient_subset_sort_prune()] to enumerate the
#' efficient sets required by the efficiency equations for time `t`.
#'
#' @param transition_probabilities A numeric list describing the transition
#'   probabilities of the finite-horizon model under study. See the
#'   documentation of [evaluate_finite_horizon_MMDP_markov_policy()] for the
#'   detailed structure of such a list.
#' @param rewards A numeric list describing the rewards. See the documentation
#'   of [evaluate_finite_horizon_MMDP_markov_policy()] for the detailed
#'   structure of such a list.
#' @param t An integer greater than or equal to 1 and less than or equal to the
#'   decision-making horizon determined by the length of `rewards`.
#' @param rho A number between (and including) 0 and 1 to use as the reward
#'   discount factor. The default value is `1` (no discount).
#' @param V_next A list with one numeric matrix per state in which each matrix
#'   represents a value set relative to some initial state. The value vectors
#'   are given by the matrices' columns.
#'
#' @return A numeric list with the same length and number of constituent
#'   matrices as `V_next`.
#' @noRd
#'
#' @references Mifrani, A. (2025). A counterexample and a corrective to the
#'   vector extension of the Bellman equations of a Markov decision process.
#' *Annals of Operations Research, 345*(1), 351-369.
#'
#'
#'
#' @examples
exact_MMDP_update <- function(transition_probabilities, rewards, t, rho, V_next) {

  # Testing the fitness and internal consistency of the arguments provided
  if (!are_valid_finite_horizon_transition_probabilities(transition_probabilities) || !are_valid_finite_horizon_rewards(rewards)) {
    stop("The rewards and/or transition probabilities specified are invalid. See the message(s) returned by are_valid_finite_horizon_transition_probabilities() and/or are_valid_finite_horizon_rewards() for an explanation.")
  }
  else if (length(transition_probabilities) != (length(rewards) - 1)) {
    stop("The rewards and transition probabilities must not span different decision-making horizons.")
  }
  else if (length(transition_probabilities[[1]]) != length(rewards[[1]])) {
    stop("The rewards and transition probabilities you have provided do not specify the same state set.")
  }
  else if (!all.equal(lengths(transition_probabilities[[1]]), lengths(rewards[[1]]))) {
    stop("The rewards and transition probabilities you have provided do not specify the same action sets.")
  }
  else if (t > length(rewards)) {
    stop("t must not be greater than the horizon allowed by the model's rewards and transition probabilities.")
  }
  else if (!is.list(V_next)) {
    stop("V_next does not possess the form required of a solution to the finite-horizon efficiency equations. See the accompanying documentation for more information.")
  }
  else if ((length(V_next) != length(rewards[[1]])) || (!all(unlist(lapply(V_next, function(v) (nrow(v) ==  length(rewards[[1]][[1]][[1]]))))))) {
    stop("V_next does not possess the form required of a solution to the finite-horizon efficiency equations. See the accompanying documentation for more information.")
  }

  no_objectives <- length(rewards[[1]][[1]][[1]])
  no_states <- length(rewards[[1]])
  action_sets <- lapply(lengths(rewards[[1]]), function(n) 1:n)

  V_current <- list()
  for (s in 1:no_states) {
    expected_total_rewards <- list()
    action_indices <- seq_along(action_sets[[s]])
    for (a_index in action_indices) {
      P_t <- transition_probabilities[[t]][[s]][[a_index]]
      expected_total_rewards[[a_index]] <- rewards[[t]][[s]][[a_index]] + rho * sum_set(transition_probabilities_times_sets(P_t, V_next))
    }
    expected_total_rewards <- Reduce("cbind", expected_total_rewards)
    V_current[[s]] <- efficient_subset_sort_prune(expected_total_rewards)[[1]]
  }
  return(V_current)
}

#' Evaluate a stationary policy in a discounted infinite-horizon multi-objective
#' Markov decision process
#'
#' `evaluate_discounted_MMDP_pure_policy()` calculates the expected discounted
#' reward of a stationary deterministic policy in an infinite-horizon
#' multi-objective Markov decision process.
#'
#' Let \eqn{S} denote the number of states. Let \eqn{\pi} be a (Markov) stationary deterministic policy and let \eqn{v^{\pi}(s) = (v_1^{\pi}(s), \dots, v_k^{\pi}(s))} denote the expected discounted reward vector it achieves over an infinite horizon assuming the initial state is \eqn{s}. For each component \eqn{v_i^{\pi}} of this value function, Puterman (2014) shows that
#' \deqn{v_i^{\pi} = (I - \rho P_{\pi})^{-1}R_{\pi, i},}
#' where \eqn{v_i^{\pi}} is viewed as an \eqn{S}-dimensional vector having entries \eqn{v_i^{\pi}(s)}; \eqn{P_{\pi}} is the transition probability matrix induced by \eqn{\pi}; \eqn{R_{\pi, i}} is the \eqn{S}-dimensional vector given by the \eqn{i}-th reward of executing policy \eqn{\pi} in each state \eqn{s}; and \eqn{I} is the identity matrix.
#'
#' @param stationary_transition_probabilities A numeric list describing the
#'   (stationary) transition probabilities of the infinite-horizon model in
#'   which `stationary_policy` is to be evaluated. The list contains a single sublist with one component per state that breaks down as follows:
#' * For each state `s`, `stationary_transition_probabilities[[1]][[s]]` should be a numeric list whose length equals the number of actions permissible in `s`.
#' * For each state `s` and each action `a`, the entry `stationary_transition_probabilities[[1]][[s]][[a]]` should be a numeric vector that gives the probability distribution of the future state if the current state is `s` and the action taken is `a`.
#' @param stationary_rewards A numeric list describing the model's (stationary)
#'   reward structure. The list contains a single sublist with one component per state that breaks down as follows:
#' * For each state `s` we expect `stationary_rewards[[1]][[s]]` to be a numeric list whose length equals the number of actions permissible in `s`.
#' * Each third-level entry `stationary_rewards[[1]][[s]][[a]]` should be a numeric vector that gives the reward for selecting action `a` in state `s` at epoch `t`.
#' @param stationary_policy An integer vector, of length equal to the number of
#'   states as determined by `stationary_transition_probabilities` and
#'   `stationary_rewards`, indicating the action prescribed by the policy for
#'   each state.
#' @param rho A number greater than or equal to 0 but strictly less than 1 to
#'   use as a reward discount factor.
#'
#' @return A numeric matrix, one row per objective and one column per state, in
#'   which the \eqn{s}-th column contains the value vector achieved from initial
#'   state \eqn{s}.
#' @export
#'
#' @examples
#' set.seed(1234)
#' stationary_MMDP <- generate_rand_MMDP(no_states = 2, action_sets = list(c(1, 2), c(1, 2, 3)),
#'                                      horizon = Inf, no_objectives = 2)
#' stationary_transition_probabilities <- stationary_MMDP$P
#' stationary_rewards <- stationary_MMDP$R
#' rho <- .7
#' # What's the value of the stationary policy that dictates action 2 in state 1
#' # and action 3 in state 2?
#' stationary_policy <- c(2, 3)
#' evaluate_discounted_MMDP_pure_policy(stationary_transition_probabilities,
#'                                     stationary_rewards, stationary_policy, rho)
evaluate_discounted_MMDP_pure_policy <- function(stationary_transition_probabilities, stationary_rewards, stationary_policy, rho) {

  # Testing the fitness and internal consistency of the arguments provided
  if (!are_valid_infinite_horizon_transition_probabilities(stationary_transition_probabilities) || !are_valid_infinite_horizon_rewards(stationary_rewards)) {
    stop("The rewards and/or transition probabilities you have specified are invalid. See the message(s) returned by are_valid_infinite_horizon_transition_probabilities() and/or are_valid_infinite_horizon_rewards() for an explanation.")
  }
  else if (length(stationary_transition_probabilities[[1]]) != length(stationary_rewards[[1]])) {
    stop("The rewards and transition probabilities you have provided do not specify the same state set.")
  }
  else if (!all.equal(lengths(stationary_transition_probabilities[[1]]), lengths(stationary_rewards[[1]]))) {
    stop("The rewards and transition probabilities you have provided do not specify the same action sets.")
  }
  else if (!is_valid_infinite_horizon_policy(stationary_policy, stationary_rewards)) {
    stop("The policy given is invalid. See the message returned by is_valid_infinite_horizon_policy().")
  }
  else if (rho >= 1) {
    stop("The discount factor rho must be less than one.")
  }
  else if (rho < 0) {
    stop("The discount factor rho must be nonnegative.")
  }

  no_objectives <- length(stationary_rewards[[1]][[1]][[1]])
  no_states <- length(stationary_rewards[[1]])

  P_d <- matrix(nrow = no_states, ncol = no_states)
  for (s in 1:no_states) {
    P_d[s, ] <- stationary_transition_probabilities[[1]][[s]][[as.numeric(stationary_policy[s])]]
  }

  R_d <- sapply(1:no_states, function(s) stationary_rewards[[1]][[s]][[as.numeric(stationary_policy[s])]])

  # v^{d, d, ...} = (I - rho*P_d)^-1 * R_d
  if (length(R_d) == 2) {no_objectives <- 1}
  else {no_objectives <- nrow(R_d)}

  V <- matrix(nrow = no_objectives, ncol = no_states)
  rownames(V) <- paste("Objective", seq_len(no_objectives))
  colnames(V) <- paste("State", seq_len(no_states))

  if (no_objectives == 1) {
    V[1, ] <- solve(diag(no_states) - rho * P_d, R_d)
  }
  else {
    for (k in 1:no_objectives) {
      V[k, ] <- solve(diag(no_states) - rho * P_d, R_d[k, ])
    }
  }


  return(V)
}


#' Optimize a discounted infinite-horizon multi-objective Markov decision
#' process through policy iteration
#'
#' `solve_discounted_MMDP_policy_iteration()` implements a vector-valued policy
#' iteration procedure to solve a discounted infinite-horizon multi-objective
#' Markov decision process.
#'
#' This function implements Kazuyoshi Wakuta's multi-objective extension of the
#' standard policy iteration method for infinite-horizon MDPs. Wakuta's
#' algorithm produces stationary deterministic (also known as pure) policies
#' which achieve a Pareto efficient expected discounted total reward vector from any initial state.
#'
#' @param stationary_transition_probabilities A numeric list describing the
#'   (stationary) transition probabilities of the infinite-horizon model under
#'   study. See the documentation of [evaluate_discounted_MMDP_pure_policy()] for the form
#'   required of such a list.
#' @param stationary_rewards A numeric list describing the (stationary) reward
#'   structure. See the documentation of [evaluate_discounted_MMDP_pure_policy()] for the form
#'   required of such a list.
#' @param rho A number greater than or equal to 0 but strictly less than 1 to
#'   use as a reward discount factor.
#'
#' @return A list with two components:
#' * An integer matrix with one column per state in which the rows define the efficient stationary policies found by policy iteration. The `s`-th entry in each row gives the action prescribed by the row's policy for state `s`.
#' * A list containing one numeric matrix per efficient policy that records the policy' value function. The column indices of each matrix represent the initial states and the columns themselves give the value vectors achieved from those states.
#'
#' @export
#'
#' @seealso [solve_discounted_MDP_policy_iteration()] for an implementation of the standard policy iteration method in a discounted single-objective MDP.
#' To ensure that the transition probability and reward lists provided are acceptable, use [are_valid_infinite_horizon_transition_probabilities()] and [are_valid_infinite_horizon_rewards()].
#'
#' @references Wakuta, K. (1995). Vector-valued Markov decision processes and
#'   the systems of linear inequalities.
#' *Stochastic processes and their applications, 56*(1), 159-169.
#'
#' @examples
#' set.seed(1234)
#' no_states <- 2
#' action_sets <- list(c(1, 2), c(1, 2, 3))
#' no_objectives <- 2
#' stationary_MMDP <- generate_rand_MMDP(no_states, action_sets, horizon = Inf, no_objectives)
#' stationary_transition_probabilities <- stationary_MMDP$P
#' stationary_rewards <- stationary_MMDP$R
#' rho <- .7
#'
#' # Let's locate the efficient stationary policies for this infinite-horizon model
#' solution <- solve_discounted_MMDP_policy_iteration(stationary_transition_probabilities,
#'                                                    stationary_rewards, rho)
#' solution$policies
#' # Inspect their value functions
#' solution$value_functions
solve_discounted_MMDP_policy_iteration <- function(stationary_transition_probabilities, stationary_rewards, rho) {

  # Testing the fitness and internal consistency of the arguments provided
  if (!are_valid_infinite_horizon_transition_probabilities(stationary_transition_probabilities) || !are_valid_infinite_horizon_rewards(stationary_rewards)) {
    stop("The rewards and/or transition probabilities you have specified are invalid. See the message(s) returned by are_valid_infinite_horizon_transition_probabilities() and/or are_valid_infinite_horizon_rewards() for an explanation.")
  }
  else if (length(stationary_transition_probabilities[[1]]) != length(stationary_rewards[[1]])) {
    stop("The rewards and transition probabilities you have provided do not specify the same state set.")
  }
  else if (!all.equal(lengths(stationary_transition_probabilities[[1]]), lengths(stationary_rewards[[1]]))) {
    stop("The rewards and transition probabilities you have provided do not specify the same action sets.")
  }
  else if (rho >= 1) {
    stop("The discount factor rho must be less than one.")
  }
  else if (rho < 0) {
    stop("The discount factor rho must be nonnegative.")
  }

  no_objectives <- length(stationary_rewards[[1]][[1]][[1]])
  no_states <- length(stationary_rewards[[1]])
  action_sets <- lapply(lengths(stationary_rewards[[1]]), function(n) 1:n)


  # E_0, F_0
  E = NULL
  D = NULL
  all_pure_policies <- expand.grid(action_sets)
  incumbent_policy <- as.numeric(all_pure_policies[1, ])

  # Steps 4 and 5
  while((is.null(E) && is.null(D)) || is.null(E) && (nrow(D) < nrow(all_pure_policies)) || is.null(D) && (nrow(E) < nrow(all_pure_policies)) || (nrow(E)+nrow(D)) < nrow(all_pure_policies)) {

    incumbent_value_function <- evaluate_discounted_MMDP_pure_policy(stationary_transition_probabilities, stationary_rewards, incumbent_policy, rho)

    other_policies_values <- list()

    if (!is.null(E) && !is.null(D)) {
      other_policies <- all_pure_policies[-row.match(rbind(E, D), all_pure_policies), ]
    }
    else if (is.null(E) && !is.null(D)) {
      other_policies <- all_pure_policies[-row.match(D, all_pure_policies), ]
    }
    else if (!is.null(E) && is.null(D)) {
      other_policies <- all_pure_policies[-row.match(E, all_pure_policies), ]
    }
    else if (is.null(E) && is.null(D)) {
      other_policies <- all_pure_policies
    }

    for (policy in 1:nrow(other_policies)) {
      other_policies_values[[policy]] <- discounted_bellman_operator(stationary_transition_probabilities, stationary_rewards, as.numeric(other_policies[policy, ]), incumbent_value_function, rho)
    }

    A <- other_policies[value_function_domination_sets(incumbent_value_function, other_policies_values)[[1]], ]
    B <- other_policies[value_function_domination_sets(incumbent_value_function, other_policies_values)[[2]], ]
    C <- other_policies[value_function_domination_sets(incumbent_value_function, other_policies_values)[[3]], ]

    if (nrow(A) > 0) {
      D <- rbind(D, B, C)
      incumbent_policy <- as.numeric(A[1,])
    }
    else {
      E <- unique(rbind(E, C))
      D <- unique(rbind(D, B))
      all_policies_but_E_and_D <- all_pure_policies[-row.match(rbind(E, D), all_pure_policies), ]
      incumbent_policy <- as.numeric(all_policies_but_E_and_D[1, ])
    }
  }

  # Step 6: Solve the linear systems (S_1), ..., (S_N) associated with each policy in E
  no_candidate_policies <- nrow(E)
  E_prime <- NULL
  for (policy in 1:no_candidate_policies) {
    Wakuta_matrices <- lapply(1:no_states, function(s) compute_Wakuta_matrix(stationary_transition_probabilities, stationary_rewards, rho, as.numeric(E[policy, ]), initial_state = s, type = 1))
    feasibility_vector <- rep(T, no_states)
    for (s in 1:no_states) {
      A <- rbind(Wakuta_matrices[[s]], -diag(no_objectives))
      feasibility_vector[s] <- is_feasible(A, b = rep(0, times = no_objectives + nrow(Wakuta_matrices[[s]])), neq = 0, nleq = nrow(Wakuta_matrices[[s]]))
    }
    if (all(feasibility_vector)) {E_prime <- rbind(E_prime, E[policy, ])}
  }

  # Step 7: Solve the linear systems (F_1), ..., (F_N) associated with each policy in E' (E_prime)
  no_candidate_policies <- nrow(E_prime)
  E_prime_prime <- NULL
  for (policy in 1:no_candidate_policies) {
    Wakuta_matrices <- lapply(1:no_states, function(s) compute_Wakuta_matrix(stationary_transition_probabilities, stationary_rewards, rho, as.numeric(E_prime[policy, ]), initial_state = s, type = 2))
    feasibility_vector <- rep(T, no_states)
    for (s in 1:no_states) {
      A <- rbind(Wakuta_matrices[[s]], -diag(no_objectives))
      feasibility_vector[s] <- is_feasible(A, b = rep(0, times = no_objectives + nrow(Wakuta_matrices[[s]])), neq = 0, nleq = nrow(Wakuta_matrices[[s]]))
    }
    if (all(feasibility_vector)) {E_prime_prime <- rbind(E_prime_prime, E_prime[policy, ])}
  }

  # If E'' = E', we are done; otherwise go to Step 8
  if (all(E_prime_prime == E_prime)) {
    message(paste0("Number of efficient pure policies found through policy iteration: ", nrow(E_prime)))
    rownames(E_prime) <- paste("Policy", seq_len(nrow(E_prime)))
    colnames(E_prime) <- paste("State", seq_len(no_states))
    value_functions <- lapply(1:nrow(E_prime), function(policy) evaluate_discounted_MMDP_pure_policy(stationary_transition_probabilities, stationary_rewards, as.numeric(E_prime[policy,]), rho))
    for (policy in 1:nrow(E_prime)) {
      rownames(value_functions[[policy]]) <- paste("Objective", seq_len(no_objectives))
      colnames(value_functions[[policy]]) <- colnames(E_prime)
    }
    return(list(policies = E_prime, value_functions = value_functions))
  }

  # Step 8 (Optional; see comment above): Solve the linear systems (U_1), ..., (U_N) for each policy in E' to the exclusion of E''
  keys1 <- row_to_key(E_prime)
  keys2 <- row_to_key(E_prime_prime)
  remaining_policies <- E_prime[!keys1 %in% keys2, ]
  no_candidate_policies <- nrow(remaining_policies)
  E_prime_prime_prime <- NULL
  for (policy in 1:no_candidate_policies) {
    Wakuta_matrices <- lapply(1:no_states, function(s) compute_Wakuta_matrix_type_3(stationary_transition_probabilities, stationary_rewards, rho, as.numeric(remaining_policies[policy, ]), initial_state = s, as.numeric(remaining_policies)))
    feasibility_vector <- rep(T, no_states)
    for (s in 1:no_states) {
      A <- rbind(Wakuta_matrices[[s]], -diag(no_objectives))
      feasibility_vector[s] <- is_feasible(A, b = rep(0, times = no_objectives + nrow(Wakuta_matrices[[s]])), neq = 0, nleq = nrow(Wakuta_matrices[[s]]))
    }
    if (all(feasibility_vector)) {E_prime_prime_prime <- rbind(E_prime_prime_prime, remaining_policies[policy, ])}
  }
  rownames(E_prime_prime_prime) <- paste("Policy", seq_len(nrow(E_prime_prime_prime)))
  colnames(E_prime_prime_prime) <- paste("State", seq_len(no_states))
  value_functions <- lapply(1:nrow(E_prime_prime_prime), function(policy) evaluate_discounted_MMDP_pure_policy(stationary_transition_probabilities, stationary_rewards, as.numeric(E_prime_prime_prime[policy,]), rho))
  for (policy in 1:nrow(E_prime_prime_prime)) {
    rownames(value_functions[[policy]]) <- paste("Objective", seq_len(no_objectives))
    colnames(value_functions[[policy]]) <- colnames(E_prime_prime_prime)
  }
  message(paste0("Number of efficient pure policies found through policy iteration: ", nrow(E_prime_prime_prime)))
  return(list(policies = E_prime_prime_prime, value_functions = value_functions))

}


#' Apply a stationary Bellman-type operator to a vector-valued value function
#'
#' `discounted_bellman_operator()` is an adjunct to
#' `solve_discounted_MMDP_policy_iteration()` and
#' `solve_discounted_MDP_policy_iteration()` that operates a standard linear
#' transformation on a value function `value_function` for a stationary
#' infinite-horizon multi-objective Markov decision process.
#'
#' This function uses `value_function` and `stationary_policy` to compute a new value function.
#' Let \eqn{R_{\pi}(s)} denote
#' the reward vector for executing policy `stationary_policy` in state \eqn{s}---
#' which is to say `stationary_rewards[[1]][[s]][[stationary_policy[s]]]`---and
#' let \eqn{R_{\pi}} denote the matrix formed by these column vectors. Similarly,
#' let \eqn{P_{\pi}} be the square matrix whose \eqn{(s, j)}-th entry is given by
#' the probability of transition from state \eqn{s} to state \eqn{j} under
#' `stationary_policy`, i.e.,
#' `stationary_probabilities[[1]][[s]][[stationary_policy[s]]][[j]]`. Then, if the letter
#' \eqn{V} denotes `value_function`, this function returns the quantity
#' \deqn{R_{\pi} + \rho V P'_{\pi},} where \eqn{P'_{\pi}} denotes the transpose of
#' \eqn{P_{\pi}}. The resulting matrix bears the same dimensions as `value_function`.
#'
#' In the special case where `value_function` stores the value function of `stationary_policy`, the transformation leaves `value_function` unchanged as per standard discounted MDP theory.
#'
#' @param stationary_transition_probabilities A numeric list defining the
#'   (stationary) transition probabilities of the infinite-horizon model under
#'   study. See the documentation of [evaluate_discounted_MMDP_pure_policy()] for the form
#'   required of such a list.
#' @param stationary_rewards A numeric list defining the (stationary) reward
#'   structure. See the documentation of [evaluate_discounted_MMDP_pure_policy()] for the form
#'   required of such a list.
#' @param stationary_policy An integer vector, of length equal to the number of
#'   states, representing a stationary policy. Each element in the vector gives
#'   the index of the action prescribed for the corresponding state.
#' @param value_function A numeric matrix, with one column per state,
#'   representing the value function to be transformed. The `s`-th column is the
#'   value vector corresponding to initial state `s`.
#' @param rho A number greater than or equal to 0 but strictly less than 1 to
#'   use as a reward discount factor.
#'
#' @return A numeric matrix having the dimensions of `value_function`,
#'   representing the new value function.
#' @export
#'
#' @references Puterman, M. L. (2014).
#' *Markov decision processes: discrete stochastic dynamic programming*.
#' John Wiley & Sons.
#'
#' @seealso [solve_discounted_MDP_policy_iteration()] for an implementation of the standard policy iteration algorithm and [solve_discounted_MMDP_policy_iteration()] for its multi-objective extension.
#'
#' To ensure that the transition probability and reward lists passed to the function are acceptable, use [are_valid_infinite_horizon_transition_probabilities()] and [are_valid_infinite_horizon_rewards()].
#'
#' @examples
#' # Generate a discounted infinite-horizon bi-objective MDP with two states
#' # and action sets that provide two actions for state 1 and three for state 2
#' set.seed(1234)
#' no_states <- 2
#' action_sets <- list(c(1, 2), c(1, 2, 3))
#' no_objectives <- 2
#' stationary_MMDP <- generate_rand_MMDP(no_states, action_sets, horizon = Inf, no_objectives)
#' stationary_transition_probabilities <- stationary_MMDP$P
#' stationary_rewards <- stationary_MMDP$R
#'
#'
#' # Consider the stationary policy that prescribes action 1 for state 1 and action 3 for state 2
#' stationary_policy <- c(1, 3)
#' # Evaluate this policy assuming a reward discount factor of .7, display the value function
#' rho <- .7
#' value_function <- evaluate_discounted_MMDP_pure_policy(stationary_transition_probabilities,
#'                                                        stationary_rewards, stationary_policy, rho)
#' value_function
#' # What happens when we subject a policy's value function
#' # to the transformation defined by the policy itself?
#' transformed_value_function <- discounted_bellman_operator(stationary_transition_probabilities,
#'                                                           stationary_rewards,
#'                                                           stationary_policy,
#'                                                           value_function, rho)
#' transformed_value_function
discounted_bellman_operator <- function(stationary_transition_probabilities, stationary_rewards, stationary_policy, value_function, rho) {

  # Testing the fitness and internal consistency of the arguments provided
  if (!are_valid_infinite_horizon_transition_probabilities(stationary_transition_probabilities) || !are_valid_infinite_horizon_rewards(stationary_rewards)) {
    stop("The rewards and/or transition probabilities you have specified are invalid. See the message(s) returned by are_valid_infinite_horizon_transition_probabilities() and/or are_valid_infinite_horizon_rewards() for an explanation.")
  }
  else if (length(stationary_transition_probabilities[[1]]) != length(stationary_rewards[[1]])) {
    stop("The rewards and transition probabilities you have provided do not specify the same state set.")
  }
  else if (!all.equal(lengths(stationary_transition_probabilities[[1]]), lengths(stationary_rewards[[1]]))) {
    stop("The rewards and transition probabilities you have provided do not specify the same action sets.")
  }
  else if (!is_valid_infinite_horizon_policy(stationary_policy, stationary_rewards)) {
    stop("The policy given is invalid. See the message returned by is_valid_infinite_horizon_policy().")
  }
  else if (!is.matrix(value_function)) {
    stop("The value function must be given as a no_objectives x S matrix, S being the number of states and no_objectives the number of objectives. Each column in the matrix should give the discounted infinite-horizon value achieved from the apposite state.")
  }
  else if (nrow(value_function) != length(stationary_rewards[[1]][[1]][[1]]) || ncol(value_function) != length(stationary_rewards[[1]])) {
    stop("The number of states and/or objectives implied by the value function is/are inconsistent with that/those implied by the reward structure provided.")
  }
  else if (rho >= 1) {
    stop("The discount factor rho must be less than one.")
  }
  else if (rho < 0) {
    stop("The discount factor rho must be nonnegative.")
  }

  no_objectives <- length(stationary_rewards[[1]][[1]][[1]])
  no_states <- length(stationary_rewards[[1]])

  P_d <- matrix(nrow = no_states, ncol = no_states)
  for (s in 1:no_states) {
    P_d[s, ] <- stationary_transition_probabilities[[1]][[s]][[stationary_policy[s]]]
  }
  temp <- value_function %*% t(P_d)
  new_value_function <- matrix(nrow = no_objectives, ncol = no_states)
  for (s in 1:no_states) {
    new_value_function[, s] <- stationary_rewards[[1]][[s]][[stationary_policy[s]]] + rho * temp[, s]
  }

  return(new_value_function)
}

#' Optimize a discounted infinite-horizon multi-objective Markov decision
#' process through linear programming
#'
#' `solve_discounted_MMDP_linear_programming()` implements a linear programming
#' approach to the solution of a discounted infinite-horizon multi-objective
#' Markov decision process. The function "maximizes" the expected discounted total reward under a fixed `initial_state_distribution` and returns all efficient stationary
#' policies together with their value vectors.
#'
#' This function extends to the multi-objective case the linear programming
#' approach described in Puterman (1994). The basis for this approach is that
#' the expected discounted reward of a policy given an initial-state
#' distribution (`initial_state_distribution`) is linear in the *state-action frequencies*
#' induced by that policy. A state-action frequency is the discounted total joint
#' probability that a particular state will be visited and a particular action will be taken at least once in the process's
#' lifetime. The collection of these frequencies forms a convex polyhedral set that is in a
#' one-to-one correspondence with Markov randomized policies, and whose extreme points uniquely determine the model's stationary
#' deterministic policies. Using Mifrani and Noll's (2025) adaptation of a
#' multi-objective simplex-type algorithm by Ecker and Kouada (1978), this
#' function enumerates those extreme points that are efficient solutions to the
#' multi-objective linear program then calculates the corresponding (efficient) stationary policies.
#'
#' @param stationary_transition_probabilities A numeric list defining the
#'   (stationary) transition probabilities of the infinite-horizon model under
#'   study. See the documentation of [evaluate_discounted_MMDP_pure_policy()] for the form
#'   required of such a list.
#' @param stationary_rewards A numeric list defining the (stationary) reward
#'   structure. See the documentation of [evaluate_discounted_MMDP_pure_policy()] for the form
#'   required of such a list.
#' @param rho A number greater than or equal to 0 but strictly less than 1 to
#'   use as a reward discount factor.
#' @param initial_state_distribution A numeric vector, whose length equals the
#'   number of states and whose entries sum to 1.0, specifying the initial
#'   probability of each state. The linear programming approach requires that
#'   all probabilities be positive.
#' @param max_iter An integer greater than or equal to 1 giving the maximum
#'   number of iterations in the optimization phase. The default value is `100`.
#'
#' @return A list with one integer matrix and one numeric matrix:
#' * `efficient_pure_policies` tabulates the efficient stationary policies found through linear programming. Each row represents a policy and thus possesses as many entries as there are states.
#' * `efficient_value_vectors` records in its columns the value vectors generated by the policies in `efficient_pure_policies`. The length of each column is the number of objectives as determined by `stationary_rewards`.
#' @export
#'
#' @seealso [solve_MOLP()] for an implementation of Mifrani and Noll's version
#'   of the Ecker-Kouada multi-objective linear programming method.
#' To ensure that the transition probability and reward lists passed are acceptable, use [are_valid_infinite_horizon_transition_probabilities()] and [are_valid_infinite_horizon_rewards()].
#'
#' @references Puterman, M. L. (2014). *Markov decision processes: discrete stochastic dynamic programming*. John Wiley & Sons.
#'
#'   Mifrani, A., & Noll, D. (2025). Linear programming for finite-horizon
#'   vector-valued Markov decision processes. *arXiv preprint arXiv:2502.13697*.
#'
#' @examples
#' set.seed(1234)
#' no_states <- 2
#' action_sets <- list(c(1, 2), c(1, 2, 3))
#' no_objectives <- 2
#' stationary_MMDP <- generate_rand_MMDP(no_states, action_sets, horizon = Inf, no_objectives)
#' stationary_transition_probabilities <- stationary_MMDP$P
#' stationary_rewards <- stationary_MMDP$R
#' rho <- .7
#'
#' # Use multi-objective linear programming to solve the model
#' # under a uniform initial-state distribution
#' initial_state_distribution <- c(.5, .5)
#' solve_discounted_MMDP_linear_programming(stationary_transition_probabilities,
#'                                          stationary_rewards, rho, initial_state_distribution)
solve_discounted_MMDP_linear_programming <- function(stationary_transition_probabilities, stationary_rewards, rho, initial_state_distribution, max_iter = 100) {

  # Testing the fitness and internal consistency of the arguments provided
  if (!are_valid_infinite_horizon_transition_probabilities(stationary_transition_probabilities) || !are_valid_infinite_horizon_rewards(stationary_rewards)) {
    stop("The rewards and/or transition probabilities you have specified are invalid. See the message(s) returned by are_valid_infinite_horizon_transition_probabilities() and/or are_valid_infinite_horizon_rewards() for an explanation.")
  }
  else if (length(stationary_transition_probabilities[[1]]) != length(stationary_rewards[[1]])) {
    stop("The rewards and transition probabilities you have provided do not specify the same state set.")
  }
  else if (!is.numeric(initial_state_distribution) || (length(initial_state_distribution) != length(stationary_rewards[[1]])) || (sum(initial_state_distribution) != 1) || !all(initial_state_distribution > 0)) {
    stop("initial_state_distribution should be a positive S-dimensional vector, S being the number of states, whose components sum to 1.0.")
  }
  else if (!all.equal(lengths(stationary_transition_probabilities[[1]]), lengths(stationary_rewards[[1]]))) {
    stop("The rewards and transition probabilities you have provided do not specify the same action sets.")
  }
  else if (rho >= 1) {
    stop("The discount factor rho must be less than one.")
  }
  else if (rho < 0) {
    stop("The discount factor rho must be nonnegative.")
  }
  else if ((max_iter%%1 != 0) || max_iter <= 0) {
    stop("max_iter should be an integer greater than or equal to one.")
  }

  no_objectives <- length(stationary_rewards[[1]][[1]][[1]])
  no_states <- length(stationary_rewards[[1]])
  action_sets <- lapply(lengths(stationary_rewards[[1]]), function(n) 1:n)

  # The linear program has no_objectives objective functions, n := (sum of action set cardinalities) decision variables and m:= no_states equality constraints
  ## The criteria matrix, C, has no_objectives rows for n columns
  n <- sum(unlist(lapply(1:no_states, function(s) length(action_sets[[s]]))))
  C <- matrix(data = rep(0, no_objectives * n), nrow = no_objectives, ncol = n)
  for (i in 1:no_objectives) {
    C_temp <- vector()
    for (s in 1:no_states) {
      C_temp <- c(C_temp, unlist(lapply(action_sets[[s]], function(a) stationary_rewards[[1]][[s]][[a]][i])))
    }
    C[i, ] <- C_temp
  }
  ## The constraint matrix, A, has no_states rows for n columns. The right-hand side vector, b, likewise has no_states rows.
  m <- no_states
  A <- matrix(data = rep(0, m*n), nrow = m, ncol = n)
  b <- rep(0, m)
  ub <- 0
  for (j in 1:m) {
    lb <- ub + 1
    ub <- lb + length(action_sets[[j]]) - 1
    A[j, lb:ub] <- 1
    A[j, ] <- A[j, ] - rho*unlist(lapply(1:no_states, function(s) unlist(lapply(1:length(action_sets[[s]]), function(a) stationary_transition_probabilities[[1]][[s]][[a]][j]))))
    b[j] <- initial_state_distribution[j]
  }

  # Enumerating the linear program's efficient basic feasible solutions using Ecker and Kouada's (1978) simplex-like algorithm
  optim_results <- solve_MOLP(C, A, b, max_iter)

  # Calculating the pure policies associated with the efficient basic feasible solutions detected above
  #efficient_pure_policies <- t(apply(optim_results[[1]], MARGIN = 1, function(x) map_bfs_to_pure_policy(no_states, action_sets, x)))
  efficient_pure_policies <- matrix(nrow = nrow(optim_results[[1]]), ncol = no_states)
  for (bfs in 1:nrow(optim_results[[1]])) {
    ub <- 0
    for (s in 1:no_states) {
      lb <- ub + 1
      ub <- lb + length(action_sets[[s]]) - 1
      efficient_pure_policies[bfs, s] <- which(optim_results[[1]][bfs, ][lb:ub] > 0)
    }
  }

  efficient_value_vectors <- optim_results[[2]]

  message(paste0("Number of efficient pure policies detected through linear programming: ", nrow(efficient_pure_policies)))

  rownames(efficient_pure_policies) <- paste("Policy", seq_len(nrow(efficient_pure_policies)))
  colnames(efficient_pure_policies) <- paste("State", seq_len(no_states))
  #rownames(efficient_value_vectors) <- paste("Objective", seq_len(no_objectives))
  colnames(efficient_value_vectors) <- rownames(efficient_pure_policies)


  return(list(efficient_pure_policies = efficient_pure_policies, efficient_value_vectors = efficient_value_vectors))

}


#' Optimize a discounted infinite-horizon multi-objective Markov decision
#' process through the weighting factor approach
#'
#' `solve_discounted_MMDP_weighting_factor()` generates efficient stationary
#' policies for a discounted infinite-horizon multi-objective Markov decision
#' process using the weighting factor approach.
#'
#' In a discounted infinite-horizon MMDP with finite state and action spaces,
#' there exist stationary policies which are efficient. Furthermore, each such
#' policy maximizes some positive linear combination of the objectives (Wakuta,
#' 1982). Conversely, if we consider the problem of maximizing a positive linear
#' combination (any one would do) of the objectives, we get a
#' single-objective MDP for which any (uniformly) optimal policy is also (uniformly) efficient in the
#' original multi-objective model. This function follows this approach, sampling
#' `no_iterations` positive weight vectors at random
#' then solving the resulting MDPs with the standard policy iteration method.
#'
#' @param stationary_transition_probabilities A numeric list describing the
#'   (stationary) transition probabilities of the infinite-horizon model under
#'   study. See the documentation of [evaluate_discounted_MMDP_pure_policy()] for the form
#'   required of such a list.
#' @param stationary_rewards A numeric list describing the (stationary) reward
#'   structure. See the documentation of [evaluate_discounted_MMDP_pure_policy()] for the form
#'   required of such a list.
#' @param rho A number greater than or equal to 0 but strictly less than 1 to
#'   use as a reward discount factor.
#' @param no_iterations The number of weighting factors, each factor being a
#'   numeric vector with one entry per objective, to be used. The default value is `100`.
#'
#' @return A list with one integer matrix and one numeric sublist:
#' * `efficient_policies`, one row per efficient policy detected and one column per state, tabulates the policies found. The \eqn{(p, s)}-th entry specifies the action prescribed by policy \eqn{p} for state \eqn{s}.
#' * `efficient_value_functions`, comprised of one numeric matrix per policy in `efficient_policies`, records each policy's value function. The \eqn{s}-th column of a value function matrix contains the expected discounted reward attained from initial state \eqn{s}.
#' @export
#'
#' @seealso [solve_discounted_MDP_policy_iteration()],
#'   [evaluate_discounted_MMDP_pure_policy()].
#' To ensure that the transition probability and reward lists passed are acceptable, use [are_valid_infinite_horizon_transition_probabilities()] and [are_valid_infinite_horizon_rewards()].
#'
#' @references Wakuta, K. (1995). Vector-valued Markov decision processes and
#'   the systems of linear inequalities. *Stochastic processes and their applications, 56*(1), 159-169.
#'
#' @examples
#' set.seed(1234)
#' no_states <- 2
#' action_sets <- list(c(1, 2), c(1, 2, 3))
#' no_objectives <- 2
#' stationary_MMDP <- generate_rand_MMDP(no_states, action_sets, horizon = Inf, no_objectives)
#' stationary_transition_probabilities <- stationary_MMDP$P
#' stationary_rewards <- stationary_MMDP$R
#' rho <- .7
#' solve_discounted_MMDP_weighting_factor(stationary_transition_probabilities, stationary_rewards, rho)
solve_discounted_MMDP_weighting_factor <- function(stationary_transition_probabilities, stationary_rewards, rho, no_iterations = 100) {

  ### Testing the fitness and internal consistency of the arguments provided
  if (!are_valid_infinite_horizon_transition_probabilities(stationary_transition_probabilities) || !are_valid_infinite_horizon_rewards(stationary_rewards)) {
    stop("The rewards and/or transition probabilities you have specified are invalid. See the message(s) returned by are_valid_infinite_horizon_transition_probabilities() and/or are_valid_infinite_horizon_rewards() for an explanation.")
  }
  else if (length(stationary_transition_probabilities[[1]]) != length(stationary_rewards[[1]])) {
    stop("The rewards and transition probabilities you have provided do not specify the same state set.")
  }
  else if (!all.equal(lengths(stationary_transition_probabilities[[1]]), lengths(stationary_rewards[[1]]))) {
    stop("The rewards and transition probabilities you have provided do not specify the same action sets.")
  }
  else if (rho >= 1) {
    stop("The discount factor rho must be less than one.")
  }
  else if (rho < 0) {
    stop("The discount factor rho must be nonnegative.")
  }

  no_objectives <- length(stationary_rewards[[1]][[1]][[1]])
  no_states <- length(stationary_rewards[[1]])
  action_sets <- lapply(lengths(stationary_rewards[[1]]), function(n) 1:n)

  weighting_factors <- sapply(1:no_iterations, function(w) rexp(no_objectives))
  weighting_factors <- apply(weighting_factors, MARGIN = 2, function(w) w / sum(w))
  efficient_policies <- matrix(nrow = ncol(weighting_factors), ncol = no_states)
  efficient_value_functions <- NULL

  # Preliminary step: Normalize the objectives to comparable magnitudes so that none of them intrinsically dominates the aggregate rewards
  for (i in 1:no_objectives) {
    reward_i <- list(lapply(1:no_states, function(s) lapply(stationary_rewards[[1]][[s]], function(r) r[i])))
    maximum_i <- solve_discounted_MDP_policy_iteration(stationary_transition_probabilities, reward_i, rho)$optimal_value_function
    weighting_factors[i, ] <- weighting_factors[i, ] / max(maximum_i)
  }

  for (w in 1:ncol(weighting_factors)) {
    weighted_reward_function <- list(lapply(1:no_states, function(s) lapply(stationary_rewards[[1]][[s]], function(r) as.numeric(weighting_factors[, w] %*% r))))
    efficient_policies[w, ] <- solve_discounted_MDP_policy_iteration(stationary_transition_probabilities, weighted_reward_function, rho)[[1]]
    efficient_value_functions[[w]] <- evaluate_discounted_MMDP_pure_policy(stationary_transition_probabilities, stationary_rewards, efficient_policies[w, ], rho)
  }

  temp <- !duplicated(efficient_policies)
  efficient_policies <- efficient_policies[temp, ]
  efficient_value_functions <- efficient_value_functions[temp]
  rownames(efficient_policies) <- paste("Policy", seq_len(nrow(efficient_policies)))
  colnames(efficient_policies) <- paste("State", seq_len(no_states))
  names(efficient_value_functions) <- rownames(efficient_policies)
  for (policy in 1:length(efficient_value_functions)) {
    rownames(efficient_value_functions[[policy]]) <- paste("Objective", seq_len(no_objectives))
    colnames(efficient_value_functions[[policy]]) <- paste("State", seq_len(no_states))
  }

  message(paste0("Number of efficient pure policies found after ", no_iterations, " iterations of the weighting factor approach: ", nrow(unique(efficient_policies))))
  return(list(efficient_policies, efficient_value_functions))

}

#' Optimize a discounted infinite-horizon Markov decision process through policy
#' iteration
#'
#' `solve_discounted_MDP_policy_iteration()` is an adjunct to
#' `solve_discounted_MMDP_weighting_factor()` that implements the policy
#' iteration method for a discounted infinite-horizon MDP. It returns an optimal
#' stationary policy together with its value function.
#'
#' Policy iteration starts from an arbitrary stationary policy then proceeds to
#' improve on it until the policy iterates generated over the course of the
#' algorithm stabilize. The algorithm is as follows, where the phrase "decision rule" refers to a mapping from states to actions:
#'
#' * \strong{Step 1}: Set \eqn{n = 0}, and select an arbitrary decision rule \eqn{d_0}.
#' * \strong{Step 2}: (Policy evaluation) Calculate the value function \eqn{v^n} of the stationary policy that always uses \eqn{d_n}.
#' * \strong{Step 3:} (Policy improvement) For each state \eqn{s}, choose \eqn{d_{n+1}(s)} from the actions that maximize the quantity \deqn{r(s, a) + \rho \sum_{j \in S} p(j | s, a)v^n(j),} where \eqn{r(s, a)} is the reward for taking action \eqn{a} in state \eqn{s} and \eqn{p(j | s, a)} is the probability of transition from \eqn{s} to \eqn{j} under \eqn{a}.
#' * \strong{Step 4:} If \eqn{d_{n+1} = d_{n}}, stop: the policy that repeats \eqn{d_n} is optimal. Otherwise increment \eqn{n} by 1 and return to step 2.
#'
#' When the state and action sets are finite, this procedure terminates in an
#' optimal policy after a finite number of iterations. To implement step 2, the
#' function calls [evaluate_discounted_MMDP_pure_policy()]. To evaluate the
#' quantities in step 3 prior to maximization, the function calls
#' [discounted_bellman_operator()].
#'
#' @param stationary_transition_probabilities A numeric list describing the
#'   (stationary) transition probabilities of the infinite-horizon model under
#'   study. See the documentation of [evaluate_discounted_MMDP_pure_policy()] for the form
#'   required of such a list.
#' @param stationary_scalar_rewards A numeric list describing the model's
#'   rewards. See the documentation of [evaluate_discounted_MMDP_pure_policy()] for the precise
#'   form of a finite-horizon reward list.
#' @param rho A number greater than or equal to 0 but strictly less than 1 to
#'   use as a reward discount factor.
#'
#' @return A list with one integer vector and one numeric vector:
#' * `optimal_policy`, of length equal to the number of states as determined by `stationary_transition_probabilities` and `stationary_scalar_rewards`, defines an optimal stationary policy such that \eqn{s}-th entry provides the action prescribed for state \eqn{s}.
#' * `optimal_value_function`, of the same length as `optimal_policy`, gives the expected discounted reward achieved by `optimal_policy` from each initial state.
#' @export
#' @seealso To ensure that the transition probability and reward lists passed are acceptable, use [are_valid_infinite_horizon_transition_probabilities()] and [are_valid_infinite_horizon_rewards()].
#'
#'
#' @references Puterman, M. L. (2014). *Markov decision processes: discrete stochastic dynamic programming*. John Wiley & Sons.
#'
#' @examples
#' set.seed(1234)
#' stationary_MDP <- generate_rand_MMDP(2, list(c(1, 2), c(1, 2, 3)), horizon = Inf, no_objectives = 1)
#' stationary_transition_probabilities <- stationary_MDP$P
#' stationary_scalar_rewards <- stationary_MDP$R
#' rho <- .7
#' solve_discounted_MDP_policy_iteration(stationary_transition_probabilities,
#'                                       stationary_scalar_rewards, rho)
solve_discounted_MDP_policy_iteration <- function(stationary_transition_probabilities, stationary_scalar_rewards, rho) {

  # Testing the fitness and internal consistency of the arguments provided
  if (!are_valid_infinite_horizon_transition_probabilities(stationary_transition_probabilities) || !are_valid_infinite_horizon_rewards(stationary_scalar_rewards)) {
    stop("The rewards and/or transition probabilities you have specified are invalid. See the message(s) returned by are_valid_infinite_horizon_transition_probabilities() and/or are_valid_infinite_horizon_rewards() for an explanation.")
  }
  else if (length(stationary_transition_probabilities[[1]]) != length(stationary_scalar_rewards[[1]])) {
    stop("The rewards and transition probabilities you have provided do not specify the same state set.")
  }
  else if (!all.equal(lengths(stationary_transition_probabilities[[1]]), lengths(stationary_scalar_rewards[[1]]))) {
    stop("The rewards and transition probabilities you have provided do not specify the same action sets.")
  }
  else if (rho >= 1) {
    stop("The discount factor rho must be less than one.")
  }
  else if (rho < 0) {
    stop("The discount factor rho must be nonnegative.")
  }

  no_states <- length(stationary_scalar_rewards[[1]])
  action_sets <- lapply(lengths(stationary_scalar_rewards[[1]]), function(n) 1:n)

  all_pure_policies <- expand.grid(action_sets)
  incumbent_policy <- all_pure_policies[1, ]
  convergence <- F

  while (!convergence) {
    incumbent_value_function <- evaluate_discounted_MMDP_pure_policy(stationary_transition_probabilities, stationary_scalar_rewards, as.numeric(incumbent_policy), rho)
    improving_policy <- t(apply(all_pure_policies, MARGIN = 1, function(policy) discounted_bellman_operator(stationary_transition_probabilities, stationary_scalar_rewards, policy, incumbent_value_function, rho)))
    improving_policy <- apply(improving_policy, MARGIN = 2, which.max)
    improving_policy <- sapply(1:no_states, function(s) all_pure_policies[improving_policy[s], s])

    if (!all(improving_policy == incumbent_policy)) {
      incumbent_policy <- improving_policy
    }
    else {
      convergence <- T
    }
  }

  optimal_policy <- t(as.matrix(improving_policy))
  colnames(optimal_policy) <- paste("State", seq_len(no_states))
  rownames(optimal_policy) <- "Policy"

  return(list(optimal_policy, optimal_value_function = evaluate_discounted_MMDP_pure_policy(stationary_transition_probabilities, stationary_scalar_rewards, as.numeric(improving_policy), rho)))

}

#' Optimize a finite-horizon multi-objective Markov decision process through the weighting factor approach
#'
#' `solve_finite_horizon_MMDP_weighting_factor()` finds Pareto efficient Markov deterministic policies for a finite-horizon multi-objective Markov decision process, with or without discount, using the weighting factor approach.
#'
#' The premise of the weighting factor approach for finite-horizon problems is identical to the infinite-horizon case as documented under [solve_discounted_MMDP_weighting_factor()], except that the efficient policies that are generated need not be stationary.
#'
#' @param transition_probabilities A numeric list defining the transition
#'   probabilities of the finite-horizon model under study. See the
#'   documentation of [evaluate_finite_horizon_MMDP_markov_policy()] for the
#'   detailed structure of such a list.
#' @param rewards A numeric list defining the model's reward structure. See the
#'   documentation of [evaluate_finite_horizon_MMDP_markov_policy()] for the
#'   detailed structure of such a list.
#' @param rho A number between 0 and 1 representing the reward discount factor.
#'   The default value `1` yields an undiscounted model.
#' @param no_iterations The number of weighting factors, each factor being a
#'   numeric vector with one positive entry per objective, to be used. One single-objective MDP will be solved per factor. The default value is `100`.
#'
#' @return A list with two sublists:
#' * `efficient_policies` is a list where each efficient policy detected by the weighting factor approach occupies an integer matrix. For each of these matrices, the \eqn{(s, t)}-th entry specifies the action prescribed for state \eqn{s} at time \eqn{t}.
#' * `efficient_value_functions`, comprised of one numeric matrix per policy in `efficient_policies`, records each policy's value function. The \eqn{s}-th column of a value function matrix contains the expected discounted reward attained from initial state \eqn{s}.
#' @export
#'
#'
#' @seealso [solve_discounted_MMDP_weighting_factor()] for an analogous implementation in the infinite-horizon case.
#' To ensure that the transition probability and reward lists provided are acceptable, use [are_valid_finite_horizon_transition_probabilities()] and [are_valid_finite_horizon_rewards()].
#' @examples
#' set.seed(1234)
#' no_states <- 2
#' action_sets <- list(c(1, 2), c(1, 2, 3))
#' no_objectives <- 2
#' horizon <- 5
#' MMDP <- generate_rand_MMDP(no_states, action_sets, horizon, no_objectives)
#' transition_probabilities <- MMDP$P
#' rewards <- MMDP$R
#' rho <- .97
#' solve_finite_horizon_MMDP_weighting_factor(transition_probabilities, rewards, rho)
solve_finite_horizon_MMDP_weighting_factor <- function(transition_probabilities, rewards, rho = 1, no_iterations = 100) {

  # Testing the fitness and internal consistency of the arguments provided
  if (!are_valid_finite_horizon_transition_probabilities(transition_probabilities) || !are_valid_finite_horizon_rewards(rewards)) {
    stop("The rewards and/or transition probabilities specified are invalid. See the message(s) returned by are_valid_finite_horizon_transition_probabilities() and/or are_valid_finite_horizon_rewards().")
  }
  else if (length(transition_probabilities) != (length(rewards) - 1)) {
    stop("The rewards and transition probabilities must not span different decision-making horizons.")
  }
  else if (length(transition_probabilities[[1]]) != length(rewards[[1]])) {
    stop("The rewards and transition probabilities you have provided do not specify the same state set.")
  }
  else if (!all.equal(lengths(transition_probabilities[[1]]), lengths(rewards[[1]]))) {
    stop("The rewards and transition probabilities you have provided do not specify the same action sets.")
  }
  else if (rho > 1) {
    stop("The discount factor rho must be less than or equal to one.")
  }
  else if (rho < 0) {
    stop("The discount factor rho must be nonnegative.")
  }

  horizon <- length(rewards)
  no_states <- length(rewards[[1]])
  no_objectives <- length(rewards[[1]][[1]][[1]])
  weighting_factors <- sapply(1:no_iterations, function(w) rexp(no_objectives))
  weighting_factors <- apply(weighting_factors, MARGIN = 2, function(w) w / sum(w))
  efficient_policies <- NULL
  efficient_value_functions <- NULL

  # Preliminary step: Normalize the objectives to comparable magnitudes so that none of them intrinsically dominates the weighted reward function
  for (i in 1:no_objectives) {
    reward_i <- c(lapply(rewards[1:(horizon-1)], function(r) {lapply(1:no_states, function(s) {lapply(r[[s]], function(p) p[i])})}), list(lapply(rewards[[horizon]], function(r) {r[i]})))
    maximum_i <- solve_finite_horizon_MDP_backward_induction(transition_probabilities, reward_i, rho)$optimal_values[, 1]
    weighting_factors[i, ] <- weighting_factors[i, ] / max(maximum_i)
  }



  for (w in 1:ncol(weighting_factors)) {
    weighted_reward_function <- c(lapply(rewards[1:(horizon-1)], function(r) {lapply(1:no_states, function(s) {lapply(r[[s]], function(p) as.numeric(weighting_factors[, w] %*% p))})}), list(lapply(rewards[[horizon]], function(r) {as.numeric(weighting_factors[, w] %*% r)})))
    efficient_policies[[w]] <- solve_finite_horizon_MDP_backward_induction(transition_probabilities, weighted_reward_function, rho)[[2]]
    efficient_value_functions[[w]] <- evaluate_finite_horizon_MMDP_markov_policy(transition_probabilities, rewards, efficient_policies[[w]], rho)
  }

  temp <- !duplicated(efficient_policies)
  efficient_policies <- efficient_policies[temp]
  efficient_value_functions <- efficient_value_functions[temp]

  message(paste0("Number of efficient Markov deterministic policies found after ", no_iterations, " iterations of the weighting factor approach: ", length(unique(efficient_policies))))
  return(list(efficient_policies = efficient_policies, efficient_value_functions = efficient_value_functions))

}

#' Optimize a finite-horizon multi-objective Markov decision
#' process through linear programming
#'
#' `solve_finite_horizon_MMDP_linear_programming()` implements a linear programming
#' approach to the solution of a finite-horizon multi-objective
#' Markov decision process. The function "maximizes" the expected total reward vector under a fixed distribution of initial states `initial_state_distribution`. It returns all efficient Markov deterministic
#' policies along with the values they generate.
#'
#' This function implements the linear programming technique proposed by Mifrani and Noll (2025). The technique rests on the observation that
#' the finite-horizon expected total reward of a policy under an initial-state
#' distribution (`initial_state_distribution`) is linear in the *state-action frequencies*
#' induced by that policy. A state-action frequency \eqn{x_{\pi, t}(s, a)} is the joint
#' probability that the process will enter state \eqn{s} at time \eqn{t} and select action \eqn{a} under policy \eqn{\pi}. The collection of these frequencies forms a convex polyhedral set that is in a
#' one-to-one correspondence with Markov randomized policies, and whose extreme points uniquely determine the model's Markov
#' deterministic policies. Using Mifrani and Noll's adaptation of a
#' multi-objective simplex-type algorithm by Ecker and Kouada (1978), this
#' function enumerates those extreme points that are efficient solutions to the
#' multi-objective linear program then calculates the corresponding (efficient) policies.
#'
#' @param transition_probabilities A numeric list specifying the
#'   transition probabilities of the finite-horizon model under
#'   study. See the documentation of [evaluate_finite_horizon_MMDP_markov_policy()] for the form
#'   required of such a list.
#' @param rewards A numeric list specifying the reward
#'   structure. See the documentation of [evaluate_finite_horizon_MMDP_markov_policy()] for the form
#'   required of such a list.
#' @param initial_state_distribution A numeric vector, whose length equals the
#'   number of states and whose entries sum to 1.0, specifying the initial
#'   probability of each state. The linear programming approach requires that
#'   all probabilities be positive.
#' @param max_iter An integer greater than or equal to 1 giving the maximum
#'   number of iterations in the optimization phase. The default value is `100`.
#'
#' @return A list of one integer sublist and one numeric matrix:
#' * `efficient_policies` contains integer matrices representing the efficient policies detected by linear programming.
#' * `efficient_value_vectors` records in its columns the value vectors generated under `initial_state_distribution` by the policies in `efficient_policies`. The length of each column is the number of objectives as determined by `rewards`.
#' @export
#'
#' @seealso [solve_MOLP()] for an implementation of Mifrani and Noll's version
#'   of the Ecker-Kouada multi-objective linear programming technique.
#' To ensure that the transition probability and reward lists passed are acceptable, use [are_valid_finite_horizon_transition_probabilities()] and [are_valid_finite_horizon_rewards()].
#'
#' @references
#'
#'   Mifrani, A., & Noll, D. (2025). Linear programming for finite-horizon
#'   vector-valued Markov decision processes. *arXiv preprint arXiv:2502.13697*.
#'
#' @examples
#' set.seed(1234)
#' no_states <- 2
#' action_sets <- list(c(1, 2), c(1, 2, 3))
#' no_objectives <- 2
#' horizon <- 5
#' MMDP <- generate_rand_MMDP(no_states, action_sets, horizon, no_objectives)
#' transition_probabilities <- MMDP$P
#' rewards <- MMDP$R
#' # Use multi-objective linear programming to solve the model
#' # under a uniform initial-state distribution
#' initial_state_distribution <- c(.5, .5)
#' solve_finite_horizon_MMDP_linear_programming(transition_probabilities,
#'                                          rewards, initial_state_distribution)
solve_finite_horizon_MMDP_linear_programming <- function(transition_probabilities, rewards, initial_state_distribution, max_iter = 100) {

  # Testing the fitness and internal consistency of the arguments provided
  if (!are_valid_finite_horizon_transition_probabilities(transition_probabilities) || !are_valid_finite_horizon_rewards(rewards)) {
    stop("The rewards and/or transition probabilities specified are invalid. See the message(s) returned by are_valid_finite_horizon_transition_probabilities() and/or are_valid_finite_horizon_rewards().")
  }
  else if (length(transition_probabilities) != (length(rewards) - 1)) {
    stop("The rewards and transition probabilities must not span different decision-making horizons.")
  }
  else if (length(transition_probabilities[[1]]) != length(rewards[[1]])) {
    stop("The rewards and transition probabilities you have provided do not specify the same state set.")
  }
  else if (!all.equal(lengths(transition_probabilities[[1]]), lengths(rewards[[1]]))) {
    stop("The rewards and transition probabilities you have provided do not specify the same action sets.")
  }
  else if (!is.numeric(initial_state_distribution) || (length(initial_state_distribution) != length(rewards[[1]])) || (sum(initial_state_distribution) != 1) || !all(initial_state_distribution > 0)) {
    stop("initial_state_distribution should be a positive S-dimensional vector, S being the number of states, summing to 1.0.")
  }
  else if ((max_iter%%1 != 0) || max_iter <= 0) {
    stop("max_iter should be an integer greater than or equal to one.")
  }

  horizon <- length(rewards)
  no_objectives <- length(rewards[[1]][[1]][[1]])
  no_states <- length(rewards[[1]])
  action_sets <- lapply(lengths(rewards[[1]]), function(n) 1:n)

  # The linear program has no_objectives criteria functions, n := [(horizon-1)*(sum of action set cardinalities) + no_states] decision variables and m:= horizon*no_states equality constraints
  ## The criteria matrix, C, has no_objectives rows for n columns
  total_no_actions <- sum(unlist(lapply(1:no_states, function(s) length(action_sets[[s]]))))
  n <- (horizon-1) * total_no_actions + no_states
  C <- matrix(data = rep(0, no_objectives * n), nrow = no_objectives, ncol = n)
  temp <- c(0, sapply(2:no_states, function(s) sum(unlist(lapply(1:(s-1), function(j) length(action_sets[[j]]))))))
  for (t in 1:(horizon-1)) {
    for (s in 1:no_states) {
      for (a in 1:length(action_sets[[s]])) {
        C[, (t-1)*total_no_actions + temp[s] + a] <- rewards[[t]][[s]][[a]]
      }
    }
  }
  for (s in 1:no_states) {
    C[, (horizon-1)*total_no_actions + s] <- rewards[[horizon]][[s]]
  }
  ## The constraint matrix, A, has m rows for n columns. The right-hand side vector, b, likewise has no_states rows.
  m <- horizon * no_states
  A <- matrix(data = rep(0, m*n), nrow = m, ncol = n)
  b <- rep(0, m)
  row <- 1
  for (j in 1:no_states) {
    for (a in 1:length(action_sets[[j]])) {
      A[row, temp[j] + a] <- 1
    }
    b[row] <- initial_state_distribution[j]
    row <- row + 1
  }
  for (t in 1:(horizon-2)) {
    for (j in 1:no_states) {
      for (a in 1:length(action_sets[[j]])) {
        A[row, t*total_no_actions + temp[j] + a] <- 1
      }
      for (s in 1:no_states) {
        for (a in 1:length(action_sets[[s]])) {
          A[row, (t-1)*total_no_actions + temp[s] + a] <- -transition_probabilities[[t]][[s]][[a]][j]
        }
      }
      b[row] <- 0
      row <- row + 1
    }
  }
  for (j in 1:no_states) {
    A[row, (horizon-1)*total_no_actions + j] <- 1
    for (s in 1:no_states) {
      for (a in 1:length(action_sets[[s]])) {
        A[row, (horizon-2)*total_no_actions + temp[s] + a] <- -transition_probabilities[[horizon-1]][[s]][[a]][j]
      }
    }
    row <- row + 1
  }


  # Enumerating the linear program's efficient basic feasible solutions using Ecker and Kouada's (1978) simplex-like algorithm
  optim_results <- solve_MOLP(C, A, b, max_iter)

  # Calculating the Markov deterministic policies associated with the efficient basic feasible solutions detected above
  efficient_policies <- list()
  for (sol in 1:nrow(optim_results[[1]])) {
    bfs <- optim_results[[1]][sol, ]
    policy <- matrix(nrow = no_states, ncol = horizon-1)
    for (t in 1:(horizon-1)) {
      for (s in 1:no_states) {
        lb <- (t-1)*total_no_actions + temp[s] + 1
        ub <- (t-1)*total_no_actions + temp[s] + length(action_sets[[s]])
        if (all(bfs[lb:ub] == 0)) { # if the b.f.s. is degenerate, set the action arbitrarily (here it is set to action 1)
          policy[s, t] <- 1
        }
        else { # otherwise, use the closed form relating state-action frequencies to the policy that induces them
          policy[s, t] <- as.numeric(which(bfs[lb:ub] > 0))
        }
      }
    }
    rownames(policy) <- paste("State", seq_len(no_states))
    colnames(policy) <- paste("t =", seq_len(horizon-1))
    efficient_policies[[sol]] <- policy
  }

  efficient_value_vectors <- optim_results[[2]]
  colnames(efficient_value_vectors) <- paste("Policy", seq_len(length(efficient_policies)))


  message(paste0("Number of efficient Markov deterministic policies detected through linear programming: ", length(efficient_policies)))


  return(list(efficient_policies = efficient_policies, efficient_value_vectors = efficient_value_vectors))

}

#' Determine whether an integer matrix represents a policy for a given class of finite-horizon problems
#'
#' `is_valid_finite_horizon_policy()` returns `TRUE` if `policy` represents a
#' valid policy for finite-horizon problems with reward list `rewards`. In the opposite case, it returns `FALSE` together with an explanation.
#'
#' This function juxtaposes `policy` against the template documented under [evaluate_finite_horizon_MMDP_markov_policy()] and in the vignette for describing finite-horizon policies. In particular, it checks whether:
#' * the number of rows equals the number of states stipulated by `rewards`;
#' * the number of columns equals the length of the decision-making horizon, minus the terminal epoch, stipulated by `rewards`;
#' * the entries conform to the action sets stipulated by `rewards`.
#'
#' @param policy An integer matrix.
#' @param rewards A numeric list providing a valid finite-horizon reward
#'   structure. See the documentation of [evaluate_finite_horizon_MMDP_markov_policy()] for the form
#'   required of such a list.
#'
#' @return A boolean: `TRUE` if `policy` defines a valid Markov deterministic policy with respect to
#'   problems in which the rewards are given by `rewards`, `FALSE` otherwise.
#' @export
#'
#' @seealso [are_valid_finite_horizon_rewards()]
#'
#' @examples
#' set.seed(1234)
#' no_states <- 2
#' action_sets <- list(c(1, 2), c(1, 2))
#' horizon <- 5
#' no_objectives <- 2
#' MMDP <- generate_rand_MMDP(no_states, action_sets, horizon, no_objectives)
#' rewards <- MMDP$R
#' # An example of a valid policy for a two-state, two-action bi-objective MDP
#' # with four decision epochs
#' policy <- matrix(nrow = no_states, ncol = (horizon - 1), data = c(2, 1, 1, 1,
#'                                                                    1, 2, 1, 2), byrow = TRUE)
#' is_valid_finite_horizon_policy(policy, rewards)
#' # The following policy is invalid because it prescribes no actions for epoch four
#' policy <- matrix(2, ncol = 3, data = c(2, 1, 1,
#'                                                                    1, 2, 1), byrow = TRUE)
#' is_valid_finite_horizon_policy(policy, rewards)
is_valid_finite_horizon_policy <- function(policy, rewards) {

  # Test for the validity of the rewards
  if (!are_valid_finite_horizon_rewards(rewards)) {
    stop("The reward structure provided is invalid.")
  }
  else if (!is.matrix(policy)) {
    message("The policy must be given as a matrix with as many rows as there are states and as many columns as there are decision-making epochs.")
    return(FALSE)
  }
  else if ((nrow(policy) != length(rewards[[1]])) || (ncol(policy) != (length(rewards) - 1))) {
    message("The dimensions of the policy matrix are inconsistent with those of rewards and transition_probabilities.")
    return(FALSE)
  }
  no_states <- length(rewards[[1]])
  action_sets <- lapply(lengths(rewards[[1]]), function(n) 1:n)

  infeasible_action <- F
  for (t in 1:ncol(policy)) {
    for (s in 1:nrow(policy)) {
      if (!(policy[s, t] %in% action_sets[[s]])) {infeasible_action <- T}
    }
  }

  return(!infeasible_action)
}

#' Determine whether an integer vector represents a stationary policy for a given class of infinite-horizon problems
#'
#' `is_valid_infinite_horizon_policy()` returns `TRUE` if `stationary_policy` represents a
#' valid stationary deterministic policy for infinite-horizon problems with reward list `stationary_rewards`. In the opposite case, it returns `FALSE` together with an explanation.
#'
#' This function juxtaposes `stationary_policy` against the template documented under [evaluate_discounted_MMDP_pure_policy()] for describing stationary policies. In particular, it checks whether:
#' * the number of entries equals the number of states implied by `stationary_rewards`;
#' * the entries themselves conform to the action sets implied by `stationary_rewards`.
#'
#' @param stationary_policy An integer vector.
#' @param stationary_rewards A numeric list providing a valid infinite-horizon reward
#'   structure. See the documentation of [evaluate_discounted_MMDP_pure_policy()] for the form
#'   required of such a list.
#'
#' @return A boolean: `TRUE` if `stationary_policy` defines a valid stationary policy with respect to
#'   problems where the rewards are given by `rewards`, `FALSE` otherwise.
#' @export
#'
#' @seealso [are_valid_infinite_horizon_rewards()]
#'
#' @examples
#' no_states <- 2
#' action_sets <- list(c(1, 2), c(1, 2))
#' no_objectives <- 2
#' stationary_MMDP <- generate_rand_MMDP(no_states, action_sets, horizon = Inf, no_objectives)
#' stationary_rewards <- stationary_MMDP$R
#' # An example of a valid stationary policy: send state 1 to action 1, state 2 to action 1
#' stationary_policy <- c(1, 1)
#' is_valid_infinite_horizon_policy(stationary_policy, stationary_rewards)
#' # An example of an invalid stationary policy: send state 1 to action 1,
#' # but send state 2 to an action that falls outside its action set
#' stationary_policy <- c(1, 3)
#' is_valid_infinite_horizon_policy(stationary_policy, stationary_rewards)
#'
is_valid_infinite_horizon_policy <- function(stationary_policy, stationary_rewards) {

  # Test for the validity of the rewards
  if (!are_valid_infinite_horizon_rewards(stationary_rewards)) {
    stop("The reward structure provided is invalid.")
  }
  else if (!is.numeric(stationary_policy)) {
    message("The policy must be given as an S-dimensional vector, S being the number of states in the model.")
    return(FALSE)
  }
  else if (length(stationary_policy) != length(stationary_rewards[[1]])) {
    message("The number of problem states implied by the policy is different than that implied by the reward structure.")
    return(FALSE)
  }

  no_states <- length(stationary_rewards[[1]])
  action_sets <- lapply(lengths(stationary_rewards[[1]]), function(n) 1:n)

  infeasible_action <- F
  for (s in 1:no_states) {
      if (!(stationary_policy[s] %in% action_sets[[s]])) {infeasible_action <- T}
  }

  return(!infeasible_action)
}

#' Determine whether a numeric list represents a valid reward structure for finite-horizon problems
#'
#' `are_valid_finite_horizon_rewards()` returns `TRUE` if `rewards` constitutes a
#' valid reward structure for a finite-horizon problem. In the opposite case, it returns `FALSE` together with an explanatory message.
#'
#' This function juxtaposes `rewards` against the template described in the documentation of [evaluate_finite_horizon_MMDP_markov_policy()] for finite-horizon reward lists.
#'
#' @param rewards A numeric list.
#'
#' @return A boolean: `TRUE` if `rewards` defines a valid finite-horizon reward structure, `FALSE` otherwise.
#' @export
#'
#' @seealso [evaluate_finite_horizon_MMDP_markov_policy()] for the template of a finite-horizon reward list.
#'
#' @examples
#' set.seed(1234)
#' MMDP <- generate_rand_MMDP(2, list(c(1, 2), c(1, 2, 3)), horizon = 5, no_objectives = 2)
#' rewards <- MMDP$R
#' are_valid_finite_horizon_rewards(rewards)
are_valid_finite_horizon_rewards <- function(rewards) {

  if (!is.list(rewards)) {
    message("The rewards should be presented as a list.")
    return(F)
  }
  else if (!is.numeric(unlist(lapply(rewards, function(t) unlist(t))))) {
    message("Please ensure the rewards are entirely numerical.")
    return(F)
  }
  else if (length(rewards) < 2) {
    message("This function is intended for finite-horizon models only. Ensure the length of the reward list (that is, the number of decision-making epochs plus the terminal epoch) is greater than or equal to 2.")
    return(F)
  }
  else if (length(unique(lengths(rewards))) != 1) {
    message("All first-level components of the reward list should be lists of equal length. The length of each such component is the number of states in the model, which does not vary by epoch.")
    return(F)
  }
  else if (nrow(unique(Reduce("rbind", lapply(rewards[1:(length(rewards)-1)], function(t) unlist(lapply(t, length)))))) != 1) {
    message("A state must have the same action set throughout the decision-making period.")
    return(F)
  }
  else if (length(unique(unlist(lapply(rewards[1:(length(rewards)-1)], function(t) unlist(lapply(t, lengths)))))) != 1) {
    message("The number of components in each rewards[[t]][[s]][[a]] (that is, the number of objectives) must not vary by state-action pair nor by epoch.")
    return(F)
  }
  else if (length(unique(unlist(lapply(rewards[[length(rewards)]], length)))) != 1) {
    message("The terminal rewards in rewards[[horizon]] (horizon is the total length of rewards) must not differ in size from the reward functions of the preceding epochs.")
    return(F)
  }
  else {return(T)}
}

#' Determine whether a numeric list represents a valid reward structure for infinite-horizon problems
#'
#' `are_valid_infinite_horizon_rewards()` returns `TRUE` if `stationary_rewards` constitutes a
#' valid reward structure for a stationary infinite-horizon problem. Otherwise, it returns `FALSE` in addition to an explanation.
#'
#' This function juxtaposes `stationary_rewards` against the template described in the documentation of [evaluate_discounted_MMDP_pure_policy()] for infinite-horizon reward lists.
#'
#' @param stationary_rewards A numeric list.
#'
#' @return A boolean: `TRUE` if `stationary_rewards` defines a valid infinite-horizon reward structure, `FALSE` otherwise.
#' @export
#'
#' @seealso [evaluate_discounted_MMDP_pure_policy()]
#'
#' @examples
#' set.seed(1234)
#' stationary_MMDP <- generate_rand_MMDP(2, list(c(1, 2), c(1, 2, 3)),
#'                                       horizon = Inf, no_objectives = 2)
#' stationary_rewards <- stationary_MMDP$R
#' are_valid_infinite_horizon_rewards(stationary_rewards)
are_valid_infinite_horizon_rewards <- function(stationary_rewards) {
  # INPUT:
  # OUTPUT:

  if (!is.list(stationary_rewards)) {
    message("The rewards should be presented as a list.")
    return(F)
  }
  else if (!is.numeric(unlist(lapply(stationary_rewards, function(t) unlist(t))))) {
    message("Please ensure the rewards are entirely numerical.")
    return(F)
  }
  else if (length(stationary_rewards) > 1) {
    message("This function is intended for stationary infinite-horizon models only. Ensure that the length of the reward list is one.")
    return(F)
  }
  else if (length(unique(unlist(lapply(stationary_rewards[[1]], function(s) unlist(lapply(s, length)))))) != 1) {
    message("The size of each stationary_rewards[[1]][[s]][[a]] (that is, the number of optimization objectives) must not vary by state-action pair.")
    return(F)
  }
  else {return(T)}
}


#' Determine whether a numeric list represents a valid transition probability structure for finite-horizon problems
#'
#' `are_valid_finite_horizon_transition_probabilities()` returns `TRUE` if `transition_probabilities` constitutes a
#' valid transition probability structure for a finite-horizon problem. In the opposite case, it returns `FALSE` in addition to an explanation.
#'
#' This function juxtaposes `transition_probabilities` against the template described in the documentation of [evaluate_finite_horizon_MMDP_markov_policy()] for finite-horizon transition probability lists.
#'
#' @param transition_probabilities A numeric list.
#'
#' @return A boolean: `TRUE` if `transition_probabilities` defines a valid finite-horizon transition probability structure, `FALSE` otherwise.
#' @export
#'
#' @seealso [evaluate_finite_horizon_MMDP_markov_policy()]
#'
#' @examples
#' set.seed(1234)
#' MMDP <- generate_rand_MMDP(2, list(c(1, 2), c(1, 2, 3)), horizon = 5, no_objectives = 2)
#' transition_probabilities <- MMDP$P
#' are_valid_finite_horizon_transition_probabilities(transition_probabilities)
are_valid_finite_horizon_transition_probabilities <- function(transition_probabilities) {

  if (!is.list(transition_probabilities)) {
    message("The transition probabilities should be presented as a list.")
    return(F)
  }
  else if (!is.numeric(unlist(lapply(transition_probabilities, function(t) unlist(t))))) {
    message("Please ensure the transition probabilities are entirely numerical.")
    return(F)
  }
  else if (length(unique(lengths(transition_probabilities))) != 1) {
    message("All first-level components of the transition probabilities' list should be lists of equal length. The length of each such component is the number of states in the model, which does not vary by epoch.")
    return(F)
  }
  else if (nrow(unique(Reduce("rbind", lapply(transition_probabilities, function(t) unlist(lapply(t, length)))))) != 1) {
    message("A state must have the same action set throughout the decision-making horizon.")
    return(F)
  }
  else if (!all(unlist(lapply(transition_probabilities, function(t) lapply(t, function(s) lapply(s, function(p) (all(p >= 0) && (all.equal(sum(p), 1))))))))) {
    message("Please ensure all entries to transition_probabilities are nonnegative and that, given a state s and a feasible action a, the transition probabilities from s and a sum to 1.0.")
    return(F)
  }
  else {return(T)}

}

#' Determine whether a numeric list represents a valid transition probability structure for infinite-horizon problems
#'
#' `are_valid_infinite_horizon_transition_probabilities()` returns `TRUE` if `stationary_rewards` constitutes a
#' valid transition probability structure for a stationary infinite-horizon problem. In the opposite case, it returns `FALSE` in addition to an explanation.
#'
#' This function juxtaposes `stationary_transition_probabilities` against the template described in the documentation of [evaluate_discounted_MMDP_pure_policy()] for infinite-horizon transition probability lists.
#'
#' @param stationary_transition_probabilities A numeric list.
#'
#' @return A boolean: `TRUE` if `stationary_transition_probabilities` defines a valid infinite-horizon transition probability structure, `FALSE` otherwise.
#' @export
#'
#' @seealso [evaluate_discounted_MMDP_pure_policy()]
#'
#' @examples
#' set.seed(1234)
#' stationary_MMDP <- generate_rand_MMDP(2, list(c(1, 2), c(1, 2, 3)),
#'                                      horizon = Inf, no_objectives = 2)
#' stationary_transition_probabilities <- stationary_MMDP$P
#' are_valid_infinite_horizon_transition_probabilities(stationary_transition_probabilities)
are_valid_infinite_horizon_transition_probabilities <- function(stationary_transition_probabilities) {

  if (!is.list(stationary_transition_probabilities)) {
    message("The transition probabilities should be furnished as a list.")
    return(F)
  }
  else if (!is.numeric(unlist(lapply(stationary_transition_probabilities, function(t) unlist(t))))) {
    message("Please ensure all entries in stationary_transition_probabilities are numerical.")
    return(F)
  }
  else if (length(stationary_transition_probabilities) > 1) {
    message("This function is intended for stationary infinite-horizon models only. Ensure that the length of the transition probabilities' list is one.")
    return(F)
  }
  else if (!all(unlist(lapply(stationary_transition_probabilities, function(t) lapply(t, function(s) lapply(s, function(p) (all(p >= 0) && (all.equal(sum(p), 1))))))))) {
    message("Please ensure all entries to stationary_transition_probabilities are nonnegative and that, given a state s and a feasible action a, the probabilities of all possible transitions from s if a is selected sum to 1.0.")
    return(F)
  }
  else {return(T)}

}
