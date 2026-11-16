library(lintools)
library(prodlim)
library(nsga2R)
library(pracma)
library(linprog)
library(dplyr)

#' Convert matrix rows into character keys
#'
#' `row_to_key()` is an adjunct to `solve_discounted_MMDP_policy_iteration()`
#' that creates a character representation of each of a matrix's rows by
#' concatenating the row entries into a single string. The entries are separated
#' by the character `"|"`, producing a convenient key for row comparison,
#' matching, or indexing.
#'
#' This function is useful for detecting duplicate rows or comparing rows
#' between matrices via string matching.
#'
#' @param M A matrix the rows of which will be converted into strings.
#'
#' @return A character vector of length `nrow(M)` where each element is a single
#'   string encoding the corresponding row of `M`.
#' @keywords internal
#' @noRd
#' @examples
#' # For purposes of illustration, let's enumerate the stationary deterministic policies for a two-state problem where two actions are available in state 1 and three are available in state 2
#' action_sets <- list(c(1, 2), c(1, 2, 3))
#' all_pure_policies <- expand.grid(action_sets) # Each row here defines a stationary policy: for instance, the first row prescribes action 1 for state 1 (column 1) and action 1 for state 2 (column 2).
#' # The following line should return c("1|1" "2|1" "1|2" "2|2" "1|3" "2|3")
#' row_to_key(all_pure_policies)
row_to_key <- function(M) {
  apply(M, 1, paste, collapse = "|")
}

#' Calculate the sum set (Minkowski sum) of two or more sets of vectors
#'
#' `sum_set()` is an adjunct to `exact_MMDP_update()` and `heuristic_MMDP_update()`
#' that forms the sum set of two or more sets viewed as matrices whose columns
#' represent the sets' elements.
#'
#' @param X A list of numeric matrices representing Euclidean sets. Each column
#'   in a matrix represents an element of the set of which the matrix
#'   is a representation. The matrices must have an identical number of rows, which is to say that the underlying sets must have the same dimension.
#'
#' @return A numeric matrix representing the sum set of the sets in `X`.
#' @export
#'
#' @examples
#' set_1 <- matrix(c(1, 2, 0, 0, 3, -6), nrow = 2, ncol = 3)
#' set_2 <- matrix(c(5, -2, 7, 0), nrow = 2, ncol = 2)
#' X <- list(set_1, set_2)
#' sum_set(X) # Returns the sum set of set_1 and set_2
sum_set <- function(X) {
  stopifnot(is.list(X), all(sapply(X, is.matrix)))

  n_rows <- unique(sapply(X, nrow))
  if (length(n_rows) != 1L) stop("All matrices must have the same number of rows.")

  # Start with the first matrix
  result <- X[[1]]

  # Iteratively add others
  for (k in 2:length(X)) {
    A <- as.matrix(result)
    B <- X[[k]]

    # Efficiently compute all columnwise sums
    result <- A[, rep(1:ncol(A), each = ncol(B))] +
      B[, rep(1:ncol(B), times = ncol(A))]
  }

  return(result)
}

#' Multiply by a scalar each of several sets of vectors
#'
#' `transition_probabilities_times_sets()` is an adjunct to `exact_MMDP_update()` and `heuristic_MMDP_update()` that takes each set in a list of sets
#' (viewed as matrices) and multiplies its elements (columns) by the
#' corresponding scalar in a numeric vector.
#'
#' @param P An \eqn{n}-dimensional numeric vector.
#' @param X A list of \eqn{n} numeric matrices each representing a set of vectors; see
#'   the documentation of [sum_set()] for a description of how sets are
#'   implemented in this package.
#'
#'
#'
#' @return A list of \eqn{n} matrices having identical dimensions to the matrices in `X`.
#' @noRd
#'
#' @examples
#' set_1 <- matrix(c(1, 2, 0, 0, 3, -6), nrow = 2, ncol = 3)
#' set_2 <- matrix(c(5, -2, 7, 0), nrow = 2, ncol = 2)
#' X <- list(set_1, set_2)
#' P <- c(.75, .25)
#' transition_probabilities_times_sets(P, X) # Multiplies each element in set_1 by .75 and each element in set_2 by .25 then stores the resulting sets in a list
transition_probabilities_times_sets <- function(P, X) {
  mapply(`*`, P, X, SIMPLIFY = FALSE)
}

#' Find the Pareto efficient subset of a set of vectors
#'
#' `efficient_subset_sort_prune()` enumerates the efficient column vectors of a
#' matrix according to the componentwise order that prefers larger components to smaller ones.
#'
#' @param X A numeric matrix.
#'
#' @return A list comprised of:
#' * A numeric matrix formed by the efficient columns of `X`, and
#' * the indices of those columns in the original matrix.
#'
#' If `X`'s columns are incomparable, the function will return `X`.
#' @export
#'
#' @examples
#' # The points in the following set are (1, 2), (0, 0) and (3, -6)
#' X <- matrix(c(1, 2, 0, 0, 3, -6), nrow = 2, ncol = 3)
#' # (1, 2) and (3, -6) are efficient whereas (0, 0) is dominated by (1, 2)
#' efficient_subset_sort_prune(X)
efficient_subset_sort_prune <- function(X) {

  m <- nrow(X)
  n <- ncol(X)

  if (n == 0) return(list(efficient = X, indices = integer(0)))

  # Sort columns descending by first objective (maximization)
  order_idx <- order(-X[1, ])
  X_sorted <- X[, order_idx, drop = FALSE]

  keep <- rep(TRUE, n)

  # Compare each column to later columns
  for (i in seq_len(n - 1)) {
    if (!keep[i]) next
    # Later columns
    Y <- X_sorted[, (i + 1):n, drop = FALSE]
    # Identify columns dominated by X_sorted[, i]
    dominated <- colSums(Y <= X_sorted[, i]) == m & colSums(Y < X_sorted[, i]) > 0
    keep[(i + 1):n][dominated] <- FALSE
  }

  # Extract efficient columns and original indices
  efficient_cols <- X_sorted[, keep, drop = FALSE]
  indices <- order_idx[keep]

  return(list(efficient = efficient_cols, indices = indices))
}

#' Compare two or more vector-valued value functions
#'
#' `value_function_domination_sets()` is an adjunct to
#' `solve_discounted_MMDP_policy_iteration()` that compares a vector-valued
#' function `V` with a set of such functions to determine which of them
#' dominate `V`, are dominated by `V` or are equal to `V`.
#'
#' Dominance here is taken with respect to the product order induced by the
#' "larger-is-better" componentwise order over column vectors. That is, if `U` denotes some matrix in `value_functions`, then `U` dominates `V` if `U[, s]` is componentwise greater than or equal to `V[, s]` for all columns `s` and there exists a column `j` such that `U[, j]` dominates `V[, j]` componentwise.
#'
#' @param V A numeric matrix in which the \eqn{s}-th column gives the image of \eqn{s} under the vector-valued function encoded in `V`.
#' @param value_functions A list of numeric matrices, having the same size as `V`, with which `V` is to be compared.
#'
#' @return A list comprising, in this order,
#' * the matrices in `value_functions` that dominate `V`,
#' * those that are dominated by `V`,
#' * and those that are identical to `V`.
#' @export
#'
#' @examples
#' V <- matrix(c(5, 5, 3, 1), nrow = 2)
#' U1 <- matrix(c(4, 4, 2, 0), nrow = 2) # Dominated by V
#' U2 <- matrix(c(6, 5, 4, 2), nrow = 2) # Dominates V
#' U3 <- V # Identical to V
#' value_functions <- list(U1, U2, U3)
#' value_function_domination_sets(V, value_functions)
value_function_domination_sets <- function(V, value_functions) {

  no_objectives <- nrow(V)
  no_states <- ncol(V)
  L <- length(value_functions)

  dominated <- integer(0)
  dominates <- integer(0)
  equal <- integer(0)

  epsilon_1 <- 10^-10
  epsilon_2 <- 10^-4

  for (l in seq_len(L)) {
    W <- value_functions[[l]]

    # Sanity check — skip inconsistent dimensions
    if (!all(dim(W) == c(no_objectives, no_states))) {
      stop("All matrices must have the same dimensions.")
    }

    # Flatten for fast vectorized comparison
    v <- as.numeric(V)
    w <- as.numeric(W)

    if (abs(mean(w-v)^2) < epsilon_1) {
      equal <- c(equal, l)
      next
    }

    differences <- v - w

    if (all(-differences >= 0) && any(-differences > epsilon_2)) {
      dominates <- c(dominates, l)
    } else if (all(differences >= 0) && any(differences > epsilon_2)) {
      dominated <- c(dominated, l)
    }
  }

  list(
    dominates_V = dominates,
    dominated_by_V = dominated,
    equal_to_V = equal
  )

}

#' Calculate the compromise solution among a set of objective vectors
#'
#' `compromise_solution()` determines which of a finite set `value_set` of
#' objective vectors is geometrically the closest to the ideal point that
#' maximizes all components simultaneously.
#'
#' @param value_set A numeric matrix whose columns contain the objective
#'   vectors.
#' @param p Either (1) an integer greater than or equal to 1 or (2) `Inf`.
#'   This argument defines the norm with respect to which distances are evaluated. The
#'   default value `2` yields the Euclidean norm.
#'
#' @return A numeric vector giving the column of `value_set` that minimizes the
#'   \eqn{l_p} distance to the ideal point.
#' @export
#'
#' @references Yu, P. L. (2013). *Multiple-criteria decision making: concepts, techniques, and extensions* (Vol. 30). Springer Science & Business Media.
#'
#' @examples
#' # Construct a set of three points: (3, 1), (2, 2) and (8, -1/2). The ideal point here is (8, 2).
#' value_set <- matrix(c(3, 1, 2, 2, 8, -1/2), nrow = 2)
#' # Which of the three points is closest to the ideal under, say, the Euclidean norm?
#' compromise_solution(value_set)
#' # What of the sup norm?
#' compromise_solution(value_set, Inf)
compromise_solution <- function(value_set, p = 2) {
  ideal_solution <- apply(value_set, 1, max)
  deviations_from_ideal <- apply(value_set, 2, function(y) pracma::Norm(y - ideal_solution, p))
  return(as.matrix(value_set[, which.min(deviations_from_ideal)]))
}

#' Compute a Wakuta matrix for the multi-objective policy iteration algorithm
#'
#' `compute_Wakuta_matrix()` is called by
#' `solve_discounted_MMDP_policy_iteration()` to perform certain matrix
#' calculations for Wakuta's multi-objective policy iteration method for
#' discounted infinite-horizon problems. The function returns a matrix that can
#' be employed in determining whether a stationary policy meets a necessary or
#' sufficient condition for efficiency.
#'
#' The final phase of Wakuta's algorithm consists in determining which of a set
#' of policies meet a necessary then a sufficient condition for efficiency. Both
#' conditions relate a policy's efficiency to the feasibility of a particular
#' group of systems of linear inequalities. This function uses the value of
#' `type` to distinguish between the conditions, calculates the left-hand side
#' coefficients of the relevant systems, then returns them in the form of a
#' matrix.
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
#' @param stationary_policy An integer vector, of length equal to the number of
#'   states, representing a stationary policy. Each element in the vector gives
#'   the index of the action prescribed by the policy for the corresponding
#'   state.
#' @param initial_state An integer greater than or equal to 1 and less than or
#'   equal to the number of states as determined by the length of
#'   `stationary_transition_probabilities[[1]]` or `stationary_rewards[[1]]`.
#'   This is in the state in which the Wakuta matrix is evaluated.
#' @param type An integer indicating which of two Wakuta matrices is warranted
#'   by the current stage of the policy iteration algorithm: `1` for the
#'   "necessary" matrix (default), or `2` for the "sufficient" one.
#'
#' @return A numeric matrix with an a priori unknown number of rows and as many
#'   columns as there are objectives (as determined by `stationary_rewards`).
#' @keywords internal
#' @noRd
#'
#' @references Wakuta, K. (1995). Vector-valued markov decision processes and
#'   the systems of linear inequalities.
#' *Stochastic processes and their applications, 56*(1), 159-169.
#'
#' @seealso [solve_discounted_MMDP_policy_iteration()] for an implementation of Wakuta's policy iteration algorithm.
#'
#' @examples
compute_Wakuta_matrix <- function(stationary_transition_probabilities, stationary_rewards, rho, stationary_policy, initial_state, type = 1) {
  # Testing the fitness and internal consistency of the arguments provided
  if (!(type %in% c(1, 2))) {
    stop("The `type` argument in compute_Wakuta_matrix() takes one of two values: 1 or 2. See the accompanying documentation for information on this helper function.")
  }
  else if (!are_valid_infinite_horizon_transition_probabilities(stationary_transition_probabilities) || !are_valid_infinite_horizon_rewards(stationary_rewards)) {
    stop("The rewards and/or transition probabilities you have specified are invalid. See the message(s) returned by are_valid_infinite_horizon_transition_probabilities() and/or are_valid_infinite_horizon_rewards() for an explanation.")
  }
  else if (!(initial_state %in% (1:length(stationary_rewards[[1]])))) {
    stop("Please ensure the starting state is in the state set determined by the transition probabilities and the rewards.")
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
  action_sets <- lapply(lengths(stationary_rewards[[1]]), function(n) 1:n)


  ## First find those states reachable under policy (after some number of epochs) when initial_state is the starting state. The total number of these states determines the size of the Q matrix.
  reachable_states <- rep(F, times = no_states)
  reachable_states[initial_state] <- T

  if (type == 1 || type == 2) {
    if (type == 1) {
      for (j in 1:no_states) {
        t <- 1
        S_t <- initial_state
        while (t <= no_states && !reachable_states[j]) {
          t <- t + 1
          # S_t is the set of states that are reachable under policy in t time periods
          S_t <- unique(unlist(lapply(S_t, function(s_t) {which(stationary_transition_probabilities[[1]][[s_t]][[stationary_policy[s_t]]] > 0)})))
          if (j %in% S_t) {reachable_states[j] <- T}
        }

      }
    }
    if (type == 2) {
      for (j in 1:no_states) {
        t <- 1
        S_t <- initial_state
        while (t <= no_states && !reachable_states[j]) {
          t <- t + 1
          # S_t is the set of states that are reachable in t time periods
          S_t <- unique(unlist(lapply(S_t, function(s_t) unique(unlist(lapply(action_sets[[s_t]], function(a) which(stationary_transition_probabilities[[1]][[s_t]][[a]] > 0)))))))
          if (T %in% (S_t == j)) {reachable_states[j] <- T}
        }
      }
    }
    # Next, for each reachable state s, calculate q(s, a) for all permissible actions a
    policy_value_function <- evaluate_discounted_MMDP_pure_policy(stationary_transition_probabilities, stationary_rewards, stationary_policy, rho)
    Wakuta_matrix <- NULL
    for (s in which(reachable_states)) {
      for (a in action_sets[[s]]) {
        successor_states_probabilities <- stationary_transition_probabilities[[1]][[s]][[a]]
        q_s_a <- stationary_rewards[[1]][[s]][[a]] + rho * successor_states_probabilities %*% t(policy_value_function) - policy_value_function[, s]
        Wakuta_matrix <- rbind(Wakuta_matrix, q_s_a)
      }
    }
  }

  return(Wakuta_matrix)
}

#' Compute a Wakuta matrix for the multi-objective policy iteration algorithm
#'
#' `compute_Wakuta_matrix_type_3()` is occasionally called by
#' `solve_discounted_MMDP_policy_iteration()` to perform certain matrix
#' calculations for Wakuta's multi-objective policy iteration method for
#' discounted infinite-horizon problems. The function returns a matrix that can
#' be employed in determining whether a stationary policy meets a necessary and
#' sufficient condition for efficiency.
#'
#' A final, optional step in Wakuta's algorithm consists in determining which of a set
#' of policies meet a necessary and sufficient condition for efficiency. The
#' condition relates a policy's efficiency to the feasibility of a particular
#' group of systems of linear inequalities. This function calculates the
#' left-hand side coefficients of each of these systems then returns them in the
#' form of a matrix.
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
#' @param stationary_policy An integer vector, of length equal to the number of
#'   states, representing a stationary policy. Each element in the vector gives
#'   the index of the action prescribed by the policy for the corresponding
#'   state.
#' @param initial_state An integer greater than or equal to 1 and less than or
#'   equal to the number of states as determined by the length of
#'   `stationary_transition_probabilities[[1]]` or `stationary_rewards[[1]]`.
#'   This is in the state in which the Wakuta matrix is evaluated.
#' @param policy_set An integer matrix with rows the length of
#'   `stationary_policy` that represents a set of possibly efficient policies
#'   with which to compare `stationary_policy`.
#'
#' @return A numeric matrix with the same number of rows as `policy_set` and as
#'   many columns as there are objectives (as determined by
#'   `stationary_rewards`).
#' @keywords internal
#' @noRd
#'
#' @references Wakuta, K. (1995). Vector-valued markov decision processes and
#'   the systems of linear inequalities.
#' *Stochastic processes and their applications, 56*(1), 159-169.
#'
#' @seealso [solve_discounted_MMDP_policy_iteration()]
#'
compute_Wakuta_matrix_type_3 <- function(stationary_transition_probabilities, stationary_rewards, rho, stationary_policy, initial_state, policy_set) {
  # Testing the fitness and internal consistency of the arguments provided
  if (!(initial_state %in% (1:no_states))) {
    stop("Please ensure the starting state is in the state set determined by no_states.")
  }
  else if (!are_valid_infinite_horizon_transition_probabilities(stationary_transition_probabilities) || !are_valid_infinite_horizon_rewards(stationary_rewards)) {
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
  action_sets <- lapply(lengths(stationary_rewards[[1]]), function(n) 1:n)


  # For each reachable state s, calculate q(s, a) for all permissible actions a
  policy_value_function <- evaluate_discounted_MMDP_pure_policy(stationary_transition_probabilities, stationary_rewards, stationary_policy, rho)
  Wakuta_matrix <- NULL
  for (pi in 1:nrow(policy_set)) {
    pi_value_function <- evaluate_discounted_MMDP_pure_policy(stationary_transition_probabilities, stationary_rewards, policy_set[pi, ], rho)
    d <- pi_value_function[, initial_state] - policy_value_function[, initial_state]
    Wakuta_matrix <- rbind(Wakuta_matrix, d)
  }

  return(Wakuta_matrix)
}

#' Solve a multi-objective linear programming problem by a simplex-type method
#'
#' `solve_MOLP()` is an adjunct to `solve_discounted_MMDP_linear_programming()`
#' that enumerates the efficient basic feasible solutions of a multi-objective
#' linear programming problem subject to equality and nonnegativity constraints.
#'
#' This function implements a particular version of Ecker and Kouada's (1978)
#' procedure for enumerating the efficient basic feasible solutions of a
#' multi-objective linear (maximization) program where candidate points are
#' tested for efficiency using a proposition from Evans and Steuer (1973). The
#' procedure requires an initial efficient extreme point, which can be found by
#' maximizing a positive linear combination of the objectives.
#'
#' @param C A numeric matrix whose \eqn{i}-th row contains the coefficient
#'   vector for linear objective function \eqn{i}.
#' @param A A numeric matrix, one row per constraint and one column per
#'   variable, giving the constraint coefficients.
#' @param b A numeric vector giving the right-hand side coefficients.
#' @param max_iter An integer greater than or equal to 1 that places an upper
#'   limit on the number of iterations performed by the linear programming
#'   procedure.
#'
#' @return A list with one integer matrix and one numeric matrix:
#' * `efficient_bfs`, whose rows contain the efficient basic feasible solutions---which are vectors of length equal to the number of variables---found;
#' * and `values`, whose columns contain the objective value vectors generated by the solutions in `efficient_bfs`.
#' @export
#'
#' @references Ecker, J. G., & Kouada, I. A. (1978). Finding all efficient
#'   extreme points for multiple objective linear programs. *Mathematical
#'   programming, 14*(1), 249-261.
#'
#'   Evans, J. P., & Steuer, R. E. (1973). A revised simplex method for linear
#'   multiple objective programs. *Mathematical programming, 5*(1), 54-72.
#'
#'   Mifrani, A., & Noll, D. (2025). Linear programming for finite-horizon
#'   vector-valued Markov decision processes. *arXiv preprint arXiv:2502.13697*.
#'
#' @seealso [solve_discounted_MMDP_linear_programming()]
#'
#' @examples
#' # Consider the multi-objective linear programming problem
#' #           Maximize 4*x_1 - x_2 + 3*x_3
#' #                    -2*x_1 + 5*x_2 - 0.5*x_3
#' #           subject to 2*x_1 + 3*x_2 - x_3 = 12
#' #                      x_1 - x_2 + 3*x_3 = 3
#' #                      x_1, x_2, x_3 nonnegative
#'
#'
#'
#' # Criteria matrix
#' C <- matrix(c(4, -1, 3,
#'                -2, 5, -0.5), nrow = 2, ncol = 3, byrow = TRUE)
#'
#' # Constraint matrix (excluding nonnegativity)
#' A <- matrix(c(2, 3, -1,
#'               1, -1, 3), nrow = 2, ncol = 3, byrow = TRUE)
#' # Right-hand side vector
#' b <- c(12, 3)
#' # Solve the multi-objective linear programming problem,
#' # display the efficient basic feasible solutions
#' result <- solve_MOLP(C, A, b, max_iter = 50)
#' result$efficient_bfs
#' result$values
solve_MOLP <- function(C, A, b, max_iter) {

  # This function implements Ecker and Kouada's (1978) procedure for enumerating the efficient basic feasible solutions of a multi-objective linear program.
  n <- ncol(C)
  m <- nrow(A)
  no_objectives <- nrow(C)

  # PRELIMINARY STEP: FIND AN INITIAL EFFICIENT BASIC FEASIBLE SOLUTION BY MAXIMIZING A POSITIVE LINEAR COMBINATION OF THE OBJECTIVES
  ## Choose some no_objectives positive scalars p_i and solve the ordinary linear program max{(pTC)x: Ax = b, x >= 0}.
  ## The simplex algorithm applied to this LP should provide an optimal basic feasible solution which is efficient for the multi-objective problem.
  p <- rep(1, times = no_objectives)
  scalar_LP_C <- p %*% C
  phase_one_tableau <- construct_tableau_phase_one(A, b)
  end_of_phase_one_tableau <- simplex_phase_one(tableau = phase_one_tableau, max_iter = max_iter)
  phase_two_tableau <- prepare_simplex_phase_two(end_of_phase_one_tableau, -scalar_LP_C)
  optimal_tableau <- simplex_phase_two(phase_two_tableau, max_iter = max_iter)

  # IMPLEMENTATION OF THE ECKER-KOUADA ALGORITHM
  optimal_tableau[[2]] <- sort(optimal_tableau[[2]])
  optimal_tableau[[3]] <- sort(optimal_tableau[[3]])

  Tableau <- optimal_tableau
  N_T <- Tableau[[3]][!(Tableau[[3]] > (ncol(Tableau[[1]]) - nrow(Tableau[[1]])))]
  V_1_bfs <- NULL
  V_1_tableaus <- NULL
  V_2_bfs <- calculate_bfs_from_tableau(Tableau, n, m)
  V_2_tableaus <- list(Tableau)
  for (step_three in 1:max_iter) {
    R_T_bfs <- find_efficient_adjacent_bfs(Tableau, C, A)[[1]]
    R_T_tableaus <- find_efficient_adjacent_bfs(Tableau, C, A)[[2]]
    temp <- tableau_difference(R_T_tableaus, V_2_tableaus)
    if (length(temp) == 0) {
      R_1_tableaus <- list()
      R_1_bfs <- NULL
    }
    else {
      R_1_tableaus <- tableau_difference(R_T_tableaus, V_2_tableaus)[[1]]
      R_1_bfs <- R_T_bfs[tableau_difference(R_T_tableaus, V_2_tableaus)[[2]], ]
    }

    if (length(R_1_bfs) == 0) {
      if (length(V_1_bfs) == 0) {
        #message(paste0("Number of pure efficient policies detected through linear programming: ", nrow(V_2_bfs)))
        break
      }
      else {
        Tableau <- V_1_tableaus[[1]]
        if (length(V_1_bfs) == n) {
          V_2_bfs <- rbind(V_2_bfs, V_1_bfs)
          V_1_bfs <- NULL
        }
        else {
          V_2_bfs <- rbind(V_2_bfs, V_1_bfs[1, ])
          V_1_bfs <- V_1_bfs[-1, ]
        }
        V_2_tableaus <- append(V_2_tableaus, list(Tableau))
        V_1_tableaus <- V_1_tableaus[-1]
        if (!is.null(V_1_bfs) && (length(V_1_bfs) > n)) { V_1_bfs <- as.matrix(V_1_bfs) }
        next
      }
    }
    else {
      if (is.null(V_1_bfs)) {
        R_2_bfs <- R_1_bfs
        R_2_tableaus <- R_1_tableaus
      }
      else if (length(V_1_bfs) > n) {
        if (is.null(nrow(R_1_bfs))) {
          temp <- tableau_difference(R_1_tableaus, V_1_tableaus)
          if (length(temp) == 0) {
            R_2_tableaus <- list()
            R_2_bfs <- NULL
          }
          else {
            R_2_tableaus <- tableau_difference(R_1_tableaus, V_1_tableaus)[[1]]
            if (length(R_2_tableaus) > 0) {
              R_2_bfs <- R_1_bfs
            }
            else {
              R_2_bfs <- NULL
            }
          }

        }
        else {
          temp <- tableau_difference(R_1_tableaus, V_1_tableaus)
          if (length(temp) == 0) {
            R_2_tableaus <- list()
            R_2_bfs <- NULL
          }
          else {
            R_2_tableaus <- tableau_difference(R_1_tableaus, V_1_tableaus)[[1]]
            if (length(R_2_tableaus) > 0) {
              R_2_bfs <- R_1_bfs[tableau_difference(R_1_tableaus, V_1_tableaus)[[2]], ]
            }
            else {
              R_2_bfs <- NULL
            }
          }

        }
      }
      else if (length(V_1_bfs) == n) {
        temp <- tableau_difference(R_1_tableaus, V_1_tableaus)
        if (length(temp) == 0) {
          R_2_tableaus <- list()
          R_2_bfs <- NULL
        }
        else {
          R_2_tableaus <- tableau_difference(R_1_tableaus, V_1_tableaus)[[1]]
          if (is.null(nrow(R_1_bfs))) {
            if (length(R_2_tableaus) > 0) {
              R_2_bfs <- R_1_bfs
            }
            else {
              R_2_bfs <- NULL
            }
          }
          else {
            R_2_bfs <- R_1_bfs[tableau_difference(R_1_tableaus, V_1_tableaus)[[2]], ]
          }
        }

      }


      else if (length(V_1_bfs) == 0) {
        R_2_bfs <- R_1_bfs
        R_2_tableaus <- R_1_tableaus
      }



      if (length(R_2_bfs) == 0) {
        # Ecker-Kouada Step 8: choose a basis in R_1 (which must lie in V_1).
        Tableau <- R_1_tableaus[[1]]
        pending_index <- which(vapply(
          V_1_tableaus,
          function(t) setequal(t[[2]], Tableau[[2]]),
          logical(1)
        ))
        if (length(pending_index) != 1L) {
          stop("Internal error: selected R_1 basis is not unique in V_1")
        }

        # Record the selected basis and its solution together.
        V_2_bfs <- rbind(V_2_bfs, calculate_bfs_from_tableau(Tableau, n, m))
        V_2_tableaus <- append(V_2_tableaus, list(Tableau))

        # Remove this same basis, not the first (possibly different) pending basis.
        if (is.null(dim(V_1_bfs))) {
          V_1_bfs <- matrix(V_1_bfs, nrow = 1L)
        }
        V_1_bfs <- V_1_bfs[-pending_index, , drop = FALSE]
        if (nrow(V_1_bfs) == 0L) V_1_bfs <- NULL
        V_1_tableaus <- V_1_tableaus[-pending_index]
        next
      } # Ecker-Kouada Step 8
      # Otherwise, if R_2 isn't empty, go to step 6
      else {
        Tableau <- R_2_tableaus[[1]]
        if (!is.null(nrow(R_2_bfs))) {
          V_2_bfs <- rbind(V_2_bfs, R_2_bfs[1, ])
          R_2_bfs <- R_2_bfs[-1, ]
        }
        else {
          V_2_bfs <- rbind(V_2_bfs, R_2_bfs) # vertex already computed
          R_2_bfs <- NULL
        }
        V_2_tableaus <- append(V_2_tableaus, list(Tableau)) # tableau already computed
        R_2_tableaus <- R_2_tableaus[-1]
        V_1_bfs <- rbind(V_1_bfs, R_2_bfs) # vertices yet to be computed
        V_1_tableaus <- append(V_1_tableaus, R_2_tableaus) # tableaus yet to be computed / examined
        if (!is.null(V_1_bfs) && (length(V_1_bfs) > n)) { V_1_bfs <- as.matrix(V_1_bfs) }
        next
      }
    }
  }

  efficient_values <- apply(V_2_bfs, MARGIN = 1, function(x) C %*% x)

  rownames(V_2_bfs) <- paste("Basic feasible solution", seq_len(nrow(V_2_bfs)))
  colnames(V_2_bfs) <- paste("x", seq_len(ncol(V_2_bfs)))
  rownames(efficient_values) <- paste("Objective", seq_len(nrow(efficient_values)))
  colnames(efficient_values) <- rownames(V_2_bfs)

  return(list(efficient_bfs = V_2_bfs, values = efficient_values))

}

#' "Read off" the basic feasible solution from a simplex tableau
#'
#' `calculate_bfs_from_tableau()` is an adjunct to `solve_MOLP()` that evaluates
#' the basic feasible solution associated with a single-objective simplex
#' tableau.
#'
#' @param tableau A list with three components:
#' * A numeric matrix with `(m+1)` rows for `(n+m+1)` columns, representing a simplex tableau. The top `m` rows represent the constraints, whereas the bottom row contains the objective function equation. Columns one through `n` correspond to the original variables, columns `(n+1)` through `(n+m)` to the artificial Phase I variables, and the last column contains the right-hand side vector at the current stage of the simplex method.
#' * An integer vector of column indices corresponding to the basic variables.
#' * An integer vector of column indices corresponding to the nonbasic variables.
#' @param n An integer greater than or equal to 1 giving the number of
#'   (original) linear programming variables.
#' @param m An integer greater than or equal to 1 giving the number of
#'   constraints.
#'
#' @return A numeric vector of length `n` giving the basic feasible solution
#'   associated with `tableau`.
#' @noRd
#'
#' @seealso [solve_MOLP()]
#'
#' @examples
calculate_bfs_from_tableau <- function(tableau, n, m) {
  bfs <- rep(0, times = n)
  basic_rows <- vector(length = m)
  for (basic_variable in 1:length(tableau[[2]])) {
    basic_rows[basic_variable] <- which(tableau[[1]][1:m, tableau[[2]][basic_variable]] == 1)
  }
  bfs[tableau[[2]]] <- tableau[[1]][1:m, ncol(tableau[[1]])][basic_rows]
  return(bfs)
}

#' Find efficient solutions to a multi-objective linear program among a basic feasible solution's neighbors
#'
#' `find_efficient_adjacent_bfs()` is an adjunct to `solve_MOLP()` that, given a basic feasible solution to a multi-objective linear program,
#' determines which of the adjacent basic feasible solutions are efficient
#' solutions.
#'
#' @param tableau A list containing a simplex tableau and the corresponding sets
#'   of basic and nonbasic variable indices. See the documentation of
#'   [calculate_bfs_from_tableau()] for further information. The tableau's
#'   bottom row (the z-equation) is ignored.
#' @param C A numeric matrix whose \eqn{i}-th row contains the coefficient
#'   vector for linear objective function \eqn{i}.
#' @param A A numeric matrix, one row per constraint and one column per
#'   variable, giving the constraint coefficients.
#'
#' @return A list with two components:
#' * `efficient_adjacent_bfs`, a numeric matrix whose rows contain the efficient basic feasible solutions that are adjacent to the basic feasible solution determined by `tableau`.
#' * `efficient_adjacent_tableaus`, a list of the tableaus associated with the basic feasible solutions recorded in `efficient_adjacent_bfs`. Each tableau bears the list structure of `tableau`.
#' @noRd
#'
#' @seealso [solve_MOLP()] and [efficiency_test_MOLP()].
#'
#' @examples
find_efficient_adjacent_bfs <- function(tableau, C, A) {
  no_objectives <- nrow(C)
  n <- ncol(A)
  m <- nrow(A)
  temp <- tableau[[1]][, 1:(ncol(tableau[[1]]) - nrow(tableau[[1]]))]
  nonbasics <- tableau[[3]][!(tableau[[3]] > (ncol(tableau[[1]]) - nrow(tableau[[1]])))]
  efficient_adjacent_bfs <- NULL
  efficient_adjacent_tableaus <- NULL
  for (j in nonbasics) {
    pivot_row_and_basic_index <- get_pivot_row_and_basic_index(tableau, j)
    adjacent_tableau <- pivot(tableau, pivot_row_and_basic_index, j)
    if (qr(A[, adjacent_tableau[[2]]])$rank == m) {
      if (efficiency_test_MOLP(adjacent_tableau, C, A)) {
        efficient_adjacent_bfs <- rbind(efficient_adjacent_bfs, calculate_bfs_from_tableau(adjacent_tableau, n, m))
        efficient_adjacent_tableaus <- c(efficient_adjacent_tableaus, list(adjacent_tableau))
      }
    }
  }


  return(list(efficient_adjacent_bfs, efficient_adjacent_tableaus))
}

#' Get the row and index of an outgoing basic variable for a simplex pivot
#'
#' `get_pivot_row_and_basic_index()` is an adjunct to `solve_MOLP()` that takes
#' a nonbasic variable (a pivot) `pivot_column` to be converted to basic status and determines an
#' outgoing basic variable. The function returns the index of this variable
#' together with the row with which it is associated in the simplex tableau.
#'
#' @param tableau A list containing a simplex tableau with the corresponding
#'   sets of basic and nonbasic variable indices. See the documentation of
#'   [calculate_bfs_from_tableau()] for further information. The tableau's
#'   bottom row (the z-equation) is immaterial to this function.
#' @param pivot_column An integer giving the index of the incoming basic (but as
#'   yet nonbasic) variable.
#'
#' @return An integer vector of the outgoing basic variable's row and index.
#' @noRd
#'
#' @examples
get_pivot_row_and_basic_index <- function(tableau, pivot_column) {
  temp <- tableau[[1]]
  lhs <- temp[1:(nrow(temp) - 1), pivot_column]
  rhs <- temp[1:(nrow(temp) - 1), ncol(temp)]
  if (max(lhs) <= 0) {
    #message("Problem is unbounded -> stop execution")
    # break
    return(-1)
  }
  else {
    theta <- vector(mode = "double", length = (nrow(temp) - 1))
    for (i in 1:(nrow(temp) - 1)) {
      if (lhs[i] > 0) {
        theta[i] <- rhs[i] / lhs[i]
      } else {
        theta[i] <- Inf
      }
    }
    pivot_row <- which.min(theta)
    # Get column (index) of the basic variable to which the chosen pivot equation is associated
    # In other words, which of the basic variables has a unitary coefficient in equation number theta?
    return(c(pivot_row, tableau[[2]][which(temp[pivot_row, tableau[[2]]] == 1)]))
  }
}

#' Execute a simplex pivot
#'
#' `pivot()` is an adjunct to `solve_MOLP()` that performs a simplex pivot
#' with respect to incoming variable `pivot_column` and outgoing variable `pivot_row_and_basic_index[1]`. The function
#' returns the ensuing tableau with the associated basic and nonbasic variable
#' index sets.
#'
#' @param tableau A list containing a simplex tableau and the corresponding sets
#'   of basic and nonbasic variable indices. See the documentation of
#'   [calculate_bfs_from_tableau()] for further information. The tableau's
#'   bottom (objective) row is immaterial to this function.
#' @param pivot_row_and_basic_index An integer vector of the row and index of
#'   the outgoing basic variable.
#' @param pivot_column An integer representing the index of the incoming basic (as of
#'   yet nonbasic) variable.
#'
#' @return A list bearing an identical structure to `tableau` that gives the
#'   tableau arising from the exchange of nonbasic variable `pivot_column` with
#'   basic variable `pivot_row_and_and_basic_index[2]`.
#' @noRd
#'
#' @examples
pivot <- function(tableau, pivot_row_and_basic_index, pivot_column) {
  temp <- tableau[[1]]
  pivot_row <- pivot_row_and_basic_index[1]
  pivot_basic_index <- pivot_row_and_basic_index[2]
  pivot_element <- temp[pivot_row, pivot_column]
  tableau[[1]][pivot_row, ] <- tableau[[1]][pivot_row, ] / pivot_element
  for (r in 1:nrow(temp)) {
    if (r != pivot_row) {
      tableau[[1]][r, ] <- tableau[[1]][r, ] - tableau[[1]][r, pivot_column] * tableau[[1]][pivot_row, ]
    }
  }
  # Updating the set of basic / nonbasic variables
  ## Get the index of the basic variable now turned nonbasic
  tableau[[2]] <- tableau[[2]][!tableau[[2]] == pivot_basic_index]
  tableau[[2]] <- c(tableau[[2]], pivot_column) # the new basic variables
  tableau[[2]] <- sort(tableau[[2]])
  tableau[[3]] <- tableau[[3]][!tableau[[3]] == pivot_column]
  tableau[[3]] <- c(tableau[[3]], pivot_basic_index) # the new nonbasics
  tableau[[3]] <- sort(tableau[[3]])
  return(tableau)
}

#' Test a basic feasible solution for efficiency in a multi-objective linear
#' program
#'
#' `efficiency_test_MOLP()` is an adjunct to `solve_MOLP()` that ascertains whether a basic feasible solution to a multi-objective linear program is an
#' efficient solution, and returns `TRUE` if such is the case. It returns `FALSE` otherwise.
#'
#' This function implements the computational test offered by Evans and Steuer
#' (1973), which links the efficiency of a basic feasible solution in a
#' multi-objective linear (maximization) program to the boundedness of an auxiliary
#' single-objective linear program.
#'
#' @param tableau A list containing a simplex tableau and the corresponding sets
#'   of basic and nonbasic variable indices. See the documentation of
#'   [calculate_bfs_from_tableau()] for further information. The tableau's
#'   bottom row (z-equation) is immaterial to this function.
#' @param C A numeric matrix whose \eqn{i}-th row contains the coefficient
#'   vector for linear objective function \eqn{i}.
#' @param A A numeric matrix, one row per constraint and one column per
#'   variable, specifying the constraint coefficients.
#'
#' @return A boolean: `TRUE` if the basic feasible solution determined by
#'   `tableau` is an efficient solution to the multi-objective linear program, `FALSE` otherwise.
#' @noRd
#'
#' @references Evans, J. P., & Steuer, R. E. (1973). A revised simplex method
#'   for linear multiple objective programs. *Mathematical programming, 5*(1),
#'   54-72.
#'
#' @seealso [solve_MOLP()], [linprog::solveLP()]
#' @examples
efficiency_test_MOLP <- function(tableau, C, A) {
  no_objectives <- nrow(C)
  n <- ncol(A)
  m <- nrow(A)
  # Get the basis matrix associated with this b.f.s.
  B <- A[, sort(tableau[[2]])]
  N <- A[, -sort(tableau[[2]])]
  R <- C[, sort(tableau[[2]])] %*% solve(B, tol=1e-50) %*% N - C[, -sort(tableau[[2]])]
  # Check the auxiliary Evans-Steuer LP for boundedness
  auxiliary_c <- c(rep(0, times = n-m), rep(1, times = no_objectives))
  auxiliary_b <- rep(0, times = no_objectives)
  auxiliary_A <- matrix(nrow = no_objectives, ncol = n-m+no_objectives)
  auxiliary_A[, 1:(n-m)] <- R
  auxiliary_A[, (n-m+1):(n-m+no_objectives)] <- diag(nrow = no_objectives, ncol = no_objectives)
  auxiliary_lp <- summary(solveLP(cvec=auxiliary_c, bvec=auxiliary_b, Amat=auxiliary_A, maximum = T, const.dir = rep("=", no_objectives), lpSolve = T))
  if (length(auxiliary_lp) != 6) {
    return(TRUE)
  }
  else {return(FALSE)}
}

#' Compute the difference of two lists
#'
#' `list_difference()` returns the elements of `list_1` that are absent from `list_2`.
#'
#'
#' @param list_1 A list.
#' @param list_2 A list.
#'
#' @return A list containing those components of `list_1` that are not
#'   `identical()` to any component of `list_2`. The relative order of
#'   the retained elements is preserved.
#' @noRd
#'
#' @examples
#' list_difference(list(1, 2, 3), list(2))
#' list_difference(list("a", "b", "c"), list("b", "d"))
list_difference <- function(list_1, list_2) {
  list_difference <- NULL
  common_elements <- intersect(list_1, list_2)

  if (length(common_elements) == 0) {
    return(list_1)
  }

  for (i in 1:length(list_1)) {
    for (j in 1:length(common_elements)) {
      if (identical(list_1[[i]], common_elements[[j]])) {
        list_difference <- c(list_difference, i)
      }
    }
  }
  return(list_1[-list_difference])
}

#' Compute the difference between two sets of tableaus
#'
#' `tableau_difference()` compares two lists of simplex tableaus and returns
#' those tableaus in `list_1` that are absent from `list_2`.
#'
#' @param list_1,list_2 Two lists of tableaus. Each tableau is a three-component
#'   list following the template described in the documentation of
#'   [evaluate_bfs_from_tableau()].
#'
#' @return A list of length two:
#' * A sublist of `list_1` containing those tableaus which are absent from `list_2`.
#' * An integer vector giving the indices of these tableaus in `list_1`.
#'
#'   If no such tableaus exist, the function returns an empty list.
#' @noRd
#'
#' @seealso [dplyr::anti_join()]
#'
#' @examples
tableau_difference <- function(list_1, list_2) {
  basic_variables_1 <- Reduce("rbind", lapply(list_1, function(t) t[[2]][order(t[[2]])]))
  basic_variables_2 <- Reduce("rbind", lapply(list_2, function(t) t[[2]][order(t[[2]])]))

  if (!is.null(nrow(basic_variables_1)) && is.null(nrow(basic_variables_2))) {
    difference_basic_variables <- as.matrix(suppressMessages(anti_join(data.frame(basic_variables_1), data.frame(t(basic_variables_2)))))
  }
  else if (is.null(nrow(basic_variables_1)) && !is.null(nrow(basic_variables_2))) {
    difference_basic_variables <- as.matrix(suppressMessages(anti_join(data.frame(t(basic_variables_1)), data.frame(basic_variables_2))))
  }
  else if (is.null(nrow(basic_variables_1)) && is.null(nrow(basic_variables_2))) {
    difference_basic_variables <- as.matrix(suppressMessages(anti_join(data.frame(t(basic_variables_1)), data.frame(t(basic_variables_2)))))
  }
  else if (!is.null(nrow(basic_variables_1)) && !is.null(nrow(basic_variables_2))) {
    difference_basic_variables <- as.matrix(suppressMessages(anti_join(data.frame(basic_variables_1), data.frame(basic_variables_2))))
  }

  if (length(difference_basic_variables) == 0) {
    return(list())
  }

  list_difference <- NULL

  for (i in 1:length(list_1)) {
    for (j in 1:nrow(difference_basic_variables)) {
      if (setequal(list_1[[i]][[2]], difference_basic_variables[j, ])) {
        list_difference <- c(list_difference, i)
      }
    }
  }
  return(list(list_1[list_difference], list_difference))
}

#' Set up a canonical tableau for Phase I of the simplex method
#'
#' `construct_tableau_phase_one()` is an adjunct to `solve_MOLP()` that, given a constraint matrix `A` and a right-hand side vector `b`, returns a tableau in which Phase I of the simplex method of linear programming can be initiated directly.
#'
#' @param A A numeric matrix, one row per constraint and one column per variable, giving the left-hand side constraint coefficients.
#' @param b A numeric vector of the right-hand side coefficients.
#'
#' @return A list with three components: a simplex tableau augmented by artificial variables, and two numeric vectors recording the indices of the basic and nonbasic variables. See the documentation of [evaluate_bfs_from_tableau()] for more information.
#' @noRd
#'
#' @examples
construct_tableau_phase_one <- function(A, b) {
  #
  #
  if (is.matrix(A)) {
    n <- ncol(A)
    m <- nrow(A)
  }
  else if (is.vector(A)) {
    n <- length(A)
    m <- 1
  }
  #
  # negative RHS
  #
  for (i in 1:m) {
    if (b[i] < 0) {
      A[i, ] <- -1 * A[i, ]
      b[i] <- -1 * b[i]
    }
  }

  artificial_variable_index_set <- c()
  artificial_variable_row_index_set <- c()
  for (i in 1:m) {
    A <- cbind(A, v = diag(m)[, i])
    artificial_variable_index_set <- c(artificial_variable_index_set, ncol(A))
    artificial_variable_row_index_set <- c(artificial_variable_row_index_set, i)
  }
  # sense = 1 ~ max, sense = -1 ~ min
  #
  #message("Minimization problem: remember to multiply the optimal objective value by (-1)!")

  artificial_c <- c(rep(0, times = n), rep(1, times = m))
  # combine c,A,b
  tableau <- rbind(cbind(A, b), w = c(artificial_c, 0))

  if (length(artificial_variable_index_set) > 0) {
    #message(
    paste("Artificial variable 'v' added at column ", artificial_variable_index_set, "and row ", artificial_variable_row_index_set, " -> phase I algorithm required")
    #)
  }
  # Subtract from the w-equation the sum of the original m equations to obtain a canonical system with respect to the artificial variables (as basic variables)
  tableau[m + 1, ] <- tableau[m + 1, ] - colSums(tableau[1:m, ])
  basic_variables_indices <- artificial_variable_index_set
  nonbasic_variables_indices <- 1:n
  return(list(tableau, basic_variables_indices, nonbasic_variables_indices))
}

#' Perform Phase I of the simplex method
#'
#' `simplex_phase_one()` is an adjunct to `solve_MOLP()` that, given an initial
#' tableau, performs Phase I of the simplex method and returns a Phase I optimal tableau in
#' canonical form.
#'
#' @param tableau A list containing a simplex tableau in which Phase I can be
#'   initiated and the corresponding index sets for the basic and nonbasic
#'   variables. See the documentation of the return value of
#'   [construct_tableau_phase_one()] for further information.
#' @param max_iter An integer greater than or equal to 1 that defines the
#'   maximum number of iterations in Phase I. The default value is `100`.
#'
#' @return A list with a Phase I optimal tableau where all artificial variables
#'   are nonbasic, along with the associated basic and nonbasic index sets,
#'   following the template of `tableau`.
#' @noRd
#'
#' @examples
simplex_phase_one <- function(tableau, max_iter = 100) {
  iter <- 1
  while (!tableau_optimality_check(tableau, phase = 1) && iter < max_iter) {
    #message(paste("Iteration", iter))
    #message("--------------------------------------------------------------------")
    pivot_column <- get_pivot_column(tableau, phase = 1)
    #message(paste("Pivot column:", pivot_column))
    pivot_row_and_basic_index <- get_pivot_row_and_basic_index(tableau, pivot_column)
    pivot_row <- pivot_row_and_basic_index[1]
    pivot_basic_index <- pivot_row_and_basic_index[2]
    if (pivot_row == -1) {
      break()
    }
    #message(paste("Pivot row:", pivot_row))
    tableau <- pivot(tableau, pivot_row_and_basic_index, pivot_column)
    #message(paste("New tableau at the end of iteration", iter))
    #message(tableau)
    iter <- iter + 1
  }
  #message("--------------------------------------------------------------------")
  #message("Status: End")
  #message("--------------------------------------------------------------------")

  return(tableau)
}

#' Determine the index of an incoming basic variable for the simplex algorithm
#'
#' `get_pivot_column()` is an adjunct to `solve_MOLP()` that, given a simplex
#' tableau, determines which (if any) nonbasic variable should be turned basic
#' and returns its index.
#'
#' @param tableau A list, following the template of
#'   [evaluate_bfs_from_tableau()], with a canonical simplex tableau and two integer
#'   vectors giving the basic and nonbasic variable indices.
#' @param phase An integer: `1` for Phase I of the simplex method (default), `2` for Phase II.
#'
#' @return An integer giving the index of the nonbasic variable (the pivot
#'   column) to be brought into the basis.
#' @noRd
#'
#' @examples
get_pivot_column <- function(tableau, phase) {
  # For Phase I of the simplex method
  if (phase == 1) {
    temp <- tableau[[1]]
    nonbasics <- tableau[[3]] # indices of nonbasic variables
  }
  # For Phase II
  else if (phase == 2) {
    temp <- tableau[[1]][, 1:(ncol(tableau[[1]]) - nrow(tableau[[1]]))]
    nonbasics <- tableau[[3]][!(tableau[[3]] > (ncol(tableau[[1]]) - nrow(tableau[[1]])))] # indices of nonbasic nonartificial variables
  }

  return(nonbasics[which.min(temp[nrow(temp), nonbasics])])
}


#' Determine whether a basic feasible solution to a single-objective linear program is optimal
#'
#' `tableau_optimality_check()` is an adjunct to `solve_MOLP()` that returns `TRUE` if the basic feasible solution associated with simplex tableau `tableau` is optimal and `FALSE` otherwise.
#'
#' @param tableau A list, following the template of
#'   [evaluate_bfs_from_tableau()], with a canonical simplex tableau and two integer
#'   vectors giving the basic and nonbasic variable indices.
#' @param phase An integer: `1` for Phase I of the simplex method (default), `2` for Phase II.
#'
#' @return A boolean: `TRUE` if the basic feasible solution determined by `tableau` is optimal, `FALSE` otherwise.
#' @noRd
#'
#' @examples
tableau_optimality_check <- function(tableau, phase) {
  # For Phase I of the simplex method
  if (phase == 1) {
    temp <- tableau[[1]]
  }
  # For Phase II
  else if (phase == 2) {
    temp <- tableau[[1]][, 1:(ncol(tableau[[1]]) - nrow(tableau[[1]]))]
  }

  if (min(temp[nrow(temp), 1:(ncol(temp) - 1)]) < 0) {
    return(FALSE) # not optimal
  }
  else {
    return(TRUE) # optimal
  }
}

#' Set up a tableau for Phase II of the simplex method
#'
#' `prepare_simplex_phase_two()` is an adjunct to `solve_MOLP()` that takes a
#' Phase I optimal simplex tableau and prepares it for Phase II.
#'
#' @param end_of_phase_1_tableau A list, following the template of
#'   [evaluate_bfs_from_tableau()], with a Phase I optimal simplex tableau and
#'   two integer vectors giving the basic and nonbasic variable indices. The
#'   tableau's bottom row contains the artificial linear form \eqn{w} (as
#'   modified by the various eliminations in Phase I).
#' @param c A numeric vector, of length equal to the number of (nonartificial)
#'   variables, giving the objective function's coefficients.
#'
#' @return A list, following the template of [evaluate_bfs_from_tableau()], with
#'   a canonical simplex tableau in which Phase II can be initiated and two
#'   integer vectors specifying the indices of the basic and nonbasic variables.
#'   The tableau's bottom row represents the objective function as modified by the various eliminations.
#' @noRd
#'
#' @examples
prepare_simplex_phase_two <- function(end_of_phase_1_tableau, c) {
  temp <- end_of_phase_1_tableau[[1]]
  n_with_artificial <- ncol(temp)
  m <- nrow(temp) - 1
  c <- c(c, rep(0, times = m))

  tableau <- list()
  tableau[[1]] <- rbind(temp[1:m, ], z = c(c, 0)) # introducing z linear form
  tableau[[2]] <- end_of_phase_1_tableau[[2]] # initial set of basic variables
  tableau[[3]] <- end_of_phase_1_tableau[[3]] # initial set of nonbasic variables

  # We now have to eliminate the basic (nonartificial) variables from the z equation in order to begin Phase II
  for (basic_variable in tableau[[2]]) {
    row <- which(tableau[[1]][1:m, basic_variable] == 1)
    tableau[[1]][nrow(tableau[[1]]), ] <- tableau[[1]][nrow(tableau[[1]]), ] - tableau[[1]][nrow(tableau[[1]]), basic_variable] * tableau[[1]][row, ]
  }

  return(tableau)
}

#' Perform Phase II of the simplex method for a minimization problem
#'
#' `simplex_phase_two()` is an adjunct to `solve_MOLP()` that, given an
#' appropriate tableau, executes Phase II of the simplex method. The function
#' returns an optimal tableau in case a minimum exists.
#'
#' @param tableau A list containing a simplex tableau in which Phase II can be
#'   initiated directly and the corresponding index sets for the basic and
#'   nonbasic variables. See the documentation of the return value of
#'   [prepare_simplex_phase_two()] for further information.
#' @param max_iter An integer greater than or equal to 1 that defines the
#'   maximum number of iterations in Phase II. The default value is `100`.
#'
#' @return If a minimum exists: a list with an optimal tableau, along with the
#'   associated basic and nonbasic index sets, following the template of
#'   `tableau`.
#' @noRd
#'
#' @seealso [prepare_simplex_phase_two()] produces the kind of tableau that is expected as input to this function.
#'
#' @examples
simplex_phase_two <- function(phase_two_tableau, max_iter = 100) {
  iter <- 1
  while (!tableau_optimality_check(phase_two_tableau, phase = 2) && iter < max_iter) {
    pivot_column <- get_pivot_column(phase_two_tableau, phase = 2)
    pivot_row_and_basic_index <- get_pivot_row_and_basic_index(phase_two_tableau, pivot_column)
    pivot_row <- pivot_row_and_basic_index[1]
    pivot_basic_index <- pivot_row_and_basic_index[2]
    if (pivot_row == -1) {
      break()
    }
    phase_two_tableau <- pivot(phase_two_tableau, pivot_row_and_basic_index, pivot_column)
    iter <- iter + 1
  }

  return(phase_two_tableau)
}
