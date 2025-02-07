from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from autodp import rdp_bank
from autodp.autodp_core import Mechanism

from scipy.optimize import minimize_scalar
from autodp import transformer_zoo
import itertools


def change_refer_pred(dr, en):
    data_references = []
    for data_refer in dr:
        for i in range(en):
            data_references.append(data_refer)
    return data_references



# =============================================compute dp parameter===============================================


def rdp_to_approxdp(rdp, alpha_max=np.inf, BBGHS_conversion=True):
    # from RDP to approx DP
    # alpha_max is an optional input which sometimes helps avoid numerical issues
    # By default, we are using the RDP to approx-DP conversion due to BBGHS'19's Theorem 21
    # paper: https://arxiv.org/pdf/1905.09982.pdf
    # if you need to use the simpler RDP to approxDP conversion for some reason, turn the flag off

    def approxdp(delta):

        """
        approxdp outputs eps as a function of delta based on rdp calculations
        :param delta:
        :return: the \epsilon with a given delta
        """

        if delta < 0 or delta > 1:
            print("Error! delta is a probability and must be between 0 and 1")
        if delta == 0:
            return rdp(np.inf)
        else:
            def fun(x):  # the input the RDP's alpha
                if x <= 1:
                    return np.inf
                else:
                    if BBGHS_conversion:
                        return np.maximum(rdp(x) + np.log((x - 1) / x) - (np.log(delta) + np.log(x)) / (x - 1), 0)
                    else:
                        return np.log(1 / delta) / (x - 1) + rdp(x)

            results = minimize_scalar(fun, method='Bounded', bracket=(1, 2), bounds=(1, alpha_max))
            if results.success:
                return results.fun
            else:
                # There are cases when certain \delta is not feasible.
                # For example, let p and q be uniform the privacy R.V. is either 0 or \infty and unless all \infty
                # events are taken cared of by \delta, \epsilon cannot be < \infty
                return np.inf

    return approxdp


def approxRDP_to_approxDP(delta, delta0, rdp_func, alpha_max=np.inf, BBGHS_conversion=True):
    if delta < delta0:
        return np.inf

    delta1 = delta - delta0

    approxdp = rdp_to_approxdp(rdp_func, alpha_max, BBGHS_conversion)

    return approxdp(delta1)


# eps is the privacy parameter for EM
# delta0 is the failure probability for PTR
class EMandPTR(Mechanism):
    def __init__(self, eps, delta0, sigma, name='PTR'):
        Mechanism.__init__(self)
        self.name = name
        self.params = {'delta0': delta0, 'sigma': sigma}

        def privloss(t, alpha):
            return (np.exp(alpha * (eps - t)) - np.exp(-alpha * t) - (
                    np.exp(alpha * eps - (alpha + 1) * t) - np.exp(eps - (alpha + 1) * t))) / (
                    np.exp(eps - t) - np.exp(-t))

        def RDP_EM(alpha):
            temp = (np.sinh(alpha * eps) - np.sinh((alpha - 1) * eps)) / np.sinh(eps)
            return min(1 / 2 * alpha * eps ** 2, np.log(temp) / (alpha - 1))

        def approxRDP_PTR(alpha, delta0):
            return rdp_bank.RDP_gaussian({'sigma': sigma}, alpha)

        def approxRDP_EMandPTR(alpha, delta0):
            return rdp_bank.RDP_gaussian({'sigma': sigma}, alpha) + RDP_EM(alpha)

        def fakeRDP_EMandPTR(alpha):
            if alpha == np.inf:
                return np.inf
            return approxRDP_EMandPTR(alpha, delta0)

        self.propagate_updates(fakeRDP_EMandPTR, 'RDP')


class compose_subsampled_EMandPTR(Mechanism):
    def __init__(self, eps, delta0, sigma, prob, niter, name='compose_subsampled_EMandPTR'):
        Mechanism.__init__(self)
        self.name = name

        subsample = transformer_zoo.AmplificationBySampling()  # by default this is using poisson sampling
        compose = transformer_zoo.Composition()

        mech = EMandPTR(eps, delta0, sigma)
        mech.neighboring = 'add_remove'

        if prob < 1:
            mech = subsample(mech, prob * (1 - delta0) / (1 - prob * delta0), improved_bound_flag=False)

        mech = compose([mech], [niter])
        rdp_total = mech.RenyiDP

        self.propagate_updates(rdp_total, type_of_update='RDP')


# params = {eps, delta0, sigma, prob, niter}
def compose_subsampled_EMandPTR_to_approxDP(params, delta):
    mech = compose_subsampled_EMandPTR(params['eps'], params['delta0'], params['sigma'], params['prob'],
                                       params['niter'])
    rdp_func = mech.RenyiDP

    return approxRDP_to_approxDP(delta, params['delta0'] * params['prob'], rdp_func, alpha_max=200,
                                 BBGHS_conversion=True)






















# ========================================= RDP to DP ==========================================




def rdp_to_approxdp(rdp, alpha_max=np.inf, BBGHS_conversion=True):
    # from RDP to approx DP
    # alpha_max is an optional input which sometimes helps avoid numerical issues
    # By default, we are using the RDP to approx-DP conversion due to BBGHS'19's Theorem 21
    # paper: https://arxiv.org/pdf/1905.09982.pdf
    # if you need to use the simpler RDP to approxDP conversion for some reason, turn the flag off

    def approxdp(delta):

        """
        approxdp outputs eps as a function of delta based on rdp calculations
        :param delta:
        :return: the \epsilon with a given delta
        """

        if delta < 0 or delta > 1:
            print("Error! delta is a probability and must be between 0 and 1")
        if delta == 0:
            return rdp(np.inf)
        else:
            def fun(x):  # the input the RDP's alpha
                if x <= 1:
                    return np.inf
                else:
                    if BBGHS_conversion:
                        return np.maximum(rdp(x) + np.log((x - 1) / x) - (np.log(delta) + np.log(x)) / (x - 1), 0)
                    else:
                        return np.log(1 / delta) / (x - 1) + rdp(x)

            results = np.min([fun(alpha) for alpha in range(1, alpha_max)])
            return results

            # results = minimize_scalar(fun, method='Bounded', bounds=(1, alpha_max) )
            # if results.success:
            #     return results.fun
            # else:
            #     # There are cases when certain \delta is not feasible.
            #     # For example, let p and q be uniform the privacy R.V. is either 0 or \infty and unless all \infty
            #     # events are taken cared of by \delta, \epsilon cannot be < \infty
            #     return np.inf

    return approxdp


def approxRDP_to_approxDP(delta, delta0, rdp_func, alpha_max=np.inf, BBGHS_conversion=True):
    if delta < delta0:
        return np.inf

    delta1 = delta - delta0

    approxdp = rdp_to_approxdp(rdp_func, alpha_max, BBGHS_conversion)

    return approxdp(delta1)


# eps is the privacy parameter for EM
# add Gumbel(sensitivity/eps), where sensitivity is the sensitivity of utility function
class EM(Mechanism):
    def __init__(self, eps, name='EM', monotone=False):
        Mechanism.__init__(self)
        self.name = name
        self.params = {'eps': eps}

        if monotone:

            def privloss(t, alpha):
                return (np.exp(alpha * (eps - t)) - np.exp(-alpha * t) - (
                        np.exp(alpha * eps - (alpha + 1) * t) - np.exp(eps - (alpha + 1) * t))) / (
                        np.exp(eps - t) - np.exp(-t))

            def RDP_EM(alpha):
                if alpha == np.infty:
                    return eps
                enegt = ((alpha - 1) * (np.exp(alpha * eps) - 1)) / ((alpha) * (np.exp(alpha * eps) - np.exp(eps)))
                return np.log(privloss(np.log(1 / enegt), alpha)) / (alpha - 1)

        else:

            def RDP_EM(alpha):
                if alpha == np.infty:
                    return eps * 2
                temp = (np.sinh(alpha * eps) - np.sinh((alpha - 1) * eps)) / np.sinh(eps)
                return min(1 / 2 * alpha * eps ** 2, np.log(temp) / (alpha - 1))

        self.propagate_updates(RDP_EM, 'RDP')


class compose_subsampled_EM(Mechanism):
    def __init__(self, eps, prob, niter, name='compose_subsampled_EM', monotone=False):
        Mechanism.__init__(self)
        self.name = name

        subsample = transformer_zoo.AmplificationBySampling()  # by default this is using poisson sampling
        compose = transformer_zoo.Composition()

        mech = EM(eps, monotone=monotone)
        mech.neighboring = 'add_remove'

        if prob < 1:
            mech = subsample(mech, prob, improved_bound_flag=False)

        mech = compose([mech], [niter])
        rdp_total = mech.RenyiDP

        self.propagate_updates(rdp_total, type_of_update='RDP')


# params = {eps, prob, niter}
def compose_subsampled_EM_to_approxDP(params, delta):
    mech = compose_subsampled_EM(params['eps'], params['prob'], params['niter'], monotone=params['monotone'])
    rdp_func = mech.RenyiDP

    # for alpha in range(1, 200):
    #   print('rdp_func({})={}'.format(alpha, rdp_func(alpha)))

    approxdp = rdp_to_approxdp(rdp_func, alpha_max=200, BBGHS_conversion=True)

    return approxdp(delta)




















def make_diff_matrix(item_counts, k):
    """Makes diff matrix where diff_matrix[i,j] = c_i - c_j + uniquifying term.

    Args:
      item_counts: Array of item counts, sorted in decreasing order.
      k: Number of top counts desired.

    Returns:
      k x d matrix diff_matrix where diff_matrix[i,j] = c_i - c_j + (d(k-i-1) + j
      + 1) / (2dk) and c_1 <= c_2 <= ... <= c_d. diff_matrix is therefore strictly
      increasing along rows and strictly decreasing down columns. Note that the
      added uniquifying term is determined entirely by d, i, j, k and is therefore
      data-independent. Moreover, since the counts are integers, their differences
      are integers as well; the uniquifying terms, which have maximum
      dk / (2dk) = 0.5, therefore do not change the relative order of any
      non-identical count differences.
    """
    d = len(item_counts)
    base_along_row = np.arange(1, d + 1)
    base_down_col = np.arange(k - 1, -1, -1) * d
    uniquifying_terms = (base_along_row[np.newaxis, :] +
                         base_down_col[:, np.newaxis]) / (2 * d * k)
    return (item_counts[:k, np.newaxis] -
            item_counts[np.newaxis, :]) + uniquifying_terms


def get_diffs_to_positions(diff_matrix):
    """Computes array a where diff_matrix[a[0][i], a[1][i]] = sorted_diffs[i].

    Args:
      diff_matrix: Matrix of distinct count differences.

    Returns:
      Array a where diff_matrix[a[0][i], a[1][i]] = sorted_diffs[i], where
      sorted_diffs contains all entries of diff_matrix in increasing order.
    """
    # The below line of the code runs in time O(d * k * log(dk)). This could be
    # implemented more efficiently, leveraging the fact that diff_matrix is
    # strictly increasing along rows. This property allows us to use k-way merging
    # (https://en.wikipedia.org/wiki/K-way_merge_algorithm), which can bring the
    # runtime down to O(d * k * log(k)). However, since this function is not a
    # practical bottleneck in the code, we leave it as-is for now.
    return np.unravel_index(np.argsort(diff_matrix, axis=None), diff_matrix.shape)


def brute_compute_log_diff_counts(diff_matrix, sorted_diffs):
    """Computes array of log(# sequences w/ diff) for diff in diff_matrix.

    Args:
      diff_matrix: Matrix of distinct count differences.
      sorted_diffs: Diffs from diff_matrix, in increasing order.

    Returns:
      Array log_counts where, for array sorted_diffs of diffs from diff_matrix
      sorted in increasing order,
      log_counts[i] = log(# of sequences where largest count difference is diff
      from sorted_diffs[i]), computed using brute force.
    """
    k, d = diff_matrix.shape
    possible_sequences = itertools.permutations(np.arange(d), k)
    diffs_to_counts = np.zeros(d * k)
    for sequence in possible_sequences:
        diff = np.amax(
            [diff_matrix[row_idx, sequence[row_idx]] for row_idx in np.arange(k)])
        diff_idx = np.searchsorted(sorted_diffs, diff)
        diffs_to_counts[diff_idx] += 1
    # Ignore warnings from taking log(0). This produces -np.inf as intended.
    with np.errstate(divide="ignore"):
        return np.log(diffs_to_counts)


def compute_log_diff_counts(diff_matrix, diffs_to_positions):
    """Computes array of log(sequence count) for each diff in diff_matrix.

    Args:
      diff_matrix: Matrix of distinct count differences.
      diffs_to_positions: Dictionary mapping diffs to positions in diff_matrix.

    Returns:
      Array log_counts where, for array sorted_diffs of diffs from diff_matrix
      sorted in decreasing order, log_counts[i] = log(# of sequences where largest
      count difference is diff from sorted_diffs[i]), computed using Lemma 3.7
      (the definition of \tilde{m}) from the paper.

    Raises:
      RuntimeError: ns vector never filled.
    """
    k, d = diff_matrix.shape
    num_diffs = d * k
    log_diff_counts = np.empty(num_diffs)
    log_ns = np.empty(k)
    indices_filled = set()
    last_diff_idx_processed = -1
    # Ignore warnings from, respectively, taking logs of 0 or negative numbers.
    # log(0) becomes -np.inf as intended, and log(<0) becomes nan and is ignored.
    with np.errstate(divide="ignore"):
        with np.errstate(invalid="ignore"):
            updates = np.log((diffs_to_positions[1] + 1) - diffs_to_positions[0])
    for (diff_idx, i, u) in zip(range(num_diffs), diffs_to_positions[0], updates):
        if np.isnan(u):
            continue
        log_ns[i] = u
        indices_filled.add(i)
        if len(indices_filled) == k:
            last_diff_idx_processed = diff_idx
            break
    if last_diff_idx_processed == -1:
        raise RuntimeError("ns vector never filled")
    log_diff_counts[:last_diff_idx_processed] = -np.inf
    log_ns_sum = np.sum(log_ns)
    for (diff_idx, i, u) in zip(
            range(last_diff_idx_processed, num_diffs),
            diffs_to_positions[0][diff_idx:], updates[diff_idx:]):
        log_ns_sum -= log_ns[i]
        log_diff_counts[diff_idx] = log_ns_sum
        log_ns[i] = u
        log_ns_sum += log_ns[i]
    return log_diff_counts


def racing_sample(log_terms):
    """Numerically stable method for sampling from an exponential distribution.

    Args:
      log_terms: Array of terms of form log(coefficient) - (exponent term).

    Returns:
      A sample from the exponential distribution determined by terms. See
      Algorithm 1 from the paper "Duff: A Dataset-Distance-Based
      Utility Function Family for the Exponential Mechanism"
      (https://arxiv.org/pdf/2010.04235.pdf) for details; each element of terms is
      analogous to a single log(lambda(A_k)) - (eps * k/2) in their algorithm.

    Raises:
      RuntimeError: encountered inf or nan min time.
    """
    race_times = np.log(np.log(
        1.0 / np.random.uniform(size=log_terms.shape))) - log_terms
    winner = np.argmin(race_times)
    min_time = race_times[winner]
    if np.isnan(min_time) or np.isinf(min_time):
        raise RuntimeError(
            "Racing sample encountered inf or nan min time: {}".format(min_time))
    return winner


def sample_diff_idx(log_diff_counts, sorted_diffs, epsilon, sensitivity):
    """Racing samples a diff index from the exponential mechanism.

    Args:
      log_diff_counts: Array of log(# sequences with diff) for each diff in
        sorted_diffs.
      sorted_diffs: Increasing array of possible diffs.
      epsilon: Privacy parameter epsilon.
      neighbor_type: Available neighbor types are defined in the NeighborType
        enum.

    Returns:
      Index idx sampled from, for diff = sorted_diffs[idx], distribution
      P(diff) ~ count[diff] * exp(-epsilon * floor(diff) / 2).
    """
    return racing_sample(log_diff_counts - (epsilon * np.floor(sorted_diffs) /
                                            (2 * sensitivity)))


def sample_max_expo_distribution(expo_lambda, log_num_trials):
    """Computes max values from (simulated) draws from the expo distribution.

    Args:
      expo_lambda: Exponential distribution parameter; PDF of the distribution is
        p(x) = expo_lambda * exp(-expo_lambda * x).
      log_num_trials: Array where each entry is the log of the number of
        (simulated) draws from the exponential distribution to use in producing a
        max value.

    Returns:
      An array where entry i represents the max over exp(log_num_trials[i]) draws
      from the exponenttial distribution with parameter expo_lambda.

    Raises:
      RuntimeError: result contains inf or nan.
    """
    # Rather than actually sampling from the exponential distribution num_trials
    # times and taking the max (an O(num_trials) operation), we can simply draw
    # from a distribution that directly represents this max (an O(1) operation).
    # The probability that num_trials independent draws from a distribution are
    # all <= x is the CDF of that distribution raised to the n-th power.  Hence,
    # the CDF for the max that we need is the exponential distribution CDF raised
    # to the n-th power.  Inverting this CDF and plugging in a sample from the
    # uniform distribution on [0, 1] gives the desired max.
    num_results = len(log_num_trials)
    # The below line is more numerically stable than 1 / np.exp(log_num_trials).
    inverse_num_trials = np.exp(-log_num_trials)
    log_uniform_draws = np.log(np.random.uniform(size=num_results))
    results = -np.log(-np.expm1(
        np.multiply(inverse_num_trials, log_uniform_draws))) / expo_lambda
    if np.any(np.isnan(results)) or np.any(np.isinf(results)):
        raise RuntimeError(
            "Max expo sampler result contains inf or nan: {}".format(results))
    return results


def sample_diff_idx_via_pnf(log_diff_counts, sorted_diffs, epsilon,
                            sensitivity):
    """Samples an index of sorted_diffs according to permute-and-flip (PNF).

    Args:
      log_diff_counts: Array of log(# sequences with diff) for each diff in
        sorted_diffs.
      sorted_diffs: Increasing array of possible diffs.
      epsilon: Privacy parameter epsilon.
      neighbor_type: Available neighbor types are defined in the NeighborType
        enum.

    Returns:
      Index idx into sorted_diffs, sampled according to the permute-and-flip
      mechanism.

    Raises:
      RuntimeError: No noised value exceeded -infinity.
    """
    expo_lambda = epsilon / (2 * sensitivity)

    # Exclude entries with a count of zero.
    nonzero_count_indicator = ~np.isneginf(log_diff_counts)

    # Permute-and-flip is identical to report-noisy-max with exponential noise;
    # see the paper "The Permute-and-Flip Mechanism Is Identical to
    # Report-Noisy-Max with Exponential Noise"
    # (https://arxiv.org/pdf/2105.07260.pdf) for details.  The latter formulation
    # is simpler to implement efficiently, so that is what we use internally here.
    # Specifically, we draw a max noise value for each of the utility values (the
    # diffs), and add this max to the utility.  We then return the index where
    # this results in the largest noisy utility value.
    utilities = -np.floor(sorted_diffs[nonzero_count_indicator])
    noisy_utilities = utilities + sample_max_expo_distribution(
        expo_lambda, log_diff_counts[nonzero_count_indicator])
    if np.all(np.isneginf(noisy_utilities)):
        raise RuntimeError("No noised value exceeded -infinity")
    nonzero_count_indices = np.flatnonzero(nonzero_count_indicator)
    return nonzero_count_indices[np.argmax(noisy_utilities)]


def sequence_from_diff(diff,
                       diff_row,
                       diff_col,
                       diff_matrix,
                       sampler=lambda ary: np.random.choice(ary, 1)):
    """Samples a sequence with given diff uniformly at random.

    Args:
      diff: Diff (negative utility) of sequence to sample.
      diff_row: diff_matrix[diff_row, diff_col] = diff.
      diff_col: diff_matrix[diff_row, diff_col] = diff.
      diff_matrix: Matrix of distinct count differences.
      sampler: Function that selects an item from an array. Default value is
        uniform random choice.

    Returns:
      Array of item indices forming a sequence with utility -diff.
    """
    k = len(diff_matrix)
    sequence = np.full(k, diff_col)
    ts = [
        np.searchsorted(diff_matrix[row, :], diff, side="right")
        for row in range(k)
    ]
    for row in range(k):
        if row != diff_row:
            # The below line of the code runs in time O(dk), which technically makes
            # the runtime of the sequence_from_diff function O(dk^2). This could be
            # implemented more efficiently, bringing the runtime of the function down
            # to O(dk). However, this is not a practical bottleneck in the code, so we
            # leave it as-is for now.
            to_sample = [i for i in range(ts[row]) if i not in sequence]
            sequence[row] = sampler(to_sample)
    return sequence


def joint(item_counts,
          k,
          epsilon,
          neighbor_type,
          sample_diff_idx_func=sample_diff_idx):
    """Applies joint exponential mechanism to return sequence of top-k items.

    Args:
      item_counts: Array of item counts.
      k: Number of items with top counts to return.
      epsilon: Privacy parameter epsilon.
      neighbor_type: Available neighbor types are defined in the NeighborType
        enum.
      sample_diff_idx_func: Function to call for sampling a utility value.

    Returns:
      Array of k item indices as estimated by the joint exponential mechanism.
    """
    # Sort the counts in non-increasing order.
    sort_indices = np.argsort(item_counts)[::-1]
    item_counts = item_counts[sort_indices]

    # Note that the diff_matrix here is the negative of the \tilde{U} matrix from
    # the paper.
    diff_matrix = make_diff_matrix(item_counts, k)
    diffs_to_positions = get_diffs_to_positions(diff_matrix)
    log_diff_counts = compute_log_diff_counts(diff_matrix, diffs_to_positions)
    sorted_diffs = diff_matrix[diffs_to_positions]
    diff_idx = sample_diff_idx_func(log_diff_counts, sorted_diffs, epsilon,
                                    neighbor_type)
    diff_row, diff_col = diffs_to_positions[0][diff_idx], diffs_to_positions[1][
        diff_idx]
    sequence = sequence_from_diff(sorted_diffs[diff_idx], diff_row, diff_col,
                                  diff_matrix)
    # Convert the indices returned by sequence_from_diff to the original item ids.
    # return sort_indices[sequence]
    return sequence


def pnf_joint(item_counts, k, epsilon, neighbor_type):
    """Applies joint permute-and-flip mechanism to return sequence of top-k items.

       Internals are identical to the exponential mechanism version of joint,
       except for the sample_diff_idx method.

    Args:
      item_counts: Array of item counts.
      k: Number of items with top counts to return.
      epsilon: Privacy parameter epsilon.
      neighbor_type: Available neighbor types are defined in the NeighborType
        enum.

    Returns:
      Array of k item indices as estimated by the joint permute-and-flip
      mechanism.
    """
    return joint(item_counts, k, epsilon, neighbor_type, sample_diff_idx_via_pnf)