import numpy as np
import itertools

"""
An implementation of the TODD algorithm for reducing the T count of a quantum circuit in the paper
[[1712.01557] An Efficient Quantum Compiler that reduces $T$ count](https://arxiv.org/abs/1712.01557) by Luke E Heyfron and Earl T Campbell (2019)
"""


def get_nullspace(A):
    """
    Method is based on the following stackexchange answer:
    [linear algebra - how to find null space basis directly by matrix calculation - Mathematics Stack Exchange](https://math.stackexchange.com/questions/1612616/how-to-find-null-space-basis-directly-by-matrix-calculation)
    """

    augmented = np.hstack((A.T, np.eye(A.shape[1], dtype=int)))

    n_rows = A.shape[1]
    n_cols = A.shape[0]

    curr_pivot = 0

    # make the first n_cols columns of the augmented matrix into rref form
    for col in range(n_cols):
        # make main diagonal non-zero
        if augmented[curr_pivot, col] == 0:
            for row in range(curr_pivot + 1, n_rows):
                if augmented[row, col] != 0:
                    augmented[curr_pivot] += augmented[row]
                    augmented[curr_pivot] %= 2
                    break
            else:
                continue

        # get other rows to have 0
        for row in range(curr_pivot + 1, n_rows):
            if augmented[row, col] != 0:
                augmented[row] += augmented[curr_pivot]
                augmented[row] %= 2

        curr_pivot += 1
        if curr_pivot == n_rows:
            break

    return augmented[curr_pivot:, n_cols:].T


def get_chi_matrix(A, z):
    if A.shape[0] != z.shape[0]:
        raise ValueError("A and z must have the same number of rows")

    seen_rows = set()

    chi_matrix = []

    for a, b, c in itertools.combinations(range(A.shape[0]), 3):
        row_a = A[a]
        row_b = A[b]
        row_c = A[c]

        new_row = (
            z[a] * (row_b * row_c) + z[b] * (row_a * row_c) + z[c] * (row_a * row_b)
        ) % 2

        if new_row.data.tobytes() in seen_rows:
            continue

        if new_row.sum() == 0:
            continue

        seen_rows.add(new_row.data.tobytes())

        chi_matrix.append(new_row)
    return np.array(chi_matrix)


def properize(A):
    cols_count = dict()
    for col in A.T:
        if col.data.tobytes() not in cols_count:
            cols_count[col.data.tobytes()] = 0
        cols_count[col.data.tobytes()] += 1

    # keep only columns that appear an odd number of times
    proper_A = np.array(
        [
            col
            for col in A.T
            if cols_count[col.data.tobytes()] % 2 == 1 and col.sum() > 0
        ]
    ).T

    # this still may include duplicate columns that appear an odd number of times
    # so we use np.unique to remove them
    if proper_A.size == 0:
        return proper_A
    return np.unique(proper_A, axis=1)


def todd_once(A, verbose=False):
    seen_z = set()

    for a, b in itertools.combinations(range(A.shape[1]), 2):
        matrix = A
        z = (A[:, a] + A[:, b]) % 2
        if z.data.tobytes() in seen_z:
            continue

        seen_z.add(z.data.tobytes())

        chi_matrix = get_chi_matrix(matrix, z)

        if chi_matrix.size > 0:
            A_tilde = np.vstack((matrix, chi_matrix))
        else:
            A_tilde = matrix

        nullspace = get_nullspace(A_tilde)

        if nullspace.size == 0:
            continue

        for y in nullspace.T:
            A_prime = matrix
            if y[a] ^ y[b] == 0:
                continue
            if np.sum(y) % 2 == 1:
                # append a zero column to A_prime
                A_prime = np.hstack(
                    (A_prime, np.zeros((A_prime.shape[0], 1), dtype=int))
                )
                y = np.hstack((y, 1))

            if verbose:
                print("Found optimization candidate")
                print(f"a, b = {a}, {b}")
                print(f"z = {z}")
                print(f"y = {y}")

            A_prime = (A_prime + np.outer(z, y)) % 2

            return properize(A_prime)

    return A


def todd(A, verbose=False):
    if A.size == 0:
        return A
    col_num = A.shape[1]

    while True:
        A = todd_once(A, verbose=verbose)
        if A.size == 0 or A.shape[1] == col_num:
            break
        col_num = A.shape[1]

    return A


def get_signature_tensor(A):
    if A.size == 0:
        return np.array([])
    n = A.shape[0]
    m = A.shape[1]

    S = np.zeros((n, n, n))
    for alpha, beta, gamma in itertools.combinations_with_replacement(range(n), 3):
        for i in range(m):
            S[alpha][beta][gamma] += A[alpha][i] * A[beta][i] * A[gamma][i]

    return S % 2


def signature_check(A, B, verbose):
    S_A = get_signature_tensor(A)
    S_B = get_signature_tensor(B)
    if A.size == 0:
        print("Initial matrix equals to Final matrix :", not np.any(S_B))
    elif B.size == 0:
        print("Initial matrix equals to Final matrix :", not np.any(S_A))
    else:
        print("Initial matrix equals to Final matrix :", np.array_equal(S_A, S_B))

    if verbose:
        print("--- S_old_A ---")
        print(S_A)
        print("--- S_new_A ---")
        print(S_B)


def main():
    from argparse import ArgumentParser

    parser = ArgumentParser(
        description='An implementation of the TODD algorithm for reducing the T count of a quantum circuit in the paper "An Efficient Quantum Compiler that reduces $T$ count" by Luke E Heyfron and Earl T Campbell (2019). arXiv link: https://arxiv.org/abs/1712.01557"',
    )

    parser.add_argument(
        "input",
        type=str,
        help="Path to a .csv file containing the matrix to be optimized",
    )

    parser.add_argument(
        "output",
        type=str,
        help="Path to a .csv file to write the optimized matrix to",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print the optimized process",
    )

    args = parser.parse_args()

    A = np.loadtxt(args.input, delimiter=",", dtype=int)

    # checks if the input matrix is binary
    for row in A:
        if not np.all(np.logical_or(row == 0, row == 1)):
            raise ValueError("Input matrix must be binary")

    # # Testing Code: random matrix
    # import random

    # random_seed = random.randint(0, 1000000)
    # print(f"Random seed: {random_seed}")
    # np.random.seed(random_seed)

    # # generate random 0-1 matrix
    # A = np.random.randint(2, size=(6, 100))

    print(
        "# terms in the initial matrix before properization: ",
        A.shape[1] if A.size > 0 else 0,
    )
    A = properize(A)
    if args.verbose:
        print("Initial Matrix after properization")
        print(A)

    print(
        "# terms in the initial matrix after properization: ",
        A.shape[1] if A.size > 0 else 0,
    )

    new_A = todd(A, verbose=args.verbose)

    print(
        "# terms after todd & properization: ", 0 if new_A.size == 0 else new_A.shape[1]
    )

    if args.verbose:
        print("Final Matrix")
        print(A)

    signature_check(A, new_A, verbose=args.verbose)

    np.savetxt(args.output, new_A, delimiter=",", fmt="%d")


if __name__ == "__main__":
    main()
