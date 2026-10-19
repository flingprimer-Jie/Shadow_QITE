"""Small spin Hamiltonians in the Pauli-term format used by qite_shadow.py.

Terms: (real_coefficient, [(zero_based_qubit_index, pauli_code), ...]);
Pauli codes are X=1, Y=2, Z=3. Identity factors are omitted.
"""


class HamiltonianFactory:
    @staticmethod
    def get_model(model_name):
        name = model_name.strip().lower()
        if name == "single_qubit_test":
            # H = -Z - 0.5 X; nontrivial for the |+> initial state.
            return [(-1.0, [(0, 3)]), (-0.5, [(0, 1)])], 1

        if name == "ising":
            # Open-boundary transverse-field Ising model, J = h = 1.
            n = 4
            terms = [(-1.0, [(i, 3), (i + 1, 3)]) for i in range(n - 1)]
            terms += [(-1.0, [(i, 1)]) for i in range(n)]
            return terms, n

        if name == "heisenberg":
            # Open-boundary antiferromagnetic XXX spin chain, J = +1.
            n = 4
            terms = [(1.0, [(i, p), (i + 1, p)])
                     for i in range(n - 1) for p in (1, 2, 3)]
            return terms, n

        if name == "lih_molecule":
            raise NotImplementedError(
                "LiH requires a validated electronic-structure Hamiltonian; "
                "no molecular model is included in this prototype."
            )

        raise ValueError(
            f"Unknown model {model_name!r}. Choose single_qubit_test, ising, or heisenberg."
        )
