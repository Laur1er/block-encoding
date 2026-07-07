import math
from typing import Optional, Sequence


class ProbabilityNode:
    """Noeud de l'arbre binaire des amplitudes/probabilités."""

    __slots__ = ("data", "left", "right", "parent")

    def __init__(self, data: float):
        self.data = data
        self.left: Optional["ProbabilityNode"] = None
        self.right: Optional["ProbabilityNode"] = None
        self.parent: Optional["ProbabilityNode"] = None

    def __repr__(self) -> str:
        return f"ProbabilityNode({self.data:.6f})"


class BinaryAmplitudeTree:
    """
    Arbre binaire utilisé pour construire l'oracle PREPARE.

    Les feuilles contiennent les coefficients normalisés (avec padding à 0
    si len(coeffs) n'est pas une puissance de 2). Chaque noeud interne
    contient la somme des données de ses deux enfants.
    """

    def __init__(self, coeffs: Sequence[float]):
        if not coeffs:
            raise ValueError("La liste de coefficients ne peut pas être vide.")
        if any(c < 0 for c in coeffs):
            raise ValueError("Les coefficients doivent être positifs ou nuls.")

        total = sum(coeffs)
        if total == 0:
            raise ValueError("La somme des coefficients ne peut pas être nulle.")

        self.depth: int = math.ceil(math.log2(len(coeffs))) if len(coeffs) > 1 else 0

        self.s: float = total
        self.coeffs: list[float] = [c / total for c in coeffs]

        n_leaves = 2**self.depth
        self.leaves: list[ProbabilityNode] = [
            ProbabilityNode(self.coeffs[i] if i < len(self.coeffs) else 0.0)
            for i in range(n_leaves)
        ]

        # On garde une trace de chaque niveau: layers[0] = [root], layers[depth] = leaves
        self.layers: list[list[ProbabilityNode]] = [None] * (self.depth + 1)
        self.layers[self.depth] = self.leaves

        previous_layer = self.leaves
        for level in range(self.depth - 1, -1, -1):
            previous_layer = self._build_layer(previous_layer)
            self.layers[level] = previous_layer

        self.root: ProbabilityNode = self.layers[0][0]

    @staticmethod
    def _build_layer(layer: list[ProbabilityNode]) -> list[ProbabilityNode]:
        new_layer = []
        for left, right in zip(layer[0::2], layer[1::2]):
            parent = ProbabilityNode(left.data + right.data)
            left.parent = parent
            right.parent = parent
            parent.left = left
            parent.right = right
            new_layer.append(parent)
        return new_layer

    def calculate_rotation_angles(self) -> list[list[float]]:
        """
        Calcule les angles de rotation Ry pour chaque qubit, niveau par niveau.

        theta_i^(k) = 2 * arccos( sqrt( p_{2i}^(k+1) / p_i^(k) ) )

        Retourne une liste de listes:
          - result[k] = angles pour le qubit k (niveau k de l'arbre)
          - result[k][i] = angle du noeud i à ce niveau (i.e. pour l'état de
            contrôle correspondant aux k premiers qubits déjà mesurés/fixés,
            en ordre binaire croissant: 0, 1, 2, ...)

        Cas limite: si p_i^(k) == 0, l'angle est fixé à 0.
        """
        result: list[list[float]] = []

        for k in range(self.depth):
            angles_k = []
            for node in self.layers[k]:
                p_i = node.data

                if p_i == 0:
                    theta = 0.0
                else:
                    p_2i = node.left.data
                    ratio = p_2i / p_i
                    # Protection contre les erreurs d'arrondi (ratio légèrement > 1 ou < 0)
                    ratio = min(1.0, max(0.0, ratio))
                    theta = 2 * math.acos(math.sqrt(ratio))

                angles_k.append(theta)
            result.append(angles_k)

        return result
