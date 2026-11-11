r"""Opérateur HWP démodulé, avec les règles de réduction associées au BandpassOperator.

Modèle de données : d = Bp · H · A · s

    A  : composantes          → ciel aux fréquences d'intégration ν_k   (n_nu, n_pix)
    H  : HWP démodulée (4f) à chaque ν_k                                 (n_nu, n_pix)
    Bp : intégration sur la bande de chaque canal                        (n_ch, n_pix)

Sous la forme calibrée (HWP idéale = identité), la HWP s'écrit à chaque fréquence

    Q' = a(ν)·Q − b(ν)·U
    U' = b(ν)·Q + a(ν)·U        soit   H(ν) = a(ν)·1 + b(ν)·J,   J(Q, U) = (−U, Q).

J ne dépend pas de ν : il commute avec les SED et avec la somme sur la bande, d'où

    (Bp @ H @ A).reduce()  =  (Bp_a @ A).reduce()  +  J @ (Bp_b @ A).reduce()

avec Bp_a, Bp_b de poids W_ck·a_k et W_ck·b_k. Chaque terme est un BlockRowOperator de
BroadcastDiagonalOperator, comme sans HWP.

La réduction se fait en trois règles, sur le modèle de BandpassOperator.py :
    1. BandpassHWPRule              Bp @ H                → HWPBandpassOperator(W·a, W·b)
    2. HWPBandpassBlockRowRule      HB @ [A₁ | A₂ | ...]  → [HB @ A₁ | HB @ A₂ | ...]
    3. HWPBandpassSEDRule           HB @ SED              → diag(W·a SED) + J @ diag(W·b SED)

Les règles ne s'appliquent qu'aux cartes QU (StokesQU), celles de la séparation de
composantes. En IQU, H s'applique normalement mais la composition n'est pas réduite.

Les coefficients a(ν), b(ν) viennent du modèle d'empilement de kwitchi (HWPStackOperator /
HWPStackHFOperator), recopié dans ce fichier : il ne dépend que de jax, numpy et furax.

Limite : H dépend de ν seulement, pas du canal. Si deux canaux de HWP différentes (MF et HF)
partageaient des fréquences d'intégration, il faudrait un H par canal. Avec les bandes des
SAT de SO (MF ≤ ~170 GHz, UHF ≥ ~190 GHz), les fréquences ne se recouvrent pas.
"""

from dataclasses import field

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, PyTree

from furax import AbstractLinearOperator, BlockRowOperator, BroadcastDiagonalOperator
from furax.core.rules import AbstractCompositionRule, NoReduction
from furax.obs.operators._seds import AbstractSEDOperator
from furax.obs.stokes import StokesIQU, StokesQU

from .BandpassOperator import BandpassOperator

__all__ = [
    'HWPOperator',
    'HWPBandpassOperator',
    'QUJOperator',
    'check_hwp_bandpass_compatibility',
    'hwp_mixing_diagonals',
    'hwp_4f_coefficients',
]



###############################################################################
# HWP STACK MODEL (copied from kwitchi/hwp/operators.py: HWPStackOperator / HWPStackHFOperator)
###############################################################################

c   = 2.998e8          # m/s
GHz = 1e9
deg = jnp.pi / 180.
mm  = 0.001

# Épaisseurs et angles des plaques de saphir (même empilement que kwitchi)
# MF
thicknesses = [0.394*mm, 0.04*mm, 0.212*mm,
               3.75*mm, 3.75*mm, 3.75*mm,
               0.212*mm, 0.04*mm, 0.394*mm]
angles = [0., 0., 0., 0., 54.*deg, 0., 0., 0., 0.]

# HF
thicknesses_HF = [0.183*mm, 0.04*mm, 0.097*mm,
                  1.60*mm, 1.60*mm, 1.60*mm,
                  0.097*mm, 0.04*mm, 0.183*mm]
angles_HF = [0., 0., 0., 0., 57.*deg, 0., 0., 0., 0.]


def get_delta(nu, theta, n, nO=3.05):
    return 2 * jnp.pi * nu * (nO - n) * theta / c


def compute_effective_index(angleIncidence, chi, nE=3.38, nO=3.05):
    sin_inc_sq = jnp.sin(angleIncidence)**2
    cos_chi_sq = jnp.cos(chi)**2
    return nE * jnp.sqrt(1 + (nE**-2 - nO**-2) * sin_inc_sq * cos_chi_sq)


def rotation_matrix_mueller(angle):
    c2 = jnp.cos(2 * angle)
    s2 = jnp.sin(2 * angle)
    return jnp.array([
        [1, 0, 0, 0],
        [0, c2, -s2, 0],
        [0, s2, c2, 0],
        [0, 0, 0, 1]
    ])


def HWP_layer(nu, theta, angleIncidence, chi, nE=3.38, nO=3.05):
    n = compute_effective_index(angleIncidence, chi, nE, nO)
    d = get_delta(nu, theta, nE, nO)
    cos_d = jnp.cos(d)
    sin_d = jnp.sin(d)

    return jnp.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, cos_d, sin_d],
        [0.0, 0.0, -sin_d, cos_d]
    ])


def hwp_stack_mueller(nu: float, *, is_hf: bool, angle_incidence: float = 5.0) -> np.ndarray:
    """Matrice de Mueller IQU (3×3) de l'empilement de 3 plaques, à ν [GHz].

    Même calcul que `HWPStackOperator.mv` (MF) et `HWPStackHFOperator.mv` (HF) de kwitchi,
    flip de U compris (il met M dans la même convention que la matrice de transfert),
    sans epsilon / phi.
    """
    thickness = (thicknesses_HF if is_hf else thicknesses)[3]
    alpha = (angles_HF if is_hf else angles)[4]

    H1 = HWP_layer(nu * GHz, thickness, angle_incidence * deg, 0.)
    H2 = HWP_layer(nu * GHz, thickness, angle_incidence * deg, alpha)

    M = H1 @ rotation_matrix_mueller(alpha).T @ H2 @ rotation_matrix_mueller(alpha) @ H1
    M = M.at[2, :].multiply(-1).at[:, 2].multiply(-1)[:-1, :-1]   # flip de U, enlève V
    return np.asarray(M)


def hwp_4f_coefficients(nu: float, *, is_hf: bool, angle_incidence: float = 5.0) -> tuple[float, float]:
    """Coefficients 4f a = (M_QQ − M_UU)/2 et b = (M_QU + M_UQ)/2 de la HWP à ν [GHz].

    Modèle de l'empilement (HWPStackOperator de kwitchi), recopié dans ce fichier.
    """
    M = hwp_stack_mueller(nu, is_hf=is_hf, angle_incidence=angle_incidence)
    return 0.5 * float(M[1, 1] - M[2, 2]), 0.5 * float(M[1, 2] + M[2, 1])


class HWPOperator(AbstractLinearOperator):
    r"""HWP démodulée (4f), forme calibrée, appliquée à chaque fréquence d'intégration.

    Entrée / sortie : StokesQU ou StokesIQU dont les champs ont la forme `(..., n_nu, n_pix)`,
    la fréquence étant l'avant-dernier axe, comme pour `BandpassOperator`.

        I' = I
        Q' = a(ν)·Q − b(ν)·U
        U' = b(ν)·Q + a(ν)·U

    Attributes:
        a: (M_QQ − M_UU)/2 à chaque fréquence, forme `(n_nu,)`.
        b: (M_QU + M_UQ)/2 à chaque fréquence, forme `(n_nu,)`.
        stacks: Nom de l'empilement utilisé à chaque fréquence (ex. 'MF', 'HF', 'none'), ou
            None s'il est inconnu. Sert à vérifier, lors de la composition avec un
            `BandpassOperator`, que chaque canal ne voit qu'une seule HWP.
    """

    a: Float[Array, ' nu']
    b: Float[Array, ' nu']
    stacks: tuple[str, ...] | None = field(default=None, metadata={'static': True})

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, 'a', jnp.asarray(self.a, dtype=float))
        object.__setattr__(self, 'b', jnp.asarray(self.b, dtype=float))
        n_nu = self.a.shape[0]
        if self.a.shape != (n_nu,) or self.b.shape != (n_nu,):
            raise ValueError(f'a and b must have shape (n_nu,), got {self.a.shape}, {self.b.shape}.')
        if self.stacks is not None:
            object.__setattr__(self, 'stacks', tuple(str(s) for s in self.stacks))
            if len(self.stacks) != n_nu:
                raise ValueError(f'stacks must have {n_nu} entries, got {len(self.stacks)}.')
        for leaf in jax.tree.leaves(self.in_structure):
            if len(leaf.shape) < 2 or leaf.shape[-2] != n_nu:
                raise ValueError(
                    f'The input maps must have {n_nu} integration frequencies on their '
                    f'second-to-last axis, got shape {leaf.shape}.'
                )

    @classmethod
    def from_kwitchi(
        cls,
        frequencies: Float[Array, ' nu'],
        in_structure: PyTree[jax.ShapeDtypeStruct],
        *,
        angle_incidence: float = 5.0,
        lf_max: float = 60.0,
        hf_min: float = 190.0,
    ) -> 'HWPOperator':
        """Coefficients a, b calculés avec le modèle d'empilement (HWPStackOperator de kwitchi).

        Choix de l'empilement selon la fréquence d'intégration ν [GHz] :
            ν < lf_max           : pas d'empilement connu, HWP idéale (a = 1, b = 0)
            lf_max ≤ ν < hf_min  : empilement MF
            ν ≥ hf_min           : empilement HF

        Les matrices de Mueller sont calculées ici, une fois : ne pas appeler cette méthode
        dans une fonction jittée ou dans la boucle d'optimisation.
        """
        nus = np.atleast_1d(np.asarray(frequencies, dtype=float))
        a = np.ones(nus.size)
        b = np.zeros(nus.size)
        stacks = []
        for k, nu in enumerate(nus):
            if nu < lf_max:
                stacks.append('none')
                continue
            is_hf = bool(nu >= hf_min)
            stacks.append('HF' if is_hf else 'MF')
            a[k], b[k] = hwp_4f_coefficients(nu, is_hf=is_hf, angle_incidence=angle_incidence)
        return cls(a, b, tuple(stacks), in_structure=in_structure)

    def mv(self, x: StokesQU | StokesIQU) -> StokesQU | StokesIQU:
        a = self.a[:, None]
        b = self.b[:, None]
        q = a * x.q - b * x.u
        u = b * x.q + a * x.u
        if isinstance(x, StokesIQU):
            return StokesIQU(i=x.i, q=q, u=u)
        if isinstance(x, StokesQU):
            return StokesQU(q=q, u=u)
        raise NotImplementedError(f'Stokes type {type(x).__name__} not supported.')


class QUJOperator(AbstractLinearOperator):
    """Rotation de 90° dans le plan QU : (Q, U) → (−U, Q). Sa transposée est (Q, U) → (U, −Q)."""

    def mv(self, x: StokesQU) -> StokesQU:
        return StokesQU(q=-x.u, u=x.q)


class HWPBandpassOperator(AbstractLinearOperator):
    r"""Composition Bp @ H réduite : intégration sur la bande d'une HWP chromatique.

        d_c^Q = Σ_k W^a_ck Q_k − W^b_ck U_k
        d_c^U = Σ_k W^b_ck Q_k + W^a_ck U_k

    avec W^a = W·a et W^b = W·b. Entrée `(..., n_nu, n_pix)`, sortie `(..., n_ch, n_pix)`.
    Uniquement pour des cartes StokesQU.
    """

    weights_a: Float[Array, 'channel nu']
    weights_b: Float[Array, 'channel nu']

    def mv(self, x: StokesQU) -> StokesQU:
        q = _integrate(self.weights_a, x.q) - _integrate(self.weights_b, x.u)
        u = _integrate(self.weights_b, x.q) + _integrate(self.weights_a, x.u)
        return StokesQU(q=q, u=u)


def _integrate(weights: Float[Array, 'c n'], x: Float[Array, '... n p']) -> Float[Array, '... c p']:
    return jnp.einsum('cn,...np->...cp', weights, x)


def _is_qu(structure: PyTree[jax.ShapeDtypeStruct]) -> bool:
    return isinstance(structure, StokesQU)


def check_hwp_bandpass_compatibility(Bp: BandpassOperator, H: HWPOperator) -> None:
    """Vérifie que chaque canal ne voit qu'une seule HWP sur sa bande.

    H ne dépend que de ν : si la bande d'un canal (fréquences de poids non nul) contient des
    fréquences auxquelles H applique des empilements différents, ce canal recevrait une HWP qui
    n'est pas la sienne sur une partie de sa bande. Cela arrive quand les bandes de canaux de HWP
    différentes se recouvrent, ou quand un seuil lf_max / hf_min tombe dans une bande.

    Ne vérifie rien si H.stacks vaut None.

    Raises:
        ValueError: si un canal voit plusieurs empilements.
    """
    if H.stacks is None:
        return
    weights = np.asarray(Bp.weights)
    for channel, row in enumerate(weights):
        support = np.nonzero(row)[0]
        stacks = {H.stacks[k] for k in support}
        if len(stacks) > 1:
            details = []
            for stack in sorted(stacks):
                idx = [int(k) for k in support if H.stacks[k] == stack]
                details.append(f'{stack} : {len(idx)} fréquences, indices {idx[0]} à {idx[-1]}')
            details = ' ; '.join(details)
            raise ValueError(
                'Une fréquence de poids non nul est utilisée par deux canaux de HWP différentes : '
                f'la bande du canal {channel} contient plusieurs empilements HWP ({details}). '
                'Les bandes de canaux de HWP différentes se recouvrent, ou un seuil lf_max / '
                'hf_min de HWPOperator.from_kwitchi tombe dans la bande de ce canal.'
            )


def hwp_mixing_diagonals(A: AbstractLinearOperator) -> tuple[Array, Array]:
    """Extrait les deux matrices de mélange de `(Bp @ H @ A).reduce()`.

    Après réduction, chaque bloc (une composante) vaut `A_a + J @ A_b`. Renvoie les SED
    intégrées des deux termes, empilées sur les composantes, dans l'ordre de `A.block_leaves`
    (cmb, dust, synchrotron), chacune de forme `(n_ch, n_comp, n_pix ou 1)` :

        matrix_A_a[c, i] = Σ_k W_ck a_k SED_i(ν_k)
        matrix_A_b[c, i] = Σ_k W_ck b_k SED_i(ν_k)
    """
    diag_a, diag_b = [], []
    for block in A.block_leaves:
        op_a, op_jb = block.operand_leaves          # A_a  et  J @ A_b
        diag_a.append(op_a._diagonal)
        diag_b.append(op_jb.operands[1]._diagonal)  # operands = [J, A_b]
    return jnp.stack(diag_a, axis=1), jnp.stack(diag_b, axis=1)


class BandpassHWPRule(AbstractCompositionRule):
    """`Bp @ H = HWPBandpassOperator(W·a, W·b)`."""

    left_operator_class = BandpassOperator
    right_operator_class = HWPOperator

    def check(self, left: AbstractLinearOperator, right: AbstractLinearOperator) -> None:
        super().check(left, right)
        if not _is_qu(right.in_structure):
            raise NoReduction

    def apply(
        self, left: AbstractLinearOperator, right: AbstractLinearOperator
    ) -> list[AbstractLinearOperator]:
        assert isinstance(left, BandpassOperator)  # ty assert
        assert isinstance(right, HWPOperator)  # ty assert
        check_hwp_bandpass_compatibility(left, right)
        return [
            HWPBandpassOperator(
                left.weights * right.a[None, :],
                left.weights * right.b[None, :],
                in_structure=right.in_structure,
            )
        ]


class HWPBandpassBlockRowRule(AbstractCompositionRule):
    """`HB @ [A₁ | A₂ | ...] = [HB @ A₁ | HB @ A₂ | ...]`."""

    left_operator_class = HWPBandpassOperator
    right_operator_class = BlockRowOperator

    def apply(
        self, left: AbstractLinearOperator, right: AbstractLinearOperator
    ) -> list[AbstractLinearOperator]:
        assert isinstance(right, BlockRowOperator)  # ty assert
        return [BlockRowOperator(right._tree_map(lambda block: left @ block)).reduce()]


class HWPBandpassSEDRule(AbstractCompositionRule):
    """`HB @ SED = diag(W·a SED) + J @ diag(W·b SED)`."""

    left_operator_class = HWPBandpassOperator
    right_operator_class = AbstractSEDOperator

    def apply(
        self, left: AbstractLinearOperator, right: AbstractLinearOperator
    ) -> list[AbstractLinearOperator]:
        assert isinstance(left, HWPBandpassOperator)  # ty assert
        assert isinstance(right, AbstractSEDOperator)  # ty assert

        def integrated_sed(weights: Float[Array, 'c n']) -> BroadcastDiagonalOperator:
            return BroadcastDiagonalOperator(
                weights @ right._diagonal,
                axis_destination=right.axis_destination,
                insert_axes=right.insert_axes,
                in_structure=right.in_structure,
            )

        A_a = integrated_sed(left.weights_a)
        A_b = integrated_sed(left.weights_b)
        J = QUJOperator(in_structure=A_b.out_structure)
        return [A_a + J @ A_b]