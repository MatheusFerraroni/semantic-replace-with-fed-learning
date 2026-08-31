"""Reprodução e persistência mínima do accountant RDP do Opacus."""

from __future__ import annotations

import importlib.metadata
import math
from typing import Any

from .dp_contracts import (
    EXPECTED_SIGMA_BY_EPSILON,
    DPAccountantState,
    DPAccountingSpec,
    PrivateTrainingError,
    accountant_state_sha256,
    validate_dp_accounting_spec,
)


def _load_accountant_class():
    try:
        from opacus.accountants import RDPAccountant
    except ImportError as error:
        raise PrivateTrainingError(
            "Opacus ausente; instale o projeto com .[model,dp]"
        ) from error
    try:
        version = importlib.metadata.version("opacus")
    except importlib.metadata.PackageNotFoundError as error:
        raise PrivateTrainingError("versão do Opacus não pôde ser validada") from error
    if version != "1.6.0":
        raise PrivateTrainingError("versão do Opacus diverge da receita privada")
    return RDPAccountant


def new_rdp_accountant(state: DPAccountantState | None = None):
    accountant = _load_accountant_class()()
    if state is not None:
        if not isinstance(state, DPAccountantState):
            raise PrivateTrainingError("estado RDP possui tipo inválido")
        try:
            accountant.load_state_dict(
                {"mechanism": "rdp", "history": [tuple(item) for item in state.history]}
            )
        except Exception as error:
            raise PrivateTrainingError("estado RDP persistido é incompatível") from error
    return accountant


def validate_accounting_profile(spec: DPAccountingSpec) -> None:
    """Recalcula os dois orçamentos antes de qualquer execução científica."""

    resolved = validate_dp_accounting_spec(spec)
    for target in resolved.target_epsilons:
        accountant = new_rdp_accountant()
        for _ in range(resolved.total_private_steps):
            accountant.step(
                noise_multiplier=resolved.sigma_for(target),
                sample_rate=resolved.sample_rate,
            )
        try:
            epsilon, order = accountant.get_privacy_spent(delta=resolved.delta)
        except Exception as error:
            raise PrivateTrainingError("accountant RDP não pôde ser reproduzido") from error
        expected_epsilon, expected_order = resolved.realized_for(target)
        if (
            not math.isfinite(float(epsilon))
            or float(epsilon) > target
            or abs(float(epsilon) - expected_epsilon) > 1e-10
            or abs(float(order) - expected_order) > 1e-10
        ):
            raise PrivateTrainingError("accountant RDP diverge do perfil fixado")


def capture_accountant_state(
    accountant: Any,
    *,
    client_id: str,
    target_epsilon: float,
    delta: float,
) -> DPAccountantState:
    try:
        state = accountant.state_dict()
        history = tuple(
            (float(sigma), float(rate), int(steps))
            for sigma, rate, steps in state["history"]
        )
        completed_steps = sum(item[2] for item in history)
        epsilon, order = accountant.get_privacy_spent(delta=delta)
    except Exception as error:
        raise PrivateTrainingError("estado do accountant RDP é inválido") from error
    if (
        state.get("mechanism") != "rdp"
        or completed_steps <= 0
        or not math.isfinite(float(epsilon))
        or not math.isfinite(float(order))
    ):
        raise PrivateTrainingError("estado do accountant RDP é inválido")
    digest = accountant_state_sha256(client_id, target_epsilon, history)
    return DPAccountantState(
        client_id=client_id,
        target_epsilon=target_epsilon,
        history=history,
        completed_steps=completed_steps,
        realized_epsilon=float(epsilon),
        optimal_order=float(order),
        state_sha256=digest,
    )


def validate_accountant_state(state: object, *, delta: float = 1e-5) -> DPAccountantState:
    """Recalcula ε e ordem antes de aceitar um estado persistido."""

    if not isinstance(state, DPAccountantState):
        raise PrivateTrainingError("estado RDP persistido possui tipo inválido")
    expected_sigma = dict(EXPECTED_SIGMA_BY_EPSILON).get(state.target_epsilon)
    if (
        not state.client_id.startswith("victim-")
        or expected_sigma is None
        or not state.history
        or any(
            sigma != expected_sigma or rate != 0.04 or steps <= 0
            for sigma, rate, steps in state.history
        )
        or state.completed_steps != sum(item[2] for item in state.history)
        or not 0 < state.completed_steps <= 2_000
        or state.state_sha256
        != accountant_state_sha256(
            state.client_id, state.target_epsilon, state.history
        )
    ):
        raise PrivateTrainingError("estado RDP persistido diverge da receita")
    accountant = new_rdp_accountant(state)
    try:
        epsilon, order = accountant.get_privacy_spent(delta=delta)
    except Exception as error:
        raise PrivateTrainingError("estado RDP persistido não pôde ser recalculado") from error
    if (
        abs(float(epsilon) - state.realized_epsilon) > 1e-10
        or abs(float(order) - state.optimal_order) > 1e-10
    ):
        raise PrivateTrainingError("estado RDP persistido possui contabilização divergente")
    return state


__all__ = [
    "capture_accountant_state",
    "new_rdp_accountant",
    "validate_accountant_state",
    "validate_accounting_profile",
]
