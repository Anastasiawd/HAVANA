"""Lightweight client for the Havana policy RPC server.

Imports stdlib + numpy only — safe to use from the Habitat environment, which
does not have torch / cosmos installed. Talks to `run_pipeline.py --serve`.

Usage:
    from fast_system.scripts.policy_client import PolicyClient

    client = PolicyClient("/tmp/havana.sock", action_names=cfg.action_names)
    client.set_instruction("Walk forward through the doorway.")
    action = client.get_action(obs, step_id)   # obs: dict of np.ndarray
    client.close()

The address may be either:
    - a UNIX socket path (e.g. "/tmp/havana.sock")
    - a TCP "host:port" string (e.g. "127.0.0.1:5555")
"""
from __future__ import annotations

import logging
import time
from multiprocessing.connection import Client
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np

LOG = logging.getLogger("havana.client")
if not LOG.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter(
        "[%(asctime)s][%(name)s][%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    LOG.addHandler(h)
    LOG.setLevel(logging.INFO)
    LOG.propagate = False


Address = Union[str, Tuple[str, int]]


def parse_address(addr: str) -> Address:
    """`/tmp/foo.sock` -> str (UNIX);  `host:port` -> (host, int(port))."""
    if isinstance(addr, str) and ":" in addr and not addr.startswith("/"):
        host, port = addr.rsplit(":", 1)
        return (host, int(port))
    return addr


def _sanitize_obs(obs: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Strip habitat-specific objects to keep the payload pickle-clean and small."""
    out: Dict[str, np.ndarray] = {}
    for k, v in obs.items():
        if isinstance(v, np.ndarray):
            out[k] = np.ascontiguousarray(v)
    return out


class PolicyClient:
    """RPC client for HavanaPolicyServer."""

    def __init__(
        self,
        address: str,
        *,
        authkey: bytes = b"havana",
        action_names: Optional[Sequence[str]] = None,
        connect_timeout: float = 30.0,
    ):
        self.address = parse_address(address)
        self.authkey = authkey
        self._conn = None
        self._connect(connect_timeout)
        if action_names is not None:
            self.attach_action_space(action_names)

    def _connect(self, timeout: float) -> None:
        deadline = time.time() + max(0.0, float(timeout))
        last_err: Optional[Exception] = None
        while time.time() < deadline:
            try:
                self._conn = Client(self.address, authkey=self.authkey)
                LOG.info("[PolicyClient] connected to %s", self.address)
                return
            except (ConnectionRefusedError, FileNotFoundError) as exc:
                last_err = exc
                time.sleep(0.5)
        raise ConnectionError(
            f"could not connect to policy server at {self.address} within {timeout:.1f}s: {last_err}"
        )

    def _call(self, op: str, *args) -> Any:
        if self._conn is None:
            raise ConnectionError("policy client not connected")
        self._conn.send((op,) + args)
        resp = self._conn.recv()
        if not isinstance(resp, dict) or not resp.get("ok"):
            err = resp.get("error") if isinstance(resp, dict) else repr(resp)
            raise RuntimeError(f"policy server error on op={op!r}: {err}")
        return resp.get("result")

    # -------------------------------------------------------------- public #
    def attach_action_space(self, action_names: Sequence[str]) -> None:
        self._call("attach_action_space", list(action_names))

    def set_instruction(self, text: str) -> None:
        self._call("set_instruction", str(text))

    def get_action(self, obs: Dict[str, Any],
                   step_id: Optional[int] = None) -> Optional[str]:
        return self._call("get_action", _sanitize_obs(obs), step_id)

    def ping(self) -> str:
        return self._call("ping")

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:  # noqa: BLE001
                pass
            self._conn = None

    def __enter__(self) -> "PolicyClient":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
