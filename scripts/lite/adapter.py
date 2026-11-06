"""Adapter: our rebuilt simulator (rwm_model.RWMEnsemble, GRU) behind the interface the lite env expects.

The lite env (robotic_world_model_lite/scripts/envs/base.py) and PolicyTraining call, on the object it was
given by `set_system_dynamics`:
    .eval()
    .forward(x_state_batch, x_action_batch, model_ids)  -> (state_means, aleatoric, epistemic,
                                                            extensions, contacts, terminations)
    .reset()                 at the start of imagination
    .reset_partial(idx)      when some robots end an episode
    .ensemble_size, .architecture_config["type"]
The rebuild has none of `model_ids`, `reset_partial`, `extensions` and `architecture_config["type"]` in
that form. Nothing here changes a weight or a formula: the heads are called exactly as
SystemDynamicsEnsemble.forward (rsl_rl_rwm/rsl_rl/modules/system_dynamics.py:85-127) calls them.

Reads only. Writes nothing.
"""
import os
import sys

import torch
import torch.nn as nn

_SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)


class LiteDynamicsAdapter(nn.Module):
    """Wraps an RWMEnsemble. `net` keeps its own parameters (state_dict is the rebuild's, under `net.`)."""

    def __init__(self, net):
        super().__init__()
        self.net = net
        self.state_dim = net.state_dim
        self.action_dim = net.action_dim
        self.extension_dim = net.extension_dim
        self.contact_dim = net.contact_dim
        self.termination_dim = net.termination_dim
        self.ensemble_size = net.ensemble_size
        self.history_horizon = net.history_horizon
        self.architecture_config = net.architecture_config
        self.prediction_type = "single"
        self.device = "cpu"

    # ------------------------------------------------------------------ forward
    def forward(self, x_state_batch, x_action_batch, model_ids=None):
        """Same outputs as SystemDynamicsEnsemble.forward. model_ids: (1, B, 1) long, or None for the head mean."""
        n = self.net
        base = n.state_base(x_state_batch, x_action_batch)
        means, stds = [], []
        for head in n.state_heads:
            m, s = head(base, x_state_batch)
            means.append(m.unsqueeze(0))
            stds.append(s.unsqueeze(0))
        means, stds = torch.cat(means, 0), torch.cat(stds, 0)

        abase = n.auxiliary_base(x_state_batch, x_action_batch)
        exts, contacts, terms = [], [], []
        for head in n.auxiliary_heads:
            e, c, t = head(abase, x_state_batch)
            exts.append(e.unsqueeze(0) if e is not None else None)
            contacts.append(c.unsqueeze(0) if c is not None else None)
            terms.append(t.unsqueeze(0) if t is not None else None)
        exts = torch.cat(exts, 0) if self.extension_dim > 0 else None
        contacts = torch.cat(contacts, 0) if self.contact_dim > 0 else None
        terms = torch.cat(terms, 0) if self.termination_dim > 0 else None

        if model_ids is None:
            out_m = means.mean(0)
            out_e = exts.mean(0) if exts is not None else None
            out_c = contacts.mean(0) if contacts is not None else None
            out_t = terms.mean(0) if terms is not None else None
        else:
            def pick(x, d):
                return torch.gather(x, 0, model_ids.repeat(1, 1, d)).squeeze(0) if x is not None else None
            out_m = pick(means, self.state_dim)
            out_e = pick(exts, self.extension_dim)
            out_c = pick(contacts, self.contact_dim)
            out_t = pick(terms, self.termination_dim)

        aleatoric = stds.mean(0).sum(1)
        epistemic = (means.std(0).sum(1) if self.ensemble_size > 1
                     else torch.zeros(out_m.shape[0], device=out_m.device))
        return out_m, aleatoric, epistemic, out_e, out_c, out_t

    # ------------------------------------------------------------------ resets
    def reset(self):
        self.net.reset()

    def reset_partial(self, batch_indices):
        """As Memory.reset_partial in rsl_rl_rwm architectures/rnn.py:49: zero the hidden state of the
        listed robots in both bases; nothing if no hidden state exists yet. The heads hold no state."""
        for base in (self.net.state_base, self.net.auxiliary_base):
            h = base.memory.hidden_states
            if h is not None:
                h[:, batch_indices] = 0.0


def load_rebuild(path, ensemble):
    """A trained rebuilt model (common.load) wrapped for the lite env. `path` is a weights file."""
    import common
    return LiteDynamicsAdapter(common.load(path, ensemble))
