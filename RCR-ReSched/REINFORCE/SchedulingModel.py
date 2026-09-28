import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from RCR import ModelInput, get_rcr_config

# Pre-processing data for model input
_DEFAULT_INPUT_DEVICE = None


def set_model_input_device(device=None):
    """Set the default tensor device without changing evaluator call sites."""
    global _DEFAULT_INPUT_DEVICE
    _DEFAULT_INPUT_DEVICE = None if device is None else torch.device(device)


def _cached_scaler(env, device, dtype=torch.float32):
    token = env.problem.duration
    key = (id(token), str(device), dtype)
    cache = getattr(env, '_resched_scaler_cache', None)
    if cache is None or getattr(env, '_resched_scaler_token', None) is not token:
        cache = {}
        env._resched_scaler_cache = cache
        env._resched_scaler_token = token
    if key not in cache:
        scaler_np = env.problem.duration.reshape(env.batch_size, -1).max(-1)
        cache[key] = torch.as_tensor(scaler_np, dtype=dtype, device=device)
    return cache[key]


def build_model_input(state, env, device=None):
    """Build the unchanged eight model inputs, optionally directly on GPU.
    """
    device = _DEFAULT_INPUT_DEVICE if device is None else torch.device(device)
    if device is None:
        rev_position = state.position_rev
        m_at = state.m_AT
        o_at = state.o_AT
        pending_mask = state.action_mask.sum(-1) > 0
        no_connect_machine = state.o2m_mask.sum(1) == 0
        o_at_mask = np.where(pending_mask, o_at, np.inf)
        masked_m_at = np.where(no_connect_machine, np.inf, m_at)
        horizon = np.concatenate((masked_m_at, o_at_mask), axis=-1).min(-1)
        o_at = o_at - horizon[:, None]
        m_at = m_at - horizon[:, None]
        scaler = env.problem.duration.reshape(env.batch_size, -1).max(-1, keepdims=True)
        m_at = m_at / scaler
        o_at = o_at / scaler
        duration = state.duration / np.expand_dims(scaler, axis=-1)
        min_duration = np.where(state.o2m_mask, state.duration, np.inf).min(-1) / scaler
        tensors = (
            torch.as_tensor(min_duration, dtype=torch.float32),
            torch.as_tensor(duration, dtype=torch.float32),
            torch.as_tensor(o_at, dtype=torch.float32),
            torch.as_tensor(m_at, dtype=torch.float32),
            torch.as_tensor(state.o2o_mask, dtype=torch.bool),
            torch.as_tensor(state.o2m_mask, dtype=torch.bool),
            torch.as_tensor(state.action_mask, dtype=torch.bool),
            torch.as_tensor(rev_position, dtype=torch.int64),
        )
        return ModelInput(tensors, env), scaler.squeeze()

    dtype = torch.float32
    m_at = torch.as_tensor(state.m_AT, dtype=dtype, device=device)
    o_at = torch.as_tensor(state.o_AT, dtype=dtype, device=device)
    # Keep the already-transferred authoritative times for RCR. Normalization
    # below creates new tensors, so these references remain unmodified.
    raw_m_at, raw_o_at = m_at, o_at
    dependency_m = torch.as_tensor(state.o2m_mask, dtype=torch.bool, device=device)
    action_mask = torch.as_tensor(state.action_mask, dtype=torch.bool, device=device)
    pending_mask = action_mask.any(-1)
    no_connect_machine = ~dependency_m.any(1)
    inf = torch.tensor(float('inf'), dtype=dtype, device=device)
    o_at_mask = torch.where(pending_mask, o_at, inf)
    masked_m_at = torch.where(no_connect_machine, m_at.new_full((), float('inf')), m_at)
    horizon = torch.cat((masked_m_at, o_at_mask), dim=-1).amin(-1)
    o_at = o_at - horizon[:, None]
    m_at = m_at - horizon[:, None]

    scaler_vec = _cached_scaler(env, device, dtype)
    scaler = scaler_vec[:, None]
    raw_duration = torch.as_tensor(state.duration, dtype=dtype, device=device)
    duration = raw_duration / scaler[:, :, None]
    min_duration = raw_duration.masked_fill(~dependency_m, float('inf')).amin(-1) / scaler

    tensors = (
        min_duration,
        duration,
        o_at / scaler,
        m_at / scaler,
        torch.as_tensor(state.o2o_mask, dtype=torch.bool, device=device),
        dependency_m,
        action_mask,
        torch.as_tensor(state.position_rev, dtype=torch.int64, device=device),
    )
    return ModelInput(tensors, env, rcr_dynamic=(raw_m_at, raw_o_at, action_mask)), scaler_vec


def edge_dtype(module):
    return module.weight.dtype


class Model(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.rcr_config = get_rcr_config(model_params.get('rcr', None))
        self.oat_embedding = nn.Linear(in_features=1,
                                       out_features=model_params['embedding_dim'],
                                       bias=True)
        self.mat_embedding = nn.Linear(in_features=1,
                                       out_features=model_params['embedding_dim'],
                                       bias=True)
        self.duration_embedding = nn.Linear(in_features=1,
                                            out_features=model_params['embedding_dim'],
                                            bias=True)
        self.layers = nn.ModuleList([CrossAttentionBlock(**model_params) for _ in range(model_params['block_num'])])

        actor_input_dim = model_params['embedding_dim'] * 3
        if self.rcr_config.use_actor:
            actor_input_dim += 4
        self.actor = Actor(num_layers=3, input_dim=actor_input_dim, hidden_dim=64, output_dim=1)

        self.decode_type = 'sampling'

    def forward(self, state, action=None, rcr_features_override=None):
        device = self.oat_embedding.weight.device
        if hasattr(state, 'device_tensors'):
            tensors = state.device_tensors(device)
        else:
            tensors = tuple(t if t.device == device else t.to(device, non_blocking=True)
                            for t in state)

        (min_duration, duration, o_at, m_at, dependency_o, dependency_m, action_mask, rev_pos) = tensors

        # Normal rollout materializes the RCR feature snapshot from the live
        # environment. Low-VRAM replay supplies that exact saved snapshot here,
        # so RCR is not recomputed after the environment has advanced.
        rcr_features = rcr_features_override
        if self.rcr_config.use_actor and rcr_features is None:
            if not hasattr(state, 'rcr_features'):
                raise RuntimeError(
                    'RCR-enabled RESCHED replay must provide rcr_features_override.')
            rcr_features = state.rcr_features(
                self.rcr_config, device=device, dtype=edge_dtype(self.duration_embedding))

        # shape: (batch, jo, embedding)
        pending_mask = action_mask.any(dim=-1)
        operation_embedding = self.duration_embedding(min_duration.unsqueeze(-1))
        operation_embedding[pending_mask] += self.oat_embedding(o_at.unsqueeze(-1))[pending_mask]

        # shape: (batch, machine, embedding)
        machine_embedding = self.mat_embedding(m_at.unsqueeze(-1))

        # shape: (batch, jo, machine, embedding)
        edge_embedding = self.duration_embedding(duration.unsqueeze(-1))

        # message passing
        for layer in self.layers:
            operation_embedding, machine_embedding = layer(operation_embedding, machine_embedding,
                                                           dependency_o, dependency_m, edge_embedding, rev_pos)

        # decision-making
        prob = self.decision_making(operation_embedding, machine_embedding, edge_embedding,
                                    action_mask, rcr_features)

        flat_prob = prob.flatten(1)
        if self.decode_type == 'sampling':
            samples = torch.multinomial(input=flat_prob, num_samples=1)
            flat_id = samples.squeeze(-1)
            seq_id = flat_id // prob.shape[-1]
            machine_id = flat_id % prob.shape[-1]
            selected_prob = flat_prob.gather(1, samples).squeeze(-1)
        elif self.decode_type == 'greedy':
            samples = torch.argmax(flat_prob, dim=-1, keepdim=True)
            flat_id = samples.squeeze(-1)
            seq_id = flat_id // prob.shape[-1]
            machine_id = flat_id % prob.shape[-1]
            selected_prob = flat_prob.gather(1, samples).squeeze(-1)
        elif self.decode_type == 'teacher_forcing':
            seq_id, machine_id = action
            selected_prob = prob[torch.arange(prob.shape[0], device=prob.device), seq_id, machine_id]
        else:
            raise ValueError('Unknown decode type')

        return [seq_id, machine_id], selected_prob

    def set_decode_type(self, decode_type):
        self.decode_type = decode_type

    def decision_making(self, operation_embedding, machine_embedding, edge_embedding,
                        action_mask, rcr_features=None):
        extra = rcr_features
        if extra is not None:
            if not isinstance(extra, torch.Tensor):
                extra = torch.as_tensor(extra, dtype=edge_embedding.dtype,
                                        device=edge_embedding.device)
            else:
                extra = extra.to(device=edge_embedding.device,
                                 dtype=edge_embedding.dtype, non_blocking=True)
            if tuple(extra.shape[:-1]) != tuple(action_mask.shape) or extra.shape[-1] != 4:
                raise ValueError('RCR Actor features must have shape action_mask + (4,).')

        # Normal RESCHED path: score only legal pairs. This removes the two
        # full [B,O,M,E] repeat() tensors and never materializes illegal pairs.
        if action_mask.dim() == 3:
            b, o, m = torch.nonzero(action_mask, as_tuple=True)
            parts = [operation_embedding[b, o],
                     machine_embedding[b, m],
                     edge_embedding[b, o, m]]
            if self.rcr_config.use_actor:
                if extra is None:
                    raise RuntimeError('RCR Actor is enabled but no analytic features were supplied.')
                parts.append(extra[b, o, m])
            pair_embedding = torch.cat(parts, dim=-1)
            score = self.actor(pair_embedding).squeeze(-1)
            score_masked = torch.full(action_mask.shape, float('-inf'),
                                      dtype=edge_embedding.dtype,
                                      device=edge_embedding.device)
            score_masked[b, o, m] = score
            prob = F.softmax(score_masked.flatten(1), dim=-1).view(action_mask.shape)
            return prob

        # Compatibility fallback for nonstandard flattened inputs.
        action_mask = action_mask.flatten(1, 2)
        operation_embedding = operation_embedding.flatten(1, 2)
        edge_embedding = edge_embedding.flatten(1, 2)
        if extra is not None:
            extra = extra.flatten(1, 2)
        operation = operation_embedding.unsqueeze(2).expand(
            -1, -1, machine_embedding.shape[1], -1)
        machine = machine_embedding.unsqueeze(1).expand(
            -1, operation_embedding.shape[1], -1, -1)
        pair = torch.cat([operation, machine, edge_embedding], dim=-1)
        pair_embedding = pair[action_mask]
        if self.rcr_config.use_actor:
            if extra is None:
                raise RuntimeError('RCR Actor is enabled but no analytic features were supplied.')
            pair_embedding = torch.cat([pair_embedding, extra[action_mask]], dim=-1)
        score = self.actor(pair_embedding)
        score_masked = torch.full(action_mask.shape, float('-inf'),
                                  dtype=edge_embedding.dtype,
                                  device=edge_embedding.device)
        score_masked[action_mask] = score.squeeze(-1)
        return F.softmax(score_masked, dim=-1)



class CrossAttentionBlock(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.embedding_dim = model_params['embedding_dim']
        self.head_num = model_params['head_num']
        self.qkv_dim = model_params['qkv_dim']

        self.Wq = nn.ModuleList(
            [nn.Linear(self.embedding_dim, self.head_num * self.qkv_dim, bias=False) for _ in range(2)])
        self.Wk = nn.ModuleList(
            [nn.Linear(self.embedding_dim, self.head_num * self.qkv_dim, bias=False) for _ in range(2)])
        self.Wv = nn.ModuleList(
            [nn.Linear(self.embedding_dim, self.head_num * self.qkv_dim, bias=False) for _ in range(2)])
        self.multi_head_combine = nn.ModuleList(
            [nn.Linear(self.head_num * self.qkv_dim, self.embedding_dim) for _ in range(2)])

        self.addAndNormalization = nn.ModuleList([AddAndNormalizationModule(**model_params) for _ in range(2*2)])
        self.feedForward = nn.ModuleList([FeedForwardModule(**model_params) for _ in range(2)])

    def forward(self, operation, machine, mask_o2o, mask_o2m, duration, rev_pos):
        # Operation Embedding [batch, operation, embedding]
        # Machine Embedding [batch, machine, embedding]

        # Attention 1: operation --> q, operation --> k, v
        # operations get information from other operations(kv)
        ope = self.attention_block(operation, operation, idx=1, mask=mask_o2o, position=rev_pos)

        # Attention 2: machine --> q, operation --> k, v
        # machines get information from operations(kv) and duration
        mac = self.attention_block(machine, operation, idx=0, mask=mask_o2m.transpose(1, 2),
                                   edge_weight=duration.transpose(1, 2),
                                   self_flag=True, edge_in_qk=True, edge_in_v=True)

        return ope, mac

    def attention_block(self, input_q, input_kv, idx, mask, position=None, edge_weight=None,
                        self_flag=False, edge_in_qk=False, edge_in_v=False):
        q = reshape_by_heads(self.Wq[idx](input_q), head_num=self.head_num)
        k = reshape_by_heads(self.Wk[idx](input_kv), head_num=self.head_num)
        v = reshape_by_heads(self.Wv[idx](input_kv), head_num=self.head_num)
        if edge_weight is not None:
            edge_weight = reshape_by_heads(edge_weight, head_num=self.head_num)
            out_concat = multi_head_attention_with_edge(q, k, v, mask, edge_weight, self_flag=self_flag,
                                                        edge_in_qk=edge_in_qk, edge_in_v=edge_in_v)
        else:
            out_concat = multi_head_attention(q, k, v, mask, position)
        multi_head_out = self.multi_head_combine[idx](out_concat)

        invalid_mask = (mask.sum(dim=-1, keepdim=True) == 0)
        multi_head_out = torch.where(invalid_mask.expand_as(multi_head_out), input_q, multi_head_out)
        out1 = self.addAndNormalization[idx*2](input_q, multi_head_out, mask)
        out2 = self.feedForward[idx](out1)
        output = self.addAndNormalization[idx*2+1](out1, out2, mask)
        output = torch.where(invalid_mask.expand_as(output), input_q, output)

        return output


class AddAndNormalizationModule(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.norm = StdNormLayer(model_params['embedding_dim'], affine=True)

    def forward(self, input1, input2, mask):
        # input.shape: (batch, *problem, embedding)
        added = input1 + input2

        orig_shape = added.shape
        batch = orig_shape[0]
        embedding = orig_shape[-1]

        # shape: (batch, problem, embedding)
        added_flat = added.reshape(batch, -1, embedding)

        # shape: (batch, problem, embedding)
        normalized = self.norm(added_flat, mask)

        # shape: (batch, *problem, embedding)
        output = normalized.reshape(orig_shape)

        return output


class StdNormLayer(nn.Module):
    def __init__(self, dim, affine=False):
        super().__init__()
        self.affine = affine
        if affine:
            self.alpha = nn.Parameter(torch.ones(dim))
            self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x, mask):
        none_connected_mask = mask.sum(dim=-1, keepdim=True) == 0
        valid_x = torch.where(none_connected_mask, torch.zeros_like(x), x)

        valid_count = (~none_connected_mask).sum(dim=(1, 2), keepdim=True)
        mean = valid_x.sum(dim=1, keepdim=True) / valid_count
        var = torch.where(none_connected_mask,
                          torch.zeros_like(x), (x - mean) ** 2).sum(dim=1, keepdim=True) / valid_count
        normalized = (x - mean) / torch.sqrt(torch.clamp(var, min=1e-8))

        output = torch.where(none_connected_mask, x, normalized)

        if self.affine:
            output = self.alpha[None, None, :] * output + self.beta[None, None, :]
        return output


class FeedForwardModule(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.W1 = nn.Linear(model_params['embedding_dim'], model_params['ff_hidden_dim'])
        self.W2 = nn.Linear(model_params['ff_hidden_dim'], model_params['embedding_dim'])

    def forward(self, input1):
        # input.shape: (batch, problem, embedding)

        return self.W2(F.relu(self.W1(input1)))


def reshape_by_heads(input_tensor, head_num):
    if input_tensor.dim() < 3:
        raise ValueError("Input tensor must have at least 3 dimensions.")

    if input_tensor.size(-1) == 1:
        # If the last dimension is 1, we can treat it as a single head.
        reshaped = input_tensor.unsqueeze(1)
        return reshaped
    elif input_tensor.size(-1) % head_num != 0:
        raise ValueError("The size of the last dimension must be divisible by head_num.")

    key_dim = input_tensor.size(-1) // head_num
    new_shape = input_tensor.shape[:-1] + (head_num, key_dim)
    reshaped = input_tensor.reshape(*new_shape)

    # shape: (batch, ..., embedding) -> (batch, ..., head_num, key_dim) -> (batch, head_num, ..., key_dim)
    tensor_l = len(new_shape)
    perm_order = [0, tensor_l - 2] + list(range(1, tensor_l - 2)) + [tensor_l - 1]
    transposed = reshaped.permute(*perm_order).contiguous()

    return transposed


def multi_head_attention(q, k, v, mask, position, rope=True):
    # q shape: (batch, head_num, length_q, key_dim)
    # k,v shape: (batch, head_num, length_kv, key_dim)
    bs, h, length_q, key_dim = q.shape
    _, _, length_kv, _ = k.shape

    if rope:
        q, k = apply_rope_mapping(q, k, position)

    score = torch.matmul(q, k.transpose(-2, -1))

    score_scaled = score / math.sqrt(key_dim)
    score_scaled = score_scaled.masked_fill(~mask.unsqueeze(1), float('-inf'))

    # shape: (batch, head_num, *length_q, *length_kv)
    weights = nn.Softmax(dim=-1)(score_scaled)
    out = torch.matmul(weights, v)

    # shape: (batch, *length_q, head_num, key_dim)
    tensor_l = out.dim()
    perm_order = [0] + list(range(2, tensor_l - 1)) + [1, tensor_l - 1]
    out_transposed = out.permute(*perm_order)

    # shape: (batch, *length_q, head_num*key_dim)
    out_concat = out_transposed.flatten(-2)

    return out_concat


def apply_rope_mapping(matrix_q, matrix_k, position):
    """
    matrix_q, matrix_k: (bs, head, jo, dk)
    """
    # bs, head, jo, dk = matrix_q.shape
    _, head, _, dk = matrix_q.shape

    pe = sinusoidal_position_embedding(head, dk, position)

    cos_pos = pe[..., dk//2:] .repeat_interleave(2, dim=-1)
    sin_pos = pe[..., :dk//2].repeat_interleave(2, dim=-1)

    q_even = matrix_q[..., ::2]
    q_odd  = matrix_q[..., 1::2]
    q2 = torch.stack([-q_odd, q_even], dim=-1).reshape_as(matrix_q)

    k_even = matrix_k[..., ::2]
    k_odd  = matrix_k[..., 1::2]
    k2 = torch.stack([-k_odd, k_even], dim=-1).reshape_as(matrix_k)

    q_rot = matrix_q * cos_pos + q2 * sin_pos
    k_rot = matrix_k * cos_pos + k2 * sin_pos

    return q_rot, k_rot


_ROPE_THETA_CACHE = {}


def sinusoidal_position_embedding(head, dk, position):
    device = position.device
    bs, jo = position.shape

    key = (str(device), dk)
    theta = _ROPE_THETA_CACHE.get(key)
    if theta is None:
        i = torch.arange(dk // 2, device=device, dtype=torch.float32)
        theta = 1.0 / (10000 ** (2 * i / dk))
        _ROPE_THETA_CACHE[key] = theta
    angles = position.float().unsqueeze(-1) * theta    # (bs, jo, dk/2)

    pe = torch.cat([angles.sin(), angles.cos()], dim=-1)  # (bs, jo, dk)
    pe = pe.unsqueeze(1).expand(bs, head, jo, dk)        # (bs, head, jo, dk)

    return pe


def multi_head_attention_with_edge(q, k, v, mask, edge_weight,
                                   self_flag=False, edge_in_qk=False, edge_in_v=False):
    """
    Memory-efficient algebraically equivalent edge-aware attention.

    Original implementation explicitly materialized:
      q_with_edge : [B,H,Q,K,D]
      k_with_edge : [B,H,Q,K,D]
      v_with_edge : [B,H,Q,K,D]

    For edge-aware scores,
        <q+e, k+e> = <q,k> + <q,e> + <k,e> + <e,e>.
    For edge-aware values,
        sum w(v+e) = w@v + sum(w*e).

    The equations and gradients are unchanged; the large 5-D expanded tensors
    are no longer allocated.
    """
    key_dim = q.size(-1)

    if edge_in_qk:
        # All outputs below are [B,H,Q,K]; einsum avoids constructing
        # expanded [B,H,Q,K,D] q/k tensors.
        score = torch.matmul(q, k.transpose(-2, -1))
        score = score + torch.einsum('bhqd,bhqkd->bhqk', q, edge_weight)
        score = score + torch.einsum('bhkd,bhqkd->bhqk', k, edge_weight)
        score = score + torch.einsum('bhqkd,bhqkd->bhqk', edge_weight, edge_weight)
    else:
        score = torch.matmul(q, k.transpose(-2, -1))

    self_k, self_v = q, q

    if self_flag:
        self_score = torch.sum(q * self_k, dim=-1, keepdim=True)
        score_add = torch.cat((score, self_score), dim=-1)
        score_scaled = score_add / math.sqrt(key_dim)

        mask_add = torch.cat(
            (mask,
             torch.ones(mask.shape[0], mask.shape[1], 1,
                        dtype=torch.bool, device=mask.device)),
            dim=-1)
        score_scaled = score_scaled.masked_fill(
            ~mask_add.unsqueeze(1), float('-inf'))

        weights = F.softmax(score_scaled, dim=-1)
        pair_weights = weights[..., :-1]

        out = torch.matmul(pair_weights, v)
        if edge_in_v:
            # Equivalent to sum(weights * (v + edge), K), without v_with_edge.
            out = out + torch.einsum(
                'bhqk,bhqkd->bhqd', pair_weights, edge_weight)
        out = out + weights[..., -1:] * self_v

    else:
        score_scaled = score / math.sqrt(key_dim)
        score_scaled = score_scaled.masked_fill(
            ~mask.unsqueeze(1), float('-inf'))

        invalid_seq = (mask.sum(dim=-1, keepdim=True) == 0)
        score_scaled = score_scaled.masked_fill(
            invalid_seq.unsqueeze(1), float('-1e9'))

        weights = F.softmax(score_scaled, dim=-1)
        out = torch.matmul(weights, v)
        if edge_in_v:
            out = out + torch.einsum(
                'bhqk,bhqkd->bhqd', weights, edge_weight)

    tensor_l = out.dim()
    perm_order = [0] + list(range(2, tensor_l - 1)) + [1, tensor_l - 1]
    out_transposed = out.permute(*perm_order)
    return out_transposed.flatten(-2)


class Actor(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        """
            the implementation of Actor network (refer to L2D)
        :param num_layers: number of layers in the neural networks (EXCLUDING the input layer).
                            If num_layers=1, this reduces to linear model.
        :param input_dim: dimensionality of input features
        :param hidden_dim: dimensionality of hidden units at ALL layers
        :param output_dim:  number of classes for prediction
        """
        super(Actor, self).__init__()

        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers

        self.activative = torch.tanh

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = self.activative((self.linears[layer](h)))
            return self.linears[self.num_layers - 1](h)


if __name__ == "__main__":
    model_params = {
        'action_space': 2,  # 0: processing-time ！= 0; 1: processing-time == 0
        'embedding_dim': 128,
        'block_num': 2,
        'head_num': 8,
        'qkv_dim': 16,
        'ff_hidden_dim': 512,
    }
    model = Model(**model_params)
    a = 1