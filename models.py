import torch
import torch.nn as nn
from configs import Config
from safetensors.torch import load_file

class AlphaSteerProjector:
    def fit_null_space(self, H_b: torch.Tensor) -> torch.Tensor:
        H_b_fp32 = H_b.to(torch.float32)
        covariance = torch.matmul(H_b_fp32, H_b_fp32.T)
        U, S, _ = torch.linalg.svd(covariance, full_matrices=True)
        null_indices = S < Config.SVD_THRESHOLD
        U_hat = U[:, null_indices]
        P = torch.matmul(U_hat, U_hat.T)
        return P.to(Config.DEVICE).to(Config.DTYPE)

class NullSpaceProjectedSAE(nn.Module):
    def __init__(self, P_matrix):
        super().__init__()
        print(f"Loading SAE weights from {Config.SAE_PATH}...")
        sae_weights = load_file(Config.SAE_PATH)
        full_dict = sae_weights["decoder.weight"].T.to(Config.DEVICE).to(Config.DTYPE)
        refusal_dict = full_dict[Config.VERIFIED_REFUSAL_INDICES, :]
        with torch.no_grad():
            projected = torch.matmul(refusal_dict, P_matrix.T)
        self.safe_dict = nn.Parameter(projected, requires_grad=False)
        
    def forward(self, routing_weights):
        return torch.matmul(routing_weights, self.safe_dict)

class CIDTRouter(nn.Module):
    def __init__(self):
        super().__init__()
        self.task_attn = nn.Sequential(
            nn.Linear(Config.ENCODER_DIM, 128, dtype=Config.DTYPE),
            nn.Tanh(),
            nn.Linear(128, 1, dtype=Config.DTYPE)
        )
        self.gru = nn.GRU(Config.ENCODER_DIM, Config.CIDT_HIDDEN_SIZE, 
                          num_layers=Config.CIDT_LAYERS, batch_first=True, dtype=Config.DTYPE)
        self.router_head = nn.Linear(Config.CIDT_HIDDEN_SIZE, 
                                     len(Config.INTERVENTION_LAYERS) * Config.NUM_SAE_FEATURES, dtype=Config.DTYPE)
        self.risk_head = nn.Linear(Config.CIDT_HIDDEN_SIZE, 1, dtype=Config.DTYPE)

    def forward(self, x, mask, h_prev=None):
        logits = self.task_attn(x).masked_fill(mask.unsqueeze(-1) == 0, float('-inf'))
        weights = torch.softmax(logits, dim=2)
        turn_embeds = torch.sum(x * weights, dim=2)
        
        out, h_next = self.gru(turn_embeds, h_prev)
        last_h = out[:, -1, :]
        
        route_w = torch.relu(self.router_head(last_h)).view(-1, len(Config.INTERVENTION_LAYERS), Config.NUM_SAE_FEATURES)
        risk = torch.sigmoid(self.risk_head(last_h))
        return route_w, risk, h_next

class SafeLLMSystem(nn.Module):
    def __init__(self, base_model, tokenizer, intent_encoder, cidt, sae_module):
        super().__init__()
        self.base_model, self.tokenizer = base_model, tokenizer
        self.intent_encoder, self.cidt, self.sae_module = intent_encoder, cidt, sae_module
        self.dynamic_steer = None
        self.step_counter = 0
        self._register_hooks()

    def _register_hooks(self):
        def get_hook(layer_idx_in_list):
            def hook_fn(module, input, output):
                if self.dynamic_steer is not None:
                    if isinstance(output, tuple):
                        h = output[0]
                    else:
                        h = output

                    if h.shape[1] == 1:
                        # 温和稳定强度 → 不乱码、自然拒绝
                        steer_vec = self.dynamic_steer[:, layer_idx_in_list, :].to(h.device)
                        modified = h.clone()
                        modified[:, -1:] += 0.5 * steer_vec.unsqueeze(1)

                        if isinstance(output, tuple):
                            return (modified,) + output[1:]
                        else:
                            return modified
                return output
            return hook_fn

        layers = self.base_model.model.layers if hasattr(self.base_model, "model") else self.base_model.layers
        for i, l_idx in enumerate(Config.INTERVENTION_LAYERS):
            layers[l_idx].register_forward_hook(get_hook(i))

    def multi_turn_generate(self, session_ids, session_masks, lengths):
        batch_size, turns, seq_len = session_ids.shape
        with torch.no_grad():
            enc_out = self.intent_encoder(session_ids.view(-1, seq_len), attention_mask=session_masks.view(-1, seq_len))
            token_embeds = enc_out

            route_w, risk, _ = self.cidt(
                token_embeds.view(batch_size, turns, seq_len, -1), 
                session_masks.view(batch_size, turns, seq_len)
            )

            # --------------------------
            # ✅ 安全阈值：只对高风险触发
            # --------------------------
            risk_score = risk.item() if batch_size == 1 else risk.mean().item()
            if risk_score > 0.5:  # 严格高风险才防御
                self.dynamic_steer = 18.0 * risk.view(batch_size, 1, 1) * self.sae_module(route_w)
            else:
                self.dynamic_steer = None  # 低风险 → 不干预（不误判）
            
            self.step_counter = 0
            
            last_idx = torch.arange(batch_size)
            curr_input_ids = session_ids[last_idx, lengths-1]
            curr_masks = session_masks[last_idx, lengths-1]
            
            outputs = self.base_model.generate(
                input_ids=curr_input_ids,
                attention_mask=curr_masks,
                max_new_tokens=128,
                do_sample=False,        # 关闭采样 → 输出稳定
                temperature=1.0,
                top_p=1.0,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
        self.dynamic_steer = None
        prompt_len = curr_input_ids.shape[1]
        generated_ids = outputs[:, prompt_len:]
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)