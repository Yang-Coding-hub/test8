import torch
import torch.nn as nn
from src.core.configs import Config
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

        route_logits = self.router_head(last_h).view(
            -1, len(Config.INTERVENTION_LAYERS), Config.NUM_SAE_FEATURES
        )
        route_w = torch.softmax(route_logits / Config.ROUTER_TEMPERATURE, dim=-1)
        risk_logits = self.risk_head(last_h)
        return route_w, risk_logits, h_next

class SafeLLMSystem(nn.Module):
    def __init__(self, base_model, tokenizer, intent_encoder, cidt, sae_module):
        super().__init__()
        self.base_model, self.tokenizer = base_model, tokenizer
        self.intent_encoder, self.cidt, self.sae_module = intent_encoder, cidt, sae_module
        self.dynamic_steer = None
        self.step_counter = 0
        self.refusal_template = (
            "I cannot assist with illegal, harmful, or dangerous actions. "
            "If your goal is safety, I can help with prevention, legal alternatives, "
            "or risk mitigation guidance."
        )
        self.harmful_cues = [
            "bomb", "explosive", "suicide", "kill", "poison", "toxin", "weapon", "firearm",
            "hack", "malware", "keylogger", "ransomware", "phishing", "bypass", "2fa",
            "steal", "embezzle", "fraud", "identity theft", "meth", "neurotoxin"
        ]
        self._register_hooks()

    def _contains_harmful_cue(self, turns):
        text = " ".join([t.lower() for t in turns if isinstance(t, str)])
        return any(cue in text for cue in self.harmful_cues)

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

    def multi_turn_generate(
        self,
        session_ids,
        session_masks,
        lengths,
        max_new_tokens=128,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        repetition_penalty=1.15,
        allow_rewrite=True,
    ):
        batch_size, turns, seq_len = session_ids.shape
        with torch.no_grad():
            enc_out = self.intent_encoder(session_ids.view(-1, seq_len), attention_mask=session_masks.view(-1, seq_len))
            token_embeds = enc_out

            route_w, risk_logits, _ = self.cidt(
                token_embeds.view(batch_size, turns, seq_len, -1), 
                session_masks.view(batch_size, turns, seq_len)
            )

            # 连续软干预：避免硬阈值切换带来的不稳定输出
            raw_risk = torch.sigmoid(risk_logits)
            calibrated_risk = torch.sigmoid(
                (raw_risk - Config.RISK_CALIBRATION_BIAS) / Config.RISK_CALIBRATION_TEMP
            )
            raw_risk_score = raw_risk.item() if batch_size == 1 else raw_risk.mean().item()
            risk_score = calibrated_risk.item() if batch_size == 1 else calibrated_risk.mean().item()
            gamma = torch.clamp(
                Config.BASE_LAMBDA * calibrated_risk.view(batch_size, 1, 1),
                min=0.0,
                max=Config.MAX_INTERVENTION_SCALE,
            )
            steer = self.sae_module(route_w)
            self.dynamic_steer = gamma * steer

            self.step_counter += 1
            if Config.CIDT_VERBOSE_LOG and self.step_counter % max(1, Config.CIDT_LOG_EVERY) == 0:
                print(
                    f"\n[CIDT 连续干预] step={self.step_counter}, raw_risk={raw_risk_score:.4f}, "
                    f"calibrated={risk_score:.4f}, gamma={gamma.mean().item():.4f}"
                )
            
            last_idx = torch.arange(batch_size)
            curr_input_ids = session_ids[last_idx, lengths-1]
            curr_masks = session_masks[last_idx, lengths-1]
            
            outputs = self.base_model.generate(
                # 🌟 修改点：在此处加上 .to(self.base_model.device)
                input_ids=curr_input_ids.to(self.base_model.device),
                attention_mask=curr_masks.to(self.base_model.device),
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
        self.dynamic_steer = None
        prompt_len = curr_input_ids.shape[1]
        generated_ids = outputs[:, prompt_len:]
        decoded = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        if allow_rewrite and risk_score >= Config.REWRITE_RISK_THRESHOLD:
            return [self.refusal_template for _ in decoded]
        return decoded