from trl import SFTTrainer, SFTConfig
import torch
import wandb

class CustomTrainerStage1(SFTTrainer):        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute training loss and additionally compute token accuracies
        """
        idx = inputs['idx']
        del inputs['idx']
        
        (ce_loss, outputs) = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )

        predict_embeddings = outputs.hidden_states
        
        # all: 620
        image_out_mask = inputs["image_out_mask"]
        shift_image_mask = image_out_mask[:, -(predict_embeddings.shape[1] - 1) :].to(predict_embeddings.device)
        shift_predict_embeddings = predict_embeddings[..., :-1, :][shift_image_mask.to(predict_embeddings.device) != 0].contiguous()

        # single-gpu
        # sim_loss = model.projector_model(idx, shift_predict_embeddings)
        # multi-gpu
        sim_loss = model.module.projector_model(idx, shift_predict_embeddings)

        # k_proj_weight = model.get_parameter('model.layers.17.self_attn.k_proj.weight')
        # gt_embeddings = input_embeddings[..., 1:, :][shift_image_mask.to(input_embeddings.device) != 0].contiguous()
        
        # sim_loss = torch.nn.functional.cosine_similarity(gt_embeddings, shift_predict_embeddings).mean()
        # sim_loss = 1 - sim_loss

        loss = 0.1 * ce_loss + sim_loss
        
        if torch.distributed.get_rank() == 0:
            wandb.log({
                "train/ce_loss": ce_loss.item(),
                "train/sim_loss": sim_loss.item(),
                "train/total_loss": loss.item(),
                "train/step": self.state.global_step,
                "train/epoch": self.state.epoch,
            })
        
        print(f"Step {self.state.global_step}: CE Loss: {ce_loss.item():.4f}, Sim Loss: {sim_loss.item():.4f}, Total Loss: {loss.item():.4f}")
        return (loss, outputs) if return_outputs else loss

class CustomTrainerStage2(SFTTrainer):
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute training loss and additionally compute token accuracies
        """
        (ce_loss, outputs) = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )

        loss = ce_loss
        if torch.distributed.get_rank() == 0:
            wandb.log({
                "train/ce_loss": ce_loss.item(),
                "train/total_loss": loss.item(),
                "train/step": self.state.global_step,
                "train/epoch": self.state.epoch,
            })
        print(f"Step {self.state.global_step}: CE Loss: {ce_loss.item():.4f}")
        return (loss, outputs) if return_outputs else loss