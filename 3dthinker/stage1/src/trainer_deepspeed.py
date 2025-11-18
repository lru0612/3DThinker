from trl import SFTTrainer, SFTConfig
import torch
import wandb
import torch.distributed as dist

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
        # print(predict_embeddings.shape)
        # for name, param in model.projector_model.named_parameters():
        #     print(param)  
        image_out_mask = inputs["image_out_mask"]
        shift_image_mask = image_out_mask[:, -(predict_embeddings.shape[1] - 1) :].to(predict_embeddings.device)
        shift_predict_embeddings = predict_embeddings[..., :-1, :][shift_image_mask.to(predict_embeddings.device) != 0].contiguous()

        sim_loss = model.projector_model(idx, shift_predict_embeddings)

        loss = 0.1 * ce_loss + sim_loss
        
        # 只在主进程记录wandb
        if dist.get_rank() == 0:
            wandb.log({
                "train/ce_loss": ce_loss.item(),
                "train/sim_loss": sim_loss.item(),
                "train/total_loss": loss.item(),
                "train/step": self.state.global_step,
                "train/epoch": self.state.epoch,
            })
        
        if dist.get_rank() == 0:
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
        
        # 只在主进程记录wandb
        if dist.get_rank() == 0:
            wandb.log({
                "train/ce_loss": ce_loss.item(),
                "train/total_loss": loss.item(),
                "train/step": self.state.global_step,
                "train/epoch": self.state.epoch,
            })
            print(f"Step {self.state.global_step}: CE Loss: {ce_loss.item():.4f}")
        
        return (loss, outputs) if return_outputs else loss
