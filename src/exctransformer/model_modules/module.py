import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import r2_score

from exctransformer.model_modules.gscgcnn import Crystal_GraphSage, Additional_feat_Embedding
from exctransformer.model_modules.CGS_transformer_encoder import TransformerEncoder
from exctransformer.model_modules import model_utils
from exctransformer.model_modules import objectives
from exctransformer.model_modules import heads
from exctransformer.utils.configchecks import set_model_load_path


def _torch_load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


class Module(LightningModule):
    @staticmethod
    def _unwrap_single_value(value):
        while isinstance(value, (list, tuple, np.ndarray)) and len(value) == 1:
            value = value[0]
        return value

    def __init__(self, config):
        super().__init__()

        self.save_hyperparameters(config)
        self.config = config
        
            
        # Initialize custom metrics.
        model_utils.set_metrics(self)
           
        # Instantiate the Normalizer for target.
        self.normalizer = model_utils.Normalizer(
            mean=config["mean"],
            std=config["std"],
            device=config["device"]
        )

        # Instantiate the Normalizer for added features.
        self.normalizer_add_feat = model_utils.Normalizer(
            mean=config["mean_a"],
            std=config["std_a"],
            device=config["device"]
        )
        # Obtain feature lengths from dataset.
        atom_fea_len, nbr_fea_len, add_fea = model_utils.get_feature_lengths_from_dataset(config)

        # Instantiate GraphSAGE embedding module.
        self.graphsage = Crystal_GraphSage(
                        atom_fea_len=atom_fea_len,
                        nbr_fea_len=nbr_fea_len,
                        max_crystal_atoms=config["max_crystal_atoms"],
                        num_layers=config["num_layers"], 
                        hidden_dims=config["hidden_dims"],
                        num_heads=config["graph_attn_heads"],
                        out_dim=config["out_dim"],
        )
        # Perform weight initialization
        self.graphsage.apply(model_utils.init_weights) 

        # Initialize additional feature embedding module if needed (Additioanl feature tokken).
        if add_fea.numel() > 0:
            self.additional_feat_embedding = Additional_feat_Embedding(
                emb_dim=config["out_dim"],
                add_feat_num=add_fea.shape[0]
            )
            self.additional_feat_embedding.apply(model_utils.init_weights)
        else:
            self.additional_feat_embedding = None
            if config.get("graph_only_mode", False):
                print("Graph-only mode enabled. This mode deactivates additional feature embeddings.")
            print("No additional features detected. Using only atomic features.")
     
        # token type embeddings
        self.token_type_embeddings = nn.Embedding(2, config["hid_dim"])
        self.token_type_embeddings.apply(objectives.init_weights)

        # Instantiate Transformer module.
        self.transformer = TransformerEncoder(
            dim=config["hid_dim"],
            depth=config["transformer_num_layers"],  
            num_heads=config["num_heads"],
            mlp_ratio=config["mlp_ratio"],
        )

        # class token
        self.cls_embeddings = nn.Linear(1, config["hid_dim"])
        self.cls_embeddings.apply(objectives.init_weights)

        # pooler
        self.pooler = heads.Pool(config["hid_dim"])
        self.pooler.apply(objectives.init_weights)

        self.write_log = True

        #--------------------Downstream Tasks-------------------
        hid_dim = config["hid_dim"]

        if self.config["loss_names"].get("regression", 0) > 0:
            self.regression_head = heads.Regression(hid_dim, config["num_targets"])
            self.regression_head.apply(objectives.init_weights)
            # normalization
            self.mean = config["mean"]
            self.std = config["std"]

        if self.config["loss_names"].get("classification", 0) > 0:
            n_classes = config["num_classes"]
            self.classification_head = heads.Classification(hid_dim, n_classes)
            self.classification_head.apply(objectives.init_weights)

        # Load model if load_path is provided and not in test_only mode
        if set_model_load_path(config["load_path"]) != "" and not config["test_only"]:
            ckpt_file = _torch_load_checkpoint(self.config["load_path"])
            state_dict = ckpt_file["state_dict"]

            self.load_state_dict(state_dict, strict=False)
            print(f"load model : {config['load_path']}")

            if set_model_load_path(config["load_pretained_weights"])!= "":
                model_utils.load_pretrained_weights(set_model_load_path(self.config["load_pretained_weights"])
                                                    , self.transformer)

        # Load model if load_path is provided and in test_only mode
        if set_model_load_path(config["load_path"]) != "" and config["test_only"]:
            ckpt_file = _torch_load_checkpoint(set_model_load_path(config["load_path"]))
            state_dict = ckpt_file["state_dict"]

            self.load_state_dict(state_dict, strict=False)
            
            print(f"load model : {config['load_path']}")

            if set_model_load_path(config["load_pretained_weights"])!= "":
                model_utils.load_pretrained_weights(set_model_load_path(config["load_pretained_weights"]), self.transformer)
        
        if set_model_load_path(config["load_pretained_weights"])!= "":
                model_utils.load_pretrained_weights(set_model_load_path(self.config["load_pretained_weights"])
                                                    , self.transformer)
    
        model_utils.set_metrics(self)
        self.current_tasks = list()

        self.test_logits = []
        self.test_labels = []
        self.test_cifid = []
        self.write_log = True
        self.no_labels = config["no_labels"]
                
        
    def infer(self, batch):
        atom_fea = batch["atom_fea"]  # [B, F]
        nbr_fea = batch["nbr_fea"]  # [B, M, F]
        nbr_fea_idx = batch["nbr_fea_idx"]  # [B, M]
        crystal_atom_idx = batch["crystal_atom_idx"]  # [B]
        num_atoms_per_crystal = batch["num_atoms_per_crystal"]  # [B]
        cif_id = batch["cif_id"]

        if self.additional_feat_embedding is not None:
            additional_features = batch["additional_features"]  # [B, A]

        # Obtain atom embeddings and mask
        graph_embeds, graph_mask = self.graphsage(
            atom_fea, nbr_fea, nbr_fea_idx, num_atoms_per_crystal
        )  # [B, N, hid_dim], [B, N]

        # Add class token to graph embeddings
        cls_tokens = torch.zeros(len(num_atoms_per_crystal)).to(graph_embeds)  # [B]
        cls_embeds = self.cls_embeddings(cls_tokens[:, None, None])  # [B, 1, hid_dim]
        cls_mask = torch.ones(len(num_atoms_per_crystal), 1).to(graph_mask)  # [B, 1]
        
        graph_embeds = torch.cat([cls_embeds, graph_embeds], dim=1)  # [B, N+1, hid_dim]
        graph_mask = torch.cat([cls_mask, graph_mask], dim=1)  # [B, N+1]

        if self.additional_feat_embedding is not None:
            additional_features = self.normalizer_add_feat.encode(additional_features)
            additional_fea_embeds, additional_fea_mask = self.additional_feat_embedding(
                additional_features
            )  # [B, 2, hid_dim], [B, 2]

            # Add token type embeddings (type=1 for additional features)
            additional_fea_embeds = additional_fea_embeds + self.token_type_embeddings(
                torch.ones_like(additional_fea_mask, device=self.device).long()
            )

        # Add token type embeddings (type=0 for graph tokens)
        graph_embeds = graph_embeds + self.token_type_embeddings(
            torch.zeros_like(graph_mask, device=self.device).long()
        )

        # Combine all embeddings
        if self.additional_feat_embedding is not None:
            combined_embeds = torch.cat(
                [graph_embeds, additional_fea_embeds], dim=1
            )  # [B, N+1+2, hid_dim]
            combined_mask = torch.cat(
                [graph_mask, additional_fea_mask], dim=1
            )  # [B, N+1+2]
        else:
            combined_embeds = graph_embeds  # [B, N+1, hid_dim]
            combined_mask = graph_mask  # [B, N+1]

        # Process through transformer
        x = combined_embeds
        for blk in self.transformer.blocks:
            x = blk(x, mask=combined_mask)
        x = self.transformer.norm(x)

        # Split features
        if self.additional_feat_embedding is not None:
            graph_feats = x[:, :graph_embeds.shape[1]]  # [B, N+1, hid_dim]
            added_feats = x[:, graph_embeds.shape[1]:]  # [B, 2, hid_dim]
        else:
            graph_feats = x  # [B, N+1, hid_dim]
            added_feats = None

        # Pool CLS token (first token)
        cls_feats = self.pooler(x)  # [B, hid_dim]

        return {
            "cif_id": cif_id,
            "num_atoms_per_crystal": num_atoms_per_crystal,
            "crystal_atom_idx": crystal_atom_idx,
            "graph_feats": graph_feats,
            "added_feats": added_feats,
            "cls_feats": cls_feats,
            "raw_cls_feats": x[:, 0],  # Unpooled CLS token
            "graph_mask": graph_mask,
        }
    
    def forward(self, batch):
        ret = dict()

        if len(self.current_tasks) == 0:
            ret.update(self.infer(batch))
            return ret
        
        # regression
        if "regression" in self.current_tasks:
            normalizer = self.normalizer
            ret.update(objectives.eval_regression(self, batch, normalizer))

        # classification
        if "classification" in self.current_tasks:
            ret.update(objectives.eval_classification(self, batch))
        return ret
    
    def on_train_start(self):
        model_utils.set_task(self)
        self.write_log = True

    def training_step(self, batch):
        output = self(batch)
        total_loss = sum([v for k, v in output.items() if "loss" in k])
        return total_loss

    def on_train_epoch_end(self):
        model_utils.epoch_wrapup(self)

    def on_validation_start(self):
        model_utils.set_task(self)
        self.write_log = True

    def validation_step(self, batch):
        output = self(batch)

    def on_validation_epoch_end(self) -> None:
        model_utils.epoch_wrapup(self)

    def on_test_start(self):
        model_utils.set_task(self)
        self.test_data = {
            'task': None,
            'cif_id': [],
            'predictions': [],
            'targets': [],
            'scores': [],
        }

    def test_step(self, batch):
        output = self(batch)
        output = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in output.items()}

        # Handle multiple targets
        if "regression_logits" in output:
            self.test_data['task'] = 'regression'
            num_targets = output["regression_logits"].shape[1] if len(output["regression_logits"].shape) > 1 else 1
            
            self.test_data['cif_id'].extend(batch['cif_id'])
            
            # Store predictions (handles both single and multi-output)
            preds = output["regression_logits"].tolist()
            if num_targets == 1:
                self.test_data['predictions'].extend(
                    self._unwrap_single_value(pred) for pred in preds
                )
            else:
                self.test_data['predictions'].extend(preds)
            
            # Store targets
            targets = output["regression_labels"].tolist()
            if num_targets == 1:
                self.test_data['targets'].extend(
                    self._unwrap_single_value(target) for target in targets
                )
            else:
                self.test_data['targets'].extend(targets)

        elif "classification_logits" in output:
            self.test_data['task'] = 'classification'
            self.test_data['cif_id'].extend(batch['cif_id'])

            logits = output["classification_logits"]
            if not torch.is_tensor(logits):
                logits = torch.tensor(logits)
            if logits.ndim == 1:
                probs = torch.sigmoid(logits).tolist()
                preds = [int(p >= 0.5) for p in probs]
                self.test_data['scores'].extend(probs)
                self.test_data['predictions'].extend(preds)
            else:
                probs = torch.softmax(logits, dim=1).tolist()
                preds = torch.argmax(logits, dim=1).tolist()
                self.test_data['scores'].extend(probs)
                self.test_data['predictions'].extend(preds)

            labels = output.get("classification_labels")
            if labels is not None:
                if torch.is_tensor(labels):
                    labels = labels.tolist()
                self.test_data['targets'].extend(
                    self._unwrap_single_value(label) for label in labels
                )
        
        return output

    def save_test_results(self):
        if not self.test_data['cif_id']:
            return

        if self.test_data.get('task') == 'classification':
            results = {'cif_id': self.test_data['cif_id']}
            first_score = self.test_data['scores'][0] if self.test_data['scores'] else None
            if isinstance(first_score, (list, tuple, np.ndarray)):
                num_classes = len(first_score)
                for i in range(num_classes):
                    results[f'class_{i}_prob'] = [score[i] for score in self.test_data['scores']]
            else:
                results['positive_class_prob'] = self.test_data['scores']
            results['predicted_class'] = self.test_data['predictions']
            if self.test_data['targets']:
                results['target_class'] = self.test_data['targets']

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = self.config.get("log_dir") or "."
            os.makedirs(output_dir, exist_ok=True)
            csv_path = os.path.join(output_dir, f"test_results_{timestamp}.csv")
            pd.DataFrame(results).to_csv(csv_path, index=False)
            print(f"Saved classification test results to {csv_path}")
            return

        # Determine number of targets
        first_pred = self.test_data['predictions'][0]
        num_targets = 1 if isinstance(first_pred, float) else len(first_pred)
        
        # Build results dictionary
        results = {'cif_id': self.test_data['cif_id']}
        
        # Add columns for each target
        for i in range(num_targets):
            results[f'pred_{i+1}'] = [
                pred[i] if num_targets > 1 else pred 
                for pred in self.test_data['predictions']
            ]
            results[f'target_{i+1}'] = [
                target[i] if num_targets > 1 else target 
                for target in self.test_data['targets']
            ]
        
        # Save to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = self.config.get("log_dir") or "."
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, f"test_results_{timestamp}.csv")

        if num_targets == 1:
            results['pred_1'] = [
                self._unwrap_single_value(pred)
                for pred in results['pred_1']
            ]
            results['target_1'] = [
                self._unwrap_single_value(target)
                for target in results['target_1']
            ]

        # Create DataFrame after single-target flattening.
        results_df = pd.DataFrame(results)
        results_df.to_csv(csv_path, index=False)
        print(f"Saved results with {num_targets} targets to {csv_path}")

    def on_test_epoch_end(self):
                # Calculate R2 and MAE for each target
        if len(self.test_data['predictions']) > 0:
            predictions = np.array(self.test_data['predictions'])
            targets = np.array(self.test_data['targets'])

            if self.test_data.get('task') == 'classification':
                if len(targets) > 0:
                    accuracy = float(np.mean(predictions == targets))
                    self.log("test/accuracy", accuracy, sync_dist=True)
                self.save_test_results()
                self.test_data = {'task': None, 'cif_id': [], 'predictions': [], 'targets': [], 'scores': []}
                return
            
            # Handle single and multi-output cases
            if predictions.ndim == 1:
                # Compute R2 score
                r2 = r2_score(targets, predictions)
                self.log("test/r2_score", r2, sync_dist=True)
                
                # Compute MAE
                mae = np.mean(np.abs(targets - predictions))
                self.log("test/mae", mae, sync_dist=True)
            else:
                for i in range(predictions.shape[1]):
                    # Compute R2 score for each target
                    r2 = r2_score(targets[:, i], predictions[:, i])
                    self.log(f"test/r2_score_target_{i}", r2, sync_dist=True)
                    
                    # Compute MAE for each target
                    mae = np.mean(np.abs(targets[:, i] - predictions[:, i]))
                    self.log(f"test/mae_target_{i}", mae, sync_dist=True)
        
        self.save_test_results()
        self.test_data = {'task': None, 'cif_id': [], 'predictions': [], 'targets': [], 'scores': []}
    

    def configure_optimizers(self):
        return model_utils.set_schedule(self)
    
    def save_predict_results(self):
        if not self.predict_data['cif_id']:
            return
        if self.predict_data.get('task') == 'classification':
            results = {'cif_id': self.predict_data['cif_id']}
            first_score = self.predict_data['scores'][0] if self.predict_data['scores'] else None
            if isinstance(first_score, (list, tuple)):
                num_classes = len(first_score)
                for i in range(num_classes):
                    results[f'class_{i}_prob'] = [score[i] for score in self.predict_data['scores']]
            else:
                results['positive_class_prob'] = self.predict_data['scores']
            results['predicted_class'] = self.predict_data['predictions']
            if not self.no_labels and self.predict_data.get('targets'):
                results['target_class'] = self.predict_data['targets']

            df = pd.DataFrame(results)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = self.config.get("log_dir") or "."
            os.makedirs(output_dir, exist_ok=True)
            path = os.path.join(output_dir, f"predict_results_{timestamp}.csv")
            df.to_csv(path, index=False)
            print(f"Saved prediction results to {path}")
            return
        # Determine if single or multi-target
        first_pred = self.predict_data['predictions'][0]
        num_targets = 1 if not isinstance(first_pred, (list, tuple)) else len(first_pred)

        results = {'cif_id': self.predict_data['cif_id']}
        if num_targets == 1:
            results['prediction'] = [
                self._unwrap_single_value(pred)
                for pred in self.predict_data['predictions']
            ]
            if not self.no_labels:
                results['target'] = [
                    self._unwrap_single_value(target)
                    for target in self.predict_data['targets']
                ]
        else:
            for i in range(num_targets):
                results[f'pred_{i+1}'] = [p[i] for p in self.predict_data['predictions']]
                if not self.no_labels:
                    results[f'target_{i+1}'] = [t[i] for t in self.predict_data['targets']]

        df = pd.DataFrame(results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = self.config.get("log_dir") or "."
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"predict_results_{timestamp}.csv")
        df.to_csv(path, index=False)
        print(f"Saved prediction results to {path}")

    def on_predict_start(self):
        self.write_log = False
        model_utils.set_task(self)
        self.predict_data = {'task': None, 'cif_id': [], 'predictions': [], 'scores': []}
        if not self.no_labels:
            self.predict_data['targets'] = []

    def predict_step(self, batch, batch_idx=None, dataloader_idx=None):
        out = self(batch)
        if "classification_logits" in out:
            if self.config.get("num_classes", 0) == 2:
                out["classification_probabilities"] = torch.sigmoid(
                    out["classification_logits"]
                )
                out["classification_logits_index"] = (
                    out["classification_probabilities"] >= 0.5
                ).to(torch.int)
            else:
                sm = torch.nn.Softmax(dim=1)
                out["classification_probabilities"] = sm(out["classification_logits"])
                out["classification_logits_index"] = torch.argmax(
                    out["classification_probabilities"], dim=1
                )
        for k, v in list(out.items()):
            if torch.is_tensor(v):
                out[k] = v.cpu().tolist()
        if "regression_logits" in out:
            self.predict_data['task'] = 'regression'
            self.predict_data['cif_id'].extend(out['cif_id'])
            self.predict_data['predictions'].extend(
                out['regression_logits']
            )
            if not self.no_labels and 'regression_labels' in out:
                self.predict_data['targets'].extend(out['regression_labels'])
        elif "classification_logits" in out:
            self.predict_data['task'] = 'classification'
            self.predict_data['cif_id'].extend(out['cif_id'])
            self.predict_data['predictions'].extend(out['classification_logits_index'])
            self.predict_data['scores'].extend(
                out.get('classification_probabilities', out['classification_logits'])
            )
            if not self.no_labels and 'classification_labels' in out:
                self.predict_data['targets'].extend(out['classification_labels'])
        result = {k: v for k, v in out.items()
                  if k == 'cif_id' or 'logits' in k or (not self.no_labels and 'labels' in k)}
        return result

    def on_predict_epoch_end(self):
        self.save_predict_results()
        self.predict_data = {'task': None, 'cif_id': [], 'predictions': [], 'scores': []}
        if not self.no_labels:
            self.predict_data['targets'] = []

    def on_predict_end(self):
        self.write_log = True
        self.predict_data.clear()

    
    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(metric)
