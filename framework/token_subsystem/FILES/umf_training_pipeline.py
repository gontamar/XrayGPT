"""
Universal Multimodal Framework (UMF) - Training Pipeline
Multi-stage training system inspired by XrayGPT methodology
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Optional, Any, Tuple
import yaml
import json
import logging
from pathlib import Path
from tqdm import tqdm
import wandb
from dataclasses import dataclass
from enum import Enum

from umf_universal_implementation import (
    UniversalMultimodalFramework, DomainType, ModalityType, 
    MultimodalInput, DomainConfig
)

# ============================================================================
# Training Configuration
# ============================================================================

@dataclass
class TrainingStage:
    """Configuration for a training stage"""
    name: str
    duration_epochs: int
    learning_rate: float
    batch_size: int
    frozen_components: List[str]
    trainable_components: List[str]
    datasets: List[str]
    objective: str
    warmup_steps: int = 0
    weight_decay: float = 0.01

class TrainingPhase(Enum):
    MODALITY_PRETRAIN = "modality_pretrain"
    CROSS_MODAL_ALIGN = "cross_modal_align"
    DOMAIN_ADAPT = "domain_adapt"
    INSTRUCTION_TUNE = "instruction_tune"

# ============================================================================
# Dataset Classes
# ============================================================================

class UniversalMultimodalDataset(Dataset):
    """Universal dataset class for multimodal training"""
    
    def __init__(self, data_path: str, domain: DomainType, stage: TrainingPhase):
        self.data_path = Path(data_path)
        self.domain = domain
        self.stage = stage
        self.samples = self._load_samples()
    
    def _load_samples(self) -> List[Dict[str, Any]]:
        """Load samples based on domain and training stage"""
        samples_file = self.data_path / f"{self.domain.value}_{self.stage.value}.json"
        
        if samples_file.exists():
            with open(samples_file, 'r') as f:
                return json.load(f)
        else:
            # Return dummy data for demonstration
            return self._generate_dummy_samples()
    
    def _generate_dummy_samples(self) -> List[Dict[str, Any]]:
        """Generate dummy samples for demonstration"""
        dummy_samples = []
        
        if self.domain == DomainType.MEDICAL:
            dummy_samples = [
                {
                    "image_path": "dummy_xray.jpg",
                    "text": "Chest X-ray shows clear lung fields with no abnormalities.",
                    "conversation": [
                        {"role": "patient", "content": "What do you see in my X-ray?"},
                        {"role": "doctor", "content": "Your chest X-ray appears normal with clear lung fields."}
                    ]
                }
            ] * 100
        
        elif self.domain == DomainType.AUTONOMOUS:
            dummy_samples = [
                {
                    "image_path": "dummy_traffic.jpg",
                    "sensor_data": "dummy_lidar.npy",
                    "text": "Traffic scene with pedestrian crossing ahead.",
                    "action": "slow_down_and_stop"
                }
            ] * 100
        
        # Add more domain-specific dummy data...
        
        return dummy_samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        
        # Create multimodal input
        multimodal_input = MultimodalInput()
        
        if "image_path" in sample:
            # In real implementation, load actual image
            multimodal_input.vision = torch.randn(3, 224, 224)  # Dummy image tensor
        
        if "audio_path" in sample:
            # In real implementation, load actual audio
            multimodal_input.audio = torch.randn(16000)  # Dummy audio tensor
        
        if "text" in sample:
            multimodal_input.text = sample["text"]
        
        if "sensor_data" in sample:
            # In real implementation, load actual sensor data
            multimodal_input.sensor = torch.randn(100)  # Dummy sensor data
        
        return {
            "multimodal_input": multimodal_input,
            "target_text": sample.get("text", ""),
            "conversation": sample.get("conversation", []),
            "metadata": sample.get("metadata", {})
        }

class DomainSpecificDataLoader:
    """Data loader factory for different domains"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_root = Path(config["data"]["root_path"])
    
    def get_dataloader(self, domain: DomainType, stage: TrainingPhase, 
                      batch_size: int, shuffle: bool = True) -> DataLoader:
        """Get domain-specific dataloader"""
        
        dataset = UniversalMultimodalDataset(
            data_path=self.data_root / domain.value,
            domain=domain,
            stage=stage
        )
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.config["data"].get("num_workers", 4),
            pin_memory=True
        )

# ============================================================================
# Training Stages Implementation
# ============================================================================

class ModalityPretrainer:
    """Stage 1: Modality-specific pretraining"""
    
    def __init__(self, framework: UniversalMultimodalFramework, config: Dict[str, Any]):
        self.framework = framework
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def train(self, dataloaders: Dict[str, DataLoader], epochs: int) -> None:
        """Train modality-specific encoders"""
        self.logger.info("Starting modality pretraining...")
        
        # Freeze LLM and Q-Former
        self._freeze_components(["language_model", "q_former"])
        
        # Setup optimizer for encoders only
        optimizer = self._setup_optimizer(["input_processor"])
        scheduler = self._setup_scheduler(optimizer, epochs)
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            for domain, dataloader in dataloaders.items():
                for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs} - {domain}"):
                    loss = self._compute_modality_loss(batch, DomainType(domain))
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
            
            scheduler.step()
            self.logger.info(f"Epoch {epoch+1} - Loss: {epoch_loss:.4f}")
    
    def _compute_modality_loss(self, batch: Dict[str, Any], domain: DomainType) -> torch.Tensor:
        """Compute modality-specific contrastive loss"""
        multimodal_input = batch["multimodal_input"]
        
        # Process inputs
        processed_features = self.framework.input_processor.process_multimodal_input(
            multimodal_input, domain
        )
        
        # Compute contrastive loss between modalities
        if len(processed_features) > 1:
            features_list = list(processed_features.values())
            loss = self._contrastive_loss(features_list[0], features_list[1])
        else:
            # Single modality reconstruction loss
            loss = torch.tensor(0.0, requires_grad=True)
        
        return loss
    
    def _contrastive_loss(self, features1: torch.Tensor, features2: torch.Tensor) -> torch.Tensor:
        """Compute contrastive loss between two feature sets"""
        # Normalize features
        features1 = F.normalize(features1, dim=-1)
        features2 = F.normalize(features2, dim=-1)
        
        # Compute similarity matrix
        similarity = torch.matmul(features1, features2.T)
        
        # Create labels (positive pairs on diagonal)
        labels = torch.arange(features1.size(0)).to(features1.device)
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(similarity / 0.1, labels)
        
        return loss
    
    def _freeze_components(self, component_names: List[str]) -> None:
        """Freeze specified components"""
        for name in component_names:
            component = getattr(self.framework, name)
            for param in component.parameters():
                param.requires_grad = False
    
    def _setup_optimizer(self, trainable_components: List[str]) -> optim.Optimizer:
        """Setup optimizer for trainable components"""
        params = []
        for name in trainable_components:
            component = getattr(self.framework, name)
            params.extend(component.parameters())
        
        return optim.AdamW(
            params,
            lr=self.config["training"]["learning_rate"],
            weight_decay=self.config["training"]["weight_decay"]
        )
    
    def _setup_scheduler(self, optimizer: optim.Optimizer, epochs: int):
        """Setup learning rate scheduler"""
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

class CrossModalAligner:
    """Stage 2: Cross-modal alignment training"""
    
    def __init__(self, framework: UniversalMultimodalFramework, config: Dict[str, Any]):
        self.framework = framework
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def train(self, dataloaders: Dict[str, DataLoader], epochs: int) -> None:
        """Train cross-modal alignment"""
        self.logger.info("Starting cross-modal alignment training...")
        
        # Freeze encoders and LLM, train Q-Former
        self._freeze_components(["input_processor", "language_model"])
        self._unfreeze_components(["q_former"])
        
        optimizer = self._setup_optimizer(["q_former"])
        scheduler = self._setup_scheduler(optimizer, epochs)
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            for domain, dataloader in dataloaders.items():
                for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs} - Alignment"):
                    loss = self._compute_alignment_loss(batch, DomainType(domain))
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
            
            scheduler.step()
            self.logger.info(f"Epoch {epoch+1} - Alignment Loss: {epoch_loss:.4f}")
    
    def _compute_alignment_loss(self, batch: Dict[str, Any], domain: DomainType) -> torch.Tensor:
        """Compute cross-modal alignment loss"""
        multimodal_input = batch["multimodal_input"]
        
        # Process inputs
        processed_features = self.framework.input_processor.process_multimodal_input(
            multimodal_input, domain
        )
        
        # Fuse features
        fused_features = self.framework.q_former(processed_features, domain)
        
        # Compute alignment loss (e.g., matching loss with text)
        if multimodal_input.text:
            text_features = self.framework.input_processor.processors[ModalityType.TEXT].process(
                multimodal_input.text, domain
            )
            loss = F.mse_loss(fused_features.mean(dim=1), text_features)
        else:
            loss = torch.tensor(0.0, requires_grad=True)
        
        return loss
    
    def _freeze_components(self, component_names: List[str]) -> None:
        """Freeze specified components"""
        for name in component_names:
            component = getattr(self.framework, name)
            for param in component.parameters():
                param.requires_grad = False
    
    def _unfreeze_components(self, component_names: List[str]) -> None:
        """Unfreeze specified components"""
        for name in component_names:
            component = getattr(self.framework, name)
            for param in component.parameters():
                param.requires_grad = True
    
    def _setup_optimizer(self, trainable_components: List[str]) -> optim.Optimizer:
        """Setup optimizer for trainable components"""
        params = []
        for name in trainable_components:
            component = getattr(self.framework, name)
            params.extend(component.parameters())
        
        return optim.AdamW(
            params,
            lr=self.config["training"]["learning_rate"],
            weight_decay=self.config["training"]["weight_decay"]
        )
    
    def _setup_scheduler(self, optimizer: optim.Optimizer, epochs: int):
        """Setup learning rate scheduler"""
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

class DomainAdapter:
    """Stage 3: Domain-specific adaptation"""
    
    def __init__(self, framework: UniversalMultimodalFramework, config: Dict[str, Any]):
        self.framework = framework
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def train(self, domain_dataloaders: Dict[DomainType, DataLoader], epochs: int) -> None:
        """Train domain-specific adapters"""
        self.logger.info("Starting domain adaptation training...")
        
        # Freeze base components, train domain adapters
        self._freeze_components(["input_processor", "q_former", "language_model"])
        self._unfreeze_components(["domain_adapters"])
        
        for domain, dataloader in domain_dataloaders.items():
            self.logger.info(f"Training domain adapter for: {domain.value}")
            
            optimizer = self._setup_domain_optimizer(domain)
            scheduler = self._setup_scheduler(optimizer, epochs)
            
            for epoch in range(epochs):
                epoch_loss = 0.0
                
                for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs} - {domain.value}"):
                    loss = self._compute_domain_loss(batch, domain)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                
                scheduler.step()
                self.logger.info(f"{domain.value} - Epoch {epoch+1} - Loss: {epoch_loss:.4f}")
    
    def _compute_domain_loss(self, batch: Dict[str, Any], domain: DomainType) -> torch.Tensor:
        """Compute domain-specific adaptation loss"""
        multimodal_input = batch["multimodal_input"]
        target_text = batch["target_text"]
        
        # Full forward pass
        response = self.framework.process(
            multimodal_input, domain, target_text
        )
        
        # Compute language modeling loss
        # This is a simplified version - in practice, you'd compute proper LM loss
        loss = torch.tensor(1.0, requires_grad=True)  # Placeholder
        
        return loss
    
    def _freeze_components(self, component_names: List[str]) -> None:
        """Freeze specified components"""
        for name in component_names:
            component = getattr(self.framework, name)
            for param in component.parameters():
                param.requires_grad = False
    
    def _unfreeze_components(self, component_names: List[str]) -> None:
        """Unfreeze specified components"""
        for name in component_names:
            component = getattr(self.framework, name)
            for param in component.parameters():
                param.requires_grad = True
    
    def _setup_domain_optimizer(self, domain: DomainType) -> optim.Optimizer:
        """Setup optimizer for specific domain adapter"""
        domain_adapter = self.framework.domain_adapters[domain.value]
        
        return optim.AdamW(
            domain_adapter.parameters(),
            lr=self.config["training"]["learning_rate"],
            weight_decay=self.config["training"]["weight_decay"]
        )
    
    def _setup_scheduler(self, optimizer: optim.Optimizer, epochs: int):
        """Setup learning rate scheduler"""
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

class InstructionTuner:
    """Stage 4: Instruction following and conversation training"""
    
    def __init__(self, framework: UniversalMultimodalFramework, config: Dict[str, Any]):
        self.framework = framework
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def train(self, dataloaders: Dict[str, DataLoader], epochs: int) -> None:
        """Train instruction following capabilities"""
        self.logger.info("Starting instruction tuning...")
        
        # Unfreeze language model and conversation components
        self._unfreeze_components(["language_model"])
        
        optimizer = self._setup_optimizer(["language_model"])
        scheduler = self._setup_scheduler(optimizer, epochs)
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            for domain, dataloader in dataloaders.items():
                for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs} - Instruction"):
                    loss = self._compute_instruction_loss(batch, DomainType(domain))
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
            
            scheduler.step()
            self.logger.info(f"Epoch {epoch+1} - Instruction Loss: {epoch_loss:.4f}")
    
    def _compute_instruction_loss(self, batch: Dict[str, Any], domain: DomainType) -> torch.Tensor:
        """Compute instruction following loss"""
        # This would implement proper instruction-following loss
        # For now, return a placeholder
        loss = torch.tensor(1.0, requires_grad=True)
        return loss
    
    def _unfreeze_components(self, component_names: List[str]) -> None:
        """Unfreeze specified components"""
        for name in component_names:
            component = getattr(self.framework, name)
            for param in component.parameters():
                param.requires_grad = True
    
    def _setup_optimizer(self, trainable_components: List[str]) -> optim.Optimizer:
        """Setup optimizer for trainable components"""
        params = []
        for name in trainable_components:
            component = getattr(self.framework, name)
            params.extend(component.parameters())
        
        return optim.AdamW(
            params,
            lr=self.config["training"]["learning_rate"],
            weight_decay=self.config["training"]["weight_decay"]
        )
    
    def _setup_scheduler(self, optimizer: optim.Optimizer, epochs: int):
        """Setup learning rate scheduler"""
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

# ============================================================================
# Main Training Pipeline
# ============================================================================

class UniversalTrainingPipeline:
    """Main training pipeline orchestrator"""
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.framework = UniversalMultimodalFramework()
        self.data_loader_factory = DomainSpecificDataLoader(self.config)
        
        # Initialize training stages
        self.modality_pretrainer = ModalityPretrainer(self.framework, self.config)
        self.cross_modal_aligner = CrossModalAligner(self.framework, self.config)
        self.domain_adapter = DomainAdapter(self.framework, self.config)
        self.instruction_tuner = InstructionTuner(self.framework, self.config)
        
        # Setup logging
        self._setup_logging()
        
        # Setup experiment tracking
        if self.config.get("wandb", {}).get("enabled", False):
            wandb.init(
                project=self.config["wandb"]["project"],
                config=self.config
            )
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load training configuration"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_logging(self) -> None:
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('umf_training.log'),
                logging.StreamHandler()
            ]
        )
    
    def train_full_pipeline(self) -> None:
        """Execute the complete training pipeline"""
        logger = logging.getLogger(__name__)
        logger.info("Starting Universal Multimodal Framework training pipeline")
        
        # Stage 1: Modality Pretraining
        if self.config["training"]["stages"]["modality_pretrain"]["enabled"]:
            logger.info("=" * 50)
            logger.info("STAGE 1: MODALITY PRETRAINING")
            logger.info("=" * 50)
            
            stage_config = self.config["training"]["stages"]["modality_pretrain"]
            dataloaders = self._get_stage_dataloaders(TrainingPhase.MODALITY_PRETRAIN)
            
            self.modality_pretrainer.train(
                dataloaders=dataloaders,
                epochs=stage_config["epochs"]
            )
            
            self._save_checkpoint("stage1_modality_pretrain")
        
        # Stage 2: Cross-Modal Alignment
        if self.config["training"]["stages"]["cross_modal_align"]["enabled"]:
            logger.info("=" * 50)
            logger.info("STAGE 2: CROSS-MODAL ALIGNMENT")
            logger.info("=" * 50)
            
            stage_config = self.config["training"]["stages"]["cross_modal_align"]
            dataloaders = self._get_stage_dataloaders(TrainingPhase.CROSS_MODAL_ALIGN)
            
            self.cross_modal_aligner.train(
                dataloaders=dataloaders,
                epochs=stage_config["epochs"]
            )
            
            self._save_checkpoint("stage2_cross_modal_align")
        
        # Stage 3: Domain Adaptation
        if self.config["training"]["stages"]["domain_adapt"]["enabled"]:
            logger.info("=" * 50)
            logger.info("STAGE 3: DOMAIN ADAPTATION")
            logger.info("=" * 50)
            
            stage_config = self.config["training"]["stages"]["domain_adapt"]
            domain_dataloaders = self._get_domain_dataloaders(TrainingPhase.DOMAIN_ADAPT)
            
            self.domain_adapter.train(
                domain_dataloaders=domain_dataloaders,
                epochs=stage_config["epochs"]
            )
            
            self._save_checkpoint("stage3_domain_adapt")
        
        # Stage 4: Instruction Tuning
        if self.config["training"]["stages"]["instruction_tune"]["enabled"]:
            logger.info("=" * 50)
            logger.info("STAGE 4: INSTRUCTION TUNING")
            logger.info("=" * 50)
            
            stage_config = self.config["training"]["stages"]["instruction_tune"]
            dataloaders = self._get_stage_dataloaders(TrainingPhase.INSTRUCTION_TUNE)
            
            self.instruction_tuner.train(
                dataloaders=dataloaders,
                epochs=stage_config["epochs"]
            )
            
            self._save_checkpoint("stage4_instruction_tune")
        
        logger.info("🎉 Training pipeline completed successfully!")
    
    def _get_stage_dataloaders(self, stage: TrainingPhase) -> Dict[str, DataLoader]:
        """Get dataloaders for a specific training stage"""
        dataloaders = {}
        
        for domain in DomainType:
            if self.config["domains"][domain.value]["enabled"]:
                dataloaders[domain.value] = self.data_loader_factory.get_dataloader(
                    domain=domain,
                    stage=stage,
                    batch_size=self.config["training"]["batch_size"]
                )
        
        return dataloaders
    
    def _get_domain_dataloaders(self, stage: TrainingPhase) -> Dict[DomainType, DataLoader]:
        """Get domain-specific dataloaders"""
        dataloaders = {}
        
        for domain in DomainType:
            if self.config["domains"][domain.value]["enabled"]:
                dataloaders[domain] = self.data_loader_factory.get_dataloader(
                    domain=domain,
                    stage=stage,
                    batch_size=self.config["training"]["batch_size"]
                )
        
        return dataloaders
    
    def _save_checkpoint(self, stage_name: str) -> None:
        """Save model checkpoint"""
        checkpoint_path = Path(self.config["training"]["output_dir"]) / f"{stage_name}.pth"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.framework.state_dict(),
            'stage': stage_name,
            'config': self.config
        }, checkpoint_path)
        
        logging.getLogger(__name__).info(f"Checkpoint saved: {checkpoint_path}")

# ============================================================================
# CLI Interface
# ============================================================================

def main():
    """Main training script"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Universal Multimodal Framework Training")
    parser.add_argument("--config", required=True, help="Path to training configuration file")
    parser.add_argument("--stage", choices=["all", "1", "2", "3", "4"], default="all",
                       help="Training stage to run")
    
    args = parser.parse_args()
    
    # Initialize training pipeline
    pipeline = UniversalTrainingPipeline(args.config)
    
    if args.stage == "all":
        pipeline.train_full_pipeline()
    else:
        # Run specific stage (implementation would go here)
        print(f"Running stage {args.stage}")

if __name__ == "__main__":
    main()