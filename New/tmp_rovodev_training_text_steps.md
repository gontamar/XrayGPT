# Exact Text Processing Steps During Training in XrayGPT

## Training Text Processing Pipeline - Step by Step

### Step 1: Training Script Launch
**File**: `train.py`
**Line**: 73-102

```bash
python train.py --cfg-path train_configs/xraygpt_mimic_pretrain.yaml
```

```python
def main():
    cfg = Config(parse_args())  # Load training configuration
    init_distributed_mode(cfg.run_cfg)
    setup_seeds(cfg)
    
    task = tasks.setup_task(cfg)  # Create ImageTextPretrainTask
    datasets = task.build_datasets(cfg)  # Build training datasets
    model = task.build_model(cfg)  # Build MiniGPT4 model
    
    runner = get_runner_class(cfg)(cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets)
    runner.train()  # Start training loop
```

**Text State**: No text processing yet, just initialization.

### Step 2: Dataset Building and Text Data Loading
**File**: `xraygpt/tasks/base_task.py`
**Lines**: 36-66

```python
def build_datasets(self, cfg):
    datasets = dict()
    datasets_config = cfg.datasets_cfg  # MIMIC/OpenI dataset configs
    
    for name in datasets_config:
        dataset_config = datasets_config[name]
        builder = registry.get_builder_class(name)(dataset_config)  # Get MIMIC/OpenI builder
        dataset = builder.build_datasets()  # Build actual datasets
        datasets[name] = dataset
    
    return datasets
```

#### A. MIMIC Dataset Text Loading
**File**: `xraygpt/datasets/datasets/mimic_dataset.py`
**Lines**: 30-45

```python
class MIMICDataset(CaptionDataset):
    def __getitem__(self, index):
        ann = self.annotation[index]  # Load annotation from JSON
        
        # Raw medical text from annotation file
        caption = ann['caption']  # "The heart size is normal. No acute cardiopulmonary process."
        
        return {
            "image": image,
            "caption": caption,  # Raw text string - NO PROCESSING YET
            "image_id": self.img_ids[ann["image_id"]],
        }
```

#### B. OpenI Dataset Text Loading
**File**: `xraygpt/datasets/datasets/openi_dataset.py`
**Lines**: 36-52

```python
class OpenIDataset(CaptionDataset):
    def __getitem__(self, index):
        ann = self.annotation[index]
        
        # Raw medical text from annotation
        caption = ann['caption']  # "Bilateral pleural effusions are present..."
        
        return {
            "image": image,
            "caption": caption,  # Raw text string - NO PROCESSING YET
            "image_id": self.img_ids[ann["image_id"]],
        }
```

**Text State**: Raw medical captions loaded as strings from annotation files.

### Step 3: Training Loop Initialization
**File**: `xraygpt/runners/runner_base.py`
**Lines**: 362-421

```python
def train(self):
    for cur_epoch in range(self.start_epoch, self.max_epoch):
        if not self.evaluate_only:
            logging.info("Start training")
            train_stats = self.train_epoch(cur_epoch)  # Call training epoch
```

**File**: `xraygpt/runners/runner_base.py`
**Lines**: 446-460

```python
def train_epoch(self, epoch):
    self.model.train()  # Set model to training mode
    
    return self.task.train_epoch(
        epoch=epoch,
        model=self.model,
        data_loader=self.train_loader,  # DataLoader with text samples
        optimizer=self.optimizer,
        scaler=self.scaler,
        lr_scheduler=self.lr_scheduler,
        cuda_enabled=self.cuda_enabled,
        log_freq=self.log_freq,
        accum_grad_iters=self.accum_grad_iters,
    )
```

### Step 4: Training Inner Loop - Batch Processing
**File**: `xraygpt/tasks/base_task.py`
**Lines**: 213-304

```python
def _train_inner_loop(self, epoch, iters_per_epoch, model, data_loader, optimizer, lr_scheduler, ...):
    for i in metric_logger.log_every(range(iters_per_epoch), log_freq, header):
        
        # GET BATCH WITH RAW TEXT
        samples = next(data_loader)  
        # samples = {
        #     "image": tensor[batch_size, 3, 224, 224],
        #     "caption": ["The heart size is normal...", "Bilateral pleural effusions..."],
        #     "image_id": tensor[batch_size]
        # }
        
        samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
        
        with torch.cuda.amp.autocast(enabled=use_amp):
            loss = self.train_step(model=model, samples=samples)  # PROCESS TEXT HERE
```

### Step 5: Train Step - Model Forward Call
**File**: `xraygpt/tasks/base_task.py`
**Lines**: 68-70

```python
def train_step(self, model, samples):
    loss = model(samples)["loss"]  # Calls MiniGPT4.forward() with text
    return loss
```

### Step 6: Model Forward Pass - FIRST TEXT PROCESSING
**File**: `xraygpt/models/mini_gpt4.py`
**Lines**: 190-250

#### A. Image Processing (No Text Yet)
```python
def forward(self, samples):
    image = samples["image"]  # Process image first
    img_embeds, atts_img = self.encode_img(image)  # Get vision features
```

#### B. Prompt Wrapping (Optional Text Processing)
```python
# Lines 194-205
if hasattr(samples, 'question_split'):  # VQA dataset
    vqa_prompt = '###Patient: <Img><ImageHere></Img> '
    img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img, vqa_prompt)
elif self.prompt_list:
    prompt = random.choice(self.prompt_list)
    img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img, prompt)
```

#### C. MAIN TEXT TOKENIZATION - Training Target Preparation
```python
# Lines 207-218 - CRITICAL TEXT PROCESSING STEP
self.llama_tokenizer.padding_side = "right"

# Raw text from dataset
text = [t + self.end_sym for t in samples["caption"]]
# text = ["The heart size is normal.\n", "Bilateral pleural effusions.\n"]

# FIRST TEXT TOKENIZATION - LLaMA Tokenizer
to_regress_tokens = self.llama_tokenizer(
    text,
    return_tensors="pt",
    padding="longest",
    truncation=True,
    max_length=self.max_txt_len,  # 32 tokens
    add_special_tokens=False
).to(image.device)

# to_regress_tokens.input_ids shape: [batch_size, seq_len]
# Contains token IDs like: [[1234, 5678, 9012, ...], [2345, 6789, 1234, ...]]
```

#### D. Training Target Preparation
```python
# Lines 220-228
targets = to_regress_tokens.input_ids.masked_fill(
    to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
)

empty_targets = (
    torch.ones([atts_img.shape[0], atts_img.shape[1]+1],
               dtype=torch.long).to(image.device).fill_(-100)  # plus one for bos
)
targets = torch.cat([empty_targets, targets], dim=1)
# targets shape: [batch_size, vision_tokens + 1 + text_tokens]
# Values: [-100, -100, ..., -100, 1234, 5678, 9012, ...]
```

#### E. Text Embedding Generation
```python
# Lines 237-238
to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids)
# Shape: [batch_size, seq_len, 4096] - LLaMA embeddings from text tokens
```

#### F. Input Sequence Construction for Training
```python
# Lines 230-239
batch_size = img_embeds.shape[0]
bos = torch.ones([batch_size, 1],
                 dtype=to_regress_tokens.input_ids.dtype,
                 device=to_regress_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
bos_embeds = self.llama_model.model.embed_tokens(bos)  # BOS token embedding
atts_bos = atts_img[:, :1]

# CONCATENATE: [BOS] + [Vision Features] + [Text Embeddings]
inputs_embeds = torch.cat([bos_embeds, img_embeds, to_regress_embeds], dim=1)
attention_mask = torch.cat([atts_bos, atts_img, to_regress_tokens.attention_mask], dim=1)

# Final sequence structure:
# [<BOS>] + [Q-Former Vision Features] + [Medical Text Tokens]
#    1    +         32 tokens          +    up to 32 tokens
```

### Step 7: LLaMA Forward Pass - Training Loss Calculation
**File**: `xraygpt/models/mini_gpt4.py`
**Lines**: 241-250

```python
with self.maybe_autocast():
    outputs = self.llama_model(
        inputs_embeds=inputs_embeds,  # [BOS + Vision + Text]
        attention_mask=attention_mask,
        return_dict=True,
        labels=targets,  # Text tokens as training targets
    )
loss = outputs.loss  # Cross-entropy loss on text generation

return {"loss": loss}
```

**Text Processing**: LLaMA model learns to predict medical text tokens given vision-conditioned context.

### Step 8: Loss Backpropagation and Optimization
**File**: `xraygpt/tasks/base_task.py`
**Lines**: 276-296

```python
# Back to training inner loop
with torch.cuda.amp.autocast(enabled=use_amp):
    loss = self.train_step(model=model, samples=samples)  # Got loss from text prediction

# Backpropagation
if use_amp:
    scaler.scale(loss).backward()
else:
    loss.backward()

# Gradient accumulation and optimization
if (i + 1) % accum_grad_iters == 0:
    if use_amp:
        scaler.step(optimizer)
        scaler.update()                     
    else:    
        optimizer.step()
    optimizer.zero_grad()
```

### Step 9: Training Statistics and Logging
**File**: `xraygpt/tasks/base_task.py`
**Lines**: 294-304

```python
metric_logger.update(loss=loss.item())
metric_logger.update(lr=optimizer.param_groups[0]["lr"])

# After epoch completion
metric_logger.synchronize_between_processes()
logging.info("Averaged stats: " + str(metric_logger.global_avg()))
return {
    k: "{:.3f}".format(meter.global_avg)
    for k, meter in metric_logger.meters.items()
}
```

## Complete Training Text Flow Summary

```
1. Raw Medical Text Loading:
   "The heart size is normal. No acute cardiopulmonary process."
   
2. Batch Formation:
   samples["caption"] = ["Text1\n", "Text2\n", ...]
   
3. LLaMA Tokenization:
   text → to_regress_tokens.input_ids [batch_size, seq_len]
   
4. Target Preparation:
   targets = [-100, -100, ..., token_id1, token_id2, ...]
   
5. Text Embedding:
   token_ids → to_regress_embeds [batch_size, seq_len, 4096]
   
6. Sequence Construction:
   [BOS_embed] + [Vision_embeds] + [Text_embeds]
   
7. LLaMA Training:
   Predict text tokens given vision-conditioned context
   
8. Loss Calculation:
   Cross-entropy loss on medical text generation
   
9. Backpropagation:
   Update model weights to improve medical text prediction
```

## Key Training Text Processing Points

1. **Text Source**: Raw medical captions from MIMIC/OpenI annotation files
2. **Tokenization**: LLaMA tokenizer (32K vocabulary, SentencePiece)
3. **Target Role**: Text tokens serve as training targets for language modeling
4. **Sequence Structure**: Vision features + Text tokens for multimodal learning
5. **Loss Function**: Cross-entropy loss on medical text prediction
6. **Learning Objective**: Generate accurate medical descriptions given X-ray images

This training pipeline teaches XrayGPT to generate medically accurate text descriptions by learning from vision-text pairs in the medical domain.