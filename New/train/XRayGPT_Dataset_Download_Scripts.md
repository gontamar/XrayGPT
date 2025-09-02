# XRayGPT Dataset Download Scripts and Code

## 🔍 **DATASET DOWNLOAD IN XRAYGPT CODEBASE**

Looking at the XRayGPT codebase, there are **NO direct download scripts** for MIMIC-CXR because:
1. **Legal restrictions** - MIMIC-CXR requires manual PhysioNet approval
2. **Credentials required** - Cannot be automated without user credentials
3. **Data use agreement** - Must be signed manually

However, I'll show you the **code structure** and create **download scripts** based on XRayGPT's architecture.

---

## 📁 **XRAYGPT DATASET HANDLING CODE**

### **1. Dataset Builder Code**
**File**: `xraygpt/datasets/builders/image_text_pair_builder.py`

```python
@registry.register_builder("mimic")
class MIMICBuilder(BaseDatasetBuilder):
    train_dataset_cls = MIMICDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/mimic/defaults.yaml"}

    def _download_ann(self):
        # NOTE: This is empty in original code
        # Manual download required
        pass

    def _download_vis(self):
        # NOTE: This is empty in original code  
        # Manual download required
        pass

    def build_datasets(self):
        # Assumes data is already downloaded
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info
        storage_path = build_info.storage

        datasets = dict()
        
        # Expects data to be at storage_path
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            ann_paths=[os.path.join(storage_path, 'filter_cap.json')],
            vis_root=os.path.join(storage_path, 'image'),
        )

        return datasets
```

### **2. Dataset Loading Code**
**File**: `xraygpt/datasets/datasets/mimic_dataset.py`

```python
class MIMICDataset(CaptionDataset):
    def __getitem__(self, index):
        # Assumes images are already downloaded
        ann = self.annotation[index]

        img_file = '{}.jpg'.format(ann["image_id"])
        image_path = os.path.join(self.vis_root, img_file)
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        caption = ann['caption']

        return {
            "image": image,
            "caption": caption,
            "image_id": self.img_ids[ann["image_id"]],
        }
```

---

## 🛠️ **CREATING DOWNLOAD SCRIPTS FOR XRAYGPT**

### **1. MIMIC-CXR Download Script**
**File**: `scripts/download_mimic_cxr.py`

```python
#!/usr/bin/env python3
"""
MIMIC-CXR Dataset Download Script for XRayGPT
Requires PhysioNet credentials and approved access
"""

import os
import subprocess
import argparse
import getpass
from pathlib import Path

def download_mimic_cxr_images(username, password, output_dir):
    """Download MIMIC-CXR images using wget"""
    
    print("🔄 Downloading MIMIC-CXR Images (37GB)...")
    
    # Create output directory
    images_dir = os.path.join(output_dir, "image")
    os.makedirs(images_dir, exist_ok=True)
    
    # Download command
    cmd = [
        "wget", "-r", "-N", "-c", "-np",
        "--user", username,
        "--password", password,
        "--directory-prefix", images_dir,
        "https://physionet.org/files/mimic-cxr-jpg/2.0.0/"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("✅ MIMIC-CXR images downloaded successfully")
        
        # Reorganize files
        reorganize_mimic_images(images_dir)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error downloading images: {e}")
        return False
    
    return True

def download_mimic_cxr_reports(username, password, output_dir):
    """Download MIMIC-CXR reports using wget"""
    
    print("🔄 Downloading MIMIC-CXR Reports (195MB)...")
    
    # Create output directory
    reports_dir = os.path.join(output_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    
    # Download command
    cmd = [
        "wget", "-r", "-N", "-c", "-np",
        "--user", username,
        "--password", password,
        "--directory-prefix", reports_dir,
        "https://physionet.org/files/mimic-cxr/2.0.0/"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("✅ MIMIC-CXR reports downloaded successfully")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error downloading reports: {e}")
        return False
    
    return True

def download_xraygpt_annotations(output_dir):
    """Download XRayGPT preprocessed annotations"""
    
    print("🔄 Downloading XRayGPT preprocessed annotations...")
    
    # XRayGPT team's preprocessed annotations
    annotation_url = "https://mbzuaiac-my.sharepoint.com/:u:/g/personal/omkar_thawakar_mbzuai_ac_ae/EZ6500itBIVMnD7sUztdMQMBVWVe7fuF7ta4FV78hpGSwg?e=wyL7Z7"
    
    cmd = [
        "wget", "-O", 
        os.path.join(output_dir, "filter_cap.json"),
        annotation_url
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("✅ XRayGPT annotations downloaded successfully")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error downloading annotations: {e}")
        return False
    
    return True

def reorganize_mimic_images(images_dir):
    """Reorganize MIMIC images to XRayGPT expected structure"""
    
    print("🔄 Reorganizing image directory structure...")
    
    # Find downloaded files
    physionet_dir = os.path.join(images_dir, "physionet.org", "files", "mimic-cxr-jpg", "2.0.0", "files")
    
    if os.path.exists(physionet_dir):
        # Move files to correct location
        import shutil
        
        for item in os.listdir(physionet_dir):
            src = os.path.join(physionet_dir, item)
            dst = os.path.join(images_dir, item)
            
            if os.path.isdir(src):
                shutil.move(src, dst)
        
        # Clean up empty directories
        shutil.rmtree(os.path.join(images_dir, "physionet.org"))
        
        print("✅ Image directory reorganized")

def verify_download(output_dir):
    """Verify that all required files are downloaded"""
    
    print("🔍 Verifying download...")
    
    # Check for images
    image_dir = os.path.join(output_dir, "image")
    if not os.path.exists(image_dir):
        print("❌ Image directory not found")
        return False
    
    # Count patient directories (should be p10-p19)
    patient_dirs = [d for d in os.listdir(image_dir) if d.startswith('p') and os.path.isdir(os.path.join(image_dir, d))]
    print(f"📊 Found {len(patient_dirs)} patient directories")
    
    # Check for annotations
    annotation_file = os.path.join(output_dir, "filter_cap.json")
    if not os.path.exists(annotation_file):
        print("❌ Annotation file not found")
        return False
    
    print("✅ Download verification successful")
    return True

def main():
    parser = argparse.ArgumentParser(description="Download MIMIC-CXR dataset for XRayGPT")
    parser.add_argument("--output-dir", default="./dataset/mimic", help="Output directory")
    parser.add_argument("--username", help="PhysioNet username")
    parser.add_argument("--skip-images", action="store_true", help="Skip image download")
    parser.add_argument("--skip-reports", action="store_true", help="Skip reports download")
    parser.add_argument("--annotations-only", action="store_true", help="Download only annotations")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.annotations_only:
        # Download only preprocessed annotations
        download_xraygpt_annotations(args.output_dir)
        return
    
    # Get credentials
    if not args.username:
        args.username = input("PhysioNet Username: ")
    
    password = getpass.getpass("PhysioNet Password: ")
    
    print(f"📁 Output directory: {args.output_dir}")
    print(f"👤 Username: {args.username}")
    
    # Download datasets
    success = True
    
    if not args.skip_images:
        success &= download_mimic_cxr_images(args.username, password, args.output_dir)
    
    if not args.skip_reports:
        success &= download_mimic_cxr_reports(args.username, password, args.output_dir)
    
    # Always download annotations
    success &= download_xraygpt_annotations(args.output_dir)
    
    if success:
        verify_download(args.output_dir)
        print("🎉 MIMIC-CXR dataset download completed!")
        print(f"📁 Dataset location: {args.output_dir}")
        print("🚀 Ready for XRayGPT training!")
    else:
        print("❌ Download failed. Please check your credentials and try again.")

if __name__ == "__main__":
    main()
```

### **2. Alternative Dataset Download Script**
**File**: `scripts/download_alternative_datasets.py`

```python
#!/usr/bin/env python3
"""
Alternative Dataset Download Script for XRayGPT
Downloads publicly accessible datasets as MIMIC-CXR alternatives
"""

import os
import subprocess
import json
import pandas as pd
from pathlib import Path

def download_nih_dataset(output_dir):
    """Download NIH Chest X-ray dataset via Kaggle"""
    
    print("🔄 Downloading NIH Chest X-ray Dataset...")
    
    # Create output directory
    nih_dir = os.path.join(output_dir, "nih")
    os.makedirs(nih_dir, exist_ok=True)
    
    # Download via Kaggle API
    cmd = [
        "kaggle", "datasets", "download", 
        "-d", "nih-chest-xrays/data",
        "-p", nih_dir,
        "--unzip"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("✅ NIH dataset downloaded successfully")
        
        # Convert to XRayGPT format
        convert_nih_to_xraygpt_format(nih_dir)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error downloading NIH dataset: {e}")
        print("💡 Make sure Kaggle API is configured: pip install kaggle")
        return False
    
    return True

def download_openi_dataset(output_dir):
    """Download OpenI dataset"""
    
    print("🔄 Downloading OpenI Dataset...")
    
    # Create output directory
    openi_dir = os.path.join(output_dir, "openi")
    os.makedirs(openi_dir, exist_ok=True)
    
    # Download images
    images_cmd = [
        "wget", "-O", 
        os.path.join(openi_dir, "NLMCXR_png.tgz"),
        "https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz"
    ]
    
    # Download reports
    reports_cmd = [
        "wget", "-O",
        os.path.join(openi_dir, "NLMCXR_reports.tgz"),
        "https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz"
    ]
    
    try:
        # Download files
        subprocess.run(images_cmd, check=True)
        subprocess.run(reports_cmd, check=True)
        
        # Extract files
        subprocess.run(["tar", "-xzf", "NLMCXR_png.tgz"], cwd=openi_dir, check=True)
        subprocess.run(["tar", "-xzf", "NLMCXR_reports.tgz"], cwd=openi_dir, check=True)
        
        print("✅ OpenI dataset downloaded successfully")
        
        # Download XRayGPT preprocessed annotations
        annotation_cmd = [
            "wget", "-O",
            os.path.join(openi_dir, "filter_cap.json"),
            "https://mbzuaiac-my.sharepoint.com/:u:/g/personal/omkar_thawakar_mbzuai_ac_ae/EVYGprPyzdhOjFlQ2aNJbykBj49SwTGBYmC1uJ7TMswaVQ?e=qdqS8U"
        ]
        subprocess.run(annotation_cmd, check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error downloading OpenI dataset: {e}")
        return False
    
    return True

def convert_nih_to_xraygpt_format(nih_dir):
    """Convert NIH dataset to XRayGPT annotation format"""
    
    print("🔄 Converting NIH dataset to XRayGPT format...")
    
    # Load NIH metadata
    metadata_file = os.path.join(nih_dir, "Data_Entry_2017.csv")
    if not os.path.exists(metadata_file):
        print("❌ NIH metadata file not found")
        return False
    
    df = pd.read_csv(metadata_file)
    
    # Create XRayGPT format annotations
    annotations = []
    
    for idx, row in df.iterrows():
        # Create medical caption from findings
        findings = row['Finding Labels']
        if findings == 'No Finding':
            caption = "No acute cardiopulmonary abnormalities detected."
        else:
            diseases = findings.split('|')
            caption = f"Chest X-ray shows evidence of {', '.join(diseases).lower()}."
        
        annotations.append({
            'image_id': row['Image Index'].replace('.png', ''),
            'caption': caption,
            'finding_labels': findings,
            'patient_age': row['Patient Age'],
            'patient_gender': row['Patient Gender'],
            'view_position': row['View Position']
        })
    
    # Save in XRayGPT format
    output_file = os.path.join(nih_dir, "filter_cap.json")
    with open(output_file, 'w') as f:
        json.dump(annotations, f, indent=2)
    
    print(f"✅ Converted {len(annotations)} images to XRayGPT format")
    return True

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Download alternative datasets for XRayGPT")
    parser.add_argument("--output-dir", default="./dataset", help="Output directory")
    parser.add_argument("--dataset", choices=["nih", "openi", "all"], default="all", help="Dataset to download")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"📁 Output directory: {args.output_dir}")
    
    success = True
    
    if args.dataset in ["nih", "all"]:
        success &= download_nih_dataset(args.output_dir)
    
    if args.dataset in ["openi", "all"]:
        success &= download_openi_dataset(args.output_dir)
    
    if success:
        print("🎉 Dataset download completed!")
        print("🚀 Ready for XRayGPT training!")
    else:
        print("❌ Download failed. Please check requirements and try again.")

if __name__ == "__main__":
    main()
```

### **3. Dataset Setup Script**
**File**: `scripts/setup_dataset.py`

```python
#!/usr/bin/env python3
"""
Dataset Setup Script for XRayGPT
Configures downloaded datasets for training
"""

import os
import yaml
import json
from pathlib import Path

def update_dataset_config(dataset_name, dataset_path):
    """Update XRayGPT dataset configuration"""
    
    config_file = f"xraygpt/configs/datasets/{dataset_name}/defaults.yaml"
    
    # Create config directory if it doesn't exist
    os.makedirs(os.path.dirname(config_file), exist_ok=True)
    
    # Create configuration
    config = {
        'datasets': {
            dataset_name: {
                'data_type': 'images',
                'build_info': {
                    'storage': dataset_path
                }
            }
        }
    }
    
    # Write configuration
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"✅ Updated configuration: {config_file}")

def verify_dataset_structure(dataset_path, dataset_name):
    """Verify dataset structure is correct for XRayGPT"""
    
    print(f"🔍 Verifying {dataset_name} dataset structure...")
    
    # Check for required files
    required_files = {
        'filter_cap.json': 'Annotation file',
        'image': 'Image directory'
    }
    
    for file_name, description in required_files.items():
        file_path = os.path.join(dataset_path, file_name)
        if os.path.exists(file_path):
            print(f"✅ {description}: {file_path}")
        else:
            print(f"❌ {description}: {file_path} - NOT FOUND")
            return False
    
    # Count annotations
    annotation_file = os.path.join(dataset_path, 'filter_cap.json')
    try:
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)
        print(f"📊 Found {len(annotations)} annotations")
    except Exception as e:
        print(f"❌ Error reading annotations: {e}")
        return False
    
    # Count images
    image_dir = os.path.join(dataset_path, 'image')
    if os.path.isdir(image_dir):
        image_count = sum([len(files) for r, d, files in os.walk(image_dir)])
        print(f"📊 Found {image_count} images")
    
    print(f"✅ {dataset_name} dataset structure verified")
    return True

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup datasets for XRayGPT")
    parser.add_argument("--dataset-path", required=True, help="Path to dataset")
    parser.add_argument("--dataset-name", required=True, choices=["mimic", "nih", "openi"], help="Dataset name")
    
    args = parser.parse_args()
    
    # Verify dataset structure
    if verify_dataset_structure(args.dataset_path, args.dataset_name):
        # Update configuration
        update_dataset_config(args.dataset_name, args.dataset_path)
        print(f"🎉 {args.dataset_name} dataset setup completed!")
        print(f"🚀 Ready for training with: {args.dataset_path}")
    else:
        print(f"❌ {args.dataset_name} dataset setup failed")

if __name__ == "__main__":
    main()
```

---

## 🚀 **USAGE EXAMPLES**

### **1. Download MIMIC-CXR (with credentials)**
```bash
# Make script executable
chmod +x scripts/download_mimic_cxr.py

# Download complete MIMIC-CXR dataset
python scripts/download_mimic_cxr.py --output-dir ./dataset/mimic --username YOUR_USERNAME

# Download only annotations (if you have images)
python scripts/download_mimic_cxr.py --annotations-only --output-dir ./dataset/mimic
```

### **2. Download Alternative Datasets**
```bash
# Download NIH dataset
python scripts/download_alternative_datasets.py --dataset nih --output-dir ./dataset

# Download OpenI dataset
python scripts/download_alternative_datasets.py --dataset openi --output-dir ./dataset

# Download all alternative datasets
python scripts/download_alternative_datasets.py --dataset all --output-dir ./dataset
```

### **3. Setup Dataset for Training**
```bash
# Setup MIMIC dataset
python scripts/setup_dataset.py --dataset-path ./dataset/mimic --dataset-name mimic

# Setup NIH dataset
python scripts/setup_dataset.py --dataset-path ./dataset/nih --dataset-name nih
```

---

## 📋 **INTEGRATION WITH XRAYGPT**

### **Add to XRayGPT Repository**
```bash
# Create scripts directory
mkdir -p scripts

# Add the download scripts
# Copy the scripts above to scripts/ directory

# Make executable
chmod +x scripts/*.py

# Update .gitignore to exclude downloaded data
echo "dataset/" >> .gitignore
```

### **Update XRayGPT Builder**
```python
# Modify: xraygpt/datasets/builders/image_text_pair_builder.py

def _download_ann(self):
    """Download annotations using our script"""
    if not os.path.exists(os.path.join(self.storage_path, 'filter_cap.json')):
        print("Running annotation download...")
        subprocess.run([
            "python", "scripts/download_mimic_cxr.py", 
            "--annotations-only", 
            "--output-dir", self.storage_path
        ])

def _download_vis(self):
    """Download images using our script"""
    if not os.path.exists(os.path.join(self.storage_path, 'image')):
        print("Images not found. Please run download script manually.")
        print(f"python scripts/download_mimic_cxr.py --output-dir {self.storage_path}")
```

**These scripts provide the code-based approach to download datasets for XRayGPT, handling both MIMIC-CXR (with credentials) and alternative public datasets!**