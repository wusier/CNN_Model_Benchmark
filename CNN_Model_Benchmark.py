import os
import shutil
import time
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224
LOAD_SIZE = 256


def create_dummy_imagenet_subset(base_path="dummy_imagenet_val", num_classes=5, images_per_class=20):
    """创建一个小型的、类似ImageNet验证集结构的虚拟数据集"""
    val_path = base_path
    if os.path.exists(val_path):
        print(f"Cleaning up existing dummy dataset at {val_path}...")
        shutil.rmtree(val_path)
    os.makedirs(val_path, exist_ok=True)

    for i in range(num_classes):
        class_name = f"n{i:08d}" 
        class_path = os.path.join(val_path, class_name)
        os.makedirs(class_path, exist_ok=True)
        for j in range(images_per_class):
            random_array = np.random.randint(0, 256, size=(IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            img = Image.fromarray(random_array)
            img.save(os.path.join(class_path, f"dummy_img_{class_name}_{j:03d}.JPEG"))
    print(f"Dummy dataset created at {val_path} with {num_classes} classes and {images_per_class} images per class.")
    return val_path

def get_model_size_mb(model):
    """计算模型大小 (MB)"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb

def add_gaussian_noise(image_tensor, mean=0., std=0.1):
    """向图像张量添加高斯噪声"""
    # image_tensor is expected to be normalized
    # We add noise in the [0,1] range and then re-normalize, or add to normalized directly
    # For simplicity, let's assume image_tensor is in [0,1] before normalization for noise addition
    # However, model expects normalized input. So, noise should be scaled appropriately or added to unnormalized.
    # A common way: add noise to normalized image, then clip to a reasonable range if needed.
    # The std here should be relative to the scale of the normalized values.
    
    # Create noise with the same shape as the image tensor
    noise = torch.randn_like(image_tensor) * std + mean
    noisy_image = image_tensor + noise
    # It's often good to clip, but since inputs are normalized around 0,
    # excessive clipping might not be necessary unless std is very high.
    # For ImageNet normalized values, range is roughly [-2, 2].
    # noisy_image = torch.clamp(noisy_image, min_val_for_normalized, max_val_for_normalized)
    return noisy_image


def calculate_accuracy(output, target, topk=(1,)):
    """计算 Top-k 准确率"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k_val in topk:
            correct_k = correct[:k_val].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res


normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
val_transform = transforms.Compose([
    transforms.Resize(LOAD_SIZE),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    normalize,
])
use_real_imagenet = False
real_imagenet_val_path = "/1/ImageNetr/imagenet/val" 

if use_real_imagenet and os.path.exists(real_imagenet_val_path):
    val_dataset_path = real_imagenet_val_path
    print(f"Using REAL ImageNet validation set from: {val_dataset_path}")
else:
    print("Real ImageNet path not found or not specified. Creating and using a DUMMY dataset.")
    val_dataset_path = create_dummy_imagenet_subset(num_classes=10, images_per_class=50) # 10 classes, 50 images/class = 500 images

val_dataset = datasets.ImageFolder(val_dataset_path, val_transform)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=32, shuffle=False, 
    num_workers=4, pin_memory=True
)
print(f"Validation dataset: {len(val_dataset)} images, {len(val_dataset.classes)} classes.")


def evaluate_model(model, model_name, data_loader, device, noise_std=0.0):
    model.eval()
    model.to(device)

    total_top1_acc = 0
    total_top5_acc = 0
    total_inference_time = 0
    num_samples = 0
    num_batches = 0

    with torch.no_grad():
        for i, (images, target) in enumerate(data_loader):
            images = images.to(device)
            target = target.to(device)

            if noise_std > 0:
                images = add_gaussian_noise(images, std=noise_std) # std relative to normalized values

            start_time = time.time()
            output = model(images)
            end_time = time.time()

            total_inference_time += (end_time - start_time)
            
            top1_acc, top5_acc = calculate_accuracy(output, target, topk=(1, 5))
            total_top1_acc += top1_acc * images.size(0)
            total_top5_acc += top5_acc * images.size(0)
            
            num_samples += images.size(0)
            num_batches += 1

    avg_top1_acc = total_top1_acc / num_samples
    avg_top5_acc = total_top5_acc / num_samples
    avg_inference_time_ms_per_image = (total_inference_time / num_samples) * 1000

    print(f"Model: {model_name}, Noise Std: {noise_std:.2f}")
    print(f"  Avg Top-1 Acc: {avg_top1_acc:.2f}%")
    print(f"  Avg Top-5 Acc: {avg_top5_acc:.2f}%")
    print(f"  Avg Inference Time: {avg_inference_time_ms_per_image:.4f} ms/image")
    
    return avg_top1_acc, avg_top5_acc, avg_inference_time_ms_per_image

model_names_to_test = ["VGG16", "ResNet50", "MobileNetV2"]
noise_levels = [0.0, 0.05, 0.1, 0.15, 0.2] 

results = {model_name: {"clean": {}, "noisy": {nl: {} for nl in noise_levels if nl > 0}}
           for model_name in model_names_to_test}
model_sizes = {}

for model_name in model_names_to_test:
    print(f"\n--- Evaluating {model_name} ---")
    if model_name == "VGG16":
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    elif model_name == "ResNet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2) # V2 is generally better
    elif model_name == "MobileNetV2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    model_sizes[model_name] = get_model_size_mb(model)
    print(f"Model Size ({model_name}): {model_sizes[model_name]:.2f} MB")

    top1, top5, inf_time = evaluate_model(model, model_name, val_loader, DEVICE, noise_std=0.0)
    results[model_name]["clean"]["top1"] = top1
    results[model_name]["clean"]["top5"] = top5
    results[model_name]["clean"]["time"] = inf_time
    robustness_top1 = []
    for noise_std in noise_levels: # Iterate through all including 0.0 for the line plot
        if noise_std == 0.0: # Already have clean results
            robustness_top1.append(results[model_name]["clean"]["top1"])
            continue
        top1_noisy, _, _ = evaluate_model(model, model_name, val_loader, DEVICE, noise_std=noise_std)
        results[model_name]["noisy"][noise_std]["top1"] = top1_noisy
        robustness_top1.append(top1_noisy)
    results[model_name]["robustness_top1_trend"] = robustness_top1


labels = model_names_to_test
clean_top1_accs = [results[m]["clean"]["top1"] for m in labels]
clean_top5_accs = [results[m]["clean"]["top5"] for m in labels]

x = np.arange(len(labels))
width = 0.35

fig1, ax1 = plt.subplots(figsize=(10, 6))
rects1 = ax1.bar(x - width/2, clean_top1_accs, width, label='Top-1 Accuracy')
rects2 = ax1.bar(x + width/2, clean_top5_accs, width, label='Top-5 Accuracy')

ax1.set_ylabel('Accuracy (%)')
ax1.set_title('Model Accuracy on Clean Data (Dummy/Subset ImageNet)')
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.legend()
ax1.bar_label(rects1, padding=3, fmt='%.1f')
ax1.bar_label(rects2, padding=3, fmt='%.1f')
fig1.tight_layout()
plt.savefig("accuracy_comparison_clean.png")
print("\nSaved accuracy_comparison_clean.png")

fig2, ax2 = plt.subplots(figsize=(10, 6))
for model_name in model_names_to_test:
    ax2.plot(noise_levels, results[model_name]["robustness_top1_trend"], marker='o', label=model_name)

ax2.set_xlabel('Gaussian Noise Standard Deviation')
ax2.set_ylabel('Top-1 Accuracy (%)')
ax2.set_title('Model Robustness to Gaussian Noise (Top-1 Accuracy)')
ax2.legend()
ax2.grid(True)
plt.savefig("robustness_to_noise.png")
print("Saved robustness_to_noise.png")


print("\n--- Model Summary ---")
print(f"{'Model':<15} | {'Size (MB)':<10} | {'Inference Time (ms/img)':<25}")
print("-" * 60)
for model_name in model_names_to_test:
    size = model_sizes[model_name]
    inf_time = results[model_name]["clean"]["time"] # Time on clean data
    print(f"{model_name:<15} | {size:<10.2f} | {inf_time:<25.4f}")

print("\nNote: Accuracy results are on the DUMMY/SUBSET dataset and will differ from actual ImageNet results.")
print("The purpose is to demonstrate the experimental procedure.")