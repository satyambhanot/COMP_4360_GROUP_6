## Slide 4 – Development Plan & Ablation Studies

---

### Framework Options Considered

Before choosing PyTorch, we considered the following options:

---

### Option 1: TensorFlow

**Benefits**
- Simple high-level API
- Faster to prototype basic models
- Large number of built-in training utilities

**Drawbacks**
- Harder to modify internal layers once the model is built
- More difficult to bypass or remove individual components
- Ablation studies often require rewriting parts of the model
- Less flexible when working with custom data pipelines and intermediate outputs

**Why We Did Not Choose It**
- Our project requires multiple ablation studies
- We need to easily disable components such as the Sobel edge detector
- TensorFlow is less convenient for fine-grained architectural changes

---

### Option 2: PyTorch

**Benefits**
- Easy to modify and experiment with model structure
- Individual layers and components can be turned on or off
- Simpler to inspect intermediate outputs during training
- Better suited for controlled ablation studies
- Flexible data loading for multiple image resolutions and formats

**Drawbacks**
- Slightly more setup code compared to Keras
- Requires more manual handling of training loops

---

### Why PyTorch Was Chosen

- Our project focuses on **understanding model behavior**, not just training accuracy
- We need to:
  - Remove layers
  - Bypass the Sobel edge detector
  - Change input formats and resolutions
- PyTorch allows us to make these changes cleanly and safely
- The extra setup cost is acceptable given the flexibility it provides

---

### Final Decision

- PyTorch was selected because it best supports:
  - Model adaptation to a new domain
  - Controlled ablation studies
  - Clear analysis of failure cases


---

### Model Implementation Approach

- The model will be implemented in a modular way so that:
  - Individual components can be turned on or off
  - Changes can be made without rewriting the full model
- The overall CNN structure from the paper will be preserved, with adaptations for:
  - Smaller image sizes
  - Binary classification (Waldo vs Not Waldo)
- The data loader will support:
  - Multiple image resolutions (64×64, 128×128, 256×256)
  - Color, grayscale, and black-and-white images

---

### Training Setup

- The model will be trained using supervised learning
- A loss function will be selected to handle class imbalance
- Training and validation settings will remain consistent across experiments
- Each ablation study will modify only **one factor at a time**

---

### Hardware Resources

- **Primary hardware:** NVIDIA GTX 1080 Ti  
  - Sufficient memory for the dataset and model size  
  - Allows multiple training runs for ablation studies
- **Fallback options:**
  - Google Colab
  - University GPU resources (if required)
- This setup ensures hardware limitations do not prevent experimentation

---

## Ablation Studies

The purpose of the ablation studies is to identify **which components are important** and **where the model fails**.

---

### Ablation 1: Image Resolution

- Train and evaluate the model using:
  - 64×64 images
  - 128×128 images
  - 256×256 images
- This tests:
  - Sensitivity to the size of Waldo in the image
  - Dependence on fine visual details
- Why this is useful:
  - Waldo is often small and partially visible
- Failure case:
  - Increased false negatives at lower resolutions

---

### Ablation 2: Color Information

- Compare performance using:
  - Color images
  - Grayscale images
  - Black-and-white images
- This tests:
  - Dependence on color patterns
- Why this matters:
  - Waldo’s red and white stripes are distinctive
- Failure case:
  - Large performance drop when color is removed

---

### Ablation 3: Sobel Edge Detector Bypass

- Train the model with:
  - Sobel edge detector enabled
  - Sobel edge detector bypassed
- This tests:
  - Importance of edge information
- Why this is useful:
  - Edges highlight outlines and repeated patterns
- Failure case:
  - Reduced performance without edge information
  - Increased confusion with background objects

---

### Ablation 4: Model Size Reduction

- Reduce:
  - Number of convolutional layers
  - Number of filters per layer
- This tests:
  - Sensitivity to model capacity
- Why this matters:
  - Detecting small objects in cluttered scenes may require higher capacity
- Failure case:
  - Model predicts “Not Waldo” for most images

---

### Ablation 5: Class Imbalance Handling

- Train the model:
  - With class-weighted loss
  - Without class-weighted loss
- This tests:
  - Sensitivity to uneven class distribution
- Why this matters:
  - Most images do not contain Waldo
- Failure case:
  - Bias toward predicting “Not Waldo”

---

gantt
    title COMP 4360 Project Gantt Chart (Mid-February to End of March)
    dateFormat  YYYY-MM-DD
    axisFormat  %b %d

    section Setup & Data
    Environment setup                         :t1, 2026-02-10, 3d
    Dataset inspection & verification         :t2, 2026-02-10, 5d
    Dataset preprocessing pipeline            :t3, 2026-02-13, 6d
    Data loader implementation                :t4, 2026-02-15, 6d

    section Model & Baseline
    CNN architecture implementation           :t5, 2026-02-17, 7d
    Training loop & evaluation metrics        :t6, 2026-02-20, 6d
    Baseline model training                   :t7, 2026-02-24, 5d
    Baseline result analysis                  :t8, 2026-02-27, 4d

    section Ablation Studies
    Resolution ablation (64/128/256)          :t9, 2026-03-01, 7d
    Color vs grayscale vs binary              :t10, 2026-03-03, 7d
    Sobel edge detector bypass                :t11, 2026-03-06, 7d
    Model size reduction                      :t12, 2026-03-09, 7d
    Class imbalance handling                  :t13, 2026-03-12, 6d

    section Analysis & Finalization
    Aggregate ablation results                :t14, 2026-03-16, 5d
    Failure case analysis                     :t15, 2026-03-18, 6d
    Final evaluation & comparison             :t16, 2026-03-22, 5d
    Final report drafting                     :t17, 2026-03-24, 6d
    Final slide preparation                   :t18, 2026-03-27, 5d

