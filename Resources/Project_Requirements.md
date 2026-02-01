# Project Description

## Aim of Project
- Explore an interesting machine learning problem  
- Gain machine learning development experience  
- See applications of theory covered in this course  
- Read published machine learning papers  

---

# Project Requirements

## Paper Selection Constraints
- Select a published paper on machine learning  
- Must be published in **2018 or later**  
- Must be from a **Q1 journal** or a **top-tier conference**  
- Examples of Q1 journals (list not exhaustive):
  - JMLR  
  - TPAMI  
  - IJCV  
- Acceptable conferences include, but are not limited to:
  - **General ML / AI**: NeurIPS, ICML, ICLR, AAAI, IJCAI  
  - **Computer Vision**: CVPR, ICCV, ECCV  
- CORE **A** or **A\*** venues are also acceptable  
- Venue rankings can be checked at:  
  https://portal.core.edu.au/conf-ranks/  
- **arXiv papers do not count**

---

## Implementation Constraints

### Domain Shift
- The model must be applied to a **completely different domain**
- Example:
  - If the paper focuses on natural image segmentation, apply it to medical imaging, satellite imagery, etc.

### Performance Audit
- Final submission must **not** just show that the model works  
- You must present:
  - A **sensitivity analysis**, or  
  - An **ablation study** showing what happens when the model breaks  

---

# Project Milestones

## Initial Project Proposal
- **Purpose**: Provide early feedback on the suitability and feasibility of the proposed paper + dataset pairing  
- Students will pitch their idea early in the course to ensure:
  - The concept meets the difficulty requirements  
  - The new dataset is compatible with the chosen architecture  
- **Presentation format**:
  - One slide  
  - 3 minutes in duration  
- Must clearly identify:
  - The specific paper (Title + Venue)  
  - A general description of the model  
  - The new dataset to be used  
  - The core technical challenge in adapting the model to the new dataset  
- **Presentation date**: January 22, 2026  
- **Submission**: Slides due on UM Learn by **5:00 PM, January 21, 2026**

---

## Full Project Proposal
- Students will present a **detailed technical proposal** via slides  
- This is a **gate presentation**
  - Used to verify careful planning and feasibility  
- **Logistics**:
  - Duration: **5–10 minutes**
  - Presentations will be stopped at the 10-minute mark  
  - **Q&A**:
    - Each student may be asked 1–2 individual questions to verify contribution and understanding  
- **Dates**: February 3 and February 5, 2026  
- Groups will be randomly selected at the start of class  
- **Submission**: Slides due on UM Learn by **5:00 PM, February 2, 2026**

---

# Project Milestones and Deliverables

## Full Project Proposal – Required Slide Content

### Title Slide
- Project name  
- Group member names  
- Course name  
- Instructor’s name  
- Date  

---

### Problem Domain & Dataset
- Introduce the specific problem being solved  
- Dataset description (must be detailed):
  - Dataset size  
  - Resolution  
  - File format  
  - Class distribution  
- You must explicitly confirm that this dataset is **different** from the one used in the original paper  

---

### Source Paper & Methodology
- Identify the paper:
  - Title  
  - Venue (2018+)  
- Architecture diagram:
  - High-level diagram of the model to be implemented  

---

### Adaptation Strategy
- Explain the **specific technical changes** required to adapt the architecture to the new dataset  
- Examples:
  - Changing input channels from 3 (RGB) to 5 (multispectral)  
  - Rewriting the data loader to handle 3D volumes instead of 2D images  

---

### Development Plan
- **Environment**:
  - Framework (e.g., PyTorch)  
  - Hardware resources (e.g., department GPUs, Google Colab)  
- **Ablation Plan**:
  - At least one ablation study
  - Example:
    - Removing a specific module  
    - Modifying a hyperparameter to test robustness  

---

### Project Management
- **Gantt Chart** with clear milestones  
- Example milestones:
  - Data pipeline working  
  - Baseline training  
  - Final analysis  

---

### Role Division
- Clearly state which student is responsible for each technical component  

---

### EDI Considerations
- Discuss **Equity, Diversity, and Inclusion** considerations relevant to the dataset or problem  
- Examples:
  - Bias in training data  
  - Accessibility concerns  
- Resource:  
  https://nserc-crsng.canada.ca/en/nserc-guide-integrating-equity-diversity-and-inclusion-considerations-research  

---

### References
- Proper citations for:
  - Source paper  
  - Dataset  
  - Any other referenced material  
- Must be formatted using **IEEE citation style**
