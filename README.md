## 기말고사 과제

- Python program (*.py)과 MS Word 파일로 작성한 보고서는 하나의 zip file로 만들어 학습관리시스템에 올려야 합니다.

- Evaluation Rubric을 잘 읽고 평가 방법을 숙지 후 과제를 해야 합니다.
- 마감: 12.19 (목) 18:00

# Final Project – Image Classification with CNNs

Python program (*.py) and a written report (MS Word file) must be compressed into **one ZIP file** and submitted to the LMS.

Read the **Evaluation Rubric** carefully before starting the assignment.

**Deadline: December 19 (Thu), 18:00**


---

## Objective  
Implement a Convolutional Neural Network (CNN) in **PyTorch** to classify images from the **CIFAR-10** dataset.

---

## Tasks

### **1. Data Preprocessing**
- Load the CIFAR-10 dataset using `torchvision.datasets`.
- Apply data augmentation:
  - Random cropping  
  - Horizontal flipping  
  - Normalization  

### **2. Model Implementation**
- Build a CNN with:
  - **At least three convolutional layers**
  - Max-pooling layers  
  - Fully connected layers  
- Include **Dropout** to reduce overfitting.

### **3. Training**
- Train the model using:
  - **CrossEntropyLoss**
  - **Adam optimizer**
- Plot:
  - Training accuracy / loss  
  - Validation accuracy / loss  

### **4. Evaluation**
- Evaluate performance on the test dataset:
  - Accuracy  
  - Precision  
  - Recall  
  - F1-score  
- Save the trained model for later use.

---

## Deliverables

- **Python script** containing your full implementation  
- Code must include **clear comments** explaining your logic  
- **Written report** (MS Word) describing:
  - Model architecture  
  - Training process  
  - Evaluation results  


---

# Evaluation Rubric

| **Criteria** | **Excellent (90–100%)** | **Good (75–89%)** | **Fair (50–74%)** | **Poor (0–49%)** |
|--------------|--------------------------|--------------------|--------------------|-------------------|
| **Code Functionality (30%)** | Code is efficient, well-documented, runs without errors; meets all requirements. | Code functional with minor inefficiencies or missing comments; meets most requirements. | Partially functional; several inefficiencies; meets some requirements. | Non-functional or missing requirements. |
| **Model Performance (25%)** | Achieves excellent accuracy or metrics; robust evaluation. | Achieves good accuracy/metrics with minor issues; adequate evaluation. | Subpar performance; evaluation lacks depth. | Performs poorly with little or no evaluation. |
| **Report Quality (25%)** | Clear, concise, well-organized; insightful analysis with visuals. | Clear and organized but lacking some depth or insights. | Unclear or poorly organized; lacks analysis. | Incomplete, unclear, or missing key elements. |
| **Creativity & Effort (20%)** | Shows innovation or extra effort (e.g., additional experiments). | Reasonable effort with some additional work. | Minimal effort; basic implementation only. | Little to no effort; no attempt beyond basics. |

---

## Summary

You must implement a full CNN classification pipeline:
- dataset loading  
- augmentation  
- CNN building  
- training visualization  
- evaluation  
- saving model  

Then submit:
- **project.py**
- **project_report.docx**
- inside **one ZIP file**
