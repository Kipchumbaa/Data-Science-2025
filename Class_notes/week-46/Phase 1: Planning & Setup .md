# Phase 1: Planning & Setup - Fake News Detection Capstone Project

**Instructor:** DataMan7 (Lead Instructor) & Co-Instructor  
**Date:** November 13, 2025  
**Duration:** 1 Hour  
**Objective:** Kick off the Fake News Detection project, assign roles, set up infrastructure, and ensure all teams are aligned for successful delivery by November 28.

---

## Agenda (60 Minutes)
1. **Introduction (10 min)**: Project Overview & Objectives
2. **Team Formation & Roles (10 min)**: Group Assignments & Responsibilities
3. **Timeline & Milestones (10 min)**: Compressed 2-Week Schedule
4. **Infrastructure & Resources (15 min)**: AWS Setup, GitHub, Tools
5. **Assignments & Handover (10 min)**: Initial Tasks & Support
6. **Q&A & Next Steps (5 min)**: Open Discussion

---

## 1. Introduction (10 min)
### Project Overview
- **Challenge**: Fake news spreads rapidly on social media, impacting elections, health, and society. Your task: Build ML classifiers to detect fake news using SVM, Random Forest, and Logistic Regression.
- **Tech Stack**: sklearn (CPU) vs. RAPIDS cuML (GPU) for performance comparison. Focus on text preprocessing (TF-IDF, stemming, stop words) and scalable pipelines.
- **Why This Matters**: Demonstrates real-world DS skills—data engineering, ML, deployment. Aligns with senior DS roles (e.g., J&J, Yassir) requiring collaboration, scalability, and innovation.
- **Success Criteria**: Accurate models (>85% accuracy), GPU speedup (>2x), deployable API, Nov 28 presentation.

### Key Modules Leveraged
- **Scalable Computing (10-13, 21)**: RAPIDS for GPU acceleration.
- **ML Classification (14)**: Model training & evaluation.
- **Text Analytics (20)**: Preprocessing pipelines.
- **Team Dynamics (22)**: Collaboration best practices.
- **Version Control (23)**: Git workflows.

---

## 2. Team Formation & Roles (10 min)
### Groups (4 Students Each)
- **Group A**: John (PM), Elsa (Data Analyst), Bismarck (DS/ML Eng), Teddy (QA)
- **Group B**: Kigen (PM), Lamech (Data Analyst), Nehemiah (ML Eng/DS), Wilberforce (QA)

### Role Responsibilities
- **Project Manager (John, Kigen)**: Sprint planning, timelines, stakeholder updates, risk management.
- **Data Analyst (Elsa, Lamech)**: EDA, preprocessing, feature engineering, data validation.
- **ML Engineer/Data Scientist (Bismarck, Nehemiah)**: Model building, tuning, GPU optimization, experimentation.
- **Quality Assurance (Teddy, Wilberforce)**: Testing, code reviews, compliance, performance audits.
- **My Roles (DataMan7)**: Data Engineering (pipelines, AWS), Overall Lead, QA Oversight.

### Collaboration
- Shared Slack/Teams channel for daily updates.
- Weekly check-ins; PRs for code reviews.

---

## 3. Timeline & Milestones (10 min)
### Compressed Schedule (Nov 13-28)
- **Phase 1: Planning & Setup (Nov 13-15)**: Roles, infrastructure, data acquisition.
- **Phase 2: Data Acquisition & Processing (Nov 16-18)**: Preprocessing, TF-IDF, train/test split.
- **Phase 3: Model Development & Experimentation (Nov 19-22)**: Train 6 models (sklearn/cuML), compare times/accuracy.
- **Phase 4: Model Deployment & Production (Nov 23-25)**: Build FastAPI, monitoring, A/B testing.
- **Phase 5: Evaluation & Presentation (Nov 26-28)**: Final testing, demo prep, Nov 28 presentation.

### Milestones
- Nov 15: AWS access granted, repos set up.
- Nov 18: Preprocessed data ready.
- Nov 22: Models trained, performance report.
- Nov 25: Deployed API.
- Nov 28: Live demo & Q&A.

---

## 4. Infrastructure & Resources (15 min)
### AWS EC2 Setup
- **Instance**: t3.medium (CPU-only) with RAPIDS CPU acceleration. GPU upgrade available via limit increase request.
- **Access Requirements**:
  - Provide Gmail emails tonight for IAM user creation.
  - Enable MFA on your Google accounts (via app/SMS).
  - OS/Device details for tailored guides.
- **Connection Demo**:
  1. Install AWS CLI: `pip install awscli`.
  2. Configure: `aws configure` (use provided keys).
  3. SSH: `ssh -i key.pem ubuntu@<ip>`.
  4. VS Code: Remote SSH extension for Jupyter dev.
  5. Activate RAPIDS: `conda activate rapids`.
- **Cost Management**: Spot instances; monitor via AWS console. Budget alerts set.

### GitHub & Tools
- **Repos**: `fake-news-groupA` & `fake-news-groupB` (private, CI/CD enabled).
- **Version Control**: Git branches (main/feature), PRs for merges.
- **Tools**: Jupyter Lab, scikit-learn, cuML, FastAPI, MLflow.

### Resources Provided
- Dataset: Kaggle Fake News (train.csv).
- Templates: Preprocessing scripts, model notebooks.
- Docs: Modules 14, 20, 21 guides.
- Support: Chat channel; 2-hour response time.

---

## 5. Assignments & Handover (10 min)
### Group A Tasks
- John: Create project plan, set up GitHub repo , Coordinate with me for AWS access.
- Elsa: Download dataset, perform initial EDA, Data cleaning scripts .
- Bismarck: Set up RAPIDS environment, baseline sklearn models, cuML model prototypes.
- Teddy: Develop test cases for preprocessing, QA checklist for data quality

### Group B Tasks
- Kigen: Coordinate with me for AWS access,Create project plan, set up GitHub repo
- Lamech: Data cleaning scripts , Download dataset, perform initial EDA.
- Nehemiah: cuML model prototypes, Set up RAPIDS environment, baseline sklearn models.
- Wilberforce: QA checklist for data quality,Develop test cases for preprocessing

### Handover
- Resources emailed post-meeting.
- Training session: 15-min AWS/RAPIDS demo tonight.

---

## 6. Q&A & Next Steps (5 min)
- Questions?
- Next: Phase 2 starts Nov 16; daily stand-ups.
- Contact: Chat or email for blockers.

**Remember**: This project builds senior DS skills—collaboration, scalability, innovation. Let's deliver excellence!

### Confirmation of Setup Completion
Based on the output, the setup is **90% complete**:
- [SUCCESS] SSH connection successful (`ec2-user@54.161.115.204`).
- [SUCCESS] Miniconda installed.
- [SUCCESS] Jupyter installed and running (token: `6952986a40885f4f9b75b0c58debe4c85355c80608a5f468`).
- [WARNING] RAPIDS install in progress (cudf not found yet; rerun `conda install -c rapidsai -c nvidia -c conda-forge rapids=24.10 python=3.10` to complete).
- [SUCCESS] Demo ready for tonight (SSH, Jupyter working; RAPIDS for Phase 2).

### Assignments & Handover Details
**Who & When Needs EC2 Access**:
- **Students**: Bismarck, Nehemiah (ML Engineers) for GPU model training (Phases 2-4: Nov 16-25).
- **When**: Access granted Nov 15 post-meeting; revoked Nov 28.
- **What They Need to Provide**:
  - Gmail email for IAM user creation.
  - MFA enabled on Google account (app/SMS).
  - OS/Device details (Windows/Mac/Linux, VS Code installed).
- **How to Access & Build Projects**:
  1. Receive IAM credentials via email.
  2. Install AWS CLI: `pip install awscli`.
  3. Configure: `aws configure` (use provided keys).
  4. SSH: `ssh -i fake-news-key.pem ec2-user@54.161.115.204`.
  5. Clone repo: `git clone https://github.com/Eldohub-data-scientists/fake-news-groupA.git`.
  6. Build: Run notebooks/scripts on instance (e.g., `jupyter lab` for dev).
  7. Push changes: `git add . && git commit -m "Update model" && git push`.

### Updated Planning & Setup .md
Updated with new EC2 details (IP 54.161.115.204, user ec2-user, key fake-news-key.pem).

### Student Version of Planning & Setup.md
Created and saved as `Phase 1: Planning & Setup - Student Version.md` in the same location. It simplifies the content for students, focuses on their roles/tasks, and includes access instructions.