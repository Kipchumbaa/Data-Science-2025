# Phase 1: Planning & Setup - Fake News Detection Capstone Project (Student Version)

**Instructor:** DataMan7 (Lead Instructor)  
**Date:** November 13, 2025  
**Duration:** 1 Hour  
**Objective:** Kick off the Fake News Detection project, understand your roles, and get set up for success by November 28.

---

## Agenda (60 Minutes)
1. **Introduction (10 min)**: Project Overview & Objectives
2. **Team Formation & Roles (10 min)**: Your Group & Responsibilities
3. **Timeline & Milestones (10 min)**: 2-Week Schedule
4. **Infrastructure & Resources (15 min)**: AWS Setup, GitHub, Tools
5. **Assignments & Handover (10 min)**: Your Initial Tasks
6. **Q&A & Next Steps (5 min)**: Questions & Support

---

## 1. Introduction (10 min)
### Project Overview
- **Challenge**: Build ML classifiers (SVM, Random Forest, Logistic Regression) to detect fake news using sklearn and RAPIDS cuML.
- **Tech Stack**: Compare CPU (sklearn) vs. GPU (cuML) performance. Focus on text preprocessing (TF-IDF, stemming).
- **Success Criteria**: Accurate models (>85%), GPU speedup (>2x), deployable API, Nov 28 presentation.

### Key Modules
- Scalable Computing (RAPIDS for GPU).
- ML Classification & Text Analytics.
- Team Dynamics & Version Control.

---

## 2. Team Formation & Roles (10 min)
### Your Groups
- **Group A**: John (PM), Elsa (Data Analyst), Bismarck (ML Eng), Teddy (QA)
- **Group B**: Kigen (PM), Lamech (Data Analyst), Nehemiah (ML Eng), Wilberforce (QA)

### Your Responsibilities
- **Project Manager**: Plan sprints, track progress.
- **Data Analyst**: EDA, preprocessing, data cleaning.
- **ML Engineer**: Build/tune models, GPU optimization.
- **QA**: Test code, ensure quality.

### Collaboration
- Slack/Teams channel for updates.
- Weekly check-ins; use PRs for code reviews.

---

## 3. Timeline & Milestones (10 min)
### Schedule (Nov 13-28)
- **Phase 1: Setup (Nov 13-15)**: Roles, infrastructure.
- **Phase 2: Data Processing (Nov 16-18)**: Preprocessing.
- **Phase 3: Model Dev (Nov 19-22)**: Training, comparison.
- **Phase 4: Deployment (Nov 23-25)**: API, monitoring.
- **Phase 5: Presentation (Nov 26-28)**: Demo & evaluation.

### Milestones
- Nov 15: AWS access, repos ready.
- Nov 18: Data preprocessed.
- Nov 22: Models trained.
- Nov 25: API deployed.
- Nov 28: Final presentation.

---

## 4. Infrastructure & Resources (15 min)
### AWS EC2 Setup
- **Instance**: t3.medium (CPU), IP 54.161.115.204, user ec2-user.
- **Access Requirements**:
  - Provide Gmail email tonight.
  - Enable MFA on Google account.
  - OS/Device details.
- **Connection**:
  1. Install AWS CLI: `pip install awscli`.
  2. Configure: `aws configure`.
  3. SSH: `ssh -i fake-news-key.pem ec2-user@54.161.115.204`.
  4. VS Code: Remote SSH extension.
  5. RAPIDS: `conda activate rapids` (install if needed).

### GitHub & Tools
- **Repos**: `fake-news-groupA` or `fake-news-groupB`.
- **Version Control**: Git branches, PRs.
- **Tools**: Jupyter Lab, sklearn, cuML, FastAPI.

### Resources
- Dataset: Kaggle Fake News.
- Templates: Preprocessing scripts.
- Support: Chat channel.

---

## 5. Assignments & Handover (10 min)
### Your Tasks
- **Group A**:
  - John: Project plan, GitHub repo setup.
  - Elsa: Dataset download, EDA.
  - Bismarck: RAPIDS setup, baseline models.
  - Teddy: Test cases.
- **Group B**:
  - Kigen: Project plan, GitHub repo setup.
  - Lamech: Dataset download, EDA.
  - Nehemiah: RAPIDS setup, baseline models.
  - Wilberforce: Test cases.

### Handover
- Resources emailed post-meeting.
- 15-min demo tonight.

---

## 6. Q&A & Next Steps (5 min)
- Questions?
- Next: Phase 2 Nov 16.
- Contact: Chat for help.

**Let's build something amazing!**