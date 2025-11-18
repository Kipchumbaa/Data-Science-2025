# Student Onboarding Guide: Fake News Detection Project

## Welcome to the Capstone Project!

Congratulations on reaching the final phase of your Data Science journey! This guide will help you get started with the Fake News Detection capstone project.

## Project Overview
- **Duration**: November 13-28 (2 weeks)
- **Objective**: Build ML classifiers to detect fake news using sklearn and RAPIDS cuML
- **Deliverables**: Working model, API, presentation, and documentation
- **Success Criteria**: >85% accuracy, GPU speedup, production-ready system

## Your Team Assignment

### Group A
- **Project Manager**: John (jnyangara303@gmail.com)
- **Data Analyst**: Elsa (elsernownex@gmail.com)
- **ML Engineer**: Bismarck (bismarkkoima844@gmail.com)
- **QA**: Teddy (amdanyteddy@gmail.com)

### Group B
- **Project Manager**: Kigen (kigenyego@gmail.com)
- **Data Analyst**: Lamech (lamechrop45@gmail.com)
- **ML Engineer**: Nehemiah (nehemiahkipchumba89@gmail.com)
- **QA**: Wilberforce (wilberforcekimutai68@gmail.com)

## Getting Started Checklist

### 1. Access Credentials (Due: November 13)
- [ ] Receive AWS access keys from instructor
- [ ] Receive SSH key (fake-news-key.pem) for EC2 access
- [ ] Set up MFA on your Gmail account (required for AWS)
- [ ] Test AWS CLI configuration

### 2. Environment Setup (Due: November 15)
- [ ] Complete environment setup using `environment_setup_guide.md`
- [ ] Install RAPIDS and required packages
- [ ] Clone project repository to your local machine
- [ ] Run `verify_setup.py` to confirm setup
- [ ] Connect to EC2 instance via SSH

### 3. Initial Setup (November 13-15)
- [ ] Join team communication channels (Slack/Teams)
- [ ] Review project documentation:
  - `fake_news_detection_workflow.md`
  - `team_management_guide.md`
  - `etl_usage_guide.md`
- [ ] Set up Git workflow (create feature branches)
- [ ] Familiarize yourself with EC2 instance

## Access Information

### EC2 Instance
- **Public IP**: 52.90.26.169
- **Username**: ec2-user
- **SSH Key**: fake-news-key.pem
- **Connect Command**:
  ```bash
  ssh -i fake-news-key.pem ec2-user@52.90.26.169
  ```

### AWS Resources
- **Region**: us-east-1 (N. Virginia)
- **S3 Bucket**: fake-news-project-data-2025
- **IAM Users**: Created for each team member
- **Credentials**: Provided in `student_credentials.txt`

### Project Repositories
- **Main Repo**: https://github.com/Eldohub-data-scientists/Data-Science-2025.git
- **Group Folders on EC2**:
  - Group A: `/home/ec2-user/groupA/`
  - Group B: `/home/ec2-user/groupB/`

## Your Role Responsibilities

### Project Manager (John/Kigen)
- [ ] Create project timeline and milestones
- [ ] Coordinate team meetings and standups
- [ ] Track progress and manage risks
- [ ] Communicate with instructor and stakeholders

### Data Analyst (Elsa/Lamech)
- [ ] Execute ETL pipeline (`etl_pipeline.py`)
- [ ] Perform exploratory data analysis
- [ ] Clean and preprocess data
- [ ] Create data visualizations

### ML Engineer (Bismarck/Nehemiah)
- [ ] Implement sklearn baseline models
- [ ] Optimize with RAPIDS cuML
- [ ] Compare CPU vs GPU performance
- [ ] Deploy model API with FastAPI

### QA (Teddy/Wilberforce)
- [ ] Test ETL pipeline reliability
- [ ] Validate model performance
- [ ] Ensure code quality and documentation
- [ ] Test API endpoints and functionality

## Communication Guidelines

### Daily Standups
- **Time**: 9:00 AM EAT (via Slack/Teams)
- **Format**:
  - What did you accomplish yesterday?
  - What will you work on today?
  - Any blockers or issues?
- **Duration**: 15 minutes

### Weekly Check-ins
- **Time**: Friday 4:00 PM EAT
- **Format**: Full team review with instructor
- **Duration**: 1 hour

### Communication Channels
- **Slack/Teams**: Daily communication and file sharing
- **GitHub**: Code reviews and issue tracking
- **Email**: Official announcements and credentials

## Technical Setup Verification

### Local Environment
Run this checklist on your local machine:
```bash
# Check Python
python --version  # Should be 3.9+

# Check conda
conda --version

# Check AWS CLI
aws --version
aws sts get-caller-identity  # Should show your IAM user
```

### EC2 Environment
Run on EC2 instance:
```bash
# Navigate to your group folder
cd /home/ec2-user/groupA  # or groupB

# Run verification
python verify_setup.py
```

## First Week Timeline (November 13-15)

### Day 1: Setup & Planning (November 13)
- [ ] Receive credentials and access information
- [ ] Set up local development environment
- [ ] Join team communication channels
- [ ] Review all project documentation

### Day 2: Environment Setup (November 14)
- [ ] Complete RAPIDS installation
- [ ] Configure AWS credentials
- [ ] Test EC2 SSH access
- [ ] Clone and explore project repository

### Day 3: Initial Development (November 15)
- [ ] Run ETL pipeline for baseline data
- [ ] Set up Git workflow
- [ ] Create initial project structure
- [ ] Plan Phase 2 data collection

## Key Files and Locations

### Project Documentation
- `fake_news_detection_workflow.md` - Complete project workflow
- `team_management_guide.md` - Management and process guide
- `environment_setup_guide.md` - Setup instructions
- `etl_usage_guide.md` - ETL pipeline usage

### Scripts and Tools
- `etl_pipeline.py` - Data ingestion pipeline
- `verify_setup.py` - Environment verification
- `student_credentials.txt` - AWS credentials

### Directory Structure
```
/home/ec2-user/
├── groupA/                    # Group A workspace
│   ├── etl_pipeline.py
│   ├── data/
│   ├── models/
│   └── notebooks/
├── groupB/                    # Group B workspace
│   ├── etl_pipeline.py
│   ├── data/
│   ├── models/
│   └── notebooks/
└── fake-news-data/           # Shared data directory
```

## Getting Help

### Primary Support
- **Instructor**: DataMan7 - Available during office hours
- **Team Members**: Collaborate within your group
- **Documentation**: Check provided guides first

### Office Hours
- **Time**: Monday-Friday, 2:00-4:00 PM EAT
- **Location**: Slack/Teams or EC2 instance
- **Booking**: Post in Slack for scheduling

### Common Issues
- **SSH Connection**: Check key permissions (`chmod 400 fake-news-key.pem`)
- **AWS Access**: Verify credentials and MFA setup
- **Conda Issues**: Try `conda clean --all` and recreate environment
- **RAPIDS Problems**: Check GPU availability (current instance is CPU-only)

## Success Metrics

### Individual Success
- [ ] Complete environment setup by November 15
- [ ] Active participation in daily standups
- [ ] Meet role-specific responsibilities
- [ ] Contribute quality code and documentation

### Team Success
- [ ] All members set up by November 15
- [ ] Functional ETL pipeline by November 16
- [ ] Baseline models trained by November 20
- [ ] API deployed by November 25

## Next Steps
1. **Today**: Start environment setup
2. **Tomorrow**: Complete setup and test access
3. **Friday**: Begin Phase 2 data collection
4. **Weekly**: Follow the management guide timeline

Remember: This is your opportunity to showcase everything you've learned. Work together, communicate openly, and build something amazing!

**Questions?** Reach out to your instructor or team lead immediately.

Let's detect some fake news! 🚀