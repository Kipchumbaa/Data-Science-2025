# Student Onboarding Guide: Fake News Detection Project (GCP Version)

## Welcome to the Fake News Detection Capstone Project!

This guide will help you get started with your role in the project on Google Cloud Platform. Follow these steps carefully to set up your development environment and begin contributing to the team.

---

## 📋 Prerequisites

### System Requirements
- **Operating System**: Windows 10/11, macOS 10.15+, or Linux
- **RAM**: Minimum 8GB (16GB recommended)
- **Storage**: 20GB free space
- **Network**: Stable internet connection

### Required Accounts
- **Google Cloud Platform**: Free tier available
- **GitHub**: For code collaboration
- **Kaggle**: For dataset access (optional)
- **Twitter Developer**: For API access (optional)

---

## 🚀 Quick Start (15 minutes)

### Step 1: Receive Your Credentials
You should have received an email with:
- **Service Account Key** (JSON file): `your-name-key.json`
- **Project Details**: GCP project ID and region
- **Team Assignment**: Group A or Group B
- **Role**: Project Manager, Data Analyst, ML Engineer, or QA

### Step 2: Install Google Cloud SDK
```bash
# Windows (PowerShell as Administrator)
(New-Object Net.WebClient).DownloadFile("https://dl.google.com/dl/cloudsdk/channels/rapid/GoogleCloudSDKInstaller.exe", "$env:Temp\GoogleCloudSDKInstaller.exe"); & "$env:Temp\GoogleCloudSDKInstaller.exe"

# macOS (Terminal)
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Linux (Terminal)
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
```

### Step 3: Authenticate and Configure
```bash
# Initialize gcloud
gcloud init

# When prompted, select "Re-initialize this configuration" and choose your project

# Authenticate with your service account
gcloud auth activate-service-account --key-file=path/to/your-key.json

# Set project and region
gcloud config set project fake-news-project
gcloud config set compute/zone us-central1-a
```

### Step 4: Test Access
```bash
# Test Compute Engine access
gcloud compute instances list

# Test Cloud Storage access
gsutil ls gs://fake-news-project-data-2025/

# SSH into the development VM
gcloud compute ssh fake-news-instance --zone=us-central1-a
```

---

## 🏗️ Development Environment Setup

### Local Development (Optional)
If you prefer to develop locally instead of on the GCP VM:

#### Install Miniconda
```bash
# Download and install Miniconda
# Windows: https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe
# macOS/Linux: curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh | bash

# Create RAPIDS environment
conda create -n rapids-env python=3.9 -y
conda activate rapids-env

# Install RAPIDS (CPU version for local development)
conda install -c rapidsai -c nvidia -c conda-forge rapids=23.10 python=3.9 -y

# Install additional packages
pip install google-cloud-storage google-auth kaggle requests tweepy fastapi uvicorn
```

#### Configure Local GCP Access
```bash
# Set environment variable for service account
export GOOGLE_APPLICATION_CREDENTIALS=path/to/your-key.json

# Test authentication
python -c "from google.cloud import storage; client = storage.Client(); print('GCP access successful')"
```

### Cloud Development (Recommended)
Use the pre-configured GCP Compute Engine VM:

#### Access the VM
```bash
# SSH into the development instance
gcloud compute ssh fake-news-instance --zone=us-central1-a

# Navigate to your group directory
cd /opt/fake-news-project/groupA  # or groupB
```

#### Activate RAPIDS Environment
```bash
# Activate the RAPIDS environment
source /opt/miniconda/bin/activate
conda activate rapids-env

# Verify RAPIDS installation
python -c "import cudf; print('RAPIDS ready!')"
```

---

## 👥 Team Structure & Communication

### Your Team Assignment

#### **Group A**
- **Project Manager**: John (jnyangara303@gmail.com)
- **Data Analyst**: Elsa (elsernownex@gmail.com)
- **ML Engineer**: Bismarck (bismarkkoima844@gmail.com)
- **QA**: Teddy (amdanyteddy@gmail.com)

#### **Group B**
- **Project Manager**: Kigen (kigenyego@gmail.com)
- **Data Analyst**: Lamech (lamechrop45@gmail.com)
- **ML Engineer**: Nehemiah (nehemiahkipchumba89@gmail.com)
- **QA**: Wilberforce (wilberforcekimutai68@gmail.com)

### Communication Channels
- **Slack/Teams**: `#fake-news-project` for daily communication
- **GitHub**: Repository for code collaboration and PR reviews
- **Email**: Instructor communication and official announcements
- **VM Access**: Direct collaboration on the GCP instance

### Daily Standups
- **Time**: 9:00 AM EAT (East Africa Time)
- **Format**: 15-minute video call
- **Agenda**:
  - What did you accomplish yesterday?
  - What are you working on today?
  - Any blockers or challenges?

---

## 🛠️ Development Workflow

### Phase 1: Setup & Planning (Nov 13-15)
- [ ] Complete environment setup
- [ ] Review project documentation
- [ ] Attend kickoff meeting
- [ ] Set up development tools

### Phase 2: Data Processing (Nov 16-18)
- [ ] Run ETL pipeline for data collection
- [ ] Perform exploratory data analysis
- [ ] Clean and preprocess datasets
- [ ] Validate data quality

### Phase 3: Model Development (Nov 19-22)
- [ ] Implement baseline sklearn models
- [ ] Compare with cuML GPU acceleration
- [ ] Optimize model performance
- [ ] Document results and benchmarks

### Phase 4: Deployment & Testing (Nov 23-25)
- [ ] Develop FastAPI REST API
- [ ] Deploy to Cloud Run
- [ ] Implement monitoring and logging
- [ ] Conduct thorough testing

### Phase 5: Presentation (Nov 26-28)
- [ ] Create final presentation
- [ ] Prepare demo environment
- [ ] Practice delivery
- [ ] Final project submission

---

## 📁 Project Directory Structure

```
/opt/fake-news-project/
├── groupA/                          # Your team directory
│   ├── data/                        # Raw and processed datasets
│   ├── models/                      # Trained models and artifacts
│   ├── notebooks/                   # Jupyter notebooks
│   ├── scripts/                     # Python scripts and utilities
│   └── docs/                        # Team documentation
├── groupB/                          # Other team directory
├── ssl/                             # SSL certificates for Jupyter
└── shared/                          # Shared resources and documentation
```

### Key Files
- `etl_pipeline_gcp.py`: Data ingestion pipeline
- `verify_setup_gcp.py`: Environment verification script
- `environment_setup_guide_gcp.md`: Detailed setup instructions
- `team_management_guide.md`: Project management guidelines

---

## 🔧 Essential Commands

### GCP Operations
```bash
# List compute instances
gcloud compute instances list

# SSH into VM
gcloud compute ssh fake-news-instance --zone=us-central1-a

# Copy files to/from VM
gcloud compute scp local-file fake-news-instance:~/remote-file --zone=us-central1-a

# Check Cloud Storage
gsutil ls gs://fake-news-project-data-2025/
```

### RAPIDS Environment
```bash
# Activate environment
conda activate rapids-env

# Check GPU status (if available)
nvidia-smi

# Start Jupyter Lab
jupyter lab --no-browser --port=8888

# Run ETL pipeline
python etl_pipeline_gcp.py
```

### Git Operations
```bash
# Clone repository
git clone https://github.com/Eldohub-data-scientists/Data-Science-2025.git

# Create feature branch
git checkout -b feature/your-feature-name

# Commit changes
git add .
git commit -m "Description of changes"

# Push to GitHub
git push origin feature/your-feature-name

# Create Pull Request
# Go to GitHub and create PR for review
```

---

## 📊 Monitoring & Cost Management

### Resource Monitoring
- **Cloud Monitoring**: Access via GCP Console → Monitoring
- **VM Metrics**: CPU, memory, disk usage
- **Cost Tracking**: Billing → Reports

### Cost Optimization Tips
- Stop VM when not in use: `gcloud compute instances stop fake-news-instance`
- Use preemptible instances for testing
- Monitor Cloud Storage usage
- Set up budget alerts

### Performance Monitoring
- Use `verify_setup_gcp.py` to check environment health
- Monitor model training times
- Track API response times
- Log errors and performance metrics

---

## 🆘 Getting Help

### Support Hierarchy

#### **Level 1: Self-Service**
- Check project documentation
- Review FAQ and troubleshooting guides
- Search existing issues on GitHub

#### **Level 2: Team Support**
- Ask team members during standups
- Post questions in Slack/Teams
- Review code together via screen sharing

#### **Level 3: Instructor Support**
- Office hours: Monday-Friday, 2-4 PM EAT
- Email for urgent issues
- Scheduled code reviews and debugging sessions

### Common Issues & Solutions

#### **Authentication Problems**
```bash
# Re-authenticate service account
gcloud auth activate-service-account --key-file=path/to/your-key.json

# Check active account
gcloud auth list
```

#### **VM Access Issues**
```bash
# Reset SSH keys
gcloud compute ssh fake-news-instance --zone=us-central1-a --ssh-key-expire-after=1h

# Check firewall rules
gcloud compute firewall-rules list
```

#### **RAPIDS Installation Issues**
```bash
# Clean and reinstall
conda env remove -n rapids-env
conda create -n rapids-env python=3.9 -y
conda activate rapids-env
conda install -c rapidsai -c nvidia -c conda-forge rapids=23.10 python=3.9 -y
```

#### **Permission Denied**
- Verify service account has correct IAM roles
- Check Cloud Storage bucket permissions
- Ensure VM has proper scopes

---

## 🎯 Success Metrics

### Individual Performance
- **Code Quality**: Clean, documented, and tested code
- **Collaboration**: Active participation in team activities
- **Learning**: Demonstrated understanding of RAPIDS and GCP
- **Delivery**: Meeting sprint commitments and deadlines

### Team Performance
- **Project Progress**: On-time delivery of milestones
- **Quality Standards**: Passing QA checks and reviews
- **Innovation**: Creative solutions and optimizations
- **Documentation**: Comprehensive project artifacts

### Technical Achievement
- **Model Performance**: Accuracy >85%, F1-score >0.85
- **GPU Utilization**: Successful RAPIDS cuML implementation
- **Scalability**: Efficient processing of large datasets
- **Deployment**: Functional API with monitoring

---

## 📅 Important Dates

- **Nov 13-15**: Environment setup and planning
- **Nov 16-18**: Data acquisition and preprocessing
- **Nov 19-22**: Model development and optimization
- **Nov 23-25**: API development and testing
- **Nov 26-28**: Final presentation and project completion

## 🏆 Final Tips for Success

1. **Start Early**: Complete setup by November 15
2. **Communicate Regularly**: Attend all standups and meetings
3. **Document Everything**: Keep detailed notes and code comments
4. **Test Frequently**: Run verification scripts regularly
5. **Ask for Help**: Don't hesitate to reach out when stuck
6. **Learn Continuously**: Take advantage of this hands-on GCP experience

---

## 📞 Contact Information

- **Instructor**: DataMan (Lead Instructor)
- **Email**: [instructor email]
- **Office Hours**: Monday-Friday, 2-4 PM EAT
- **Slack/Teams**: #fake-news-project channel

Welcome aboard! Your contributions to this fake news detection project will make a real impact. Let's build something amazing together on Google Cloud Platform! 🚀