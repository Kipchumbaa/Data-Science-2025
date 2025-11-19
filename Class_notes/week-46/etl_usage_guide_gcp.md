# ETL Pipeline Usage Guide: Fake News Detection Project (GCP Version)

## Overview
The `etl_pipeline_gcp.py` script provides a comprehensive data ingestion pipeline for collecting fake news data from multiple sources and storing it in Google Cloud Storage. This guide explains how, when, and where to use the script during the project lifecycle on GCP.

## File Location
The `etl_pipeline_gcp.py` script should be placed in the root of each group's project folder on the GCP Compute Engine VM:

```
/opt/fake-news-project/groupA/etl_pipeline_gcp.py    # For Group A
/opt/fake-news-project/groupB/etl_pipeline_gcp.py    # For Group B
```

## When to Use the ETL Pipeline

### Phase 2: Data Acquisition & Processing (Nov 16-18)
- **Day 1 (Nov 16)**: Initial data collection and baseline dataset setup
- **Day 2-3 (Nov 17-18)**: Additional data sources and preprocessing validation
- **Milestone**: Complete data preprocessing by Nov 18

### Key Usage Points
1. **Project Kickoff**: Download baseline Kaggle dataset
2. **Data Exploration**: Fetch additional news sources for diversity
3. **Real-time Updates**: Collect fresh data during development
4. **Testing**: Validate pipeline before model training

## How to Run the ETL Pipeline

### Prerequisites
1. **GCP Authentication**: Service account key or default credentials
2. **Environment**: Run on GCP Compute Engine VM with RAPIDS installed
3. **Permissions**: Cloud Storage admin access for data uploads
4. **API Keys**: Kaggle, Twitter, NewsAPI credentials

### Step-by-Step Execution

#### 1. Navigate to Project Directory
```bash
# For Group A
cd /opt/fake-news-project/groupA

# For Group B
cd /opt/fake-news-project/groupB
```

#### 2. Configure GCP Authentication
```bash
# Option 1: Service Account Key (recommended for students)
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/your-service-account-key.json

# Option 2: Use default credentials (if running on GCP VM with proper IAM)
# No additional setup needed
```

#### 3. Set Up API Credentials
Create environment variables or credentials file:

```bash
# Set environment variables
export KAGGLE_USERNAME="your_username"
export KAGGLE_KEY="your_api_key"
export NEWSAPI_KEY="your_newsapi_key"
export TWITTER_API_KEY="your_twitter_key"
export TWITTER_API_SECRET="your_twitter_secret"
export TWITTER_ACCESS_TOKEN="your_access_token"
export TWITTER_ACCESS_TOKEN_SECRET="your_access_token_secret"

# Or create a credentials file
cat > gcp_credentials.py << EOF
# GCP and API Credentials
GCP_PROJECT_ID = 'fake-news-project'
GCS_BUCKET_NAME = 'fake-news-project-data-2025'
SERVICE_ACCOUNT_KEY = '/path/to/your-service-account-key.json'

KAGGLE_CREDS = {'username': 'your_username', 'key': 'your_api_key'}
NEWSAPI_KEY = 'your_newsapi_key'
TWITTER_CREDS = {
    'api_key': 'your_twitter_key',
    'api_secret': 'your_twitter_secret',
    'access_token': 'your_access_token',
    'access_token_secret': 'your_access_token_secret'
}
EOF
```

#### 4. Run the Pipeline
```bash
# Activate RAPIDS environment
source /opt/miniconda/bin/activate
conda activate rapids-env

# Basic usage with all sources
python etl_pipeline_gcp.py

# Or run programmatically
python -c "
from etl_pipeline_gcp import FakeNewsETLGCP
etl = FakeNewsETLGCP()
results = etl.run_full_pipeline(
    gcp_key_path='/path/to/your-service-account-key.json',
    kaggle_creds={'username': 'your_username', 'key': 'your_api_key'},
    newsapi_key='your_newsapi_key',
    twitter_creds={
        'api_key': 'your_twitter_key',
        'api_secret': 'your_twitter_secret',
        'access_token': 'your_access_token',
        'access_token_secret': 'your_access_token_secret'
    }
)
print('ETL Results:', results)
"
```

#### 5. Individual Data Source Collection
```python
from etl_pipeline_gcp import FakeNewsETLGCP

# Initialize with GCP credentials
etl = FakeNewsETLGCP()
etl.authenticate_gcp('/path/to/service-account-key.json')

# Download Kaggle dataset only
local_file, gcs_file = etl.download_kaggle_dataset()
print(f'Local: {local_file}, GCS: {gcs_file}')

# Fetch news from NewsAPI
local_file, gcs_file = etl.fetch_news_api('your_newsapi_key', query='fake news')
print(f'Local: {local_file}, GCS: {gcs_file}')

# Collect Twitter data
local_file, gcs_file = etl.fetch_twitter_data('fake news', count=100)
print(f'Local: {local_file}, GCS: {gcs_file}')

# Preprocess existing data
df = etl.load_from_gcs('gs://fake-news-project-data-2025/datasets/kaggle_fake_news_20251118.csv')
processed_df = etl.preprocess_text(df)
```

## Data Sources and Configuration

### 1. Kaggle Fake News Dataset
- **Purpose**: Baseline training data with labels
- **Requirements**: Kaggle account and API key
- **Output**: `kaggle_fake_news_{date}.csv` in Cloud Storage
- **GCS Path**: `gs://fake-news-project-data-2025/datasets/kaggle_fake_news_{date}.csv`
- **Usage**: Primary training dataset

### 2. NewsAPI
- **Purpose**: Current news articles from major sources
- **Requirements**: NewsAPI key (free tier available)
- **Output**: `newsapi_{query}_{date}.csv` in Cloud Storage
- **GCS Path**: `gs://fake-news-project-data-2025/datasets/newsapi_{query}_{date}.csv`
- **Usage**: Real-time news collection

### 3. Twitter API
- **Purpose**: Social media content and trending topics
- **Requirements**: Twitter Developer account and API keys
- **Output**: `twitter_{query}_{date}.csv` in Cloud Storage
- **GCS Path**: `gs://fake-news-project-data-2025/datasets/twitter_{query}_{date}.csv`
- **Usage**: User-generated content analysis

### 4. Web Scraping (Future Extension)
- **Purpose**: Custom news sources
- **Requirements**: BeautifulSoup, Scrapy libraries
- **Output**: Custom CSV files in Cloud Storage
- **Usage**: Specialized data collection

## Group-Specific Usage

### Group A Workflow
```bash
# Data Analyst (Elsa): Initial data exploration
cd /opt/fake-news-project/groupA
python etl_pipeline_gcp.py  # Download baseline data
# Analyze data quality and distribution in Cloud Storage

# ML Engineer (Bismarck): Model training data
# Load processed data from GCS for sklearn/cuML training
```

### Group B Workflow
```bash
# Data Analyst (Lamech): Data preprocessing
cd /opt/fake-news-project/groupB
python etl_pipeline_gcp.py  # Collect diverse datasets
# Clean and preprocess for modeling, store in GCS

# ML Engineer (Nehemiah): GPU optimization
# Compare sklearn vs cuML performance with GCS data
```

## Output Data Structure

### Cloud Storage Organization
```
gs://fake-news-project-data-2025/
├── datasets/
│   ├── kaggle_fake_news_20251118.csv          # Combined Kaggle dataset
│   ├── newsapi_fake_news_20251118.csv         # NewsAPI results
│   ├── twitter_fake_news_20251118.csv         # Twitter data
│   └── processed_data_20251118.csv            # Preprocessed data
├── models/
│   ├── sklearn_model_20251122.pkl             # Trained sklearn models
│   ├── cuml_model_20251122.pkl                # Trained cuML models
│   └── model_artifacts/                        # Model metadata
└── results/
    ├── performance_metrics_20251122.json      # Benchmarking results
    └── evaluation_reports/                     # Analysis reports
```

### Data Format Standards
- **CSV files**: Consistent column naming and data types
- **cuDF DataFrames**: GPU-accelerated processing
- **Labeled data**: Binary classification (0=fake, 1=real)
- **Metadata preservation**: Source tracking and timestamps

## Error Handling and Troubleshooting

### Common Issues

#### GCP Authentication Problems
```bash
# Check service account key
cat $GOOGLE_APPLICATION_CREDENTIALS | head -5

# Test authentication
python -c "from google.cloud import storage; client = storage.Client(); print('GCP auth successful')"

# Re-authenticate if needed
gcloud auth activate-service-account --key-file=path/to/key.json
```

#### Cloud Storage Permission Issues
```bash
# Check bucket permissions
gsutil iam get gs://fake-news-project-data-2025

# Test access
gsutil ls gs://fake-news-project-data-2025/datasets/

# Add permissions if needed
gsutil iam ch serviceAccount:your-service-account@fake-news-project.iam.gserviceaccount.com:roles/storage.admin gs://fake-news-project-data-2025
```

#### API Rate Limits
```bash
# Implement delays between API calls
# Check API documentation for rate limits
# Use caching for repeated requests
```

#### RAPIDS GPU Issues
```bash
# Verify GPU availability
nvidia-smi

# Check cuDF installation
python -c "import cudf; print('cuDF version:', cudf.__version__)"

# Fallback to CPU if GPU unavailable
# cuML will automatically use CPU if GPU not available
```

### Logging and Monitoring
```python
import logging
logging.basicConfig(level=logging.INFO)

# Pipeline operations are automatically logged
# Check logs for debugging
tail -f etl_pipeline_gcp.log
```

### Data Validation
```python
# Check data quality after ETL
from etl_pipeline_gcp import FakeNewsETLGCP

etl = FakeNewsETLGCP()
df = etl.load_from_gcs('gs://fake-news-project-data-2025/datasets/kaggle_fake_news_20251118.csv')

print(f"Rows: {len(df)}, Columns: {len(df.columns)}")
print(f"Missing values: {df.isnull().sum().sum()}")
print(f"Label distribution: {df['label'].value_counts()}")
print(f"Data types: {df.dtypes}")
```

## Integration with Project Workflow

### Phase 2 Integration
- **Data Analyst Role**: Execute ETL, validate GCS uploads
- **ML Engineer Role**: Load data from GCS for model training
- **QA Role**: Test pipeline reliability and data integrity

### Version Control
```bash
# Track ETL scripts and configurations
git add etl_pipeline_gcp.py
git add gcp_credentials.py
git commit -m "ETL pipeline GCP integration - Phase 2"

# Note: Never commit actual credentials or keys
```

### Collaboration
- **Shared GCS Bucket**: Store processed data for team access
- **Documentation**: Log data sources and preprocessing steps
- **Reviews**: Peer review ETL code and GCS configurations

## Performance Optimization

### RAPIDS Acceleration
- **GPU Processing**: Automatic cuDF usage for large datasets
- **Memory Management**: Monitor GPU memory usage on GCP
- **Batch Processing**: Efficient handling of large data volumes

### GCP Optimization
- **Storage Classes**: Use appropriate GCS storage tiers
- **Network Optimization**: Minimize data transfer costs
- **Caching**: Implement appropriate caching strategies

### Scaling Considerations
- **Instance Type**: Current n1-standard-4, upgradeable to GPU instances
- **Storage**: Use GCS for large datasets, local storage for active work
- **Parallel Processing**: Distribute across team members

## Security and Compliance

### GCP Security
- **Service Account Keys**: Secure key management and rotation
- **VPC Security**: Network-level access control
- **Cloud Audit Logs**: Complete activity logging

### API Security
- **Key Management**: Environment variables, never in code
- **Rate Limiting**: Respect API provider limits
- **Data Privacy**: Strip PII from collected data

### Data Governance
- **Access Control**: IAM roles for appropriate data access
- **Data Retention**: Automatic cleanup policies
- **Compliance**: GCP compliance certifications

## Support and Resources

### Getting Help
- **GCP Documentation**: https://cloud.google.com/storage/docs
- **RAPIDS Documentation**: https://docs.rapids.ai/
- **Instructor Office Hours**: ETL pipeline debugging
- **Team Collaboration**: Share successful GCP configurations

### Useful Commands
```bash
# Check GCS bucket contents
gsutil ls gs://fake-news-project-data-2025/datasets/

# Monitor data transfer
gsutil -m cp -r local_data gs://fake-news-project-data-2025/

# Check bucket size
gsutil du -sh gs://fake-news-project-data-2025/

# Set up lifecycle policies
gsutil lifecycle set lifecycle_config.json gs://fake-news-project-data-2025/
```

### Monitoring ETL Performance
```bash
# Check VM resource usage
gcloud compute instances describe fake-news-instance --zone=us-central1-a --format="table(name,status,machineType,cpuPlatform)"

# Monitor Cloud Storage usage
gsutil ls -l gs://fake-news-project-data-2025/datasets/

# Check costs
gcloud billing accounts list
```

This GCP ETL pipeline provides the foundation for your fake news detection project, enabling efficient data collection, processing, and storage in Google Cloud Platform with RAPIDS GPU acceleration.