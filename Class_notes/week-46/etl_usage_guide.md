# ETL Pipeline Usage Guide: Fake News Detection Project

## Overview
The `etl_pipeline.py` script provides a comprehensive data ingestion pipeline for collecting fake news data from multiple sources. This guide explains how, when, and where to use the script during the project lifecycle.

## File Location
The `etl_pipeline.py` script should be placed in the root of each group's project folder on the EC2 instance:

```
/home/ec2-user/groupA/etl_pipeline.py    # For Group A
/home/ec2-user/groupB/etl_pipeline.py    # For Group B
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
1. **Credentials Setup**: Each group needs API keys for external services
2. **Environment**: Run on EC2 instance with RAPIDS installed
3. **Permissions**: Ensure write access to data directories

### Step-by-Step Execution

#### 1. Navigate to Project Directory
```bash
# For Group A
cd /home/ec2-user/groupA

# For Group B
cd /home/ec2-user/groupB
```

#### 2. Configure Credentials
Create a credentials file or set environment variables:

```bash
# Option 1: Environment variables
export KAGGLE_USERNAME="your_username"
export KAGGLE_KEY="your_api_key"
export NEWSAPI_KEY="your_newsapi_key"
export TWITTER_API_KEY="your_twitter_key"
export TWITTER_API_SECRET="your_twitter_secret"
export TWITTER_ACCESS_TOKEN="your_access_token"
export TWITTER_ACCESS_TOKEN_SECRET="your_access_token_secret"

# Option 2: Create credentials.py
cat > credentials.py << EOF
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

#### 3. Run the Pipeline
```bash
# Basic usage with all sources
python etl_pipeline.py

# Or import and use programmatically
python -c "
from etl_pipeline import FakeNewsETL
etl = FakeNewsETL()
results = etl.run_full_pipeline(
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

#### 4. Individual Data Source Collection
```python
from etl_pipeline import FakeNewsETL

etl = FakeNewsETL()

# Download Kaggle dataset only
kaggle_file = etl.download_kaggle_dataset()

# Fetch news from NewsAPI
news_file = etl.fetch_news_api('your_newsapi_key', query='fake news')

# Collect Twitter data
twitter_file = etl.fetch_twitter_data('fake news', count=100)

# Preprocess existing data
etl.preprocess_text(cudf.read_csv('data.csv'))
```

## Data Sources and Configuration

### 1. Kaggle Fake News Dataset
- **Purpose**: Baseline training data
- **Requirements**: Kaggle account and API key
- **Output**: `kaggle_fake_news.csv` with labeled data
- **Usage**: Primary training dataset

### 2. NewsAPI
- **Purpose**: Current news articles
- **Requirements**: NewsAPI key (free tier available)
- **Output**: `newsapi_{query}_{date}.csv`
- **Usage**: Real-time news collection

### 3. Twitter API
- **Purpose**: Social media content analysis
- **Requirements**: Twitter Developer account and API keys
- **Output**: `twitter_{query}_{date}.csv`
- **Usage**: Trending topics and user-generated content

### 4. Web Scraping (Future Extension)
- **Purpose**: Custom news sources
- **Requirements**: BeautifulSoup, Scrapy
- **Output**: Custom CSV files
- **Usage**: Specialized data collection

## Group-Specific Usage

### Group A Workflow
```bash
# Data Analyst (Elsa): Initial data exploration
cd /home/ec2-user/groupA
python etl_pipeline.py  # Download baseline data
# Analyze data quality and distribution

# ML Engineer (Bismarck): Model training data
# Use collected data for sklearn/cuML training
```

### Group B Workflow
```bash
# Data Analyst (Lamech): Data preprocessing
cd /home/ec2-user/groupB
python etl_pipeline.py  # Collect diverse datasets
# Clean and preprocess for modeling

# ML Engineer (Nehemiah): GPU optimization
# Compare sklearn vs cuML performance
```

## Output Data Structure

### Expected File Outputs
```
/home/ec2-user/fake-news-data/
├── kaggle_fake_news.csv          # Combined Kaggle dataset
├── newsapi_fake_news_20251118.csv # NewsAPI results
├── twitter_fake_news_20251118.csv # Twitter data
└── processed_data.csv            # Preprocessed data
```

### Data Format
- **CSV files** with consistent column naming
- **CUDF DataFrames** for GPU processing
- **Labeled data** for supervised learning (0=fake, 1=real)
- **Metadata** preserved for feature engineering

## Error Handling and Troubleshooting

### Common Issues
1. **API Rate Limits**: Implement delays between requests
2. **Missing Credentials**: Verify environment variables
3. **Disk Space**: Monitor EC2 storage usage
4. **Network Issues**: Retry failed API calls

### Logging and Monitoring
```python
import logging
logging.basicConfig(level=logging.INFO)
# Pipeline operations are logged automatically
```

### Data Validation
```python
# Check data quality after ETL
df = cudf.read_csv('output.csv')
print(f"Rows: {len(df)}, Columns: {len(df.columns)}")
print(f"Missing values: {df.isnull().sum().sum()}")
print(f"Label distribution: {df['label'].value_counts()}")
```

## Integration with Project Workflow

### Phase 2 Integration
- **Data Analyst Role**: Execute ETL, validate data quality
- **ML Engineer Role**: Use processed data for model training
- **QA Role**: Test pipeline reliability and data integrity

### Version Control
```bash
# Track ETL scripts and outputs
git add etl_pipeline.py
git add fake-news-data/
git commit -m "ETL pipeline execution - Phase 2"
```

### Collaboration
- **Shared Data**: Store processed data in group folder
- **Documentation**: Log data sources and preprocessing steps
- **Reviews**: Peer review ETL code and data quality

## Performance Optimization

### RAPIDS Acceleration
- **GPU Processing**: Automatic cuDF usage for large datasets
- **Memory Management**: Monitor GPU memory usage
- **Batch Processing**: Handle large data volumes efficiently

### Scaling Considerations
- **Instance Type**: Current t3.medium (CPU) - consider GPU upgrade
- **Storage**: Use S3 for large datasets
- **Parallel Processing**: Distribute across team members

## Security and Compliance

### API Key Management
- **Environment Variables**: Never commit credentials to Git
- **Access Control**: Limit API permissions to read-only
- **Rate Limiting**: Respect API provider limits

### Data Privacy
- **PII Removal**: Strip personal information from datasets
- **Compliance**: Follow data usage terms from providers
- **Storage**: Secure data handling on EC2

## Support and Resources

### Getting Help
- **Instructor Office Hours**: ETL pipeline debugging
- **Documentation**: Check inline code comments
- **Team Collaboration**: Share successful configurations

### Useful Commands
```bash
# Check data directory
ls -la /home/ec2-user/fake-news-data/

# Monitor disk usage
df -h

# Check running processes
ps aux | grep python

# View logs
tail -f etl_pipeline.log
```

This ETL pipeline provides the foundation for your fake news detection project, enabling efficient data collection and preprocessing for high-quality model development.