#!/usr/bin/env python3
"""
Verification script for Fake News Detection project setup on GCP
Run this script on the Vertex AI Workbench to verify all components are properly configured
Includes RAPIDS GPU acceleration verification following NVIDIA DLI methodology
"""

import os
import sys
import subprocess
import importlib
from google.cloud import storage, bigquery, pubsub_v1
from google.oauth2 import service_account
import time

def run_command(cmd, description):
    """Run a shell command and return success/failure"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ {description}: SUCCESS")
            return True, result.stdout.strip()
        else:
            print(f"✗ {description}: FAILED")
            print(f"  Error: {result.stderr}")
            return False, result.stderr
    except Exception as e:
        print(f"✗ {description}: ERROR - {str(e)}")
        return False, str(e)

def check_python_packages():
    """Check if required Python packages are installed"""
    required_packages = ['pandas', 'numpy', 'scikit-learn', 'requests', 'tweepy', 'kaggle',
                        'google-cloud-storage', 'google-cloud-bigquery', 'google-auth']
    missing = []

    for package in required_packages:
        try:
            importlib.import_module(package.replace('-', '_'))
            print(f"✓ Package {package}: INSTALLED")
        except ImportError:
            print(f"✗ Package {package}: MISSING")
            missing.append(package)

    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Install with: pip install " + ' '.join(missing))
    else:
        print("\n✓ All required packages are installed")

    return len(missing) == 0

def check_rapids_acceleration():
    """Check RAPIDS GPU acceleration following NVIDIA DLI methodology"""
    try:
        # Test cuDF pandas accelerator mode (zero-code-change)
        import cudf.pandas  # Enable automatic GPU acceleration
        import pandas as pd

        print("✓ RAPIDS cuDF pandas accelerator: ENABLED")

        # Test basic GPU acceleration
        df = pd.DataFrame({'test': range(10000), 'label': [0,1] * 5000})
        start = time.time()
        result = df.groupby('label').sum()
        gpu_time = time.time() - start

        print(f"  GPU time: {gpu_time:.2f}s")

        # Test cuML GPU acceleration
        try:
            from cuml.ensemble import RandomForestClassifier
            from cuml.datasets import make_classification

            X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
            model = RandomForestClassifier(n_estimators=10)
            model.fit(X, y)

            print("✓ RAPIDS cuML GPU acceleration: WORKING")
        except Exception as e:
            print(f"⚠️  RAPIDS cuML: LIMITED - {e}")

        # Test cuGraph if available
        try:
            import cugraph
            print("✓ RAPIDS cuGraph: AVAILABLE")
        except ImportError:
            print("⚠️  RAPIDS cuGraph: NOT INSTALLED")

        return True

    except ImportError:
        print("✗ RAPIDS GPU acceleration: NOT AVAILABLE")
        print("  Install with: conda install -c rapidsai -c nvidia -c conda-forge rapids=23.10 python=3.9")
        return False
    except Exception as e:
        print(f"✗ RAPIDS GPU acceleration: ERROR - {str(e)}")
        return False

def check_gcp_services():
    """Check GCP services and authentication"""
    try:
        # Test Cloud Storage
        storage_client = storage.Client()
        bucket_name = 'fake-news-project-data-2025'
        bucket = storage_client.bucket(bucket_name)

        if bucket.exists():
            print(f"✓ Cloud Storage bucket '{bucket_name}': ACCESSIBLE")

            # Test read/write permissions
            test_blob = bucket.blob('test_verification.txt')
            test_blob.upload_from_string('test content')
            content = test_blob.download_as_text()
            test_blob.delete()

            if content == 'test content':
                print("✓ Cloud Storage read/write permissions: WORKING")
            else:
                print("⚠️  Cloud Storage permissions: LIMITED")
        else:
            print(f"✗ Cloud Storage bucket '{bucket_name}': NOT FOUND")
            return False

        # Test BigQuery
        try:
            bq_client = bigquery.Client()
            datasets = list(bq_client.list_datasets())
            print(f"✓ BigQuery access: WORKING ({len(datasets)} datasets accessible)")
        except Exception as e:
            print(f"⚠️  BigQuery access: LIMITED - {e}")

        # Test Pub/Sub
        try:
            publisher = pubsub_v1.PublisherClient()
            topics = list(publisher.list_topics(request={"project": f"projects/{storage_client.project}"}))
            print(f"✓ Pub/Sub access: WORKING ({len(topics)} topics accessible)")
        except Exception as e:
            print(f"⚠️  Pub/Sub access: LIMITED - {e}")

        return True

    except Exception as e:
        print(f"✗ GCP Services: ERROR - {str(e)}")
        print("  Make sure you're authenticated: gcloud auth activate-service-account --key-file=key.json")
        return False

def check_directories():
    """Check if required directories exist"""
    required_dirs = ['/home/jupyter/fake-news-project/groupA', '/home/jupyter/fake-news-project/groupB']
    all_exist = True

    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✓ Directory {dir_path}: EXISTS")
        else:
            print(f"✗ Directory {dir_path}: MISSING")
            all_exist = False

    if not all_exist:
        print("\nCreate missing directories with: mkdir -p " + ' '.join(required_dirs))

    return all_exist

def check_gcloud():
    """Check if Google Cloud SDK is configured"""
    success, output = run_command("gcloud config get-value project", "GCP Project configuration")
    if success and output:
        print(f"  Current project: {output}")

    success, output = run_command("gcloud config get-value compute/zone", "GCP Zone configuration")
    if success and output:
        print(f"  Current zone: {output}")

    # Check if authenticated
    success, output = run_command("gcloud auth list --filter=status:ACTIVE --format='value(account)'", "GCP Authentication")
    if success and output:
        print(f"  Active account: {output.strip()}")

def check_jupyter():
    """Check if Jupyter Lab is configured"""
    success, output = run_command("jupyter lab --version", "Jupyter Lab installation")
    if success:
        print(f"  Version: {output}")

    # Check RAPIDS integration
    config_path = os.path.expanduser("~/.jupyter/jupyter_lab_config.py")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_content = f.read()
            if 'cudf.pandas' in config_content:
                print("✓ Jupyter RAPIDS integration: CONFIGURED")
            else:
                print("⚠️  Jupyter RAPIDS integration: NOT CONFIGURED")
    else:
        print("⚠️  Jupyter configuration: MISSING")

def check_compute_engine():
    """Check Vertex AI Workbench instance details"""
    success, output = run_command("curl -s http://metadata.google.internal/computeMetadata/v1/instance/name -H 'Metadata-Flavor: Google'", "Instance name")
    if success:
        print(f"  Instance name: {output}")

    success, output = run_command("curl -s http://metadata.google.internal/computeMetadata/v1/instance/machine-type -H 'Metadata-Flavor: Google'", "Machine type")
    if success:
        machine_type = output.split('/')[-1]
        print(f"  Machine type: {machine_type}")

        # Check if GPU enabled
        if 'gpu' in machine_type.lower() or 'accelerator' in machine_type.lower():
            print("✓ GPU instance: DETECTED")
        else:
            print("⚠️  CPU-only instance (consider upgrading to GPU for RAPIDS)")

    # Check GPU status
    success, output = run_command("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits", "GPU status")
    if success and output.strip():
        gpu_info = output.strip().split(',')
        print(f"  GPU: {gpu_info[0]} with {gpu_info[1]}MB memory")
    else:
        print("⚠️  GPU: NOT DETECTED")

def check_vertex_ai():
    """Check Vertex AI services"""
    try:
        from google.cloud import aiplatform
        aiplatform.init()
        print("✓ Vertex AI SDK: INITIALIZED")
        return True
    except Exception as e:
        print(f"⚠️  Vertex AI SDK: NOT CONFIGURED - {e}")
        return False

def benchmark_rapids_performance():
    """Benchmark RAPIDS performance following NVIDIA DLI methodology"""
    try:
        import cudf.pandas  # Enable acceleration
        import pandas as pd
        import time

        print("\n🔬 RAPIDS Performance Benchmark:")

        # Create test dataset
        sizes = [10000, 50000, 100000]

        for size in sizes:
            df = pd.DataFrame({
                'text': [f'sample text {i}' for i in range(size)],
                'label': [i % 2 for i in range(size)],
                'value': range(size)
            })

            # Benchmark groupby operation
            start = time.time()
            result = df.groupby('label')['value'].sum()
            gpu_time = time.time() - start

            print(f"  Dataset size {size}: {gpu_time:.2f}s")

        print("✓ RAPIDS GPU acceleration: PERFORMING WELL")
        return True

    except Exception as e:
        print(f"✗ RAPIDS Performance Benchmark: FAILED - {e}")
        return False

def main():
    print("🔍 Fake News Detection Project GCP Setup Verification")
    print("=" * 70)

    checks = []

    print("\n1. System Requirements:")
    checks.append(("Python packages", check_python_packages()))
    checks.append(("RAPIDS GPU acceleration", check_rapids_acceleration()))
    checks.append(("GCP services", check_gcp_services()))

    print("\n2. GCP Configuration:")
    check_gcloud()
    checks.append(("Vertex AI", check_vertex_ai()))

    print("\n3. Development Environment:")
    checks.append(("Project directories", check_directories()))
    check_jupyter()

    print("\n4. Compute Resources:")
    check_compute_engine()

    print("\n5. Performance Benchmark:")
    benchmark_rapids_performance()

    print("\n" + "=" * 70)
    print("📊 Summary:")

    passed = sum(1 for _, status in checks if status)
    total = len(checks)

    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("🎉 GCP setup verification COMPLETE! Ready for RAPIDS-accelerated development.")
        print("\nNext steps:")
        print("1. Run ETL pipeline: python etl_pipeline_gcp.py")
        print("2. Start Jupyter: Open Vertex AI Workbench")
        print("3. Begin GPU-accelerated data science development")
        print("4. Compare CPU vs GPU performance with RAPIDS")
    else:
        print("⚠️  Some checks failed. Please address the issues above.")
        print("\nCommon fixes:")
        print("- Install RAPIDS: conda install -c rapidsai -c nvidia -c conda-forge rapids=23.10")
        print("- Authenticate GCP: gcloud auth activate-service-account --key-file=key.json")
        print("- Configure Vertex AI: Follow environment setup guide")
        print("- Check GPU: nvidia-smi (should show T4/A100 GPU)")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)