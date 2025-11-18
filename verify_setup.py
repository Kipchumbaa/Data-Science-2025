#!/usr/bin/env python3
"""
Verification script for Fake News Detection project setup on EC2
Run this script on the EC2 instance to verify all components are properly configured
"""

import os
import sys
import subprocess
import importlib

def run_command(cmd, description):
    """Run a shell command and return success/failure"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"[SUCCESS] {description}: SUCCESS")
            return True, result.stdout.strip()
        else:
            print(f"[FAIL] {description}: FAILED")
            print(f"  Error: {result.stderr}")
            return False, result.stderr
    except Exception as e:
        print(f"[ERROR] {description}: ERROR - {str(e)}")
        return False, str(e)

def check_python_packages():
    """Check if required Python packages are installed"""
    required_packages = ['pandas', 'numpy', 'scikit-learn', 'requests', 'tweepy', 'kaggle']
    missing = []

    for package in required_packages:
        try:
            importlib.import_module(package.replace('-', '_'))
            print(f"[OK] Package {package}: INSTALLED")
        except ImportError:
            print(f"[MISSING] Package {package}: MISSING")
            missing.append(package)

    if missing:
        print(f"\n[WARNING] Missing packages: {', '.join(missing)}")
        print("Install with: pip install " + ' '.join(missing))
    else:
        print("\n[OK] All required packages are installed")

    return len(missing) == 0

def check_rapids():
    """Check if RAPIDS is properly installed"""
    try:
        import cudf
        print("[OK] RAPIDS cuDF: INSTALLED")
        # Test basic functionality
        df = cudf.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        print(f"  cuDF test: Created DataFrame with {len(df)} rows")
        return True
    except ImportError:
        print("[NOT INSTALLED] RAPIDS cuDF: NOT INSTALLED")
        print("  Install with: conda install -c rapidsai -c nvidia -c conda-forge rapids")
        return False
    except Exception as e:
        print(f"[ERROR] RAPIDS cuDF: ERROR - {str(e)}")
        return False

def check_directories():
    """Check if required directories exist"""
    required_dirs = ['/home/ec2-user/fake-news-data', '/home/ec2-user/groupA', '/home/ec2-user/groupB']
    all_exist = True

    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"[OK] Directory {dir_path}: EXISTS")
        else:
            print(f"[MISSING] Directory {dir_path}: MISSING")
            all_exist = False

    if not all_exist:
        print("\nCreate missing directories with: mkdir -p " + ' '.join(required_dirs))

    return all_exist

def check_aws_access():
    """Check AWS access and resources"""
    # Check AWS CLI
    success, output = run_command("aws sts get-caller-identity", "AWS CLI access")
    if success:
        print(f"  Current user: {output.split()[1] if 'UserId' in output else 'Unknown'}")

    # Check S3 bucket
    success, output = run_command("aws s3 ls s3://fake-news-project-data-2025", "S3 bucket access")
    if success:
        print("  Bucket is accessible")

    # Check EC2 permissions
    success, output = run_command("aws ec2 describe-instances --instance-ids $(curl -s http://169.254.169.254/latest/meta-data/instance-id)", "EC2 permissions")
    if success:
        print("  EC2 access confirmed")

def check_git():
    """Check if Git is configured"""
    success, output = run_command("git --version", "Git installation")
    if success:
        print(f"  Version: {output}")

    # Check if in a git repo
    success, output = run_command("git status", "Git repository")
    if success:
        print("  In a Git repository")
    else:
        print("  Not in a Git repository (create one for the project)")

def main():
    print("🔍 Fake News Detection Project Setup Verification")
    print("=" * 50)

    checks = []

    print("\n1. System Requirements:")
    checks.append(("Python packages", check_python_packages()))
    checks.append(("RAPIDS installation", check_rapids()))
    checks.append(("Git setup", check_git()[0] if isinstance(check_git(), tuple) else check_git()))

    print("\n2. Directory Structure:")
    checks.append(("Project directories", check_directories()))

    print("\n3. AWS Resources:")
    check_aws_access()  # This doesn't return a boolean, just prints

    print("\n" + "=" * 50)
    print("📊 Summary:")

    passed = sum(1 for _, status in checks if status)
    total = len(checks)

    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("[COMPLETE] Setup verification COMPLETE! Ready for project development.")
    else:
        print("[WARNING] Some checks failed. Please address the issues above.")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)