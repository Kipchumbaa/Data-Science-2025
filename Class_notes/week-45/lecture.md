# Module 23: Code Backup and Version Control

## Overview
This module introduces essential practices for code backup, version control, and collaborative software development. Version control systems are fundamental tools for modern software development, enabling teams to work together efficiently, track changes, and maintain code quality.

## Learning Objectives
By the end of this module, students will be able to:
- Understand the importance of version control in software development
- Use Git for basic and advanced version control operations
- Implement effective branching and merging strategies
- Collaborate effectively using GitHub/GitLab
- Set up automated backups and deployment pipelines
- Follow best practices for code versioning and release management

## Key Topics

### 1. Introduction to Version Control

#### What is Version Control?
- **Definition:** A system that records changes to files over time
- **Benefits:**
  - Track changes and history
  - Collaborate with multiple developers
  - Backup and recovery
  - Branching and experimentation
  - Code review and quality control

#### Types of Version Control Systems
- **Centralized VCS:** SVN, CVS (single central repository)
- **Distributed VCS:** Git, Mercurial (every developer has full repository copy)
- **Modern Platforms:** GitHub, GitLab, Bitbucket

### 2. Git Fundamentals

#### Basic Git Concepts
- **Repository:** Project folder under version control
- **Commit:** Snapshot of changes at a point in time
- **Working Directory:** Current files on your computer
- **Staging Area:** Files ready to be committed
- **HEAD:** Pointer to current branch/commit

#### Essential Git Commands
```bash
# Initialize repository
git init

# Clone existing repository
git clone <repository-url>

# Check status
git status

# Add files to staging
git add <filename>
git add .  # Add all files

# Commit changes
git commit -m "Commit message"

# View history
git log
git log --oneline
```

### 3. Branching and Merging

#### Branching Strategy
- **Main/Master Branch:** Production-ready code
- **Develop Branch:** Integration branch for features
- **Feature Branches:** Individual feature development
- **Release Branches:** Pre-production testing
- **Hotfix Branches:** Emergency bug fixes

#### Branching Commands
```bash
# Create and switch to new branch
git checkout -b feature/new-feature

# Switch between branches
git checkout main
git checkout develop

# Merge branches
git merge feature/completed-feature

# Delete branch
git branch -d feature/completed-feature
```

#### Merge Conflicts
- **Causes:** Simultaneous changes to same lines
- **Resolution:**
  1. Identify conflicting files
  2. Edit files to resolve conflicts
  3. Stage resolved files
  4. Complete merge commit

### 4. Remote Repositories and Collaboration

#### Working with Remotes
```bash
# Add remote repository
git remote add origin <repository-url>

# Push local changes
git push origin main

# Pull latest changes
git pull origin main

# Fetch without merging
git fetch origin
```

#### Collaboration Workflow
- **Fork and Pull Request Model:**
  1. Fork repository
  2. Create feature branch
  3. Make changes and commit
  4. Push to your fork
  5. Create pull request
  6. Code review and merge

### 5. Code Review and Quality Control

#### Pull Request Best Practices
- **Clear Title and Description:** Explain what and why
- **Small, Focused Changes:** Easier to review
- **Link to Issues:** Reference related tasks
- **Request Reviewers:** Tag appropriate team members

#### Code Review Guidelines
- **Reviewer Responsibilities:**
  - Check for bugs and logic errors
  - Ensure code follows standards
  - Suggest improvements
  - Verify tests are included

- **Author Responsibilities:**
  - Respond to feedback constructively
  - Make requested changes
  - Update documentation

### 6. Backup and Recovery

#### Backup Strategies
- **Local Backups:** External drives, cloud storage
- **Remote Repositories:** GitHub, GitLab as backup
- **Automated Backups:** Scheduled scripts
- **Offsite Storage:** Geographic redundancy

#### Recovery Scenarios
- **Accidental Deletions:** Restore from git history
- **Corrupted Repositories:** Clone from remote
- **Lost Commits:** Reflog recovery
- **Disaster Recovery:** Multiple backup locations

### 7. Advanced Git Features

#### Rebasing vs. Merging
- **Merge:** Creates merge commit, preserves history
- **Rebase:** Linear history, cleaner but riskier
- **Interactive Rebase:** Edit, squash, or reorder commits

#### Git Hooks
- **Pre-commit:** Run tests before committing
- **Pre-push:** Validate before pushing
- **Post-commit:** Update documentation

#### Git LFS (Large File Storage)
- Handle large files efficiently
- Store pointers instead of actual files
- Useful for datasets, models, images

### 8. CI/CD Integration

#### Continuous Integration
- **Automated Testing:** Run tests on every push
- **Code Quality Checks:** Linting, formatting
- **Build Verification:** Ensure code compiles

#### Continuous Deployment
- **Automated Deployment:** Push to staging/production
- **Rollback Capability:** Quick reversion to previous versions
- **Environment Management:** Separate dev/staging/prod

### 9. Version Control Best Practices

#### Commit Message Conventions
- **Clear and Descriptive:** Explain what and why
- **Imperative Mood:** "Add feature" not "Added feature"
- **Reference Issues:** Link to task numbers
- **Keep it Short:** 50 characters or less

#### Repository Organization
- **README.md:** Project description and setup
- **CONTRIBUTING.md:** Contribution guidelines
- **LICENSE:** Open source license
- **.gitignore:** Exclude unnecessary files

### 10. Industry Tools and Platforms

#### Git Hosting Platforms
- **GitHub:** Most popular, extensive features
- **GitLab:** Self-hosted option, CI/CD built-in
- **Bitbucket:** Atlassian integration

#### Development Tools
- **Git Clients:** GitKraken, Sourcetree, VS Code
- **Code Review Tools:** GitHub PRs, Gerrit
- **Project Management:** Jira, Trello integration

## Practical Applications

### Git Workflow Exercise
Students will practice:
- Creating and managing branches
- Making commits and writing messages
- Merging and resolving conflicts
- Working with remote repositories

### Collaborative Project
- Team-based development project
- Code review sessions
- Pull request management
- Release management simulation

## Assessment Methods
- Git command proficiency quiz
- Branching and merging exercises
- Collaborative coding assignment
- Code review participation
- Repository management project

## Resources
- "Pro Git" by Scott Chacon and Ben Straub
- Git documentation: https://git-scm.com/doc
- GitHub Guides: https://guides.github.com
- Atlassian Git Tutorials: https://www.atlassian.com/git
- Interactive Git Learning: https://learngitbranching.js.org

## Key Takeaways
- Version control is essential for modern software development
- Git enables efficient collaboration and change tracking
- Proper branching strategies prevent conflicts and maintain code quality
- Regular commits and clear messages improve project maintainability
- Code review and automated testing ensure quality and reliability
- Backup strategies protect against data loss and enable recovery