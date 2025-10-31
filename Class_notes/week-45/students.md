# Module 23: Code Backup and Version Control - Student Notes

## Introduction to Version Control

### Why Version Control Matters
Version control systems are essential for:
- **Collaboration:** Multiple developers working on the same codebase
- **History Tracking:** See what changed, when, and why
- **Backup:** Protect against data loss
- **Experimentation:** Try new ideas without breaking existing code
- **Accountability:** Know who made what changes

### Types of Version Control Systems

#### Centralized Version Control (CVCS)
- Single central repository
- Examples: SVN, CVS
- All changes go through central server
- Requires constant network connection

#### Distributed Version Control (DVCS)
- Every developer has full repository copy
- Examples: Git, Mercurial
- Work offline, sync when ready
- More flexible and robust

## Git Fundamentals

### Repository Structure
```
Working Directory ──── git add ──── Staging Area ──── git commit ──── Repository
       │                       │                        │
   (files you edit)      (files ready to commit)     (committed snapshots)
```

### Basic Git Workflow

#### 1. Initialize Repository
```bash
# Create new repository
git init

# Clone existing repository
git clone https://github.com/username/repo.git
```

#### 2. Check Status
```bash
# See current state
git status

# See commit history
git log
git log --oneline  # Compact view
git log --graph    # Visual branch history
```

#### 3. Make Changes
```bash
# Add specific file
git add filename.py

# Add all changes
git add .

# Add interactively
git add -p
```

#### 4. Commit Changes
```bash
# Commit with message
git commit -m "Add user authentication feature"

# Amend last commit
git commit --amend -m "Updated commit message"

# Commit all tracked files
git commit -a -m "Quick fix"
```

### Ignoring Files
Create `.gitignore` file:
```
# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
venv/

# Data files
*.csv
*.xlsx
data/
models/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
```

## Branching and Merging

### Branch Management

#### Creating Branches
```bash
# Create and switch to new branch
git checkout -b feature/user-login

# Create branch without switching
git branch feature/data-validation

# Switch to existing branch
git checkout main
git checkout develop
```

#### Listing Branches
```bash
# List all branches
git branch

# List remote branches
git branch -r

# List all branches (local and remote)
git branch -a
```

### Merging Strategies

#### Fast-Forward Merge
```
A ── B ── C (main)
           │
           D ── E (feature)
```
Becomes:
```
A ── B ── C ── D ── E (main)
```

#### Three-Way Merge
```
A ── B ── C (main)
     │
     D ── E (feature)
```
Becomes:
```
A ── B ── C ── F (main)
     │       │
     D ── E ─┘
```

### Resolving Merge Conflicts

#### When Conflicts Occur
Git cannot automatically merge changes. Manual resolution needed.

#### Resolution Steps
1. **Identify conflicts:**
   ```bash
   git status
   ```

2. **Edit conflicting files:**
   Look for conflict markers:
   ```
   <<<<<<< HEAD
   Current branch changes
   =======
   Incoming branch changes
   >>>>>>> feature-branch
   ```

3. **Resolve conflicts:**
   - Choose one version
   - Combine both versions
   - Consult with team members

4. **Complete merge:**
   ```bash
   git add resolved-file.py
   git commit
   ```

## Remote Repositories

### Working with Remotes

#### Add Remote
```bash
# Add origin remote
git remote add origin https://github.com/username/repo.git

# View remotes
git remote -v
```

#### Push and Pull
```bash
# Push local branch to remote
git push origin main

# Push new branch
git push -u origin feature/new-feature

# Pull latest changes
git pull origin main

# Fetch without merging
git fetch origin
```

### Collaboration Workflow

#### Fork and Pull Request
1. **Fork repository** on GitHub
2. **Clone your fork:**
   ```bash
   git clone https://github.com/yourusername/repo.git
   ```
3. **Add upstream remote:**
   ```bash
   git remote add upstream https://github.com/original/repo.git
   ```
4. **Create feature branch:**
   ```bash
   git checkout -b feature/improvement
   ```
5. **Make changes and commit**
6. **Push to your fork:**
   ```bash
   git push origin feature/improvement
   ```
7. **Create Pull Request** on GitHub

## Advanced Git Features

### Rebasing

#### Interactive Rebase
```bash
# Rebase last 3 commits
git rebase -i HEAD~3
```

Commands in interactive rebase:
- `pick`: Use commit as-is
- `reword`: Change commit message
- `edit`: Amend commit
- `squash`: Combine with previous commit
- `fixup`: Combine and discard message
- `drop`: Remove commit

#### Rebase vs. Merge
- **Merge:** Preserves history, creates merge commit
- **Rebase:** Creates linear history, cleaner but riskier

### Git Stash
```bash
# Save current work
git stash

# List stashes
git stash list

# Apply latest stash
git stash apply

# Apply specific stash
git stash apply stash@{1}

# Drop stash
git stash drop
```

### Git Reflog
```bash
# View reference log
git reflog

# Recover lost commit
git checkout <commit-hash>
```

## Code Review Best Practices

### Pull Request Guidelines

#### Creating Good PRs
- **Clear title:** "Add user authentication" not "Fix stuff"
- **Detailed description:** What, why, how
- **Link issues:** Reference related tasks
- **Small scope:** Easier to review
- **Tested code:** Include tests and verification

#### Review Checklist
- [ ] Code compiles without errors
- [ ] Tests pass
- [ ] No obvious bugs or security issues
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] Commit messages are clear
- [ ] Changes are minimal and focused

### Code Review Comments
- **Be constructive:** Focus on code, not person
- **Explain reasoning:** Why suggest a change?
- **Provide examples:** Show better alternatives
- **Ask questions:** Understand intent
- **Praise good code:** Encourage best practices

## Backup and Recovery

### Backup Strategies

#### Local Backups
- External hard drives
- Network attached storage (NAS)
- Cloud storage (Dropbox, Google Drive)

#### Repository Backups
- GitHub/GitLab as primary backup
- Mirror repositories
- Regular exports/archives

#### Automated Backups
```bash
# Cron job for daily backup
0 2 * * * tar -czf /backup/repo-$(date +%Y%m%d).tar.gz /path/to/repo
```

### Recovery Scenarios

#### Lost Local Changes
```bash
# Recover uncommitted changes
git checkout HEAD -- file.txt

# Recover from stash
git stash apply
```

#### Repository Corruption
```bash
# Clone fresh copy
git clone <remote-url> temp-repo
cp -r temp-repo/.git .git
rm -rf temp-repo
```

#### Lost Commits
```bash
# Find lost commits
git reflog

# Recover commit
git checkout <commit-hash>
git checkout -b recovered-branch
```

## Git Best Practices

### Commit Message Conventions

#### Good Commit Messages
```
feat: add user authentication system

- Implement JWT token validation
- Add password hashing with bcrypt
- Create user registration endpoint

Fixes #123
```

#### Bad Commit Messages
- "fix bug"
- "update code"
- "changes"
- Very long paragraphs

### Branch Naming
- `feature/user-login`
- `bugfix/null-pointer-exception`
- `hotfix/security-patch`
- `release/v1.2.0`

### Repository Structure
```
project/
├── .gitignore
├── README.md
├── LICENSE
├── pyproject.toml
├── src/
│   └── package/
├── tests/
├── docs/
└── scripts/
```

## GitHub/GitLab Features

### Issues and Projects
- **Issues:** Bug reports, feature requests, tasks
- **Projects:** Kanban boards for organization
- **Milestones:** Group issues by release/version

### GitHub Actions (CI/CD)
```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.8'
    - name: Install dependencies
      run: pip install -r requirements.txt
    - name: Run tests
      run: pytest
```

### Protected Branches
- Require pull request reviews
- Require status checks to pass
- Restrict force pushes
- Require linear history

## Common Git Problems and Solutions

### Problem: "fatal: remote origin already exists"
```bash
# Remove existing remote
git remote remove origin
git remote add origin <new-url>
```

### Problem: "Your branch is ahead by X commits"
```bash
# Push your changes
git push origin main

# Or reset if you want to discard
git reset --hard origin/main
```

### Problem: "Please commit your changes or stash them"
```bash
# Stash changes
git stash

# Pull, then restore
git pull
git stash pop
```

### Problem: "Merge conflict in file.txt"
See merge conflict resolution section above.

## Tools and Resources

### Git Clients
- **Command Line:** Most powerful, learn fundamentals
- **VS Code:** Integrated Git support
- **GitKraken:** Visual Git client
- **Sourcetree:** Free Git GUI

### Learning Resources
- [Pro Git Book](https://git-scm.com/book/en/v2)
- [GitHub Learning Lab](https://lab.github.com)
- [Learn Git Branching](https://learngitbranching.js.org)
- [Git Cheat Sheet](https://education.github.com/git-cheat-sheet-education.pdf)

### Practice Platforms
- [GitHub](https://github.com) - Host repositories
- [GitLab](https://gitlab.com) - Alternative platform
- [Bitbucket](https://bitbucket.org) - Atlassian platform

## Assessment Preparation

### Git Commands Quiz
- Initialize repository
- Create and switch branches
- Stage and commit changes
- Push/pull from remote
- Resolve merge conflicts

### Practical Exercises
1. Fork a repository
2. Create feature branch
3. Make changes and commit
4. Create pull request
5. Review team member's code

### Common Scenarios
- Lost uncommitted work
- Need to undo last commit
- Working with large files
- Collaborating on open source projects

## Key Takeaways
- Version control is essential for modern development
- Git enables powerful collaboration workflows
- Regular commits with clear messages improve project history
- Branching strategies prevent conflicts and organize work
- Code review improves quality and knowledge sharing
- Backup strategies protect against data loss
- Continuous learning of Git features increases productivity