# Module 23 Assignment: Git Workflow and Version Control

## Overview
In this assignment, you will demonstrate proficiency in Git version control and collaborative development practices. You will work individually and in teams to complete various Git operations, establish proper workflows, and implement best practices for code versioning.

## Learning Objectives
- Master fundamental Git operations
- Implement branching and merging strategies
- Practice collaborative development workflows
- Set up automated backups and recovery procedures
- Follow industry best practices for version control

## Assignment Components

### Part 1: Individual Git Proficiency (40%)

#### Task 1.1: Repository Setup and Basic Operations
1. **Create a new Git repository** for a data science project
2. **Initialize with proper structure:**
   ```
   project/
   ├── .gitignore
   ├── README.md
   ├── requirements.txt
   ├── src/
   │   ├── __init__.py
   │   └── analysis.py
   ├── tests/
   │   └── test_analysis.py
   └── notebooks/
       └── exploration.ipynb
   ```

3. **Perform basic operations:**
   - Add and commit files with descriptive messages
   - Modify files and create multiple commits
   - View commit history and changes
   - Create and manage `.gitignore`

#### Task 1.2: Branching and Merging
1. **Create feature branches:**
   - `feature/data-preprocessing`
   - `feature/model-development`
   - `bugfix/data-loading`

2. **Implement changes on each branch:**
   - Add data preprocessing functions
   - Implement machine learning model
   - Fix data loading bug

3. **Merge branches back to main:**
   - Use appropriate merge strategies
   - Handle any merge conflicts
   - Maintain clean commit history

#### Task 1.3: Working with Remotes
1. **Create GitHub repository**
2. **Push local repository to remote**
3. **Set up remote tracking branches**
4. **Simulate collaborative workflow:**
   - Create pull request
   - Review and merge changes

### Part 2: Team Collaboration (30%)

#### Task 2.1: Team Repository Setup
1. **Create team repository** on GitHub
2. **Establish contribution guidelines:**
   - Branch naming conventions
   - Commit message standards
   - Code review requirements

3. **Set up branch protection rules:**
   - Require pull request reviews
   - Require status checks
   - Prevent force pushes to main

#### Task 2.2: Collaborative Development
1. **Assign team roles:**
   - Repository maintainer
   - Code reviewers
   - Contributors

2. **Implement team workflow:**
   - Create feature branches
   - Submit pull requests
   - Conduct code reviews
   - Merge approved changes

#### Task 2.3: Conflict Resolution
1. **Create intentional conflicts:**
   - Multiple team members edit same file
   - Different approaches to same problem

2. **Resolve conflicts collaboratively:**
   - Communicate resolution approach
   - Implement agreed solution
   - Document conflict resolution process

### Part 3: Advanced Git Features (20%)

#### Task 3.1: Rebasing and History Management
1. **Practice interactive rebasing:**
   - Squash related commits
   - Reorder commits
   - Edit commit messages

2. **Clean up commit history:**
   - Remove unnecessary commits
   - Create logical commit progression
   - Maintain meaningful history

#### Task 3.2: Git Hooks and Automation
1. **Implement pre-commit hooks:**
   - Code formatting checks
   - Linting validation
   - Test execution

2. **Set up GitHub Actions:**
   - Automated testing on push
   - Code quality checks
   - Deployment automation

### Part 4: Backup and Recovery (10%)

#### Task 4.1: Backup Strategies
1. **Implement multiple backup methods:**
   - Local repository backups
   - Remote repository mirroring
   - Cloud storage backups

2. **Create backup scripts:**
   - Automated daily backups
   - Repository archiving
   - Recovery procedures

#### Task 4.2: Recovery Simulation
1. **Simulate data loss scenarios:**
   - Accidental file deletion
   - Repository corruption
   - Lost commits

2. **Demonstrate recovery procedures:**
   - Restore from backups
   - Use git reflog
   - Recover lost work

## Deliverables

### Individual Repository
- **GitHub Repository:** Public repository with complete project
- **README.md:** Project description, setup instructions, usage
- **Commit History:** Clean, logical progression of changes
- **Branch Structure:** Proper branching strategy implementation
- **Documentation:** Git workflow documentation

### Team Repository
- **Team Project Repository:** Collaborative data science project
- **Contribution Guidelines:** Clear rules for team collaboration
- **Pull Requests:** Evidence of code review process
- **Issues:** Task tracking and bug reports
- **Wiki:** Team documentation and processes

### Documentation Package
- **Git Workflow Guide:** Step-by-step instructions for team processes
- **Backup Strategy Document:** Comprehensive backup and recovery plan
- **Code Review Checklist:** Standards for reviewing code changes
- **Troubleshooting Guide:** Solutions for common Git problems

## Assessment Criteria

### Technical Proficiency (40%)
- **Git Commands:** Correct use of Git operations
- **Branching Strategy:** Appropriate branch management
- **Commit Quality:** Clear messages and logical grouping
- **Conflict Resolution:** Effective handling of merge conflicts
- **Remote Operations:** Proper use of GitHub/GitLab features

### Collaboration Skills (30%)
- **Team Communication:** Clear and timely communication
- **Code Reviews:** Constructive feedback and improvements
- **Workflow Adherence:** Following established processes
- **Conflict Management:** Professional handling of disagreements
- **Documentation:** Clear and comprehensive documentation

### Best Practices (20%)
- **Code Quality:** Following coding standards
- **Security:** Proper handling of sensitive information
- **Automation:** Use of Git hooks and CI/CD
- **Backup Strategy:** Comprehensive data protection
- **Version Management:** Proper tagging and releases

### Innovation and Problem Solving (10%)
- **Creative Solutions:** Novel approaches to challenges
- **Process Improvement:** Suggestions for workflow enhancement
- **Tool Integration:** Effective use of Git ecosystem tools
- **Scalability:** Solutions that work for larger teams

## Tools and Resources

### Required Tools
- **Git:** Version control system
- **GitHub/GitLab:** Remote repository hosting
- **VS Code:** Code editor with Git integration
- **Terminal/Command Line:** Git command execution

### Recommended Tools
- **GitKraken/Sourcetree:** Visual Git clients
- **GitHub Desktop:** Simplified Git operations
- **Pre-commit:** Git hooks framework
- **Git LFS:** Large file storage

### Learning Resources
- [Pro Git Book](https://git-scm.com/book)
- [GitHub Learning Lab](https://lab.github.com)
- [Learn Git Branching](https://learngitbranching.js.org)
- [Git Documentation](https://git-scm.com/doc)

## Timeline

### Week 1: Individual Setup
- Repository creation and basic operations
- Branching and merging practice
- Remote repository setup

### Week 2: Team Collaboration
- Team repository setup
- Collaborative workflow implementation
- Code review processes

### Week 3: Advanced Features
- Rebasing and history management
- Automation setup
- Backup strategy development

### Week 4: Finalization
- Recovery testing
- Documentation completion
- Final presentations

## Grading Rubric

| Criteria | Excellent (90-100%) | Good (80-89%) | Satisfactory (70-79%) | Needs Improvement (<70%) |
|----------|-------------------|---------------|---------------------|-------------------------|
| **Git Proficiency** | Expert command of all Git operations, creative use of advanced features | Solid understanding of core Git operations, proper use of standard workflows | Basic Git operations working, some errors in complex scenarios | Frequent errors in basic Git operations, lack of understanding |
| **Collaboration** | Proactive communication, excellent code reviews, strong team contribution | Good communication, constructive reviews, balanced team participation | Adequate communication, basic reviews, minimal team interaction | Poor communication, unhelpful reviews, minimal team engagement |
| **Best Practices** | Exemplary adherence to standards, innovative improvements to processes | Good adherence to standards, proper tool usage | Basic adherence to standards, some tool usage | Poor adherence to standards, incorrect tool usage |
| **Documentation** | Comprehensive, clear, professional documentation throughout | Good documentation with clear explanations | Basic documentation provided | Minimal or unclear documentation |

## Important Notes

### Academic Integrity
- All work must be original and properly attributed
- No copying of code without citation
- Individual assessments must reflect personal work
- Team assessments must reflect collaborative effort

### Late Policy
- 10% deduction per day late
- Extensions granted only with documentation
- Incomplete assignments receive proportional grading

### Support
- Office hours for Git questions
- Online resources and tutorials provided
- Peer mentoring encouraged
- Instructor code reviews available

### Success Tips
- Start early and practice regularly
- Document your Git learning journey
- Participate actively in team activities
- Ask questions when stuck
- Review and learn from feedback

Remember, mastering version control is a crucial skill for any data science professional. This assignment provides practical experience that will serve you throughout your career. Focus on understanding concepts deeply rather than just completing tasks.