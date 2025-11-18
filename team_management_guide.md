# Team Management Guide: Fake News Detection Capstone Project

## Overview
This guide provides a comprehensive framework for managing the Fake News Detection capstone project from start to end. As the Lead Instructor, you'll oversee two teams (Group A and Group B) through a 2-week intensive project cycle, ensuring successful delivery of ML models, APIs, and presentations by November 28.

## Project Structure
- **Duration**: November 13-28 (2 weeks)
- **Teams**: 2 groups of 4 students each
- **Roles per Team**: Project Manager, Data Analyst, ML Engineer, QA
- **Deliverables**: Source code, project report, presentation
- **Infrastructure**: AWS EC2 instance, GitHub repos, RAPIDS GPU acceleration

## Phase 1: Planning & Setup (Nov 13-15)
### Day 1: Kickoff Meeting
- **Agenda**:
  - Project overview and objectives
  - Team introductions and role assignments
  - Infrastructure setup walkthrough
  - Initial task assignments
- **Actions**:
  - Distribute AWS credentials and SSH keys
  - Ensure GitHub repos are created
  - Verify RAPIDS environment setup
  - Assign initial tasks (data download, EDA)

### Team Management Tasks
- **Monitor Setup**: Ensure all students can access EC2 instance
- **Role Clarity**: Confirm each student understands their responsibilities
- **Resource Distribution**: Provide ETL pipeline scripts and workflow documents
- **Communication Setup**: Establish Slack/Teams channels

## Phase 2: Data Processing (Nov 16-18)
### Daily Check-ins
- **Morning Standups**: 15-minute video calls
  - What was completed yesterday?
  - What is planned for today?
  - Any blockers?
- **Progress Tracking**: Use shared spreadsheets or project boards

### Management Focus
- **Data Pipeline Oversight**: Ensure ETL scripts are working
- **Quality Control**: Review initial data preprocessing
- **Collaboration**: Facilitate cross-team knowledge sharing
- **Risk Mitigation**: Address early technical issues

### Milestones
- Nov 15: AWS access confirmed
- Nov 18: Data preprocessed and validated

## Phase 3: Model Development (Nov 19-22)
### Weekly Rhythm
- **Daily Standups**: Continue morning check-ins
- **Mid-week Review**: Assess progress against timeline
- **Technical Support**: Office hours for code reviews

### Management Responsibilities
- **Performance Monitoring**: Track model accuracy targets (>85%)
- **GPU Utilization**: Ensure RAPIDS cuML is properly configured
- **Code Reviews**: Mandatory PR reviews before merging
- **Experiment Tracking**: Guide proper logging of training times

### Milestones
- Nov 22: Models trained and compared (sklearn vs cuML)

## Phase 4: Deployment & Testing (Nov 23-25)
### Intensive Support Phase
- **Daily Check-ins**: Increase frequency as deadline approaches
- **API Development**: Guide FastAPI implementation
- **Testing Coordination**: Oversee QA testing procedures
- **Performance Optimization**: Focus on latency (<500ms) and scalability

### Management Focus
- **Integration Testing**: Ensure end-to-end pipeline works
- **Documentation**: Require comprehensive code documentation
- **Security Review**: Basic security checks for API
- **Backup Plans**: Prepare contingencies for technical issues

### Milestones
- Nov 25: API deployed and tested

## Phase 5: Presentation & Evaluation (Nov 26-28)
### Final Push
- **Content Review**: Review presentation slides and demos
- **Rehearsals**: Schedule dry-run presentations
- **Feedback Sessions**: Provide constructive feedback
- **Final Adjustments**: Last-minute improvements

### Management Focus
- **Stakeholder Prep**: Ensure presentations meet requirements
- **Evaluation Setup**: Prepare scoring rubrics
- **Celebration Planning**: Recognize team achievements
- **Lessons Learned**: Capture insights for future cohorts

### Milestones
- Nov 28: Final presentations and project completion

## Communication Strategy
### Channels
- **Slack/Teams**: Daily communication and file sharing
- **GitHub**: Code reviews and issue tracking
- **Email**: Official announcements and credentials
- **Video Calls**: Standups and reviews

### Frequency
- **Daily**: Standup meetings (15 mins)
- **Weekly**: Full team reviews (1 hour)
- **As Needed**: Technical support and blocker resolution

## Risk Management
### Common Risks
- **Technical Blockers**: RAPIDS setup issues, API development challenges
- **Team Dynamics**: Communication breakdowns, unequal workload distribution
- **Time Management**: Scope creep, missed deadlines
- **Resource Issues**: AWS limits, data access problems

### Mitigation Strategies
- **Technical**: Maintain backup CPU-only implementations
- **Team**: Regular one-on-one check-ins, role rotations if needed
- **Time**: Strict milestone adherence, early warning systems
- **Resources**: Monitor usage, have escalation paths

## Escalation Procedures
### Issue Levels
- **Level 1**: Student can resolve independently (provide guidance)
- **Level 2**: Requires instructor intervention (schedule office hours)
- **Level 3**: Team-wide impact (immediate meeting)
- **Level 4**: Project-threatening (stakeholder involvement)

### Response Times
- **Urgent**: Same-day response
- **High**: Within 24 hours
- **Medium**: Within 48 hours
- **Low**: Weekly review

## Success Metrics
### Team Performance
- **Delivery**: All milestones met on time
- **Quality**: Code passes QA, models meet accuracy targets
- **Collaboration**: Positive peer feedback, effective communication
- **Innovation**: Creative solutions, proper RAPIDS utilization

### Individual Performance
- **Role Fulfillment**: Meets responsibilities as PM/Data Analyst/ML Engineer/QA
- **Technical Growth**: Demonstrates learning and application of skills
- **Professionalism**: Timely communication, quality work
- **Team Contribution**: Supports colleagues, shares knowledge

## Tools and Resources
### Management Tools
- **Project Tracking**: GitHub Projects or Trello
- **Communication**: Slack/Teams channels
- **Code Review**: GitHub PRs
- **Documentation**: Shared Google Docs/OneDrive

### Technical Resources
- **RAPIDS Docs**: https://docs.rapids.ai/
- **AWS Guides**: EC2, S3, IAM documentation
- **ML Resources**: Scikit-learn, cuML comparisons
- **API Frameworks**: FastAPI tutorials

## Instructor Responsibilities
### Daily
- Monitor progress via standups
- Provide technical guidance
- Review code and architectures
- Address blockers promptly

### Weekly
- Assess overall project health
- Provide feedback on deliverables
- Adjust timelines as needed
- Communicate with stakeholders

### Throughout
- Maintain positive team morale
- Ensure fair workload distribution
- Document lessons learned
- Prepare for final evaluation

## Student Support Framework
### Technical Support
- **Office Hours**: Scheduled and on-demand
- **Code Reviews**: Mandatory for all PRs
- **Debugging Sessions**: Pair programming for complex issues
- **Resource Provision**: Provide scripts, templates, examples

### Motivational Support
- **Regular Feedback**: Positive reinforcement and constructive criticism
- **Goal Setting**: Help teams set achievable objectives
- **Celebration**: Recognize milestones and achievements
- **Encouragement**: Support through challenging phases

## Final Evaluation
### Assessment Components
- **Source Code (30%)**: Quality, documentation, functionality
- **Project Report (40%)**: Technical depth, analysis, presentation
- **Presentation (30%)**: Communication, demonstration, Q&A

### Grading Criteria
- **Technical Excellence**: Model performance, implementation quality
- **Project Management**: Timeline adherence, team collaboration
- **Business Impact**: Problem-solving effectiveness, scalability
- **Professional Skills**: Communication, documentation, ethics

## Post-Project Activities
### Wrap-up
- **Retrospectives**: Team and individual reflections
- **Feedback Collection**: Student surveys and instructor notes
- **Certificate Distribution**: Recognize completion
- **Showcase**: Share successful projects with broader community

### Continuous Improvement
- **Lesson Documentation**: Update curriculum based on insights
- **Resource Updates**: Improve scripts and guides
- **Best Practices**: Develop templates for future cohorts
- **Community Building**: Connect alumni for networking

This guide ensures systematic management of the capstone project, balancing technical oversight with team development and successful delivery of high-quality fake news detection solutions.