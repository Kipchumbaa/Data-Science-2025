# Team Management Guide: Fake News Detection Capstone Project (GCP Version)

## Overview
This guide provides a comprehensive framework for managing the Fake News Detection capstone project on Google Cloud Platform from start to end. As the Lead Instructor, you'll oversee two teams (Group A and Group B) through a 2-week intensive project cycle, ensuring successful delivery of ML models, APIs, and presentations.

## Project Structure
- **Duration**: November 13-28 (2 weeks)
- **Teams**: 2 groups of 4 students each
- **Roles per Team**: Project Manager, Data Analyst, ML Engineer, QA
- **Deliverables**: Source code, project report, presentation
- **Infrastructure**: GCP Compute Engine VM, Cloud Storage, RAPIDS GPU acceleration

## Phase 1: Planning & Setup (Nov 13-15)
### Day 1: Kickoff Meeting
- **Agenda**:
  - Project overview and objectives
  - Team introductions and role assignments
  - GCP infrastructure walkthrough
  - Initial task assignments
- **Actions**:
  - Distribute GCP service account credentials
  - Ensure Cloud Storage access is working
  - Verify RAPIDS environment setup
  - Assign initial tasks (data download, EDA)

### Team Management Tasks
- **Monitor Setup**: Ensure all students can access GCP VM
- **Role Clarity**: Confirm each student understands their GCP service account permissions
- **Resource Distribution**: Provide ETL pipeline scripts and GCP workflow guides
- **Communication Setup**: Establish Slack/Teams channels

## Phase 2: Data Processing (Nov 16-18)
### Daily Check-ins
- **Morning Standups**: 15-minute video calls
  - What was completed yesterday?
  - What is planned for today?
  - Any blockers?
- **Progress Tracking**: Use shared Google Docs or GitHub Projects

### Management Focus
- **Data Pipeline Oversight**: Ensure ETL scripts are working with Cloud Storage
- **Quality Control**: Review initial data preprocessing and GCS uploads
- **Collaboration**: Facilitate cross-team knowledge sharing
- **Risk Mitigation**: Address early GCP authentication or permission issues

### Milestones
- Nov 15: GCP access confirmed, service accounts working
- Nov 18: Data preprocessed and uploaded to Cloud Storage

## Phase 3: Model Development (Nov 19-22)
### Weekly Rhythm
- **Daily Standups**: Continue morning check-ins
- **Mid-week Review**: Assess progress against timeline
- **Technical Support**: Office hours for GCP and RAPIDS debugging

### Management Responsibilities
- **Performance Monitoring**: Track model accuracy targets (>85%)
- **GPU Utilization**: Ensure RAPIDS cuML is properly configured on GCP
- **Code Reviews**: Mandatory PR reviews before merging
- **Experiment Tracking**: Guide proper logging of training times

### Milestones
- Nov 22: Models trained and sklearn vs cuML comparison completed

## Phase 4: Deployment & Testing (Nov 23-25)
### Intensive Support Phase
- **Daily Check-ins**: Increase frequency as deadline approaches
- **API Development**: Guide FastAPI deployment to Cloud Run
- **Testing Coordination**: Oversee QA testing procedures on GCP
- **Performance Optimization**: Focus on latency (<500ms) and Cloud Run scaling

### Management Focus
- **Integration Testing**: Ensure end-to-end pipeline works with GCP services
- **Documentation**: Require comprehensive code documentation
- **Security Review**: GCP IAM and VPC security verification
- **Backup Plans**: Prepare contingencies for GCP service issues

### Milestones
- Nov 25: API deployed to Cloud Run and tested

## Phase 5: Presentation & Evaluation (Nov 26-28)
### Final Push
- **Content Review**: Review presentation slides and Cloud Run demos
- **Rehearsals**: Schedule dry-run presentations
- **Feedback Sessions**: Provide constructive feedback
- **Final Adjustments**: Last-minute improvements

### Management Focus
- **Stakeholder Prep**: Ensure presentations highlight GCP architecture
- **Evaluation Setup**: Prepare scoring rubrics
- **Celebration Planning**: Recognize team achievements
- **Lessons Learned**: Capture insights for future GCP projects

### Milestones
- Nov 28: Final presentations and project completion

## Communication Strategy
### Channels
- **Slack/Teams**: Daily communication and file sharing
- **GitHub**: Code reviews and issue tracking
- **Google Drive**: Shared documentation and progress tracking
- **GCP Console**: Resource monitoring and cost tracking

### Frequency
- **Daily**: Standup meetings (15 mins)
- **Weekly**: Full team reviews (1 hour)
- **As Needed**: GCP technical support and blocker resolution

## Risk Management
### Common Risks
- **GCP Authentication**: Service account key management issues
- **GPU Access**: RAPIDS GPU configuration problems
- **Cost Overruns**: Unexpected GCP charges
- **Data Storage**: Cloud Storage permission or quota issues
- **API Limits**: External service rate limits

### Mitigation Strategies
- **GCP Access**: Provide backup authentication methods
- **GPU Fallback**: CPU RAPIDS implementation available
- **Cost Monitoring**: Set up budget alerts and monitoring
- **Data Backup**: Multiple storage locations and snapshots
- **API Management**: Implement rate limiting and caching

## Escalation Procedures
### Issue Levels
- **Level 1**: Student can resolve independently (provide GCP documentation)
- **Level 2**: Requires instructor intervention (schedule office hours)
- **Level 3**: Team-wide impact (immediate meeting)
- **Level 4**: GCP service outage (alternative platform consideration)

### Response Times
- **Urgent**: Same-day response for GCP access issues
- **High**: Within 24 hours for technical blockers
- **Medium**: Within 48 hours for general issues
- **Low**: Weekly review for minor concerns

## Success Metrics
### Team Performance
- **Delivery**: All milestones met on schedule
- **Quality**: Code passes QA, models meet accuracy targets
- **Collaboration**: Positive peer feedback, effective GCP usage
- **Innovation**: Creative solutions using GCP services

### Individual Performance
- **Role Fulfillment**: Meets responsibilities as PM/Data Analyst/ML Engineer/QA
- **Technical Growth**: Demonstrates GCP and RAPIDS proficiency
- **Professionalism**: Timely communication, quality work
- **Team Contribution**: Supports colleagues, shares GCP knowledge

## Tools and Resources
### Management Tools
- **GitHub Projects**: Task tracking and milestone management
- **Google Docs**: Shared documentation and progress reports
- **GCP Console**: Resource monitoring and cost management
- **Cloud Monitoring**: Performance metrics and alerting

### Technical Resources
- **GCP Documentation**: https://cloud.google.com/docs
- **RAPIDS Documentation**: https://docs.rapids.ai/
- **Cloud IAM**: Service account management guides
- **Cloud Storage**: Data management best practices

## Instructor Responsibilities
### Daily
- Monitor progress via standups
- Provide GCP and RAPIDS technical guidance
- Review code and Cloud Architecture
- Address authentication and permission blockers

### Weekly
- Assess overall project health
- Provide feedback on deliverables
- Adjust timelines as needed
- Communicate with stakeholders

### Throughout
- Maintain positive team morale
- Ensure fair GCP resource allocation
- Document GCP lessons learned
- Prepare for final evaluation

## Student Support Framework
### Technical Support
- **Office Hours**: Scheduled GCP assistance
- **Code Reviews**: Mandatory for all PRs
- **Debugging Sessions**: GCP configuration troubleshooting
- **Resource Provision**: Provide service account keys and setup scripts

### Motivational Support
- **Regular Feedback**: Positive reinforcement and constructive criticism
- **Goal Setting**: Help teams set achievable GCP-based objectives
- **Celebration**: Recognize GCP architecture achievements
- **Encouragement**: Support through GCP learning curve

## GCP-Specific Management
### Resource Management
- **Cost Monitoring**: Daily budget checks via GCP Console
- **Quota Management**: Monitor API limits and resource usage
- **Access Control**: Regular IAM permission audits
- **Backup Strategy**: Automated VM snapshots and data backups

### Performance Optimization
- **Instance Selection**: Right-size Compute Engine instances
- **Storage Classes**: Optimize Cloud Storage costs
- **Network Optimization**: Minimize data transfer costs
- **Caching Strategy**: Implement appropriate caching layers

### Security Management
- **IAM Best Practices**: Least-privilege access principles
- **Key Management**: Secure service account key handling
- **Network Security**: VPC firewall rule management
- **Audit Logging**: Regular security log reviews

## Continuous Improvement
### Feedback Collection
- **Daily**: Quick pulse surveys on GCP experience
- **Weekly**: Structured feedback on tools and processes
- **End of Project**: Comprehensive GCP retrospective

### Process Updates
- **Documentation**: Regular GCP guide updates
- **Tools**: Evaluation of GCP service effectiveness
- **Training**: Instructor GCP skill development
- **Templates**: Reusable GCP project templates

### Lessons Learned
- **GCP Patterns**: Common setup and configuration issues
- **Cost Optimization**: Budget management insights
- **Security**: IAM and access control learnings
- **Performance**: GCP service selection and configuration

## Emergency Procedures
### GCP Service Outage
1. **Assessment**: Determine scope and impact on project
2. **Communication**: Notify all affected teams immediately
3. **Workaround**: Implement local development alternatives
4. **Resolution**: Monitor GCP status and restore when available

### Authentication Crisis
1. **Immediate Response**: Provide backup access methods
2. **Key Regeneration**: Create new service account keys if compromised
3. **Access Restoration**: Update all team members with new credentials
4. **Security Review**: Audit access patterns and permissions

### Cost Overrun
1. **Alert Response**: Immediate cost monitoring activation
2. **Resource Shutdown**: Stop non-essential GCP resources
3. **Budget Adjustment**: Implement stricter cost controls
4. **Root Cause Analysis**: Identify and fix cost drivers

### Team Crisis
1. **Private Discussion**: Individual conversations with team members
2. **Mediation**: Facilitated conflict resolution sessions
3. **Rebalancing**: Role or team adjustments as needed
4. **Support Resources**: Additional GCP training and mentoring

## GCP Cost Management
### Budget Setup
```bash
# Create project budget
gcloud billing budgets create fake-news-budget \
    --billing-account=YOUR_BILLING_ACCOUNT \
    --display-name="Fake News Project Budget" \
    --amount=300.00 \
    --currency=USD \
    --threshold-rule=percent=80 \
    --threshold-rule=percent=100
```

### Cost Monitoring
- **Daily Checks**: Review GCP Console billing dashboard
- **Alerts**: Automatic notifications at 80% and 100% budget
- **Resource Optimization**: Right-size instances and storage
- **Cost Allocation**: Label resources for team attribution

### Optimization Strategies
- **Scheduled Shutdowns**: Stop VM during non-development hours
- **Preemptible Instances**: Use for non-critical workloads
- **Storage Classes**: Choose appropriate Cloud Storage tiers
- **Commitment Discounts**: Consider committed use discounts for longer projects

This GCP-focused team management framework ensures the project stays on track with proactive issue prevention, rapid resolution, and continuous improvement throughout the 2-week development cycle on Google Cloud Platform.