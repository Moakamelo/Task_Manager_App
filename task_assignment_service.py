# task_assignment_service.py
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
import numpy as np
import os
import requests
import pickle
import json
from datetime import datetime
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from collections import defaultdict
# from email_services import send_task_assignment

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configuration
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev_secret_key_for_task_service')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'postgresql://postgres:password@localhost:5432/task_db')
db = SQLAlchemy(app)

# Employee service configuration
EMPLOYEE_SERVICE_URL = os.environ.get('EMPLOYEE_SERVICE_URL', 'http://localhost:5001/api')
API_KEY = os.environ.get('API_KEY', 'dev_api_key')

class Task(db.Model):
    __tablename__ = 'tasks'
    
    task_id = db.Column(db.String(50), primary_key=True)
    project_type = db.Column(db.String(100), nullable=False)
    skills = db.Column(db.ARRAY(db.String(50)), default=[])
    complexity = db.Column(db.String(20), nullable=False)
    priority = db.Column(db.String(20), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Assignment details
    email_assigned_to =db.Column(db.String(50))
    assigned_to = db.Column(db.String(50), nullable=True)  # employee ID
    assigned_at = db.Column(db.DateTime, nullable=True)
    status = db.Column(db.String(20), default='assigned')  # assigned, in_progress, pending_approval, completed, rejected
    start_date = db.Column(db.DateTime, nullable=True)
    due_date = db.Column(db.DateTime, nullable=True)

    # Submission and approval details
    submitted_at = db.Column(db.DateTime, nullable=True)
    approved_by = db.Column(db.String(50), nullable=True)  # manager ID
    approved_at = db.Column(db.DateTime, nullable=True)
    approval_notes = db.Column(db.Text, nullable=True)
    
    # Metrics
    completion_date = db.Column(db.DateTime, nullable=True)
    success_rating = db.Column(db.Integer, nullable=True)  # 1-10 rating
    
    def to_dict(self):
        return {
            'task_id': self.task_id,
            'project_type': self.project_type,
            'skills': self.skills,
            'complexity': self.complexity,
            'priority': self.priority,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'assigned_to': self.assigned_to,
            'email_assigned_to': self.email_assigned_to,
            'assigned_at': self.assigned_at.isoformat() if self.assigned_at else None,
            'status': self.status,
            'submitted_at': self.submitted_at.isoformat() if self.submitted_at else None,
            'approved_by': self.approved_by,
            'approved_at': self.approved_at.isoformat() if self.approved_at else None,
            'approval_notes': self.approval_notes,
            'completion_date': self.completion_date.isoformat() if self.completion_date else None,
            'success_rating': self.success_rating
        }

class TaskHistory(db.Model):
    __tablename__ = 'task_history'
    
    id = db.Column(db.Integer, primary_key=True)
    task_id = db.Column(db.String(50), db.ForeignKey('tasks.task_id'), nullable=False)
    action = db.Column(db.String(50), nullable=False)  # task_created, task_assigned, task_started, task_submitted, task_approved, task_rejected, etc.
    performed_by = db.Column(db.String(50), nullable=False)  # employee ID who performed the action
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    details = db.Column(db.Text, nullable=True)  # Additional context about the action
    
    # Relationship with Task
    task = db.relationship('Task', backref=db.backref('history', lazy='dynamic'))
    
    def to_dict(self):
        return {
            'id': self.id,
            'task_id': self.task_id,
            'action': self.action,
            'performed_by': self.performed_by,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'details': self.details
        }

@app.cli.command("create_tables")
def create_tables():
    db.create_all()
    print("Task tables created!")

# Define project types and their required skills
PROJECT_TYPES = {
    "website_development": [
        "HTML", "CSS", "JavaScript", "React", "Vue", "Angular", "Node.js", 
        "PHP", "UI/UX", "Responsive Design", "Web Security"
    ],
    "mobile_app_development": [
        "Swift", "Kotlin", "React Native", "Flutter", "Java", "Mobile UI/UX", 
        "Firebase", "App Store Optimization"
    ],
    "machine_learning": [
        "Python", "TensorFlow", "PyTorch", "Scikit-learn", "NLP", "Computer Vision", 
        "Data Mining", "Statistics", "Feature Engineering"
    ],
    "data_engineering": [
        "SQL", "ETL", "Data Warehouse", "Spark", "Hadoop", "Data Modeling", 
        "MongoDB", "PostgreSQL", "AWS Redshift"
    ],
    "api_development": [
        "REST API", "GraphQL", "Node.js", "Django", "Flask", "API Security",
        "API Testing", "API Documentation", "Microservices"
    ],
    "devops": [
        "Docker", "Kubernetes", "CI/CD", "AWS", "Azure", "GCP", "Jenkins",
        "Terraform", "Ansible", "System Administration"
    ],
    "blockchain": [
        "Solidity", "Smart Contracts", "Ethereum", "Web3.js", "DApps",
        "Blockchain Security", "Consensus Algorithms"
    ],
    "cybersecurity": [
        "Network Security", "Penetration Testing", "Vulnerability Assessment",
        "Security Auditing", "Encryption", "Ethical Hacking", "OWASP"
    ],
    "game_development": [
        "Unity", "Unreal Engine", "C#", "C++", "Game Design", "3D Modeling",
        "Animation", "Physics Simulation", "Multiplayer Networking"
    ]
}

# Helper functions for API calls to employee service
def api_headers():
    return {'X-API-KEY': API_KEY, 'Content-Type': 'application/json'}

def get_all_employees():
    """Get all employees from the employee service"""
    try:
        response = requests.get(f'{EMPLOYEE_SERVICE_URL}/employees', headers=api_headers())
        if response.status_code == 200:
            return response.json()
        print(f"Failed to get employees: {response.status_code}, {response.text}")
        return []
    except Exception as e:
        print(f"Exception while getting employees: {str(e)}")
        return []

def get_employee_by_id(emp_id):
    """Get a specific employee's details from the employee service"""
    try:
        response = requests.get(f'{EMPLOYEE_SERVICE_URL}/employees/{emp_id}', headers=api_headers())
        if response.status_code == 200:
            return response.json()
        print(f"Failed to get employee {emp_id}: {response.status_code}, {response.text}")
        return None
    except Exception as e:
        print(f"Exception while getting employee {emp_id}: {str(e)}")
        return None

# Function to load or train the model
def train_scoring_model(employees_data):
    """Train a model to predict assignment success scores (no scaling needed)"""
    
    # Create synthetic assignment history with success scores
    training_data = generate_training_data(employees_data)
    
    if len(training_data) < 10:
        print("Insufficient training data, falling back to rule-based scoring")
        return None
    
    df = pd.DataFrame(training_data)
    
    # Features: task and employee characteristics
    feature_cols = [
        'skill_match_percentage', 'experience', 'success_rate', 
        'tasks_completed', 'complexity_score', 'priority_score',
        'workload_factor'
    ]
    
    X = df[feature_cols]
    y = df['assignment_success_score']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model (RandomForest doesn't need scaling)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"Model Performance - Train: {train_score:.3f}, Test: {test_score:.3f}")
    
    return {
        'model': model,
        'feature_columns': feature_cols
        # No scaler needed!
    }

def generate_training_data(employees):
    """Generate synthetic training data with realistic success scores"""
    training_data = []
    
    # Create synthetic tasks
    complexities = ['Low', 'Medium', 'High']
    priorities = ['Low', 'Medium', 'High']
    
    # Extract all skills
    all_skills = set()
    for emp in employees:
        if isinstance(emp, dict) and 'skills' in emp:
            all_skills.update(emp.get('skills', []))
    
    all_skills = list(all_skills)
    
    # Generate training examples
    for _ in range(1000):  # Generate 1000 synthetic assignments
        # Create random task
        num_required_skills = np.random.randint(1, min(4, len(all_skills)) + 1)
        required_skills = np.random.choice(all_skills, num_required_skills, replace=False)
        complexity = np.random.choice(complexities)
        priority = np.random.choice(priorities)
        
        # Pick random employee
        emp = np.random.choice(employees)
        if not isinstance(emp, dict):
            continue
            
        emp_skills = emp.get('skills', [])
        if not isinstance(emp_skills, list):
            emp_skills = []
        
        # Calculate features
        skill_match = len(set(required_skills).intersection(set(emp_skills))) / len(required_skills)
        
        # Skip if no skill match
        if skill_match == 0:
            continue
        
        experience = emp.get('experience', 0)
        success_rate = emp.get('success_rate', 0.5)
        tasks_completed = emp.get('tasks_completed', 0)
        
        complexity_score = 0 if complexity == 'Low' else 1 if complexity == 'Medium' else 2
        priority_score = 0 if priority == 'Low' else 1 if priority == 'Medium' else 2
        workload_factor = np.random.uniform(0.5, 1.0)  # Random workload
        
        # Calculate realistic success score based on factors
        success_score = calculate_realistic_success_score(
            skill_match, experience, success_rate, tasks_completed,
            complexity_score, workload_factor
        )
        
        training_data.append({
            'skill_match_percentage': skill_match * 100,
            'experience': experience,
            'success_rate': success_rate,
            'tasks_completed': tasks_completed,
            'complexity_score': complexity_score,
            'priority_score': priority_score,
            'workload_factor': workload_factor,
            'assignment_success_score': success_score
        })
    
    return training_data


def calculate_realistic_success_score(skill_match, experience, success_rate, 
                                    tasks_completed, complexity_score, workload_factor):
    """Calculate a realistic success score for training data"""
    
    # Base score from skill match
    base_score = skill_match
    
    # Experience factor (helps with complex tasks)
    exp_factor = 1.0 + (experience / 20.0)  # Max 1.5x boost
    if complexity_score == 2:  # High complexity
        exp_factor = 1.0 + (experience / 10.0)  # More experience needed
    
    # Success rate directly influences outcome
    success_factor = 0.5 + (success_rate / 2.0)  # 0.5 to 1.0 multiplier
    
    # Task completion experience
    completion_factor = 1.0 + min(tasks_completed / 200.0, 0.3)  # Max 1.3x
    
    # Workload penalty
    workload_penalty = workload_factor
    
    # Calculate final score with some randomness
    final_score = (base_score * exp_factor * success_factor * completion_factor * workload_penalty)
    
    # Add some noise and clamp to [0, 1]
    final_score += np.random.normal(0, 0.1)
    final_score = max(0, min(1, final_score))
    
    return final_score

def assign_tasks_ml_scoring(tasks, employees, model_data=None):
    """Assign tasks using ML model (no scaler version)"""
    
    if not employees or not tasks:
        return {"error": "Invalid input data"}
    
    assignments = {}
    current_workloads = defaultdict(int)
    
    for emp in employees:
        if isinstance(emp, dict) and 'emp_id' in emp:
            current_workloads[emp['emp_id']] = emp.get('current_workload', 0)
    
    for task in tasks:
        if not isinstance(task, dict):
            continue
            
        task_id = task.get('task_id', 'unknown')
        required_skills = task.get('skills', [])
        
        candidate_scores = []
        
        for emp in employees:
            if not isinstance(emp, dict) or 'emp_id' not in emp:
                continue
            
            emp_skills = emp.get('skills', [])
            if not isinstance(emp_skills, list):
                emp_skills = []
            
            # Calculate skill match
            skill_match = 0
            if required_skills and emp_skills:
                skill_match = len(set(required_skills).intersection(set(emp_skills))) / len(required_skills)
            
            if skill_match == 0:
                continue
            
            # Prepare features for ML model
            features = {
                'skill_match_percentage': skill_match * 100,
                'experience': emp.get('experience', 0),
                'success_rate': emp.get('success_rate', 0.5),
                'tasks_completed': emp.get('tasks_completed', 0),
                'complexity_score': 0 if task.get('complexity') == 'Low' else 1 if task.get('complexity') == 'Medium' else 2,
                'priority_score': 0 if task.get('priority') == 'Low' else 1 if task.get('priority') == 'Medium' else 2,
                'workload_factor': max(0.1, 1.0 - (current_workloads[emp['emp_id']] / 5.0))
            }
            
            # Predict score using ML model if available
            if model_data and model_data.get('model'):
                try:
                    # Create feature DataFrame (keeps column names)
                    feature_df = pd.DataFrame([features])[model_data['feature_columns']]
                    predicted_score = model_data['model'].predict(feature_df)[0]
                except Exception as e:
                    print(f"ML prediction error: {e}")
                    # Fallback to rule-based scoring
                    predicted_score = calculate_assignment_score(emp, task, skill_match, current_workloads[emp['emp_id']])
            else:
                # Use rule-based scoring
                predicted_score = calculate_assignment_score(emp, task, skill_match, current_workloads[emp['emp_id']])
            
            candidate_scores.append({
                'emp_id': emp['emp_id'],
                'name': emp.get('name', ''),
                'score': predicted_score,
                'skill_match': skill_match
            })
        
        if not candidate_scores:
            assignments[task_id] = "No eligible employees found"
            continue
        
        # Sort by score and assign to best candidate
        candidate_scores.sort(key=lambda x: x['score'], reverse=True)
        best_candidate = candidate_scores[0]
        
        assignments[task_id] = {
            "emp_id": best_candidate['emp_id'],
            "name": best_candidate['name'],
            "skill_match_percentage": f"{best_candidate['skill_match'] * 100:.1f}%",
            "predicted_success_score": f"{best_candidate['score']:.3f}"
        }
        
        current_workloads[best_candidate['emp_id']] += 1
    
    return assignments


def calculate_assignment_score(employee, task, skill_match, current_workload):
    """Calculate assignment score for an employee-task pair (fallback function)"""
    
    # Base score from skill match (0-1)
    skill_score = skill_match
    
    # Experience factor (normalize to 0-1, assuming max 10 years)
    experience = employee.get('experience', 0)
    experience_score = min(experience / 10.0, 1.0)
    
    # Success rate (already 0-1)
    success_rate = employee.get('success_rate', 0.5)
    
    # Task completion history (normalize to 0-1, assuming max 100 tasks)
    tasks_completed = employee.get('tasks_completed', 0)
    completion_score = min(tasks_completed / 100.0, 1.0)
    
    # Workload penalty (0-1, where 1 is best)
    # Assuming max reasonable workload is 5 concurrent tasks
    workload_penalty = max(0, 1 - (current_workload / 5.0))
    
    # Priority boost for high-priority tasks
    priority_boost = 1.0
    if task.get('priority') == 'High':
        priority_boost = 1.2
    elif task.get('priority') == 'Low':
        priority_boost = 0.9
    
    # Complexity matching - experienced employees get boost for complex tasks
    complexity_factor = 1.0
    if task.get('complexity') == 'High' and experience > 5:
        complexity_factor = 1.1
    elif task.get('complexity') == 'Low' and experience < 2:
        complexity_factor = 1.1
    
    # Weighted final score
    final_score = (
        0.40 * skill_score +           # 40% skill match
        0.20 * success_rate +          # 20% success rate  
        0.15 * experience_score +      # 15% experience
        0.10 * completion_score +      # 10% task history
        0.15 * workload_penalty        # 15% workload consideration
    ) * priority_boost * complexity_factor
    
    return final_score


def save_model(model_data, path='task_scoring_model.pkl'):
    """Save the trained model"""
    with open(path, 'wb') as f:
        pickle.dump(model_data, f)

def load_model(path='task_scoring_model.pkl'):
    """Load the trained model"""
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None


def get_skills_for_project_type(project_type):
    """Get the required skills for a given project type"""
    normalized_type = project_type.lower().replace(' ', '_')
    
    if normalized_type in PROJECT_TYPES:
        return PROJECT_TYPES[normalized_type]
    
    # If the exact project type is not found, try to find a similar one
    for pt, skills in PROJECT_TYPES.items():
        if normalized_type in pt or pt in normalized_type:
            return skills
    
    # If no match, return general programming skills
    return ["Programming", "Problem Solving", "Communication"]


def update_employee_metrics(emp_id):
    """Updates employee metrics based on completed tasks"""
    try:
        # Get completed tasks for the employee
        completed_tasks = Task.query.filter_by(assigned_to=emp_id, status='completed').all()
        
        if not completed_tasks:
            return True  # No tasks to process
        
        # Calculate new metrics
        tasks_completed = len(completed_tasks)
        
        # Calculate success rate based on ratings
        rated_tasks = [task for task in completed_tasks if task.success_rating is not None]
        if rated_tasks:
            avg_rating = sum(task.success_rating for task in rated_tasks) / len(rated_tasks)
            # Convert to a percentage (assuming ratings are 1-10)
            success_rate = (avg_rating / 10) * 100
        else:
            success_rate = 0
        
        # Update employee service with new metrics
        api_response = requests.put(
            f'{EMPLOYEE_SERVICE_URL}/employees/{emp_id}/metrics',
            headers=api_headers(),
            json={
                'tasks_completed': tasks_completed,
                'success_rate': success_rate
            }
        )
        
        if api_response.status_code != 200:
            print(f"Failed to update employee metrics: {api_response.text}")
            return False
            
        return True
    except Exception as e:
        print(f"Error updating employee metrics: {str(e)}")
        return False



@app.route('/api/get_project_skills', methods=['GET'])
def get_project_skills():
    """Get skills required for a specific project type"""
    project_type = request.args.get('project_type')
    if not project_type:
        return jsonify({'success': False, 'error': 'Project type is required'}), 400
        
    skills = get_skills_for_project_type(project_type)
    return jsonify({'success': True, 'skills': skills})


@app.route('/api/task-service/project-types', methods=['GET'])
def get_project_types():
    """Return all available project types and their skills"""
    return jsonify({
        'project_types': list(PROJECT_TYPES.keys()),
        'project_type_details': PROJECT_TYPES
    })

@app.route('/api/task-service/tasks/<task_id>', methods=['GET'])
def get_task(task_id):
    """Get details for a specific task"""
    try:
        # Verify API key if needed
        api_key = request.headers.get('X-API-Key')
        if not api_key or api_key != 'dev_api_key':  # Replace with your actual API key validation
            return jsonify({
                'success': False,
                'error': 'Invalid or missing API key'
            }), 401
            
        # Find the task using SQLAlchemy 2.0-compatible syntax
        from sqlalchemy import select
        task = db.session.execute(select(Task).filter_by(task_id=task_id)).scalar_one_or_none()
        
        if not task:
            app.logger.warning(f"Task not found: {task_id}")
            return jsonify({
                'success': False,
                'error': 'Task not found'
            }), 404
            
        app.logger.info(f"Returning task data for: {task_id}")
        return jsonify({
            'success': True,
            'task': task.to_dict()
        })
    except Exception as e:
        app.logger.error(f"Error retrieving task {task_id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Add this endpoint with '/api' prefix
@app.route('/api/task-service/tasks/<task_id>/review', methods=['PUT'])
def review_task_api(task_id):
    """Move a submitted task to pending_approval status for manager review"""
    try:
        print(f"Received review request for task {task_id}")
        
        task = Task.query.get(task_id)
        if not task:
            return jsonify({
                'success': False,
                'error': 'Task not found'
            }), 404
            
        # Check if task is submitted
        if task.status != 'submitted':
            return jsonify({
                'success': False,
                'error': f'Cannot review task with status: {task.status}. Task must be submitted.'
            }), 400
            
        # Update task status to pending_approval
        task.status = 'pending_approval'
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Task moved to pending approval',
            'task': task.to_dict()
        })
    except Exception as e:
        db.session.rollback()
        print(f"Error moving task to pending approval: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Keep the original for backward compatibility
@app.route('/task-service/tasks/<task_id>/review', methods=['PUT'])
def review_task(task_id):
    """Redirect to main review function"""
    return review_task_api(task_id)

@app.route('/api/task-service/skills-for-project', methods=['GET'])
def get_skills_for_project():
    """Return the required skills for a given project type"""
    project_type = request.args.get('project_type')
    
    if not project_type:
        return jsonify({'error': 'Project type is required'}), 400
    
    skills = get_skills_for_project_type(project_type)
    return jsonify({'skills': skills})


@app.route('/api/task-service/assign-tasks', methods=['POST'])
def assign_tasks():
    """Assign tasks to employees using ML model, and save to database"""
    try:
        data = request.json
        
        if not data or 'tasks' not in data:
            return jsonify({'success': False, 'error': 'Tasks are required'}), 400
        
        tasks = data['tasks']
        
        # Process tasks to ensure they have skills
        for task in tasks:
            if 'project_type' in task and ('skills' not in task or not task['skills']):
                task['skills'] = get_skills_for_project_type(task['project_type'])
        
        # Get all employees
        employees = get_all_employees()
        
        # Load or train the model
        model_data = load_model() or train_scoring_model(employees)
        return assign_tasks_ml_scoring(tasks, employees, model_data)
        
        if isinstance(assignments, dict) and "error" in assignments:
            return jsonify({
                'success': False,
                'error': assignments["error"]
            }), 500
        
        # Save assignments to database and send emails
        saved_tasks = []
        failed_tasks = []
        email_results = []
        
        for task in tasks:
            task_id = task.get('task_id')
            if not task_id:
                failed_tasks.append({"task": task, "error": "Missing task_id"})
                continue
                
            assignment = assignments.get(task_id)
            if not assignment or 'emp_id' not in assignment:
                failed_tasks.append({"task": task, "error": "No valid assignment found"})
                continue
            
            try:
                emp_id = assignment.get('emp_id')
                
                # Get employee details including email
                employee = get_employee_by_id(emp_id)
                employee_email = None
                employee_name = "Employee"
                
                if employee:
                    employee_email = employee.get('email')
                    employee_name = employee.get('name', 'Employee')
                else:
                    print(f"Warning: Could not fetch employee details for {emp_id}")
                
                # Check if task already exists
                existing_task = Task.query.get(task_id)
                
                if existing_task:
                    # Update existing task
                    existing_task.assigned_to = emp_id
                    existing_task.email_assigned_to = employee_email
                    existing_task.assigned_at = datetime.utcnow()
                    existing_task.status = 'assigned'
                else:
                    # Create new task
                    new_task = Task(
                        task_id=task_id,
                        project_type=task.get('project_type', 'unknown'),
                        skills=task.get('skills', []),
                        complexity=task.get('complexity', 'Medium'),
                        priority=task.get('priority', 'Medium'),
                        assigned_to=emp_id,
                        email_assigned_to=employee_email,
                        assigned_at=datetime.utcnow(),
                        status='assigned'
                    )
                    db.session.add(new_task)
                
                saved_tasks.append(task_id)
                
                # Send email notification if email is available
                email_sent = False
                if employee_email:
                    try:
                        email_sent = send_task_assignment(
                            email=employee_email,
                            task_id=task_id,
                            project_type=task.get('project_type'),
                            complexity=task.get('complexity'),
                            priority=task.get('priority'),
                            assigned_to=emp_id,
                            employee_name=employee_name,
                            assigned_at = task.get('assigned_at')
                          
                        )
                        
                        if not email_sent:
                            print(f"Warning: Email could not be sent to {employee_email} for task {task_id}")
                            
                    except Exception as email_error:
                        print(f"Email sending error for task {task_id}: {str(email_error)}")
                        email_sent = False
                else:
                    print(f"Warning: No email address found for employee {emp_id}")
                
                # Record email result
                email_results.append({
                    'task_id': task_id,
                    'emp_id': emp_id,
                    'employee_name': employee_name,
                    'email': employee_email,
                    'email_sent': email_sent,
                    'reason': 'Email sent successfully' if email_sent else ('No email address' if not employee_email else 'Email sending failed')
                })
                
            except Exception as e:
                db.session.rollback()
                failed_tasks.append({"task": task, "error": str(e)})
        
        # Commit all successful tasks
        if saved_tasks:
            try:
                db.session.commit()
            except Exception as commit_error:
                db.session.rollback()
                print(f"Database commit error: {str(commit_error)}")
                return jsonify({
                    'success': False,
                    'error': f'Database commit failed: {str(commit_error)}'
                }), 500
        
        return jsonify({
            'success': True,
            'assignments': assignments,
            'saved_tasks': saved_tasks,
            'failed_tasks': failed_tasks,
            'email_notifications': email_results
        })
        
    except Exception as e:
        import traceback
        print(f"Error in assign_tasks: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': f'Exception: {str(e)}'
        }), 500



@app.route('/api/task-service/tasks_recommendation', methods=['POST'])
def tasks_recommendation_endpoint():
    """Get task assignment recommendations WITHOUT saving to database or sending emails"""
    try:
        data = request.json
        
        if not data or 'tasks' not in data:
            return jsonify({'success': False, 'error': 'Tasks are required'}), 400
        
        tasks = data['tasks']
        
        # Process tasks to ensure they have skills
        for task in tasks:
            if 'project_type' in task and ('skills' not in task or not task['skills']):
                task['skills'] = get_skills_for_project_type(task['project_type'])
        
        # Get all employees
        employees = get_all_employees()
        
        # Load or train the model
        model_data = load_model() or train_scoring_model(employees)
        
        # Get recommendations using ML model (but don't save anything)
        assignments = assign_tasks_ml_scoring(tasks, employees, model_data)
        
        if isinstance(assignments, dict) and "error" in assignments:
            return jsonify({
                'success': False,
                'error': assignments["error"]
            }), 500
        
        # Return recommendations WITHOUT saving to database or sending emails
        return jsonify({
            'success': True,
            'assignments': assignments
        })
        
    except Exception as e:
        import traceback
        print(f"Error in tasks_recommendation: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': f'Exception: {str(e)}'
        }), 500


    
@app.route('/api/task-service/tasks/pending-review', methods=['GET'])
def get_pending_review_tasks():
    """Get all tasks that are submitted and waiting for review"""
    try:
        # The correct status should be 'submitted', not 'pending_review'
        pending_tasks = Task.query.filter_by(status='submitted').all()
        
        # Convert to dictionary format
        tasks_list = [task.to_dict() for task in pending_tasks]
        
        return jsonify({
            'success': True,
            'tasks': tasks_list
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/task-service/tasks', methods=['GET'])
def get_all_tasks():
    """Get all tasks or filter by employee, status, etc."""
    try:
        # Get query parameters for filtering
        emp_id = request.args.get('emp_id')
        status = request.args.get('status')
        project_type = request.args.get('project_type')
        
        # Start with base query
        query = Task.query
        
        # Apply filters if provided
        if emp_id:
            query = query.filter_by(assigned_to=emp_id)
        if status:
            query = query.filter_by(status=status)
        if project_type:
            query = query.filter_by(project_type=project_type)
            
        # Execute query and convert to dict
        tasks = query.all()
        task_list = [task.to_dict() for task in tasks]
        
        return jsonify({
            'success': True,
            'tasks': task_list,
            'count': len(task_list)
        })
    except Exception as e:
        print(f"Error getting tasks: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
    



@app.route('/api/task-service/tasks/<task_id>/status', methods=['PUT'])
def update_task_status(task_id):
    """Update a task's status"""
    try:
        data = request.json
        if not data or 'status' not in data:
            return jsonify({
                'success': False,
                'error': 'Status is required'
            }), 400
            
        task = Task.query.get(task_id)
        if not task:
            return jsonify({
                'success': False,
                'error': 'Task not found'
            }), 404
            
        # Update task status
        task.status = data['status']
        
        # For completed tasks, set completion date and update metrics
        if data['status'] == 'completed':
            task.completion_date = datetime.utcnow()
            
            # Set success rating if provided
            if 'rating' in data:
                task.success_rating = data['rating']
                
            # Update employee metrics
            if task.assigned_to:
                update_employee_metrics(task.assigned_to)
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'task': task.to_dict()
        })
    except Exception as e:
        db.session.rollback()
        print(f"Error updating task status: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/task-service/tasks/<task_id>/submit', methods=['POST'])
def submit_task(task_id):
    try:
        data = request.get_json()
        if not data or 'emp_id' not in data:
            return jsonify({'success': False, 'error': 'Employee ID required'}), 400

        # Get and validate task
        task = Task.query.filter_by(task_id=task_id).first()
        if not task:
            return jsonify({'success': False, 'error': 'Task not found'}), 404

        if task.status != 'in_progress':
            return jsonify({
                'success': False,
                'error': f'Task must be in progress (current: {task.status})'
            }), 400

        # Update task status and submission time
        task.status = 'submitted'
        task.submitted_at = datetime.utcnow()
        db.session.commit()

        return jsonify({
            'success': True,
            'message': 'Task submitted successfully',
            'task': {
                'task_id': task.task_id,
                'new_status': task.status,
                'submitted_at': task.submitted_at.isoformat()
            }
        })

    except Exception as e:
        db.session.rollback()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/task-service/tasks/<task_id>/approve', methods=['PUT'])
def approve_task(task_id):
    """Approve or reject a submitted task"""
    try:
        data = request.json
        if not data or 'approved' not in data:
            return jsonify({
                'success': False,
                'error': 'Approval decision is required'
            }), 400
            
        manager_id = data.get('manager_id')
        if not manager_id:
            return jsonify({
                'success': False,
                'error': 'Manager ID is required'
            }), 400
            
        task = Task.query.get(task_id)
        if not task:
            return jsonify({
                'success': False,
                'error': 'Task not found'
            }), 404
            
        # Check if task is pending approval
        if task.status != 'pending_approval':
            return jsonify({
                'success': False,
                'error': f'Cannot approve/reject task with status: {task.status}. Task must be pending approval.'
            }), 400
            
        # Handle approval or rejection
        is_approved = data['approved']
        task.approved_by = manager_id
        task.approved_at = datetime.utcnow()
        
        if 'notes' in data:
            task.approval_notes = data['notes']
            
        if is_approved:
            # Mark as completed
            task.status = 'completed'
            task.completion_date = datetime.utcnow()
            
            # Set success rating if provided
            if 'rating' in data:
                task.success_rating = data['rating']
                
            # Update employee metrics
            if task.assigned_to:
                update_employee_metrics(task.assigned_to)
        else:
            # If rejected, set status back to in_progress
            task.status = 'in_progress'
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Task ' + ('approved' if is_approved else 'rejected and returned for revisions'),
            'task': task.to_dict()
        })
    except Exception as e:
        db.session.rollback()
        print(f"Error processing task approval: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/task-service/dashboard', methods=['GET'])
def get_dashboard_data():
    """Get dashboard data including task statistics"""
    try:
        # Get employee ID if filtering for a specific employee
        emp_id = request.args.get('emp_id')
        
        # Base query
        query = Task.query
        
        # Filter for specific employee if provided
        if emp_id:
            query = query.filter_by(assigned_to=emp_id)
            
        # Get all matching tasks
        tasks = query.all()
        
        # Calculate statistics
        total_tasks = len(tasks)
        completed_tasks = sum(1 for task in tasks if task.status == 'completed')
        in_progress_tasks = sum(1 for task in tasks if task.status == 'in_progress')
        assigned_tasks = sum(1 for task in tasks if task.status == 'assigned')
        pending_approval_tasks = sum(1 for task in tasks if task.status == 'pending_approval')
        
        # Calculate completion rate
        completion_rate = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
        
        # Group by project type
        project_counts = {}
        for task in tasks:
            project_type = task.project_type
            if project_type not in project_counts:
                project_counts[project_type] = 0
            project_counts[project_type] += 1
            
        # Group by priority
        priority_counts = {
            'Low': sum(1 for task in tasks if task.priority == 'Low'),
            'Medium': sum(1 for task in tasks if task.priority == 'Medium'),
            'High': sum(1 for task in tasks if task.priority == 'High')
        }
        
        # Calculate recent completion trend (last 5 completions)
        recent_completions = Task.query.filter_by(status='completed')\
            .order_by(Task.completion_date.desc())\
            .limit(5)\
            .all()
            
        completion_trend = [
            {
                'task_id': task.task_id,
                'completion_date': task.completion_date.isoformat() if task.completion_date else None,
                'rating': task.success_rating
            } for task in recent_completions
        ]
        
        # Return dashboard data
        return jsonify({
            'success': True,
            'total_tasks': total_tasks,
            'tasks_by_status': {
                'completed': completed_tasks,
                'in_progress': in_progress_tasks,
                'assigned': assigned_tasks,
                'pending_approval': pending_approval_tasks
            },
            'completion_rate': completion_rate,
            'tasks_by_project': project_counts,
            'tasks_by_priority': priority_counts,
            'recent_completions': completion_trend
        })
    except Exception as e:
        print(f"Error generating dashboard data: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/task-service/retrain-model', methods=['POST'])
def retrain_model():
    """Force retraining of the ML model"""
    # Delete existing model if it exists
    model_path = 'task_assignment_model.pkl'
    if os.path.exists(model_path):
        os.remove(model_path)
    
    # Train new model
    model_data = train_scoring_model(model_path)
    
    if model_data:
        return jsonify({
            'success': True,
            'message': 'Model retrained successfully'
        })
    else:
        return jsonify({
            'success': False,
            'error': 'Failed to retrain model'
        }), 500

if __name__ == '__main__':
    app.run(port=5002, debug=True)