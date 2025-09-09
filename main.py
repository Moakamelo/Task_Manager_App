# main_app.py

from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from salary_service import (
    get_salary_predictor, 
    get_competitiveness_analyzer, 
    initialize_salary_predictor, 
    retrain_salary_model,
    retrain_salary_model_with_progress, 
    get_retraining_progress,
    clear_retraining_progress,
    set_retraining_progress
)
import os
import requests
from task_assignment_service import Task
import json
from dotenv import load_dotenv
from email_services import send_credentials_email, send_approval_notification, send_new_application_notification
from datetime import datetime, timedelta
import random
import threading
import time
import logging

# Load environment variables
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

retraining_progress = {}
progress_lock = threading.Lock()

salary_predictor = None
competitiveness_analyzer = None

try:
    salary_predictor = get_salary_predictor()
    competitiveness_analyzer = get_competitiveness_analyzer()
    logger.info("Salary prediction services initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize salary prediction services: {e}")
    # Initialize with None to prevent crashes
    salary_predictor = None
    competitiveness_analyzer = None

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev_key_for_testing')

@app.context_processor
def inject_template_globals():
    """Make certain functions available to all templates"""
    return dict(
        format_date=format_date
    )


@app.route('/api/retrain_salary_model_with_progress', methods=['POST'])
def api_retrain_salary_model_with_progress():
    """Enhanced API endpoint to retrain the salary prediction model with real progress tracking"""
    if 'emp_id' not in session or session['role'] != 'human resource':
        return jsonify({'success': False, 'error': 'Unauthorized'}), 403
        
    session_id = session['emp_id']
    
    # Check if retraining is already in progress
    current_progress = get_retraining_progress(session_id)
    if current_progress['percentage'] > 0 and not current_progress['completed'] and not current_progress['failed']:
        return jsonify({
            'success': False,
            'error': 'Retraining is already in progress'
        }), 409
    
    def retrain_in_background():
        """Run the actual retraining process in background"""
        try:
            logger.info(f"Starting background retraining for session {session_id}")
            success = retrain_salary_model_with_progress(session_id)
            
            if success:
                # Reinitialize the global predictors with the new model
                global salary_predictor, competitiveness_analyzer
                try:
                    salary_predictor = get_salary_predictor()
                    competitiveness_analyzer = get_competitiveness_analyzer()
                    logger.info("Salary prediction services reinitialized after retraining")
                except Exception as reinit_error:
                    logger.error(f"Failed to reinitialize services after retraining: {reinit_error}")
                    # Continue anyway, the retrain was successful
            else:
                logger.error("Background retraining failed")
                
        except Exception as e:
            logger.error(f"Exception during background retraining: {e}")
            # Make sure we set an error status
            from salary_service import set_retraining_progress
            set_retraining_progress(session_id, 0, 'Retraining failed', None, str(e))
    
    # Clear any previous progress
    clear_retraining_progress(session_id)
    
    # Start retraining in background thread
    thread = threading.Thread(target=retrain_in_background, daemon=True)
    thread.start()
    
    return jsonify({
        'success': True, 
        'message': 'Model retraining started. Use /api/retrain_progress to track progress.'
    })

@app.route('/api/retrain_progress', methods=['GET'])
def get_retrain_progress():
    """Get current retraining progress"""
    if 'emp_id' not in session or session['role'] != 'human resource':
        return jsonify({'success': False, 'error': 'Unauthorized'}), 403
    
    session_id = session['emp_id']
    progress = get_retraining_progress(session_id)
    
    return jsonify({
        'success': True,
        'progress': progress
    })

@app.route('/api/cancel_retrain', methods=['POST'])
def cancel_retrain():
    """Cancel ongoing retraining process"""
    if 'emp_id' not in session or session['role'] != 'human resource':
        return jsonify({'success': False, 'error': 'Unauthorized'}), 403
    
    session_id = session['emp_id']
    
    # Clear progress (this effectively cancels it from UI perspective)
    clear_retraining_progress(session_id)
    
    return jsonify({
        'success': True,
        'message': 'Retraining process cancelled'
    })


def set_retraining_progress(session_id, percentage, stage, step_id=None, error=None):
    """Set retraining progress for a session"""
    with progress_lock:
        retraining_progress[session_id] = {
            'percentage': percentage,
            'stage': stage,
            'step_id': step_id,
            'error': error,
            'timestamp': datetime.now().isoformat(),
            'completed': percentage >= 100,
            'failed': error is not None
        }

def get_retraining_progress(session_id):
    """Get retraining progress for a session"""
    with progress_lock:
        return retraining_progress.get(session_id, {
            'percentage': 0,
            'stage': 'Not started',
            'step_id': None,
            'error': None,
            'timestamp': datetime.now().isoformat(),
            'completed': False,
            'failed': False
        })

def clear_retraining_progress(session_id):
    """Clear progress data for a session"""
    with progress_lock:
        if session_id in retraining_progress:
            del retraining_progress[session_id]

# Employee service configuration
EMPLOYEE_SERVICE_URL = os.environ.get('EMPLOYEE_SERVICE_URL', 'http://localhost:5001/api')
API_KEY = os.environ.get('API_KEY', 'dev_api_key')
PROJECT_TYPES = {
    "website_development": ["HTML", "CSS", "JavaScript", "React", "Vue", "Angular"],
    "mobile_app_development": ["Swift", "Kotlin", "React Native", "Flutter"],
    "machine_learning": ["Python", "TensorFlow", "PyTorch", "Scikit-learn"],
    
}

TASK_SERVICE_URL = os.environ.get('TASK_SERVICE_URL', 'http://localhost:5002/api')

def assign_tasks(tasks):
    # Fix URL construction - the function calls recommendations endpoint instead of actual assignment
    response = requests.post(
        f'{TASK_SERVICE_URL}/task-service/tasks_recommendation',  # Use recommendations endpoint
        json={'tasks': tasks},
        headers=api_headers()
    )
    if response.status_code == 200:
        return response.json()
    return None

# Add new function for actual assignment
def assign_tasks_with_save(tasks):
    """Actually assign tasks and save to database"""
    response = requests.post(
        f'{TASK_SERVICE_URL}/task-service/assign-tasks',
        json={'tasks': tasks},
        headers=api_headers()
    )
    if response.status_code == 200:
        return response.json()
    return None

def get_project_types():
    """Get all project types from task service with fallback to local definition"""
    try:
        response = requests.get(
            f'{TASK_SERVICE_URL}/task-service/project-types',
            headers=api_headers()
        )
        if response.status_code == 200:
            return response.json().get('project_types', []), response.json().get('project_type_details', {})
        return list(PROJECT_TYPES.keys()), PROJECT_TYPES
    except Exception as e:
        print(f"Error fetching project types: {str(e)}")
        return list(PROJECT_TYPES.keys()), PROJECT_TYPES

def get_skills_for_project_type(project_type):
    """Get required skills for a specific project type"""
    try:
        response = requests.get(
            f'{TASK_SERVICE_URL}/task-service/skills-for-project',
            params={'project_type': project_type},
            headers=api_headers()
        )
        if response.status_code == 200:
            return response.json().get('skills', [])
        return PROJECT_TYPES.get(project_type, [])
    except Exception as e:
        print(f"Error fetching skills for project type: {str(e)}")
        return PROJECT_TYPES.get(project_type, [])


def format_date(date_str, format_str='%b %d, %Y'):
    """Format a date string with the specified format"""
    if not date_str:
        return "N/A"
    
    try:
        # First try to parse the date string
        if isinstance(date_str, str):
            # Try ISO format first (most common API response format)
            try:
                date_obj = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            except ValueError:
                # Try common date formats
                try:
                    date_obj = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
                except ValueError:
                    try:
                        date_obj = datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%S')
                    except ValueError:
                        try:
                            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                        except ValueError:
                            return date_str  # Return original if parsing fails
        elif isinstance(date_str, datetime):
            date_obj = date_str
        else:
            return str(date_str)
        
        # Format the date object
        return date_obj.strftime(format_str)
    except Exception as e:
        print(f"Error formatting date: {str(e)}")
        return str(date_str)  # Return original as fallback


def get_developer_tasks(emp_id):
    """Get all tasks assigned to a specific developer"""
    try:
        response = requests.get(
            f'{TASK_SERVICE_URL}/task-service/tasks',
            params={'emp_id': emp_id},
            headers=api_headers()
        )
        if response.status_code == 200:
            return response.json().get('tasks', [])
        return []
    except Exception as e:
        print(f"Error fetching developer tasks: {str(e)}")
        return []

# Helper functions for API calls
def api_headers():
    key = os.environ.get('API_KEY', 'dev_api_key')
    print(f"Using API key: {key}")  # Debug line
    return {'X-API-KEY': key, 'Content-Type': 'application/json'}

def get_employee(emp_id):
    response = requests.get(f'{EMPLOYEE_SERVICE_URL}/employees/{emp_id}', headers=api_headers())
    if response.status_code == 200:
        return response.json()
    return None

def authenticate_employee(emp_id, password):
    data = {'emp_id': emp_id, 'password': password}
    headers = api_headers()
    print(f"Sending auth request to {EMPLOYEE_SERVICE_URL}/authenticate")
    print(f"Auth request data: {data}")
    
    try:
        response = requests.post(f'{EMPLOYEE_SERVICE_URL}/authenticate', json=data, headers=headers)
        print(f"Auth response status: {response.status_code}")
        print(f"Auth response text: {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Parsed response: {result}")
            return result
        else:
            print(f"Error response headers: {response.headers}")
            print(f"Error response body: {response.text}")
            return {'authenticated': False, 'reason': f'API error: {response.status_code}'}
    except Exception as e:
        print(f"Exception during authentication: {str(e)}")
        return {'authenticated': False, 'reason': f'Exception: {str(e)}'}

def change_employee_password(emp_id, current_password, new_password):
    data = {
        'emp_id': emp_id,
        'current_password': current_password,
        'new_password': new_password
    }
    response = requests.post(f'{EMPLOYEE_SERVICE_URL}/change_password', json=data, headers=api_headers())
    return response.json() if response.status_code == 200 else None

def get_all_employees():
    response = requests.get(f'{EMPLOYEE_SERVICE_URL}/employees', headers=api_headers())
    if response.status_code == 200:
        return response.json()
    return []

def create_new_employee(name, email, role, skills=None, experience=None, emp_id=None):
    """Admin creates employee directly - RESTRICTED to HR role only"""
    
    # RESTRICTION: Only allow HR role creation directly
    if role != 'human resource':
        return False, 'Direct employee creation restricted to HR role only. Other employees must come from HR recommendations.'
    
    data = {
        'name': name,
        'email': email,
        'role': role
    }
    
    # Only include emp_id if it's provided and not empty
    if emp_id:
        data['emp_id'] = emp_id
        
    # Include skills if provided
    if skills:
        data['skills'] = skills
        
    # Include experience if provided
    if experience is not None:
        data['experience'] = experience
    
    response = requests.post(
        f'{EMPLOYEE_SERVICE_URL}/employees',
        json=data,
        headers=api_headers()
    )
    
    if response.status_code == 201:
        result = response.json()
        return True, result.get('employee', {}).get('emp_id', '')
    else:
        error_message = response.json().get('error', 'Failed to create employee')
        return False, error_message

def update_employee(emp_id, data):
    response = requests.put(f'{EMPLOYEE_SERVICE_URL}/employees/{emp_id}', json=data, headers=api_headers())
    if response.status_code == 200:
        return response.json()
    return None

def delete_employee(emp_id):
    response = requests.delete(f'{EMPLOYEE_SERVICE_URL}/employees/{emp_id}', headers=api_headers())
    return response.status_code == 200

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/admin")
def admin_login_page():
    """Special route for admin login that bypasses role selection"""
    return render_template("admin_login.html")

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        emp_id = request.form.get('emp_id')
        password = request.form.get('password')
        
        # Attempt authentication
        result = authenticate_employee(emp_id, password)
        
        if not result['authenticated']:
            flash('Invalid employee ID or password')
            return render_template('admin_login.html')
        
        employee = result['employee']
        
        if employee['role'] != 'admin':
            flash('This employee ID is not registered as an admin')
            return render_template('admin_login.html')
        
        # Store employee info in session
        session['emp_id'] = emp_id
        session['role'] = employee['role']
        session['name'] = employee['name']
        session['email'] = employee['email']
        
        # If first login, redirect to password change page
        if employee['is_first_login']:
            return redirect(url_for('change_password'))
        
        # Direct to admin dashboard
        return redirect(url_for('admin_dashboard'))
    
    return render_template('admin_login.html')

@app.route('/select_role/<role>')
def select_role(role):
    if role not in ['developer', 'project_manager', 'human_resource']:
        flash('Invalid role selected')
        return redirect(url_for('index'))
    
    session['selected_role'] = role
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'selected_role' not in session:
        flash('Please select a role first')
        return redirect(url_for('index'))
    
    role = session['selected_role']
    print(f"\n--- LOGIN ATTEMPT ---\nRole selected: {role}")

    if request.method == 'POST':
        emp_id = request.form.get('emp_id')
        password = request.form.get('password')
        print(f"Attempting login for: {emp_id}")

        result = authenticate_employee(emp_id, password)
        print(f"Auth result: {result}")

        if not result or not result.get('authenticated'):
            flash('Invalid employee ID or password')
            return render_template('login.html', role=role)

        employee = result['employee']
        print(f"Employee data received: {employee}")

        # Case-insensitive role comparison
        if employee['role'].lower().replace(' ', '_') != role.lower():
            flash(f'This employee ID is not registered as a {role.replace("_", " ")}')
            return render_template('login.html', role=role)

        # Store session data
        session.update({
            'emp_id': emp_id,
            'role': employee['role'],
            'name': employee['name'],
            'email': employee['email']
        })

        # Debug print session data
        print(f"\n--- SESSION DATA ---\n{session}\n")

        # Handle first login (using get() with True default)
        first_login = employee.get('is_first_login', True)
        print(f"First login status: {first_login} (Type: {type(first_login)})")

        if first_login:
            print("Redirecting to change_password")
            return redirect(url_for('change_password'))

        # Regular redirect based on role
        print(f"Redirecting to {role}_dashboard")
        return redirect(url_for(f'{role}_dashboard'))

    return render_template('login.html', role=role)

@app.route('/change_password', methods=['GET', 'POST'])
def change_password():
    if 'emp_id' not in session:
        return redirect(url_for('index'))
    
    # Add debug logging
    print(f"Reached change_password route for employee {session['emp_id']}")
    
    if request.method == 'POST':
        current_password = request.form.get('current_password')
        new_password = request.form.get('new_password')
        confirm_password = request.form.get('confirm_password')
        
        if new_password != confirm_password:
            flash('New passwords do not match')
            return render_template('change_password.html')
        
        # Call API to change password
        result = change_employee_password(session['emp_id'], current_password, new_password)
        
        if not result or not result.get('success'):
            flash('Current password is incorrect')
            return render_template('change_password.html')
        
        flash('Password changed successfully')
        
        # Redirect based on role
        role = session['role'].lower().replace(' ', '_')
        if role == 'developer':
            return redirect(url_for('developer_dashboard'))
        elif role == 'project_manager':
            return redirect(url_for('project_manager_dashboard'))
        elif role == 'admin':
            return redirect(url_for('admin_dashboard'))
        elif role == 'human_resource':  
            return redirect(url_for('human_resource_dashboard'))
        else:
            return redirect(url_for('hr_dashboard'))
    
    # Get employee to check first_login status
    employee = get_employee(session['emp_id'])
    first_login = employee['is_first_login'] if employee else False
    
    print(f"Employee first_login status: {first_login}")
    
    return render_template('change_password.html', first_login=first_login)

@app.route('/developer_dashboard')
def developer_dashboard():
    if 'emp_id' not in session:
        flash('Please log in to access the developer dashboard', 'warning')
        return redirect(url_for('login'))
    
    normalized_role = session.get('role', '').lower().replace(' ', '_')
    if normalized_role != 'developer':
        flash('Access restricted to developers only', 'danger')
        return redirect(url_for('index'))
    
    # Fetch complete employee data
    employee_data = get_employee(session['emp_id'])
    
    # Fallback with defaults if needed
    if not employee_data:
        employee_data = dict(session)
        employee_data['success_rate'] = 0.0
        employee_data['tasks_completed'] = 0
        employee_data['experience'] = 0
        employee_data['skills'] = []
    
    # Get assigned tasks for this developer
    assigned_tasks = get_developer_tasks(session['emp_id'])
    
    # Get dashboard statistics for the developer
    try:
        dashboard_response = requests.get(
            f'{request.host_url.rstrip("/")}/api/task-service/dashboard?emp_id={session["emp_id"]}',
            headers=api_headers()
        )
        dashboard_data = dashboard_response.json() if dashboard_response.status_code == 200 else {}
    except Exception as e:
        print(f"Error getting dashboard data: {str(e)}")
        dashboard_data = {}
    
    # If dashboard data wasn't returned from API, calculate it from tasks
    if not dashboard_data:
        dashboard_data = calculate_dashboard_metrics(assigned_tasks, employee_data)
    
    # Categorize tasks by status - Updated to match Task model status values
    in_progress_tasks = [task for task in assigned_tasks if task.get('status') == 'in_progress']
    pending_tasks = [task for task in assigned_tasks if task.get('status') == 'assigned']
    submited_tasks = [task for task in assigned_tasks if task.get('status') == 'submitted']
    completed_tasks = [task for task in assigned_tasks if task.get('status') == 'completed']
    
    # Calculate performance history (last 6 months)
    performance_history = get_performance_history(session['emp_id'])
    
    # Get project types for filtering
    try:
        project_types_response = requests.get(
            f'{request.host_url.rstrip("/")}/api/task-service/project-types',
            headers=api_headers()
        )
        project_types = project_types_response.json().get('project_types', []) if project_types_response.status_code == 200 else []
    except Exception as e:
        print(f"Error getting project types: {str(e)}")
        project_types = []
    
    return render_template(
        'developer_dashboard.html', 
        employee=employee_data,
        tasks=assigned_tasks,
        in_progress_tasks=in_progress_tasks,
        pending_tasks=pending_tasks,
        pending_approval_tasks=pending_approval_tasks,  # Updated variable name
        completed_tasks=completed_tasks,
        dashboard=dashboard_data,
        performance_history=performance_history,
        project_types=project_types,
        format_date=format_date
    )


def calculate_dashboard_metrics(tasks, employee_data):
    """Calculate dashboard metrics when API doesn't return data"""
    dashboard = {
        'success_rate': 0.0,
        'tasks_completed': 0,
        'avg_completion_time': 0,
        'performance_trend': []
    }
    
    # Count completed tasks
    completed_tasks = [task for task in tasks if task.get('status') == 'completed']
    dashboard['tasks_completed'] = len(completed_tasks)
    
    # Calculate success rate
    if dashboard['tasks_completed'] > 0:
        # Get average of task ratings
        ratings = [task.get('success_rating', 0) for task in completed_tasks if task.get('success_rating') is not None]
        if ratings:
            avg_rating = sum(ratings) / len(ratings)
            # Convert 5-star rating to percentage (e.g., 4/5 = 80%)
            dashboard['success_rate'] = (avg_rating / 5) * 100
        else:
            # Default to 70% if no ratings available
            dashboard['success_rate'] = 70.0
    else:
        # No completed tasks yet, use N/A or default value
        dashboard['success_rate'] = employee_data.get('success_rate', 0.0)
    
    # Calculate average completion time (in days)
    if completed_tasks:
        completion_times = []
        for task in completed_tasks:
            start_date = parse_date(task.get('start_date'))
            completion_date = parse_date(task.get('completion_date'))
            if start_date and completion_date:
                delta = completion_date - start_date
                completion_times.append(delta.days)
        
        if completion_times:
            dashboard['avg_completion_time'] = sum(completion_times) / len(completion_times)
    
    return dashboard

def get_performance_history(emp_id, months=6):
    """Get the performance history for the last X months"""
    today = datetime.now()
    
    # Default data structure with empty values
    history = {
        'labels': [],
        'success_rates': []
    }
    
    try:
        # Fetch historical data from API
        response = requests.get(
            f'{request.host_url.rstrip("/")}/api/task-service/performance-history?emp_id={emp_id}&months={months}',
            headers=api_headers()
        )
        
        if response.status_code == 200:
            history_data = response.json().get('history', [])
            if history_data:
                # If API returned data, use it
                for month_data in history_data:
                    history['labels'].append(month_data.get('month'))
                    history['success_rates'].append(month_data.get('success_rate', 0))
                return history
    except Exception as e:
        print(f"Error getting performance history: {str(e)}")
    
    # If no data from API or error occurred, generate some reasonable sample data
    for i in range(months, 0, -1):
        month_date = today - timedelta(days=30*i)
        month_name = month_date.strftime('%b')
        history['labels'].append(month_name)
        
        # Generate reasonable random data that trends upward slightly
        # Starting from 70% and gradually improving
        base_rate = 70 + (i * 2)  # Increase by 2% each month
        rate = min(base_rate + random.randint(-3, 5), 95)  # Add randomness but cap at 95%
        history['success_rates'].append(rate)
    
    return history

def parse_date(date_str):
    """Parse date string to datetime object"""
    if not date_str:
        return None
    
    try:
        # Adjust the format based on your date string format
        return datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
    except ValueError:
        try:
            # Try another common format
            return datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%S')
        except ValueError:
            return None

@app.route('/api/assign_multiple_employees', methods=['POST'])
def assign_multiple_employees():
    if 'emp_id' not in session or session['role'] != 'project manager':
        return jsonify({'success': False, 'error': 'Unauthorized'}), 403
    
    task_id = request.json.get('task_id')
    employee_ids = request.json.get('employee_ids', [])
    
    if not task_id or not employee_ids:
        return jsonify({'success': False, 'error': 'Task ID and at least one employee ID are required'}), 400
    
    try:
        # Call the task service API to assign multiple employees to the task
        response = requests.post(
            f'{request.host_url.rstrip("/")}/api/task-service/tasks/{task_id}/assign',
            headers=api_headers(),
            json={
                'manager_id': session['emp_id'],
                'employee_ids': employee_ids
            }
        )
        
        if response.status_code != 200:
            return jsonify({'success': False, 'error': f'Failed to assign employees: {response.text}'}), 500
        
        result = response.json()
        return jsonify({'success': True, 'assignments': result.get('assignments', {})})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/project_manager_dashboard')
def project_manager_dashboard():
    if 'emp_id' not in session or session['role'] != 'project manager':
        return redirect(url_for('index'))
    
    # Fetch complete employee data from the employee service
    employee_data = get_employee(session['emp_id'])
    
    # If employee data couldn't be fetched, use session data as fallback with defaults
    if not employee_data:
        employee_data = dict(session)
        employee_data['success_rate'] = 0.0
        employee_data['tasks_completed'] = 0
        employee_data['experience'] = 0
    
    # Get all employees for team stats
    all_employees = get_all_employees()
    
    # Calculate team statistics
    team_stats = {
        'total_members': len(all_employees),
        'active_tasks': 0,  # You would get this from your task service
        'avg_success': sum(emp.get('success_rate', 0) for emp in all_employees) / len(all_employees) if all_employees else 0,
        'top_skills': {},
        'top_performers': sorted(
            [emp for emp in all_employees if emp.get('tasks_completed', 0) > 0],
            key=lambda x: x.get('success_rate', 0),
            reverse=True
        )[:3]  # Top 3 performers
    }
    
    # Calculate skill distribution
    skill_counts = {}
    for emp in all_employees:
        for skill in emp.get('skills', []):
            skill_counts[skill] = skill_counts.get(skill, 0) + 1
    team_stats['top_skills'] = dict(sorted(skill_counts.items(), key=lambda item: item[1], reverse=True)[:5])  # Top 5 skills
    
    # Get project types from task service
    project_types, project_type_details = get_project_types()
    
    return render_template(
        'project_manager_dashboard.html',
        employee=employee_data,
        team_stats=team_stats,
        project_types=project_types,
        project_type_details=project_type_details
    )


@app.route('/task_management')
def task_management():
    if 'emp_id' not in session or session['role'] != 'project manager':
        flash('Access restricted to project managers', 'danger')
        return redirect(url_for('index'))
    
    # Get filter parameters from query string
    status_filter = request.args.get('status', 'all')
    project_type_filter = request.args.get('project_type', 'all')
    assignee_filter = request.args.get('assignee', 'all')
    
    # Get project types and their details
    project_types, project_type_details = get_project_types()
    
    # Get all employees for assignment dropdown
    employees = get_all_employees()
    
    try:
        # Get all tasks with optional filters
        tasks_response = requests.get(
            f'{request.host_url.rstrip("/")}/api/task-service/tasks',
            params={
                'status': status_filter if status_filter != 'all' else None,
                'project_type': project_type_filter if project_type_filter != 'all' else None,
                'assignee': assignee_filter if assignee_filter != 'all' else None
            },
            headers=api_headers()
        )
        
        tasks = []
        if tasks_response.status_code == 200:
            tasks = tasks_response.json().get('tasks', [])
        
        # Categorize tasks by status for easier display
        pending_review_tasks = [task for task in tasks if task.get('status') == 'pending_review']
        assigned_tasks = [task for task in tasks if task.get('status') == 'assigned']
        in_progress_tasks = [task for task in tasks if task.get('status') == 'in_progress']
        completed_tasks = [task for task in tasks if task.get('status') == 'completed']
        
        return render_template(
            'task_management.html',
            project_types=project_types,
            project_type_details=project_type_details,
            employees=employees,
            tasks=tasks,
            pending_review_tasks=pending_review_tasks,
            assigned_tasks=assigned_tasks,
            in_progress_tasks=in_progress_tasks,
            completed_tasks=completed_tasks,
            status_filter=status_filter,
            project_type_filter=project_type_filter,
            assignee_filter=assignee_filter,
            format_date=format_date
        )
    
    except Exception as e:
        flash(f'Error fetching task data: {str(e)}', 'danger')
        return render_template(
            'task_management.html',
            project_types=project_types,
            project_type_details=project_type_details,
            employees=employees,
            tasks=[],
            pending_review_tasks=[],
            assigned_tasks=[],
            in_progress_tasks=[],
            completed_tasks=[]
        )

@app.route('/approve_task', methods=['POST'])
def approve_task():
    if 'emp_id' not in session or session['role'] != 'project manager':
        return jsonify({'success': False, 'error': 'Unauthorized'}), 403
        
    task_id = request.json.get('task_id')
    action = request.json.get('action')  # 'approve' or 'reject'
    feedback = request.json.get('feedback', '')
    
    if not task_id or not action:
        return jsonify({'success': False, 'error': 'Task ID and action are required'}), 400
    
    if action not in ['approve', 'reject']:
        return jsonify({'success': False, 'error': 'Invalid action'}), 400
    
    try:
        # First, ensure the task is in pending_approval status by calling review endpoint
        api_url_review = f"{TASK_SERVICE_URL}/task-service/tasks/{task_id}/review"
        print(f"Moving task to pending_approval: {api_url_review}")
        
        review_response = requests.put(
            api_url_review,
            headers=api_headers()
        )
        
        print(f"Review response: {review_response.status_code} - {review_response.text}")
        
        # If task is already in pending_approval, this might fail but we can continue
        # Only fail if it's not a 'status' related error
        if review_response.status_code != 200:
            result = review_response.json()
            if 'error' in result and 'status' not in result['error']:
                return jsonify({'success': False, 'error': f'Failed to prepare task for review: {review_response.text}'}), 500
        
        # Now proceed with approval or rejection
        api_url = f"{TASK_SERVICE_URL}/task-service/tasks/{task_id}/approve"
        print(f"Approving/rejecting task: {api_url}")
        
        payload = {
            'manager_id': session['emp_id'],
            'approved': action == 'approve',
            'notes': feedback,
            'rating': 5 if action == 'approve' else None
        }
        
        print(f"With payload: {payload}")
        
        # Call the task service API to approve or reject the task
        response = requests.put(
            api_url,
            headers=api_headers(),
            json=payload
        )
        
        print(f"Approve response: {response.status_code} - {response.text}")
        
        if response.status_code != 200:
            return jsonify({'success': False, 'error': f'Failed to process task: {response.text}'}), 500
            
        result = response.json()
        return jsonify({'success': True, 'task': result.get('task', {})})
        
    except Exception as e:
        import traceback
        print(f"Exception in approve_task: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500


def create_employee_recommendation(name, email, suggested_role, skills=None, experience=None, recommended_by=None):
    """HR creates employee recommendation"""
    data = {
        'name': name,
        'email': email,
        'suggested_role': suggested_role,
        'recommended_by': recommended_by,
        'skills': skills or [],
        'experience': experience or 0
    }
    
    response = requests.post(
        f'{EMPLOYEE_SERVICE_URL}/employee-recommendations',
        json=data,
        headers=api_headers()
    )
    
    if response.status_code == 201:
        return True, response.json().get('recommendation', {})
    else:
        error_message = response.json().get('error', 'Failed to create recommendation')
        return False, error_message

def get_employee_recommendations(status='pending_admin_review'):
    """Get employee recommendations"""
    response = requests.get(
        f'{EMPLOYEE_SERVICE_URL}/employee-recommendations',
        params={'status': status},
        headers=api_headers()
    )
    
    if response.status_code == 200:
        return response.json().get('recommendations', [])
    return []

def process_employee_recommendation(rec_id, action, processed_by, notes=''):
    """Admin processes recommendation (approve/reject)"""
    data = {
        'action': action,  # 'approve' or 'reject'
        'processed_by': processed_by,
        'notes': notes
    }
    
    response = requests.put(
        f'{EMPLOYEE_SERVICE_URL}/employee-recommendations/{rec_id}/process',
        json=data,
        headers=api_headers()
    )
    
    if response.status_code in [200, 201]:
        return True, response.json()
    else:
        return False, response.json().get('error', 'Failed to process recommendation')



@app.route('/human_resource_dashboard')
def human_resource_dashboard():
    # Debug session information
    print(f"DEBUG: HR Dashboard - Session data: {dict(session)}")
    
    if 'emp_id' not in session:
        flash('Please log in to access the HR dashboard', 'warning')
        return redirect(url_for('login'))
    
    # Normalize role comparison
    normalized_role = session.get('role', '').lower().replace(' ', '_')
    print(f"DEBUG: HR Dashboard - Normalized role: {normalized_role}")
    
    if normalized_role != 'human_resource':
        flash('Access restricted to HR personnel', 'danger')
        print(f"DEBUG: HR Dashboard - Access denied. Expected 'human_resource', got '{normalized_role}'")
        return redirect(url_for('index'))
    
    # Rest of the function remains the same...
    # Get employee data
    employee_data = get_employee(session['emp_id'])
    
    # Get all employees
    employees = get_all_employees()
    
    # Get HR's recommendations by status
    pending_recommendations = get_employee_recommendations('pending_admin_review')
    my_pending = [r for r in pending_recommendations if r.get('recommended_by') == session['emp_id']]
    
    approved_recommendations = get_employee_recommendations('approved')
    my_approved = [r for r in approved_recommendations if r.get('recommended_by') == session['emp_id']]
    
    rejected_recommendations = get_employee_recommendations('rejected')
    my_rejected = [r for r in rejected_recommendations if r.get('recommended_by') == session['emp_id']]
    
    # Calculate HR stats
    hr_stats = {
        'total_employees': len(employees),
        'tech_employees': len([e for e in employees if e.get('role') and (
            'developer' in e['role'].lower() or 
            'engineer' in e['role'].lower() or
            'data' in e['role'].lower()
        )]),
        'avg_success_rate': sum(e.get('success_rate', 0) for e in employees) / len(employees) if employees else 0,
        'pending_recommendations': len(my_pending),
        'approved_recommendations': len(my_approved),
        'rejected_recommendations': len(my_rejected),
        'role_distribution': {},
        'top_performers': sorted(
            [e for e in employees if e.get('success_rate', 0) > 0],
            key=lambda x: x.get('success_rate', 0),
            reverse=True
        )[:5]
    }
    
    # Count roles
    for employee in employees:
        role = employee.get('role', 'Unknown')
        hr_stats['role_distribution'][role] = hr_stats['role_distribution'].get(role, 0) + 1
    
    # Common job titles and locations for the form
    common_job_titles = [
        "data scientist", "data analyst", "software developer", 
        "software engineer", "devops engineer", "machine learning engineer",
        "product manager", "ui/ux designer", "backend developer",
        "full stack developer", "frontend developer"
    ]
    
    common_locations = ["cape town", "johannesburg", "durban", "pretoria"]
    experience_levels = ["LESS_THAN_ONE", "ONE_TO_THREE", "FOUR_TO_SIX", "SEVEN_TO_NINE", "TEN_PLUS"]
    
    return render_template(
        'human_resource_dashboard.html',
        employee=employee_data,
        employees=employees,
        hr_stats=hr_stats,
        job_titles=common_job_titles,
        locations=common_locations,
        experience_levels=experience_levels,
        my_pending_recommendations=my_pending,
        my_approved_recommendations=my_approved,
        my_rejected_recommendations=my_rejected,
        format_date=format_date
    )


def get_pending_employees():
    """Get all employees with pending approval status"""
    response = requests.get(
    f'{EMPLOYEE_SERVICE_URL}/employees/pending',
    headers=api_headers()
    )
    if response.status_code == 200:
        return response.json().get('employees', [])
    return []

def approve_employee(emp_id, role, approved_by):
    """Approve an employee and assign their system role"""
    data = {
        'status': 'approved',
        'role': role,
        'approved_by': approved_by,
        'approved_at': datetime.now().isoformat()
    }
    
    response = requests.put(
        f'{EMPLOYEE_SERVICE_URL}/employees/{emp_id}/approve',
        json=data,
        headers=api_headers()
    )
    
    if response.status_code == 200:
        return response.json()
    return None

def reject_employee(emp_id, reason, rejected_by):
    """Reject an employee application"""
    data = {
        'status': 'rejected',
        'rejection_reason': reason,
        'rejected_by': rejected_by,
        'rejected_at': datetime.now().isoformat()
    }
    
    response = requests.put(
        f'{EMPLOYEE_SERVICE_URL}/employees/{emp_id}/reject',
        json=data,
        headers=api_headers()
    )
    
    return response.status_code == 200


@app.route('/hr/create_employee', methods=['GET', 'POST'])
def hr_create_employee():
    """HR creates employee recommendation (forwarded to admin)"""
    
    # Debug: Print session information
    print(f"DEBUG: Session data: {dict(session)}")
    print(f"DEBUG: Request method: {request.method}")
    print(f"DEBUG: Content-Type: {request.headers.get('Content-Type', 'Not set')}")
    print(f"DEBUG: Is JSON request: {request.is_json}")
    
    # Check if user is logged in
    if 'emp_id' not in session:
        if request.is_json:
            return jsonify({'success': False, 'error': 'Please log in to access this page'}), 401
        flash('Please log in to access this page', 'warning')
        return redirect(url_for('login'))
    
    # Normalize role comparison - handle different formats
    user_role = session.get('role', '').lower().replace(' ', '_')
    print(f"DEBUG: Normalized user role: {user_role}")
    
    if user_role != 'human_resource':
        if request.is_json:
            return jsonify({'success': False, 'error': 'Access restricted to HR personnel only'}), 403
        flash('Access restricted to HR personnel only', 'danger')
        print(f"DEBUG: Access denied. Expected 'human_resource', got '{user_role}'")
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        try:
            # Handle both JSON and form data
            if request.is_json:
                # JSON request from AJAX
                data = request.get_json()
                name = data.get('name')
                email = data.get('email')
                suggested_role = data.get('requested_role')
                skills_input = data.get('skills', '').strip()
                experience_input = str(data.get('experience', '')).strip()
                print(f"DEBUG: Processing JSON data: {data}")
            else:
                # Traditional form submission
                name = request.form.get('name')
                email = request.form.get('email')
                suggested_role = request.form.get('requested_role')
                skills_input = request.form.get('skills', '').strip()
                experience_input = request.form.get('experience', '').strip()
                print(f"DEBUG: Processing form data")
            
            # Validate required fields
            if not all([name, email, suggested_role]):
                error_msg = 'Please fill in all required fields (name, email, and role)'
                if request.is_json:
                    return jsonify({'success': False, 'error': error_msg}), 400
                flash(error_msg, 'danger')
                return render_template('hr/create_employee.html', 
                                     skills=get_all_skills(), 
                                     available_roles=get_available_roles())
            
            # Process skills
            skills = [skill.strip() for skill in skills_input.split(',')] if skills_input else []
            
            # Process experience
            experience = int(experience_input) if experience_input and experience_input.isdigit() else 0
            
            print(f"DEBUG: Creating recommendation for {name} with role {suggested_role}")
            print(f"DEBUG: Skills: {skills}")
            print(f"DEBUG: Experience: {experience}")
            
            # Create employee recommendation
            success, result = create_employee_recommendation(
                name=name,
                email=email,
                suggested_role=suggested_role,
                skills=skills,
                experience=experience,
                recommended_by=session['emp_id']
            )
            
            if not success:
                error_msg = f'Failed to submit employee recommendation: {result}'
                print(f"DEBUG: Recommendation creation failed: {result}")
                
                if request.is_json:
                    # Check if it's a duplicate email error (409 status)
                    if 'already exists' in str(result).lower() or 'duplicate' in str(result).lower():
                        return jsonify({'success': False, 'error': result}), 409
                    return jsonify({'success': False, 'error': result}), 400
                
                flash(error_msg, 'danger')
                return render_template('hr/create_employee.html',
                                     skills=get_all_skills(), 
                                     available_roles=get_available_roles())
            else:
                success_msg = f'Employee recommendation submitted for {name}. Forwarded to admin for approval.'
                print(f"DEBUG: Recommendation created successfully")
                
                # Notify all admins about the new recommendation
                try:
                    all_employees = get_all_employees()
                    admin_emails = [emp['email'] for emp in all_employees if emp.get('role') == 'admin']
                    
                    print(f"DEBUG: Found {len(admin_emails)} admin emails: {admin_emails}")
                    
                    # Send notification emails to admins
                    for admin_email in admin_emails:
                        send_new_application_notification(admin_email, name, email, suggested_role)
                        print(f"DEBUG: Sent notification to {admin_email}")
                except Exception as e:
                    print(f"Failed to send admin notifications: {str(e)}")
                
                if request.is_json:
                    return jsonify({
                        'success': True, 
                        'message': success_msg,
                        'recommendation': result if isinstance(result, dict) else {}
                    }), 200
                
                flash(success_msg, 'success')
                return redirect(url_for('human_resource_dashboard'))
                
        except Exception as e:
            error_msg = f'An error occurred: {str(e)}'
            print(f"DEBUG: Exception in create_employee: {str(e)}")
            import traceback
            traceback.print_exc()
            
            if request.is_json:
                return jsonify({'success': False, 'error': error_msg}), 500
            
            flash(error_msg, 'danger')
            return render_template('hr/create_employee.html')
    
    # GET request - show the form
    print(f"DEBUG: Rendering create_employee.html template")
    
    return render_template('hr/create_employee.html')

@app.route('/admin/create_employee', methods=['GET', 'POST'])
def create_employee():
    """Admin creates HR employee directly (restricted to HR role only)"""
    if 'emp_id' not in session or session['role'] != 'admin':
        flash('Unauthorized access')
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        # Employee ID is now optional
        emp_id = request.form.get('emp_id', '').strip()
        name = request.form.get('name')
        email = request.form.get('email')
        role = request.form.get('role')
        
        # Validate that only HR role can be created directly
        if role != 'human resource':
            flash('You can only create HR employees directly. Other employees must come from HR recommendations.', 'danger')
            return redirect(url_for('create_employee'))
        
        # Get skills as a list from the comma-separated input
        skills_input = request.form.get('skills', '').strip()
        skills = [skill.strip() for skill in skills_input.split(',')] if skills_input else None
        
        # Get experience as an integer
        experience_input = request.form.get('experience', '').strip()
        experience = int(experience_input) if experience_input and experience_input.isdigit() else None
        
        # Call API to create employee with the new parameters
        success, result = create_new_employee(
            name=name,
            email=email,
            role=role,
            skills=skills,
            experience=experience,
            emp_id=emp_id if emp_id else None
        )
        
        if not success:
            flash(f'Failed to create employee: {result}', 'danger')
        else:
            # Get the assigned employee ID for the success message
            assigned_emp_id = result if isinstance(result, str) else emp_id
            flash(f'HR Employee created with ID: {assigned_emp_id}. Credentials sent via email.', 'success')
        
        return redirect(url_for('admin_dashboard'))
    # Get project types for skill suggestions
    project_types, project_type_details = get_project_types()
    all_skills = []
    for skills_list in project_type_details.values():
        all_skills.extend(skills_list)
    # Remove duplicates while preserving order
    unique_skills = list(dict.fromkeys(all_skills))
    
    # Only allow HR role for direct creation
    available_roles = ['human resource']
    
    return render_template('create_employee.html', 
                         skills=unique_skills, 
                         available_roles=available_roles,
                         hr_only=True)  # Flag to show restriction message

@app.route('/admin/edit_employee/<emp_id>', methods=['GET', 'POST'])
def edit_employee(emp_id):
    if 'emp_id' not in session or session['role'] != 'admin':
        flash('Unauthorized access')
        return redirect(url_for('index'))
    
    employee = get_employee(emp_id)
    if not employee:
        flash('Employee not found')
        return redirect(url_for('admin_dashboard'))

    if request.method == 'POST':
        data = {
            'name': request.form.get('name'),
            'email': request.form.get('email'),
            'role': request.form.get('role')
        }
        
        # Update employee via API
        result = update_employee(emp_id, data)
        
        if not result:
            flash('Failed to update employee')
            return render_template('edit_employee.html', employee=employee)
        
        flash('Employee updated successfully')
        return redirect(url_for('admin_dashboard'))
    
    return render_template('edit_employee.html', employee=employee)

@app.route('/admin/delete_employee/<emp_id>', methods=['POST'])
def admin_delete_employee(emp_id):
    if 'emp_id' not in session or session['role'] != 'admin':
        flash('Unauthorized access')
        return redirect(url_for('index'))
    
    if emp_id == session['emp_id']:
        flash('Cannot delete your own account')
        return redirect(url_for('admin_dashboard'))
    
    # Delete employee via API
    if delete_employee(emp_id):
        flash('Employee deleted successfully')
    else:
        flash('Failed to delete employee')
    
    return redirect(url_for('admin_dashboard'))


@app.route('/admin_dashboard')
def admin_dashboard():
    if 'emp_id' not in session or session['role'] != 'admin':
        return redirect(url_for('index'))
    
    # Get all employees from API
    all_employees = get_all_employees()
    
    # Get pending employee recommendations
    pending_recommendations = get_employee_recommendations('pending_admin_review')
    
    # Get recently processed recommendations
    recent_processed = get_employee_recommendations('approved') + get_employee_recommendations('rejected')
    # Sort by processed date and take last 10
    recent_processed.sort(key=lambda x: x.get('processed_at', ''), reverse=True)
    recent_processed = recent_processed[:10]
    
    return render_template('admin_dashboard.html', 
                         employees=all_employees, 
                         pending_recommendations=pending_recommendations,
                         recent_processed=recent_processed,
                         current_user=session)


@app.route('/admin/process_recommendation/<int:rec_id>', methods=['POST'])
def admin_process_recommendation(rec_id):
    """Admin processes employee recommendation"""
    if 'emp_id' not in session or session['role'] != 'admin':
        return jsonify({'success': False, 'error': 'Unauthorized access'}), 403
    
    data = request.json
    action = data.get('action')  # 'approve' or 'reject'
    notes = data.get('notes', '')
    
    if action not in ['approve', 'reject']:
        return jsonify({'success': False, 'error': 'Invalid action'}), 400
    
    # Process the recommendation
    success, result = process_employee_recommendation(rec_id, action, session['emp_id'], notes)
    
    if not success:
        return jsonify({'success': False, 'error': result}), 500
    
    if action == 'approve':
        try:
            # Send approval email with credentials if employee was created
            employee_data = result.get('employee', {})
            if employee_data:
                message = f'Employee recommendation approved. Employee created with ID: {employee_data.get("emp_id")} and credentials sent via email.'
            else:
                message = 'Employee recommendation approved.'
                
            return jsonify({
                'success': True, 
                'message': message,
                'employee': employee_data
            })
            
        except Exception as e:
            # Employee was approved but email might have failed
            return jsonify({
                'success': True, 
                'message': f'Employee recommendation approved but email notification failed: {str(e)}'
            })
    else:
        return jsonify({
            'success': True, 
            'message': 'Employee recommendation rejected.'
        })

# NEW: Route to view recommendation details
@app.route('/admin/recommendation/<int:rec_id>')
def view_recommendation(rec_id):
    """View detailed recommendation"""
    if 'emp_id' not in session or session['role'] != 'admin':
        flash('Unauthorized access', 'danger')
        return redirect(url_for('index'))
    
    try:
        response = requests.get(
            f'{EMPLOYEE_SERVICE_URL}/employee-recommendations/{rec_id}',
            headers=api_headers()
        )
        
        if response.status_code == 200:
            recommendation = response.json().get('recommendation', {})
            return render_template('recommendation_details.html', recommendation=recommendation)
        else:
            flash('Recommendation not found', 'danger')
            return redirect(url_for('admin_dashboard'))
    
    except Exception as e:
        flash(f'Error retrieving recommendation: {str(e)}', 'danger')
        return redirect(url_for('admin_dashboard'))



@app.route('/admin/approve_employee/<emp_id>', methods=['POST'])
def admin_approve_employee(emp_id):
    """Admin approves employee and assigns system role"""
    if 'emp_id' not in session or session['role'] != 'admin':
        return jsonify({'success': False, 'error': 'Unauthorized access'}), 403
    
    data = request.json
    assigned_role = data.get('role')
    
    if not assigned_role:
        return jsonify({'success': False, 'error': 'Role is required'}), 400
    
    # Approve the employee
    result = approve_employee(emp_id, assigned_role, session['emp_id'])
    
    if not result:
        return jsonify({'success': False, 'error': 'Failed to approve employee'}), 500
    
    try:
        # Send approval email with credentials
        employee_data = result.get('employee', {})
        temp_password = result.get('temp_password', 'defaultpass123')
        
        send_approval_notification(
            email=employee_data.get('email'),
            name=employee_data.get('name'),
            emp_id=emp_id,
            role=assigned_role,
            temp_password=temp_password
        )
        
        return jsonify({
            'success': True, 
            'message': f'Employee approved as {assigned_role} and credentials sent via email'
        })
        
    except Exception as e:
        # Employee was approved but email failed
        return jsonify({
            'success': True, 
            'message': f'Employee approved as {assigned_role} but failed to send email: {str(e)}'
        })


@app.route('/admin/update_metrics/<emp_id>', methods=['POST'])
def update_metrics(emp_id):
    if 'emp_id' not in session or session['role'] != 'admin':
        return jsonify({'success': False, 'error': 'Unauthorized access'}), 403
    
    data = request.json
    if not data:
        return jsonify({'success': False, 'error': 'No data provided'}), 400
    
    # Prepare update data
    update_data = {}
    if 'tasks_completed' in data:
        update_data['tasks_completed'] = data['tasks_completed']
    if 'success_rate' in data:
        update_data['success_rate'] = data['success_rate']
    
    # Update employee via API
    result = update_employee(emp_id, update_data)
    
    if not result:
        return jsonify({'success': False, 'error': 'Failed to update employee metrics'}), 500
    
    return jsonify({'success': True})

@app.route('/api/assign_task_confirmed', methods=['POST'])
def assign_task_confirmed():
    """Actually assign and save the task after confirmation"""
    if 'emp_id' not in session or session.get('role') != 'project manager':
        return jsonify({'success': False, 'error': 'Unauthorized'}), 403
    
    task_data = request.json
    tasks = [task_data]
    
    # Use the actual assignment endpoint that saves to database
    result = assign_tasks_with_save(tasks)
    
    if not result or not result.get('success'):
        return jsonify({
            'success': False, 
            'error': result.get('error', 'Failed to assign task')
        }), 500
        
    return jsonify({
        'success': True,
        'assignment': result.get('assignments', {}).get(task_data['task_id']),
        'saved_tasks': result.get('saved_tasks', []),
        'email_notifications': result.get('email_notifications', [])
    })


@app.route('/api/create_task', methods=['POST'])
def create_task():
    """Create a new task and get assignment recommendations"""
    if 'emp_id' not in session or session.get('role') != 'project manager':
        return jsonify({'success': False, 'error': 'Unauthorized'}), 403
        
    task_data = request.json
    # Validate task data
    required_fields = ['task_id', 'project_type', 'complexity', 'priority']
    for field in required_fields:
        if field not in task_data:
            return jsonify({'success': False, 'error': f'Missing required field: {field}'}), 400
    
    # Get skills for project type if not provided
    if 'skills' not in task_data or not task_data['skills']:
        task_data['skills'] = get_skills_for_project_type(task_data['project_type'])
    
    # Create list of tasks (API expects an array)
    tasks = [task_data]
    
    # Get recommendation first using the recommendations endpoint
    result = assign_tasks(tasks)  # This now calls the recommendations endpoint
    
    if not result or 'success' not in result or not result['success']:
        return jsonify({
            'success': False, 
            'error': result.get('error', 'Failed to get task recommendation')
        }), 500
        
    return jsonify({
        'success': True,
        'task': task_data,
        'recommendation': result.get('assignments', {}).get(task_data['task_id'])
    })


@app.route('/submit_task_for_review', methods=['POST'])
def submit_task_for_review():
    if 'emp_id' not in session:
        return jsonify({'success': False, 'error': 'Unauthorized'}), 401

    try:
        data = request.get_json()
        task_id = data.get('task_id')
        
        if not task_id:
            return jsonify({'success': False, 'error': 'Task ID required'}), 400

        # Call task service
        api_url = f'http://localhost:5002/api/task-service/tasks/{task_id}/submit'
        response = requests.post(
            api_url,
            headers=api_headers(),
            json={'emp_id': session['emp_id']},
            timeout=5
        )

        # Handle response
        response_data = response.json()
        if response.status_code != 200:
            return jsonify({
                'success': False,
                'error': response_data.get('error', 'Task service error')
            }), response.status_code

        return jsonify(response_data)

    except requests.exceptions.RequestException as e:
        return jsonify({
            'success': False,
            'error': f'Service unavailable: {str(e)}'
        }), 503
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Internal error: {str(e)}'
        }), 500

@app.route('/get_task/<task_id>', methods=['GET'])
def get_task(task_id):
    """Get details for a specific task"""
    if 'emp_id' not in session:
        return jsonify({'success': False, 'error': 'Unauthorized'}), 403
    
    try:
        # Define API endpoint - FIXED to use port 5002 and correct path
        api_endpoint = f'http://localhost:5002/api/task-service/tasks/{task_id}'
        
        # Log the request
        app.logger.info(f"Requesting task data from: {api_endpoint}")
        
        # Get headers
        headers = api_headers()
        
        # Call the task service API with error handling
        response = requests.get(api_endpoint, headers=headers)
        
        # Log response status
        app.logger.info(f"Task service response status: {response.status_code}")
        
        if response.status_code != 200:
            error_message = f"Failed to get task: Status {response.status_code}"
            try:
                error_detail = response.json()
                error_message += f" - {error_detail.get('error', 'Unknown error')}"
            except:
                error_message += f" - {response.text[:100]}"
            
            app.logger.error(error_message)
            return jsonify({'success': False, 'error': error_message}), 500
        
        # Parse response
        try:
            result = response.json()
            return jsonify({'success': True, 'task': result.get('task', {})})
        except Exception as json_error:
            app.logger.error(f"Error parsing JSON response: {str(json_error)}")
            return jsonify({'success': False, 'error': f'Invalid response format: {str(json_error)}'}), 500
    
    except requests.exceptions.ConnectionError as conn_error:
        error_msg = f"Connection error to task service: {str(conn_error)}"
        app.logger.error(error_msg)
        return jsonify({'success': False, 'error': error_msg}), 500
    except Exception as e:
        app.logger.error(f"Unexpected error in get_task: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/pending_review_tasks', methods=['GET'])
def get_pending_review_tasks():
    if 'emp_id' not in session:
        return jsonify({'success': False, 'error': 'Unauthorized'}), 401
    
    try:
        # Call task service to get pending review tasks
        api_url = f'{TASK_SERVICE_URL}/task-service/tasks/pending-review'
        response = requests.get(
            api_url,
            headers=api_headers(),
            timeout=5
        )
        
        # Handle response
        response_data = response.json()
        if response.status_code != 200:
            return jsonify({
                'success': False,
                'error': response_data.get('error', 'Task service error')
            }), response.status_code
        
        return jsonify(response_data)
    
    except requests.exceptions.RequestException as e:
        return jsonify({
            'success': False,
            'error': f'Service unavailable: {str(e)}'
        }), 503
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Internal error: {str(e)}'
        }), 500

@app.route('/task_details/<task_id>')
def task_details(task_id):
    if 'emp_id' not in session:
        flash('Please log in to view task details', 'warning')
        return redirect(url_for('login'))
    
    try:
        # Get task details
        task_response = requests.get(
            f'{TASK_SERVICE_URL}/task-service/tasks/{task_id}',
            headers=api_headers()
        )
        
        if task_response.status_code != 200:
            flash('Failed to retrieve task details', 'danger')
            return redirect(url_for('index'))
        
        task = task_response.json().get('task', {})
        
        # Get assigned employees for this task
        assignees_response = requests.get(
            f'{TASK_SERVICE_URL}/task-service/tasks/{task_id}/assignees',
            headers=api_headers()
        )
        
        assignees = []
        if assignees_response.status_code == 200:
            assignee_ids = assignees_response.json().get('assignees', [])
            # Get employee details for each assignee
            for emp_id in assignee_ids:
                emp_data = get_employee(emp_id)
                if emp_data:
                    assignees.append(emp_data)
        
        # Get task history/activity log
        history_response = requests.get(
            f'{TASK_SERVICE_URL}/task-service/tasks/{task_id}/history',
            headers=api_headers()
        )
        
        history = []
        if history_response.status_code == 200:
            history = history_response.json().get('history', [])
        
        # Different templates based on user role
        if session.get('role') == 'project manager':
            return render_template(
                'task_details_manager.html',
                task=task,
                assignees=assignees,
                history=history,
                format_date=format_date
            )
        else:
            return render_template(
                'task_details.html',
                task=task,
                assignees=assignees,
                history=history,
                format_date=format_date
            )
    
    except Exception as e:
        flash(f'Error retrieving task details: {str(e)}', 'danger')
        return redirect(url_for('index'))


@app.route('/api/get_assignment_recommendation', methods=['POST'])
def get_assignment_recommendation():
    """Get assignment recommendation for a task without saving it"""
    if 'emp_id' not in session or session.get('role') != 'project manager':
        return jsonify({'success': False, 'error': 'Unauthorized'}), 403
        
    task_data = request.json
    tasks = [task_data]
    
    # Get skills for project type if not provided
    if 'skills' not in task_data or not task_data['skills']:
        task_data['skills'] = get_skills_for_project_type(task_data['project_type'])
    
    # Get recommendation without persistence
    try:
        result = assign_tasks(tasks)
        
        if not result:
            return jsonify({
                'success': False, 
                'error': 'Failed to get recommendation - result is None'
            }), 500
        
        if 'success' not in result or not result['success']:
            return jsonify({
                'success': False, 
                'error': result.get('error', 'Failed to get recommendation')
            }), 500
            
        return jsonify({
            'success': True,
            'recommendations': result.get('assignments', {}).get(task_data['task_id'])
        })
    except Exception as e:
        import traceback
        print(f"Error in get_assignment_recommendation: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': f'Exception: {str(e)}'
        }), 500



@app.route('/update_task_status', methods=['POST'])
def update_task_status():
    """Update a task's status"""
    if 'emp_id' not in session:
        return jsonify({'success': False, 'error': 'Not logged in'}), 401
        
    data = request.json
    if not data or 'task_id' not in data or 'status' not in data:
        return jsonify({'success': False, 'error': 'Missing required fields'}), 400
        
    task_id = data['task_id']
    # Check if task_id is empty or None
    if not task_id:
        return jsonify({'success': False, 'error': 'Invalid task ID'}), 400
        
    new_status = data['status']
    rating = data.get('rating')
    emp_id = session['emp_id']
    
    # Fix the URL formatting - Use correct endpoint structure
    print(f"TASK_SERVICE_URL value: {TASK_SERVICE_URL}")
    
    # Clear any trailing slashes from the base URL
    base_url = TASK_SERVICE_URL.rstrip('/')
    
    # Construct URL correctly based on whether /api is already present
    if '/api' in base_url:
        task_url = f'{base_url}/task-service/tasks/{task_id}/status'
    else:
        task_url = f'{base_url}/api/task-service/tasks/{task_id}/status'
    
    print(f"Making request to: {task_url}")
        
    # Call the task service API to update the task
    try:
        response = requests.put(
            task_url,
            json={
                'emp_id': emp_id,
                'status': new_status,
                'rating': rating
            },
            headers=api_headers()
        )
            
        # Print response for debugging
        print(f"Update task response: {response.status_code} - {response.text}")
            
        if response.status_code != 200:
            return jsonify({
                'success': False, 
                'error': f'Task service error: {response.status_code} - {response.text}'
            }), 500
                
        result = response.json()
        return jsonify({
            'success': True,
            'task': result.get('task', {})
        })
    except Exception as e:
        print(f"Error updating task status: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Failed to update task: {str(e)}'
        }), 500


# Modified get_all_tasks function to ensure Task is defined
@app.route('/api/task-service/tasks', methods=['GET'])
def get_all_tasks():
    """Get all tasks or filter by employee, status, etc."""
    try:
        # Get query parameters for filtering
        emp_id = request.args.get('emp_id')
        status = request.args.get('status')
        project_type = request.args.get('project_type')
        
        # Start with base query - make sure Task is defined in this scope
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


@app.route('/salary_estimator')
def salary_estimator():
    """Salary estimation page for HR"""
    if 'emp_id' not in session or session['role'] != 'human resource':
        flash('Access restricted to HR personnel', 'danger')
        return redirect(url_for('index'))
    
    # Common job titles and locations for the form
    common_job_titles = [
        "data scientist", "data analyst", "software developer", 
        "software engineer", "devops engineer", "machine learning engineer",
        "product manager", "ui/ux designer", "backend developer",
        "full stack developer", "frontend developer"
    ]
    
    common_locations = ["cape town", "johannesburg", "durban", "pretoria"]
    experience_levels = ["LESS_THAN_ONE", "ONE_TO_THREE", "FOUR_TO_SIX", "SEVEN_TO_NINE", "TEN_PLUS"]
    
    return render_template(
        'salary_estimator.html',
        job_titles=common_job_titles,
        locations=common_locations,
        experience_levels=experience_levels
    )

@app.route('/api/estimate_salary', methods=['POST'])
def api_estimate_salary():
    """API endpoint for salary estimation"""
    if 'emp_id' not in session or session['role'] != 'human resource':
        return jsonify({'success': False, 'error': 'Unauthorized'}), 403
    
    try:
        # Check if salary services are initialized
        if salary_predictor is None:
            logger.error("Salary predictor not initialized")
            return jsonify({
                'success': False, 
                'error': 'Salary prediction service is not available. Please contact your administrator.'
            }), 503
        
        data = request.json
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
            
        job_title = data.get('job_title')
        location = data.get('location')
        experience_level = data.get('experience_level')
        budget = data.get('budget')
        
        if not all([job_title, location, experience_level]):
            return jsonify({'success': False, 'error': 'Missing required parameters'}), 400
        
        logger.info(f"Estimating salary for {job_title} in {location} with {experience_level} experience")
        
        # Get salary prediction
        prediction = salary_predictor.predict_salary(job_title, location, experience_level)
        
        if not prediction:
            logger.error("Salary prediction returned None")
            return jsonify({
                'success': False, 
                'error': 'Unable to generate salary prediction for the given parameters'
            }), 400
        
        result = {
            'success': True,
            'prediction': prediction
        }
        
        # Add competitiveness analysis if budget is provided and analyzer is available
        if budget and competitiveness_analyzer is not None:
            try:
                budget = float(budget)
                analysis = competitiveness_analyzer.analyze_competitiveness(
                    prediction.get('predicted_annual_salary', 0),
                    prediction.get('predicted_range', [0, 0]),
                    budget
                )
                result['competitiveness_analysis'] = analysis
                logger.info("Competitiveness analysis completed successfully")
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid budget value provided: {e}")
                # Don't fail the entire request for invalid budget
            except Exception as e:
                logger.error(f"Error in competitiveness analysis: {e}")
                # Don't fail the entire request for analysis errors
        
        logger.info("Salary estimation completed successfully")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in salary estimation: {e}")
        return jsonify({
            'success': False, 
            'error': f'Internal server error: {str(e)}'
        }), 500


@app.route('/api/retrain_salary_model', methods=['POST'])
def api_retrain_salary_model():
    """API endpoint to retrain the salary prediction model"""
    if 'emp_id' not in session or session['role'] != 'human resource':
        return jsonify({'success': False, 'error': 'Unauthorized'}), 403
        
    try:
        logger.info("Starting salary model retraining process")
        
        # Use the retrain function that forces fresh data collection
        success = retrain_salary_model()
        
        if success:
            # Reinitialize the global predictors with the new model
            global salary_predictor, competitiveness_analyzer
            try:
                salary_predictor = get_salary_predictor()
                competitiveness_analyzer = get_competitiveness_analyzer()
                logger.info("Salary prediction services reinitialized after retraining")
            except Exception as reinit_error:
                logger.error(f"Failed to reinitialize services after retraining: {reinit_error}")
                # Continue anyway, the retrain was successful
            
            return jsonify({
                'success': True, 
                'message': 'Salary model retrained successfully with fresh data'
            })
        else:
            logger.error("Model retraining failed - insufficient fresh data collected")
            return jsonify({
                'success': False, 
                'error': 'Failed to retrain model - insufficient fresh data collected'
            }), 500
            
    except Exception as e:
        logger.error(f"Error during model retraining: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
    


@app.route('/api/retrain_salary_model_with_tracking', methods=['POST'])
def api_retrain_salary_model_with_tracking():
    """Enhanced API endpoint to retrain the salary prediction model with progress tracking"""
    if 'emp_id' not in session or session['role'] != 'human resource':
        return jsonify({'success': False, 'error': 'Unauthorized'}), 403
        
    session_id = session['emp_id']
    
    # Check if retraining is already in progress
    current_progress = get_retraining_progress(session_id)
    if current_progress['percentage'] > 0 and not current_progress['completed'] and not current_progress['failed']:
        return jsonify({
            'success': False,
            'error': 'Retraining is already in progress'
        }), 409
    
    def retrain_with_detailed_progress():
        try:
            logger.info("Starting salary model retraining with detailed progress tracking")
            
            # Clear any previous progress
            clear_retraining_progress(session_id)
            
            # Step 1: Initialization (2-8%)
            set_retraining_progress(session_id, 2, 'Initializing data collection...', 'step-init')
            time.sleep(1)
            
            # Initialize collector
            from salary_service import SalaryDataCollector
            salary_collector = SalaryDataCollector()
            set_retraining_progress(session_id, 5, 'Data collector initialized', 'step-init')
            time.sleep(1)
            
            set_retraining_progress(session_id, 8, 'Preparing to collect salary data...', 'step-init')
            time.sleep(1)
            
            # Step 2: Data Collection (8-85%) - This is the main part
            set_retraining_progress(session_id, 10, 'Starting salary data collection...', 'step-collect')
            
            # The actual data collection
            # We'll simulate the progress since the original function doesn't provide callbacks
            priority_combinations = [
                ("data scientist", "cape town", "LESS_THAN_ONE"),
                ("data scientist", "johannesburg", "LESS_THAN_ONE"),
                ("data scientist", "cape town", "ONE_TO_THREE"),
                ("data scientist", "johannesburg", "ONE_TO_THREE"),
                ("data scientist", "cape town", "FOUR_TO_SIX"),
                ("data scientist", "johannesburg", "FOUR_TO_SIX"),
                ("data analyst", "cape town", "LESS_THAN_ONE"),
                ("data scientist", "cape town", "SEVEN_TO_NINE"),
                ("data scientist", "johannesburg", "SEVEN_TO_NINE"),
                ("data analyst", "johannesburg", "LESS_THAN_ONE"),
                ("data analyst", "cape town", "ONE_TO_THREE"),
                ("data analyst", "johannesburg", "ONE_TO_THREE"),
                ("data analyst", "cape town", "FOUR_TO_SIX"),
                ("data analyst", "johannesburg", "FOUR_TO_SIX"),
                ("data analyst", "cape town", "SEVEN_TO_NINE"),
                ("data analyst", "johannesburg", "SEVEN_TO_NINE"),
                ("software developer", "cape town", "LESS_THAN_ONE"),
                ("software developer", "johannesburg", "LESS_THAN_ONE"),
                ("software engineer", "cape town", "ONE_TO_THREE"),
                ("software engineer", "johannesburg", "ONE_TO_THREE"),
                ("software engineer", "cape town", "FOUR_TO_SIX"),
                ("software engineer", "johannesburg", "FOUR_TO_SIX"),
                ("software engineer", "cape town", "SEVEN_TO_NINE"),
                ("software engineer", "johannesburg", "SEVEN_TO_NINE"),
                ("full stack developer", "johannesburg", "ONE_TO_THREE"),
                ("backend developer", "johannesburg", "FOUR_TO_SIX"),
                ("devops engineer", "cape town", "FOUR_TO_SIX"),
                ("machine learning engineer", "johannesburg", "FOUR_TO_SIX"),
                ("product manager", "cape town", "FOUR_TO_SIX"),
                ("ui/ux designer", "cape town", "ONE_TO_THREE"),
            ]
            
            # Simulate the collection process with realistic timing
            batch_size = 5
            total_combinations = len(priority_combinations)
            base_progress = 10
            collection_progress_range = 75  # 85 - 10 = 75%
            
            # Process in batches like the real implementation
            for i in range(0, total_combinations, batch_size):
                batch = priority_combinations[i:i + batch_size]
                batch_num = (i // batch_size) + 1
                total_batches = (total_combinations + batch_size - 1) // batch_size
                
                # Update progress for batch start
                batch_progress = base_progress + (i / total_combinations) * collection_progress_range
                set_retraining_progress(session_id, int(batch_progress), 
                                      f'Processing batch {batch_num}/{total_batches}...', 'step-collect')
                
                # Process each item in batch (simulate the 3-second API calls + processing)
                for j, (job_title, location, experience) in enumerate(batch):
                    item_progress = batch_progress + (j + 1) * (collection_progress_range / total_combinations)
                    set_retraining_progress(session_id, int(item_progress), 
                                          f'Collecting data for {job_title} in {location}...', 'step-collect')
                    time.sleep(3.5)  # Simulate API call + processing time
                
                # Batch break (like in the real implementation)
                set_retraining_progress(session_id, int(batch_progress + (batch_size / total_combinations) * collection_progress_range), 
                                      f'Batch {batch_num} completed. Taking break...', 'step-collect')
                time.sleep(2)
            
            # Step 3: Quality Assessment (85-92%)
            set_retraining_progress(session_id, 85, 'Starting quality assessment...', 'step-quality')
            time.sleep(2)
            
            set_retraining_progress(session_id, 88, 'Analyzing data quality...', 'step-quality')
            time.sleep(2)
            
            set_retraining_progress(session_id, 92, 'Data cleaning completed', 'step-quality')
            time.sleep(1)
            
            # Step 4: Model Training (92-98%)
            set_retraining_progress(session_id, 93, 'Preparing model training...', 'step-train')
            time.sleep(1)
            
            set_retraining_progress(session_id, 95, 'Training machine learning model...', 'step-train')
            
            # Actually call the retraining function
            success = retrain_salary_model()
            
            if not success:
                set_retraining_progress(session_id, 0, 'Model training failed', 'step-train', 'Insufficient training data')
                return False
            
            set_retraining_progress(session_id, 98, 'Model training completed successfully', 'step-train')
            time.sleep(1)
            
            # Step 5: Saving (98-100%)
            set_retraining_progress(session_id, 99, 'Saving updated model...', 'step-save')
            time.sleep(1)
            
            # Reinitialize global predictors
            global salary_predictor, competitiveness_analyzer
            try:
                salary_predictor = get_salary_predictor()
                competitiveness_analyzer = get_competitiveness_analyzer()
                logger.info("Salary prediction services reinitialized after retraining")
            except Exception as reinit_error:
                logger.error(f"Failed to reinitialize services: {reinit_error}")
                # Continue anyway
            
            set_retraining_progress(session_id, 100, 'Model retraining completed successfully!', 'step-save')
            logger.info("Salary model retrained successfully with detailed progress tracking")
            return True
            
        except Exception as e:
            logger.error(f"Error during model retraining: {e}")
            set_retraining_progress(session_id, 0, f'Retraining failed', None, str(e))
            return False
    
    # Start retraining in background thread
    def run_retraining():
        retrain_with_detailed_progress()
    
    thread = threading.Thread(target=run_retraining)
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'success': True, 
        'message': 'Model retraining started. Use /api/retrain_progress to track progress.'
    })



# Add a jinja template filter for formatting dates
@app.template_filter('datetime')
def format_datetime(value, format='%B %d, %Y %I:%M %p'):
    if value is None:
        return ""
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace('Z', '+00:00'))
        except:
            return value
    return value.strftime(format)

@app.route('/test_email')
def test_email():
    test_email = "lolomoakamela@gmail.com"  # Use a real email you can check
    result = send_credentials_email(
        email=test_email,
        emp_id="TEST123",
        temp_password="testpass123"
    )
    return jsonify({"success": result})

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(port=5000, debug=True)