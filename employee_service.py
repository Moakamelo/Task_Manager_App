# employee_service.py
from email_services import send_credentials_email
from flask import Flask, request, jsonify, url_for
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import os
import string
import random
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Database configuration
DATABASE_URL = os.environ.get('DATABASE_URL', 'postgresql://postgres:password@localhost:5432/employee_db')
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev_secret_key_for_api')

# Initialize SQLAlchemy
db = SQLAlchemy(app)


class EmployeeRecommendation(db.Model):
    __tablename__ = 'employee_recommendations'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), nullable=False)
    suggested_role = db.Column(db.String(50), nullable=False)
    skills = db.Column(db.ARRAY(db.String(50)), default=[])
    experience = db.Column(db.Integer, default=0)
    
    # Recommendation metadata
    recommended_by = db.Column(db.String(50), nullable=False)  # HR emp_id
    status = db.Column(db.String(30), default='pending_admin_review')  # pending_admin_review, approved, rejected, processed
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    processed_at = db.Column(db.DateTime, nullable=True)
    processed_by = db.Column(db.String(50), nullable=True)  # Admin emp_id
    notes = db.Column(db.Text, nullable=True)
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'email': self.email,
            'suggested_role': self.suggested_role,
            'skills': self.skills or [],
            'experience': self.experience or 0,
            'recommended_by': self.recommended_by,
            'status': self.status,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'processed_at': self.processed_at.isoformat() if self.processed_at else None,
            'processed_by': self.processed_by,
            'notes': self.notes
        }



class Employee(db.Model):
    __tablename__ = 'employees'
    
    emp_id = db.Column(db.String(50), primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    is_first_login = db.Column(db.Boolean, default=True)
    role = db.Column(db.String(20), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime, nullable=True)
    
    # Additional fields
    skills = db.Column(db.ARRAY(db.String(50)), default=[])
    experience = db.Column(db.Integer, default=0)
    tasks_completed = db.Column(db.Integer, default=0)
    success_rate = db.Column(db.Float, default=0.0)
    
    # NEW: Track if employee was created from recommendation
    created_from_recommendation = db.Column(db.Integer, db.ForeignKey('employee_recommendations.id'), nullable=True)
    
    def __repr__(self):
        return f'<Employee {self.emp_id} - {self.name}>'
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def to_dict(self):
        return {
            'emp_id': self.emp_id,
            'name': self.name,
            'email': self.email,
            'role': self.role.strip().lower(),
            'is_first_login': bool(self.is_first_login),
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'skills': self.skills or [],
            'experience': self.experience or 0,
            'tasks_completed': self.tasks_completed or 0,
            'success_rate': self.success_rate or 0.0,
            'created_from_recommendation': self.created_from_recommendation
        }


# Function to generate random password
def generate_random_password(length=5):  
    characters = string.ascii_letters + string.digits + string.punctuation
    return ''.join(random.choice(characters) for _ in range(length))

# Function to get the next available employee ID
# Fix for the get_next_employee_id function
# Alternative simpler version if regexp_replace is not supported in your PostgreSQL version
def get_next_employee_id():
    try:
        # Print debug information
        print("Fetching all employees to find highest ID")
        
        # Get all employees
        employees = Employee.query.all()
        
        # Debug: Print all employee IDs
        all_ids = [emp.emp_id for emp in employees]
        print(f"All employee IDs in database: {all_ids}")
        
        # Start with 1000 as the next ID
        next_id = 1000
        
        # Process each employee ID
        for emp in employees:
            try:
                # Try to convert the ID to an integer if it's numeric
                emp_id_int = int(emp.emp_id)
                
                # If this ID is higher than our current next_id, update next_id
                if emp_id_int >= next_id:
                    next_id = emp_id_int + 1
                    
                print(f"Found numeric ID: {emp_id_int}, next_id now: {next_id}")
            except ValueError:
                # If the ID is not a number, just skip it
                print(f"Skipping non-numeric ID: {emp.emp_id}")
                continue
        
        # Convert final result to string
        result = str(next_id)
        print(f"Final next employee ID: {result}")
        return result
            
    except Exception as e:
        # If anything goes wrong, log it and use timestamp
        print(f"Error in get_next_employee_id: {str(e)}")
        timestamp_id = str(int(datetime.now().timestamp()))
        print(f"Falling back to timestamp ID: {timestamp_id}")
        return timestamp_id
    
# API Authentication - Simple API key check
def authenticate_request():
    api_key = request.headers.get('X-API-KEY')
    expected_key = os.environ.get('API_KEY', 'dev_api_key')
    
    print(f"Received headers: {dict(request.headers)}")
    print(f"Received API key: '{api_key}'")
    print(f"Expected API key: '{expected_key}'")
    print(f"Keys equal? {api_key == expected_key}")
    print(f"Request method: {request.method}")
    print(f"Request path: {request.path}")
    
    if not api_key or api_key != expected_key:
        return False
    return True


@app.route('/api/employee-recommendations', methods=['POST'])
def create_recommendation():
    """HR creates employee recommendation"""
    if not authenticate_request():
        return jsonify({'error': 'Unauthorized'}), 401
    
    data = request.json
    required_fields = ['name', 'email', 'suggested_role', 'recommended_by']
    
    if not all(k in data for k in required_fields):
        return jsonify({'error': f'Missing required fields: {required_fields}'}), 400
    
    # Check if email already exists in employees or pending recommendations
    existing_employee = Employee.query.filter_by(email=data['email']).first()
    if existing_employee:
        return jsonify({'error': 'Email already exists in employee database'}), 409
    
    existing_recommendation = EmployeeRecommendation.query.filter_by(
        email=data['email'], 
        status='pending_admin_review'
    ).first()
    if existing_recommendation:
        return jsonify({'error': 'Email already has pending recommendation'}), 409
    
    try:
        recommendation = EmployeeRecommendation(
            name=data['name'],
            email=data['email'],
            suggested_role=data['suggested_role'],
            skills=data.get('skills', []),
            experience=data.get('experience', 0),
            recommended_by=data['recommended_by']
        )
        
        db.session.add(recommendation)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Employee recommendation submitted successfully',
            'recommendation': recommendation.to_dict()
        }), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/api/employee-recommendations', methods=['GET'])
def get_recommendations():
    """Get employee recommendations (filtered by status)"""
    if not authenticate_request():
        return jsonify({'error': 'Unauthorized'}), 401
    
    status = request.args.get('status', 'pending_admin_review')
    recommended_by = request.args.get('recommended_by')
    
    query = EmployeeRecommendation.query
    
    if status != 'all':
        query = query.filter_by(status=status)
    
    if recommended_by:
        query = query.filter_by(recommended_by=recommended_by)
    
    recommendations = query.order_by(EmployeeRecommendation.created_at.desc()).all()
    
    return jsonify({
        'success': True,
        'recommendations': [rec.to_dict() for rec in recommendations],
        'count': len(recommendations)
    })

@app.route('/api/employee-recommendations/<int:rec_id>', methods=['GET'])
def get_recommendation(rec_id):
    """Get specific recommendation"""
    if not authenticate_request():
        return jsonify({'error': 'Unauthorized'}), 401
    
    recommendation = EmployeeRecommendation.query.get(rec_id)
    if not recommendation:
        return jsonify({'error': 'Recommendation not found'}), 404
    
    return jsonify({
        'success': True,
        'recommendation': recommendation.to_dict()
    })

@app.route('/api/employee-recommendations/<int:rec_id>/process', methods=['PUT'])
def process_recommendation(rec_id):
    """Admin processes recommendation (approve/reject)"""
    if not authenticate_request():
        return jsonify({'error': 'Unauthorized'}), 401
    
    data = request.json
    action = data.get('action')  # 'approve' or 'reject'
    processed_by = data.get('processed_by')  # Admin emp_id
    notes = data.get('notes', '')
    
    if action not in ['approve', 'reject']:
        return jsonify({'error': 'Action must be approve or reject'}), 400
    
    recommendation = EmployeeRecommendation.query.get(rec_id)
    if not recommendation:
        return jsonify({'error': 'Recommendation not found'}), 404
    
    if recommendation.status != 'pending_admin_review':
        return jsonify({'error': 'Recommendation already processed'}), 409
    
    try:
        if action == 'approve':
            # Create employee from recommendation
            emp_id = get_next_employee_id()
            temp_password = generate_random_password()
            
            new_employee = Employee(
                emp_id=emp_id,
                name=recommendation.name,
                email=recommendation.email,
                role=recommendation.suggested_role,
                skills=recommendation.skills,
                experience=recommendation.experience,
                created_from_recommendation=rec_id
            )
            new_employee.set_password(temp_password)
            
            db.session.add(new_employee)
            
            # Update recommendation status
            recommendation.status = 'approved'
            recommendation.processed_at = datetime.utcnow()
            recommendation.processed_by = processed_by
            recommendation.notes = notes
            
            db.session.commit()
            
            # Send credentials email
            send_credentials_email(
                email=recommendation.email,
                emp_id=emp_id,
                temp_password=temp_password
            )
            
            return jsonify({
                'success': True,
                'message': 'Employee created successfully from recommendation',
                'employee': new_employee.to_dict(),
                'recommendation': recommendation.to_dict()
            }), 201
            
        else:  # reject
            recommendation.status = 'rejected'
            recommendation.processed_at = datetime.utcnow()
            recommendation.processed_by = processed_by
            recommendation.notes = notes
            
            db.session.commit()
            
            return jsonify({
                'success': True,
                'message': 'Recommendation rejected',
                'recommendation': recommendation.to_dict()
            })
            
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

# API Routes
@app.route('/api/employees', methods=['POST'])
def create_employee():
    """Admin creates employee directly (restricted to HR role only)"""
    if not authenticate_request():
        return jsonify({'error': 'Unauthorized'}), 401
    
    data = request.json
    
    # RESTRICTION: Only allow HR role creation directly
    if data.get('role') != 'human resource':
        return jsonify({
            'error': 'Direct employee creation restricted to HR role only. Other employees must come from HR recommendations.'
        }), 403
    
    # Validate required fields
    required_fields = ['name', 'email', 'role']
    if not all(k in data for k in required_fields):
        return jsonify({'error': f'Missing required fields: {required_fields}'}), 400
    
    # Check if email already exists
    existing_employee = Employee.query.filter_by(email=data['email']).first()
    if existing_employee:
        return jsonify({'error': 'Email already exists'}), 409
    
    try:
        # Generate employee ID if not provided
        if 'emp_id' not in data or not data['emp_id']:
            emp_id = get_next_employee_id()
        else:
            emp_id = str(data['emp_id'])
            if Employee.query.get(emp_id):
                return jsonify({'error': 'Employee ID already exists'}), 409
        
        temp_password = generate_random_password()
        
        new_employee = Employee(
            emp_id=emp_id,
            name=data['name'],
            email=data['email'],
            role=data['role'],
            skills=data.get('skills', []),
            experience=data.get('experience', 0),
            tasks_completed=data.get('tasks_completed', 0),
            success_rate=data.get('success_rate', 0.0)
        )
        new_employee.set_password(temp_password)
        
        db.session.add(new_employee)
        db.session.commit()
        
        # Send email with credentials
        send_credentials_email(
            email=data['email'],
            emp_id=emp_id,
            temp_password=temp_password
        )
        
        return jsonify({
            'success': True,
            'message': 'HR employee created successfully',
            'employee': new_employee.to_dict()
        }), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500
    

@app.route('/api/employees', methods=['GET'])
def get_all_employees():
    if not authenticate_request():
        return jsonify({'error': 'Unauthorized'}), 401
    
    employees = Employee.query.all()
    return jsonify([employee.to_dict() for employee in employees])

@app.route('/api/employees/<emp_id>', methods=['GET'])
def get_employee(emp_id):
    if not authenticate_request():
        return jsonify({'error': 'Unauthorized'}), 401
    
    employee = Employee.query.get(emp_id)
    if not employee:
        return jsonify({'error': 'Employee not found'}), 404
    
    return jsonify(employee.to_dict())

        
@app.route('/api/employees/<emp_id>', methods=['PUT'])
def update_employee(emp_id):
    if not authenticate_request():
        return jsonify({'error': 'Unauthorized'}), 401
    
    employee = Employee.query.get(emp_id)
    if not employee:
        return jsonify({'error': 'Employee not found'}), 404
    
    data = request.json
    
    # Update fields
    if 'name' in data:
        employee.name = data['name']
    if 'email' in data:
        # Check if email already exists on another account
        existing_email = Employee.query.filter_by(email=data['email']).first()
        if existing_email and existing_email.emp_id != emp_id:
            return jsonify({'error': 'Email already exists'}), 409
        employee.email = data['email']
    if 'role' in data:
        employee.role = data['role']
    if 'is_first_login' in data:
        employee.is_first_login = data['is_first_login']
    if 'password' in data:
        employee.set_password(data['password'])
    if 'skills' in data:
        employee.skills = data['skills']
    if 'experience' in data:
        employee.experience = data['experience']
    if 'tasks_completed' in data:
        employee.tasks_completed = data['tasks_completed']
    if 'success_rate' in data:
        employee.success_rate = data['success_rate']
    
    try:
        db.session.commit()
        return jsonify(employee.to_dict()), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/api/employees/<emp_id>/metrics', methods=['PUT'])
def update_metrics(emp_id):
    """Update only the metrics of an employee"""
    if not authenticate_request():
        return jsonify({'error': 'Unauthorized'}), 401
    
    employee = Employee.query.get(emp_id)
    if not employee:
        return jsonify({'error': 'Employee not found'}), 404
    
    data = request.json
    
    # Update only metrics fields
    if 'tasks_completed' in data:
        employee.tasks_completed = data['tasks_completed']
    if 'success_rate' in data:
        employee.success_rate = data['success_rate']
    
    try:
        db.session.commit()
        return jsonify({'success': True, 'employee': employee.to_dict()}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/api/employees/<emp_id>', methods=['DELETE'])
def delete_employee(emp_id):
    if not authenticate_request():
        return jsonify({'error': 'Unauthorized'}), 401
    
    employee = Employee.query.get(emp_id)
    if not employee:
        return jsonify({'error': 'Employee not found'}), 404
    
    try:
        db.session.delete(employee)
        db.session.commit()
        return jsonify({'message': f'Employee {emp_id} deleted successfully'}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/api/authenticate', methods=['POST'])
def authenticate_employee():
    print(f"Request received at /api/authenticate")
    
    if not authenticate_request():
        print("API key authentication failed")
        return jsonify({'error': 'Unauthorized', 'message': 'API key authentication failed'}), 401
    
    print("API key authentication successful")
    
    try:
        data = request.json
        print(f"Received authentication data: {data}")
        
        # Validate required fields
        if not all(k in data for k in ['emp_id', 'password']):
            print("Missing required fields")
            return jsonify({'error': 'Missing required fields'}), 400
        
        employee = Employee.query.get(data['emp_id'])
        print(f"Found employee: {employee}")
        
        if not employee:
            print("Employee not found")
            return jsonify({'authenticated': False, 'reason': 'Employee not found'}), 200
        
        password_check = employee.check_password(data['password'])
        print(f"Password check result: {password_check}")
        
        if not password_check:
            print("Invalid password")
            return jsonify({'authenticated': False, 'reason': 'Invalid password'}), 200
        
        # Update last login time
        employee.last_login = datetime.utcnow()
        db.session.commit()
        
        result = {
            'authenticated': True,
            'employee': employee.to_dict()
        }
        print(f"Authentication successful: {result}")
        
        return jsonify(result)
    except Exception as e:
        print(f"Exception during authentication: {str(e)}")
        return jsonify({'error': 'Server error', 'message': str(e)}), 500
    
@app.route('/api/change_password', methods=['POST'])
def change_password():
    if not authenticate_request():
        return jsonify({'error': 'Unauthorized'}), 401
    
    data = request.json
    
    # Validate required fields
    if not all(k in data for k in ['emp_id', 'current_password', 'new_password']):
        return jsonify({'error': 'Missing required fields'}), 400
    
    employee = Employee.query.get(data['emp_id'])
    
    if not employee or not employee.check_password(data['current_password']):
        return jsonify({'success': False, 'error': 'Invalid credentials'}), 401
    
    employee.set_password(data['new_password'])
    employee.is_first_login = False
    
    try:
        db.session.commit()
        return jsonify({'success': True, 'employee': employee.to_dict()})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500

# Command to create tables
@app.cli.command("create_tables")
def create_tables():
    db.create_all()
    print("Tables created!")

# Command to create admin user
@app.cli.command("create_admin")
def create_admin():
    admin = Employee.query.filter_by(emp_id='admin').first()
    if not admin:
        temp_password = generate_random_password()
        admin = Employee(
            emp_id='admin', 
            name='Administrator',
            email='admin@company.com',
            role='admin'
        )
        admin.set_password(temp_password)
        db.session.add(admin)
        db.session.commit()
        print(f"Admin created with temporary password: {temp_password}")
    else:
        print("Admin already exists")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)