# salary_service_optimized.py
import os
import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import time
import json
from datetime import datetime, timedelta
import threading
import logging
from dotenv import load_dotenv
import random
from collections import deque

# Load environment variables
load_dotenv()
retraining_progress = {}
progress_lock = threading.Lock()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

class AdaptiveRateLimiter:
    """Advanced rate limiter that adapts to API behavior"""
    
    def __init__(self, initial_delay=2.0, max_delay=60.0, backoff_factor=1.5):
        self.current_delay = initial_delay
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.success_count = 0
        self.failure_count = 0
        self.last_request_time = None
        self.request_times = deque(maxlen=100)  # Track recent request times
        
    def wait_before_request(self):
        """Wait appropriate time before making next request"""
        if self.last_request_time:
            elapsed = time.time() - self.last_request_time
            wait_time = max(0, self.current_delay - elapsed)
            if wait_time > 0:
                logger.debug(f"Rate limiter waiting {wait_time:.2f}s")
                time.sleep(wait_time)
        
        self.last_request_time = time.time()
        self.request_times.append(self.last_request_time)
    
    def on_success(self):
        """Called when request succeeds"""
        self.success_count += 1
        self.failure_count = 0  # Reset failure count on success
        
        # Gradually reduce delay if we're having consistent success
        if self.success_count >= 5:
            self.current_delay = max(
                self.initial_delay,
                self.current_delay * 0.9  # Reduce by 10%
            )
            self.success_count = 0
            logger.debug(f"Rate limiter delay reduced to {self.current_delay:.2f}s")
    
    def on_rate_limit(self):
        """Called when rate limit is hit"""
        self.failure_count += 1
        self.success_count = 0  # Reset success count on failure
        
        # Increase delay significantly
        old_delay = self.current_delay
        self.current_delay = min(
            self.max_delay,
            self.current_delay * self.backoff_factor
        )
        
        logger.warning(f"Rate limit hit. Delay increased from {old_delay:.2f}s to {self.current_delay:.2f}s")
        
        # Add immediate cooldown period
        cooldown = min(30, self.current_delay * 2)
        logger.info(f"Cooling down for {cooldown}s due to rate limit")
        time.sleep(cooldown)
    
    def on_error(self):
        """Called when other errors occur"""
        self.failure_count += 1
        # Slight increase in delay for general errors
        self.current_delay = min(
            self.max_delay,
            self.current_delay * 1.2
        )
    
    def get_stats(self):
        """Get rate limiter statistics"""
        requests_per_minute = 0
        if len(self.request_times) > 1:
            time_span = self.request_times[-1] - self.request_times[0]
            if time_span > 0:
                requests_per_minute = len(self.request_times) / (time_span / 60)
        
        return {
            'current_delay': self.current_delay,
            'success_count': self.success_count,
            'failure_count': self.failure_count,
            'requests_per_minute': requests_per_minute,
            'total_requests': len(self.request_times)
        }

class SalaryDataCollector:
    """Enhanced Data Collection with Adaptive Rate Limiting and Caching"""
    
    def __init__(self, api_base_url=None, api_key=None, host=None, session_id=None):
        self.api_base_url = api_base_url or os.getenv("SALARY_API_URL")
        self.api_key = api_key or os.getenv("SALARY_API_KEY")
        self.host = host or os.getenv("HOST")
        self.quality_threshold = {
            "min_sample_size": 2,  # Reduced threshold for more data
            "required_confidence": ["MEDIUM", "HIGH", "VERY_HIGH", "LOW"]
        }
        self.session_id = session_id
        self.rate_limiter = AdaptiveRateLimiter(initial_delay=3.0, max_delay=45.0)
        self.cache = {}  # Simple in-memory cache
        self.cache_duration = timedelta(hours=6)  # Cache for 6 hours
        
    def update_progress(self, percentage, stage, step_id=None):
        """Update progress if session_id is provided"""
        if self.session_id:
            set_retraining_progress(self.session_id, percentage, stage, step_id)
            logger.info(f"Progress: {percentage}% - {stage}")
    
    def _get_cache_key(self, job_title, location, experience_level):
        """Generate cache key for request"""
        return f"{job_title.lower()}_{location.lower()}_{experience_level.upper()}"
    
    def _is_cache_valid(self, cache_entry):
        """Check if cache entry is still valid"""
        if not cache_entry:
            return False
        return datetime.now() - cache_entry['timestamp'] < self.cache_duration
    
    def make_api_call(self, job_title, location, experience_level, location_type="CITY"):
        """Enhanced API call with adaptive rate limiting and caching"""
        
        # Check cache first
        cache_key = self._get_cache_key(job_title, location, experience_level)
        if cache_key in self.cache and self._is_cache_valid(self.cache[cache_key]):
            logger.debug(f"Using cached data for {cache_key}")
            return self.cache[cache_key]['data']
        
        # Wait before making request
        self.rate_limiter.wait_before_request()
        
        try:
            params = {
                "job_title": job_title,
                "location": location,
                "location_type": location_type,
                "years_of_experience": experience_level
            }
            
            headers = {
                "x-rapidapi-key": self.api_key,
                "x-rapidapi-host": self.host
            }
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            response = requests.get(
                self.api_base_url, 
                params=params, 
                headers=headers,
                timeout=15  # Add timeout
            )
            
            if response.status_code == 429:
                self.rate_limiter.on_rate_limit()
                return None
            
            response.raise_for_status()
            data = response.json()
            
            # Cache successful response
            self.cache[cache_key] = {
                'data': data,
                'timestamp': datetime.now()
            }
            
            self.rate_limiter.on_success()
            return data
            
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:
                self.rate_limiter.on_rate_limit()
            else:
                self.rate_limiter.on_error()
                logger.error(f"HTTP error {response.status_code} for {job_title}, {location}: {e}")
            return None
            
        except requests.exceptions.RequestException as e:
            self.rate_limiter.on_error()
            logger.error(f"API call failed for {job_title}, {location}: {e}")
            return None
    
    def assess_data_quality(self, api_response):
        """Enhanced data quality assessment with more flexible thresholds"""
        if not api_response or api_response.get("status") != "OK":
            return False, "API call failed"
        
        data = api_response.get("data", [])
        if not data:
            return False, "No data returned"
        
        sample_data = data[0]
        sample_size = sample_data.get("salary_count", 0)
        confidence = sample_data.get("confidence", "LOW")
        
        # More flexible quality assessment
        has_salary_data = any([
            sample_data.get("median_salary"),
            sample_data.get("min_salary"),
            sample_data.get("max_salary")
        ])
        
        quality_score = {
            "sample_size": sample_size,
            "confidence": confidence,
            "has_salary_data": has_salary_data,
            "meets_threshold": (
                sample_size >= self.quality_threshold["min_sample_size"] and 
                confidence in self.quality_threshold["required_confidence"] and
                has_salary_data
            ) or (
                # Accept even low confidence if sample size is decent
                sample_size >= 10 and has_salary_data
            )
        }
        
        return quality_score["meets_threshold"], quality_score
    
    def extract_data_point(self, api_response, job_title, location, experience):
        """Enhanced data extraction with fallback values"""
        try:
            data = api_response["data"][0]
            
            # Handle missing median by calculating from min/max
            median_salary = data.get("median_salary")
            if not median_salary:
                min_sal = data.get("min_salary")
                max_sal = data.get("max_salary")
                if min_sal and max_sal:
                    median_salary = (min_sal + max_sal) / 2
            
            # Only include if we have some salary data
            if not median_salary:
                return None
            
            return {
                "job_title": job_title,
                "location": location,  
                "experience_level": experience,
                "min_salary": data.get("min_salary") or median_salary * 0.8,
                "max_salary": data.get("max_salary") or median_salary * 1.2,
                "median_salary": median_salary,
                "min_base_salary": data.get("min_base_salary"),
                "max_base_salary": data.get("max_base_salary"), 
                "median_base_salary": data.get("median_base_salary"),
                "salary_period": data.get("salary_period", "MONTH"),
                "salary_currency": data.get("salary_currency", "ZAR"),
                "salary_count": data.get("salary_count", 1),
                "confidence": data.get("confidence", "LOW"),
                "collected_at": datetime.now().isoformat()
            }
        except (KeyError, IndexError) as e:
            logger.error(f"Error extracting data point: {e}")
            return None
    
    def collect_strategic_data(self, max_failures=15, target_data_points=20):
        """Optimized data collection with dynamic stopping criteria"""
        
        # Prioritized combinations - start with most likely to succeed
        priority_combinations = [
            # High-demand tech roles in major cities
            ("data scientist", "cape town", "LESS_THAN_ONE"),
            ("data scientist", "johannesburg", "LESS_THAN_ONE"),
            ("data scientist", "cape town", "ONE_TO_THREE"),
            ("data scientist", "johannesburg", "ONE_TO_THREE"),
            ("data scientist", "cape town", "FOUR_TO_SIX"),
            ("data scientist", "johannesburg", "FOUR_TO_SIX"),
            ("data analyst", "cape town", "LESS_THAN_ONE"),
            ("data scientist", "cape town", "SEVEN_TO_NINE"),
            ("data scientist", "johannesburg", "SEVEN_TO_NINE"),

            # Data analyst roles
            ("data analyst", "cape town", "LESS_THAN_ONE"),
            ("data analyst", "johannesburg", "LESS_THAN_ONE"),
            ("data analyst", "cape town", "ONE_TO_THREE"),
            ("data analyst", "johannesburg", "ONE_TO_THREE"),
            ("data analyst", "cape town", "FOUR_TO_SIX"),
            ("data analyst", "johannesburg", "FOUR_TO_SIX"),
            ("data analyst", "cape town", "SEVEN_TO_NINE"),
            ("data analyst", "johannesburg", "SEVEN_TO_NINE"),

            # Software roles
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
            
            # Other tech roles
            ("devops engineer", "cape town", "FOUR_TO_SIX"),
            ("machine learning engineer", "johannesburg", "FOUR_TO_SIX"),
            ("product manager", "cape town", "FOUR_TO_SIX"),
            ("ui/ux designer", "cape town", "ONE_TO_THREE"),
        ]
        
        collected_data = []
        quality_report = {}
        failed_requests = 0
        consecutive_failures = 0
        max_consecutive_failures = 5
        
        total_combinations = len(priority_combinations)
        self.update_progress(5, 'Starting optimized data collection...', 'step-collect')
        
        logger.info(f"Starting optimized data collection for up to {total_combinations} combinations")
        logger.info(f"Target: {target_data_points} data points, Max failures: {max_failures}")
        
        for i, (job_title, location, experience) in enumerate(priority_combinations):
            # Calculate progress (5% to 85% range for data collection)
            base_progress = 5
            collection_progress_range = 80
            current_progress = base_progress + int((i / total_combinations) * collection_progress_range)
            
            # Check stopping criteria
            if len(collected_data) >= target_data_points:
                self.update_progress(
                    85,
                    f'Target reached: {len(collected_data)} high-quality data points collected',
                    'step-collect'
                )
                logger.info(f"Target of {target_data_points} data points reached. Stopping collection.")
                break
            
            if failed_requests >= max_failures:
                self.update_progress(
                    current_progress,
                    f'Max failures reached ({failed_requests}). Stopping with {len(collected_data)} data points',
                    'step-collect'
                )
                logger.warning(f"Max failures ({max_failures}) reached. Stopping collection.")
                break
                
            if consecutive_failures >= max_consecutive_failures:
                self.update_progress(
                    current_progress,
                    f'Too many consecutive failures ({consecutive_failures}). Taking extended break...',
                    'step-collect'
                )
                logger.warning(f"Too many consecutive failures. Taking 60s break...")
                time.sleep(60)
                consecutive_failures = 0
            
            self.update_progress(
                current_progress,
                f'Collecting {i+1}/{total_combinations}: {job_title} in {location} ({len(collected_data)} collected)',
                'step-collect'
            )
            
            logger.info(f"Collecting {i+1}/{total_combinations}: {job_title} in {location}")
            
            # Make API call
            api_response = self.make_api_call(job_title, location, experience)
            
            if api_response:
                consecutive_failures = 0  # Reset consecutive failure counter
                
                # Assess quality with more flexible criteria
                is_quality, quality_info = self.assess_data_quality(api_response)
                quality_report[(job_title, location, experience)] = quality_info
                
                if is_quality:
                    data_point = self.extract_data_point(api_response, job_title, location, experience)
                    if data_point:
                        collected_data.append(data_point)
                        logger.info(f"✓ Quality data collected: {quality_info['sample_size']} samples, {quality_info['confidence']} confidence")
                    else:
                        logger.warning("Failed to extract data point")
                        failed_requests += 1
                else:
                    logger.warning(f"✗ Low quality data: {quality_info}")
                    failed_requests += 1
            else:
                failed_requests += 1
                consecutive_failures += 1
                logger.warning(f"Failed to collect data (failure {failed_requests}/{max_failures}, consecutive: {consecutive_failures})")
            
            # Show rate limiter stats periodically
            if (i + 1) % 5 == 0:
                stats = self.rate_limiter.get_stats()
                logger.info(f"Rate limiter stats: {stats['requests_per_minute']:.1f} req/min, delay: {stats['current_delay']:.1f}s")
        
        # Final progress update
        self.update_progress(
            85,
            f'Data collection complete: {len(collected_data)} high-quality data points',
            'step-collect'
        )
        
        logger.info(f"Data collection complete: {len(collected_data)} data points collected")
        logger.info(f"Total failures: {failed_requests}, Success rate: {((len(priority_combinations) - failed_requests) / len(priority_combinations) * 100):.1f}%")
        
        # Log final rate limiter stats
        final_stats = self.rate_limiter.get_stats()
        logger.info(f"Final rate limiter stats: {final_stats}")
        
        return collected_data, quality_report

class SalaryPredictor:
    """Enhanced ML Model with better fallback handling"""
    
    def __init__(self, model_path="salary_model_v2.joblib"):
        self.model = None
        self.label_encoders = {}
        self.is_trained = False
        self.feature_columns = ['job_title', 'location', 'experience_level']
        self.model_path = model_path
        # self.fallback_data = {}  # Store averages for fallback predictions
        
        # Try to load existing model on initialization
        self._try_load_model()
    
    def _try_load_model(self):
        """Try to load an existing model on initialization"""
        if os.path.exists(self.model_path):
            try:
                self.load_model(self.model_path)
                logger.info("Existing salary model loaded successfully")
                return True
            except Exception as e:
                logger.warning(f"Failed to load existing model: {e}")
                return False
        return False
    
    # def _calculate_fallback_data(self, df):
    #     """Calculate fallback averages for unseen categories"""
    #     self.fallback_data = {
    #         'overall_median': df['annual_median_salary'].median(),
    #         'job_medians': df.groupby('job_title')['annual_median_salary'].median().to_dict(),
    #         'location_medians': df.groupby('location')['annual_median_salary'].median().to_dict(),
    #         'experience_medians': df.groupby('experience_level')['annual_median_salary'].median().to_dict()
    #     }
    #     logger.info(f"Calculated fallback data with overall median: {self.fallback_data['overall_median']:,.0f}")
    
    def prepare_features(self, df):
        """Enhanced feature encoding with better handling of unseen categories"""
        df_encoded = df.copy()
        
        for column in self.feature_columns:
            if column not in self.label_encoders:
                self.label_encoders[column] = LabelEncoder()
                df_encoded[f'{column}_encoded'] = self.label_encoders[column].fit_transform(df[column])
            else:
                # Handle unseen categories during prediction
                le = self.label_encoders[column]
                mask = df[column].isin(le.classes_)
                df_encoded[f'{column}_encoded'] = -1  # Default for unseen categories
                df_encoded.loc[mask, f'{column}_encoded'] = le.transform(df.loc[mask, column])
        
        return df_encoded
    
    def train_model(self, training_data):
        """Enhanced model training with better validation"""
        df = pd.DataFrame(training_data)
        
        if len(df) < 5:
            raise ValueError(f"Insufficient training data. Need at least 5 data points, got {len(df)}.")
        
        # Enhanced data cleaning
        original_len = len(df)
        df = df.dropna(subset=['median_salary'])
        df = df[df['median_salary'] > 0]
        
        if len(df) < 3:
            raise ValueError("Insufficient valid training data after cleaning.")
        
        logger.info(f"Data cleaning: {original_len} -> {len(df)} samples")
        
        # Normalize salary period (convert monthly to annual)
        df['annual_median_salary'] = df.apply(lambda row: 
            row['median_salary'] * 12 if row['salary_period'] == 'MONTH' 
            else row['median_salary'], axis=1)
        
        # Enhanced outlier removal with IQR method
        Q1 = df['annual_median_salary'].quantile(0.25)
        Q3 = df['annual_median_salary'].quantile(0.75)
        IQR = Q3 - Q1
        
        # More conservative outlier removal for small datasets
        multiplier = 2.0 if len(df) < 20 else 1.5
        lower_bound = max(50000, Q1 - multiplier * IQR)  # Minimum reasonable salary
        upper_bound = min(5000000, Q3 + multiplier * IQR)  # Maximum reasonable salary
        
        before_outlier_removal = len(df)
        df = df[(df['annual_median_salary'] >= lower_bound) & 
                (df['annual_median_salary'] <= upper_bound)]
        
        logger.info(f"Outlier removal: {before_outlier_removal} -> {len(df)} samples")
        logger.info(f"Salary range: {df['annual_median_salary'].min():,.0f} - {df['annual_median_salary'].max():,.0f}")
        
        if len(df) < 3:
            raise ValueError("Insufficient training data after outlier removal.")
        
        # Calculate fallback data before encoding
        # self._calculate_fallback_data(df)
        
        # Prepare features
        df_encoded = self.prepare_features(df)
        
        # Features and target
        feature_cols = [f'{col}_encoded' for col in self.feature_columns]
        X = df_encoded[feature_cols]
        y = df_encoded['annual_median_salary']
        
        # Dynamic test size based on dataset size
        if len(df) < 8:
            # For very small datasets, don't split - use all data for training
            X_train, y_train = X, y
            X_test, y_test = X, y  # Use training data for evaluation
            logger.warning(f"Very small dataset ({len(df)} samples): Using all data for training and evaluation")
        elif len(df) < 15:
            test_size = 0.1
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            logger.info(f"Small dataset ({len(df)} samples): {test_size*100}% for testing")
        else:
            test_size = 0.2
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            logger.info(f"Standard split ({len(df)} samples): {test_size*100}% for testing")
        
        logger.info(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        # Enhanced model with better parameters for small datasets
        n_estimators = min(100, max(10, len(df) * 3))  # Scale with dataset size
        max_depth = min(10, max(3, int(np.log2(len(df)) + 1)))  # Prevent overfitting
        
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=42,
            max_depth=max_depth,
            min_samples_split=max(2, len(df) // 10),
            min_samples_leaf=max(1, len(df) // 20),
            bootstrap=True
        )
        
        logger.info(f"Model params: n_estimators={n_estimators}, max_depth={max_depth}")
        
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        
        evaluation = {
            "mae": mean_absolute_error(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "r2": r2_score(y_test, y_pred),
            "training_samples": len(df),
            "test_samples": len(y_test),
            "feature_importance": dict(zip(feature_cols, self.model.feature_importances_))
        }
        
        self.is_trained = True
        logger.info(f"Model trained successfully: R² = {evaluation['r2']:.3f}, MAE = {evaluation['mae']:,.0f}")
        
        # Save the enhanced model
        self.save_model(self.model_path)
        
        return evaluation
    
    # def _fallback_prediction(self, job_title, location, experience_level):
    #     """Enhanced fallback prediction using stored averages"""
    #     if not self.fallback_data:
    #         # Ultra-basic fallback if no data available
    #         base_salary = 400000  # Conservative base salary for South Africa
    #         experience_multipliers = {
    #             'LESS_THAN_ONE': 0.8,
    #             'ONE_TO_THREE': 1.0,
    #             'FOUR_TO_SIX': 1.3,
    #             'SEVEN_TO_NINE': 1.6,
    #             'TEN_PLUS': 2.0
    #         }
    #         multiplier = experience_multipliers.get(experience_level.upper(), 1.0)
    #         prediction = base_salary * multiplier
    #     else:
    #         # Use calculated fallback data
    #         job_key = job_title.lower().strip()
    #         location_key = location.lower().strip()
    #         exp_key = experience_level.upper().strip()
            
    #         # Try to find most relevant fallback
    #         prediction = self.fallback_data['overall_median']
            
    #         # Adjust based on available data
    #         if job_key in self.fallback_data['job_medians']:
    #             prediction = self.fallback_data['job_medians'][job_key]
    #         elif location_key in self.fallback_data['location_medians']:
    #             prediction = self.fallback_data['location_medians'][location_key]
    #         elif exp_key in self.fallback_data['experience_medians']:
    #             prediction = self.fallback_data['experience_medians'][exp_key]
        
    #     prediction_range = (prediction * 0.7, prediction * 1.3)
        
    #     return {
    #         "predicted_annual_salary": round(prediction),
    #         "predicted_range": (round(prediction_range[0]), round(prediction_range[1])),
    #         "market_insights": f"Fallback prediction for {job_title} in {location}. Based on similar roles and market averages.",
    #         "source": "FALLBACK_MODEL",
    #         "confidence": "LOW"
    #     }
    
    def predict_salary(self, job_title, location, experience_level):
        """Enhanced prediction with better error handling"""
        if not self.is_trained:
            if not self._try_load_model():
                return self._fallback_prediction(job_title, location, experience_level)
        
        # Create input DataFrame
        input_data = pd.DataFrame([{
            'job_title': job_title.lower().strip(),
            'location': location.lower().strip(),
            'experience_level': experience_level.upper().strip()
        }])
        
        try:
            # Check if we have encoders for all inputs
            for column in self.feature_columns:
                if column not in self.label_encoders:
                    logger.warning(f"No encoder for feature: {column}")
                    return self._fallback_prediction(job_title, location, experience_level)
            
            # Encode features
            input_encoded = self.prepare_features(input_data)
            feature_cols = [f'{col}_encoded' for col in self.feature_columns]
            X_input = input_encoded[feature_cols]
            
            # Check for unseen categories (encoded as -1)
            if (X_input == -1).any().any():
                logger.warning("Unseen category detected, using enhanced fallback prediction")
                return self._fallback_prediction(job_title, location, experience_level)
            
            # Make prediction
            prediction = self.model.predict(X_input)[0]
            
            # Get prediction confidence based on model performance
            confidence = "MEDIUM"  # Default confidence
            
            # Estimate range with confidence intervals
            prediction_range = (prediction * 0.85, prediction * 1.15)
            
            # Add market insights
            insights = self._generate_market_insights(job_title, location, experience_level, prediction)
            
            return {
                "predicted_annual_salary": round(prediction),
                "predicted_range": (round(prediction_range[0]), round(prediction_range[1])),
                "market_insights": insights,
                "source": "ML_MODEL",
                "confidence": confidence
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return self._fallback_prediction(job_title, location, experience_level)
    
    def _generate_market_insights(self, job_title, location, experience_level, prediction):
        """Enhanced market insights generation"""
        insights = []
        
        # Experience-based insights
        exp_insights = {
            "LESS_THAN_ONE": "Entry-level position with strong growth potential",
            "ONE_TO_THREE": "Junior to mid-level role with competitive market demand",
            "FOUR_TO_SIX": "Mid-level role with solid market positioning",
            "SEVEN_TO_NINE": "Senior-level position commanding premium salaries",
            "TEN_PLUS": "Expert-level role with top-tier compensation"
        }
        
        if experience_level in exp_insights:
            insights.append(exp_insights[experience_level])
        
        # Location-based insights
        if "cape town" in location.lower():
            insights.append("Cape Town market offers competitive tech salaries with lifestyle benefits")
        elif "johannesburg" in location.lower():
            insights.append("Johannesburg commands premium for financial and tech sectors")
        elif "durban" in location.lower():
            insights.append("Durban market growing rapidly in tech sector")
        
        # Job title insights
        if any(term in job_title.lower() for term in ["senior", "lead", "principal"]):
            insights.append("Leadership role with elevated compensation expectations")
        elif any(term in job_title.lower() for term in ["data scientist", "machine learning"]):
            insights.append("High-demand AI/ML skill set commands premium pricing")
        elif "full stack" in job_title.lower():
            insights.append("Versatile skill set valued across multiple industries")
        
        # Salary range insights
        if prediction > 1200000:
            insights.append("Above-average salary range for this combination")
        elif prediction < 300000:
            insights.append("Entry-level salary range with growth opportunities")
        elif prediction > 800000:
            insights.append("Competitive mid-to-senior level compensation")
        
        return ". ".join(insights) if insights else f"Market analysis for {job_title} in {location}"
    
    def save_model(self, filepath):
        """Enhanced model saving with metadata"""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        model_data = {
            "model": self.model,
            "label_encoders": self.label_encoders,
            "feature_columns": self.feature_columns,
            # "fallback_data": self.fallback_data,
            "version": "2.0",
            "trained_at": datetime.now().isoformat(),
            "model_params": self.model.get_params() if self.model else None
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Enhanced model saved to {filepath}")
    
    def load_model(self, filepath):
        """Enhanced model loading with backward compatibility"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
            
        try:
            model_data = joblib.load(filepath)
            
            self.model = model_data["model"]
            self.label_encoders = model_data["label_encoders"]
            self.feature_columns = model_data["feature_columns"]
            # self.fallback_data = model_data.get("fallback_data", {})
            self.is_trained = True
            
            logger.info(f"Model loaded from {filepath}")
            
            # Log model info if available
            if "trained_at" in model_data:
                logger.info(f"Model was trained at: {model_data['trained_at']}")
            if "version" in model_data:
                logger.info(f"Model version: {model_data['version']}")
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model from {filepath}: {e}")
            raise

class CompetitivenessAnalyzer:
    """Enhanced Competitiveness Analysis with market context"""
    
    @staticmethod
    def analyze_competitiveness(predicted_salary, predicted_range, user_budget, confidence="MEDIUM"):
        """Enhanced competitiveness analysis with detailed recommendations"""
        
        min_market, max_market = predicted_range
        pct_vs_median = (user_budget - predicted_salary) / predicted_salary * 100
        
        # Determine competitiveness level with more nuanced categories
        if user_budget > max_market * 1.1:
            level = "highly_competitive"
            flag = f"Budget is {abs(pct_vs_median):.1f}% above market - Highly Competitive"
            color = "green"
            recommendation = "Excellent positioning to attract top talent. Consider emphasizing total compensation package."
            market_position = "Well above market rate"
            
        elif user_budget > predicted_salary * 1.05:
            level = "competitive"
            flag = f"Budget is {abs(pct_vs_median):.1f}% above median - Competitive"
            color = "blue"
            recommendation = "Good positioning for quality candidates. Highlight career growth opportunities."
            market_position = "Above market median"
            
        elif user_budget >= min_market:
            level = "somewhat_competitive"
            flag = f"Budget is {abs(pct_vs_median):.1f}% below median but within market range"
            color = "yellow"
            recommendation = "May attract motivated candidates. Consider non-monetary benefits like remote work or flexible hours."
            market_position = "Within market range"
            
        elif user_budget >= min_market * 0.9:
            level = "below_market"
            flag = f"Budget is {abs(pct_vs_median):.1f}% below market - Below Market"
            color = "orange"
            recommendation = "Consider increasing budget or targeting junior candidates. Emphasize learning opportunities."
            market_position = "Below market range"
            
        else:
            level = "not_competitive"
            flag = f"Budget is {abs(pct_vs_median):.1f}% below market - Not Competitive"
            color = "red"
            recommendation = "Strongly consider increasing budget or significantly adjusting job requirements."
            market_position = "Well below market"
        
        # Additional insights based on confidence level
        confidence_notes = {
            "HIGH": "High confidence in market data",
            "MEDIUM": "Moderate confidence - consider additional market research",
            "LOW": "Limited market data available - recommendations are preliminary"
        }
        
        return {
            "level": level,
            "flag": flag,
            "color": color,
            "pct_vs_median": round(pct_vs_median, 1),
            "recommendation": recommendation,
            "market_position": {
                "user_budget": user_budget,
                "market_median": predicted_salary,
                "market_range": predicted_range,
                "within_range": min_market <= user_budget <= max_market,
                "position_description": market_position
            },
            "confidence": confidence,
            "confidence_note": confidence_notes.get(confidence, "Unknown confidence level"),
            "additional_insights": CompetitivenessAnalyzer._generate_additional_insights(
                user_budget, predicted_salary, predicted_range, confidence
            )
        }
    
    @staticmethod
    def _generate_additional_insights(user_budget, predicted_salary, predicted_range, confidence):
        """Generate additional market insights"""
        insights = []
        
        min_market, max_market = predicted_range
        budget_position = (user_budget - min_market) / (max_market - min_market) if max_market != min_market else 0.5
        
        # Position-based insights
        if budget_position > 0.8:
            insights.append("Budget positions you in the top 20% of the market range")
        elif budget_position > 0.5:
            insights.append("Budget is in the upper half of the typical market range")
        elif budget_position > 0.2:
            insights.append("Budget is in the lower half of the typical market range")
        else:
            insights.append("Budget is below the typical market range")
        
        # Confidence-based insights
        if confidence == "LOW":
            insights.append("Consider validating with additional salary surveys or market research")
        elif confidence == "HIGH":
            insights.append("Market data is well-supported and reliable")
        
        # Market volatility insights
        range_spread = (max_market - min_market) / predicted_salary
        if range_spread > 0.4:
            insights.append("Large salary range indicates high market variability - experience and skills heavily influence compensation")
        else:
            insights.append("Narrow salary range indicates stable market pricing")
        
        return insights

# Global instances
salary_collector = None
salary_predictor = None
competitiveness_analyzer = CompetitivenessAnalyzer()

def initialize_salary_predictor_with_progress(session_id, force_retrain=False):
    """Initialize the salary predictor with enhanced progress tracking"""
    global salary_collector, salary_predictor
    
    try:
        set_retraining_progress(session_id, 2, 'Initializing enhanced salary predictor...', 'step-init')
        
        # Initialize instances with session_id for progress tracking
        salary_collector = SalaryDataCollector(session_id=session_id)
        salary_predictor = SalaryPredictor()
        
        set_retraining_progress(session_id, 5, 'Initialized enhanced data collector and predictor', 'step-init')
        
        # If force_retrain is False, check if model is already loaded
        if not force_retrain and salary_predictor.is_trained:
            set_retraining_progress(session_id, 100, 'Enhanced salary predictor model already loaded and ready', 'step-save')
            logger.info("Enhanced salary predictor model already loaded and ready")
            return True
        
        # If force_retrain is False, try to load existing model first
        if not force_retrain and os.path.exists(salary_predictor.model_path):
            try:
                salary_predictor.load_model(salary_predictor.model_path)
                set_retraining_progress(session_id, 100, 'Existing enhanced model loaded successfully', 'step-save')
                logger.info("Enhanced salary predictor model loaded successfully from file")
                return True
            except Exception as e:
                logger.warning(f"Failed to load existing model: {e}")
                set_retraining_progress(session_id, 8, 'Failed to load existing model, collecting fresh data with enhanced methods...', 'step-collect')
        
        # Collect fresh data and train model with enhanced methods
        if force_retrain:
            set_retraining_progress(session_id, 8, 'Force retrain requested. Collecting fresh data with adaptive rate limiting...', 'step-collect')
            logger.info("Force retrain requested. Using enhanced data collection with adaptive rate limiting...")
        else:
            set_retraining_progress(session_id, 8, 'No existing model found. Starting enhanced data collection...', 'step-collect')
            logger.info("No existing model found. Using enhanced data collection to train new model...")
            
        # Enhanced data collection with progress tracking
        training_data, quality_report = salary_collector.collect_strategic_data(
            max_failures=20,  # More lenient failure threshold
            target_data_points=15  # Lower target for faster completion
        )
        
        if len(training_data) >= 5:  # Lowered minimum requirement
            set_retraining_progress(session_id, 87, f'Enhanced data collection complete with {len(training_data)} points. Preparing for model training...', 'step-quality')
            time.sleep(1)
            
            set_retraining_progress(session_id, 90, 'Starting enhanced model training...', 'step-train')
            evaluation = salary_predictor.train_model(training_data)
            
            set_retraining_progress(session_id, 95, 'Enhanced model training complete. Saving model...', 'step-save')
            time.sleep(1)
            
            action = "retrained" if force_retrain else "trained"
            set_retraining_progress(session_id, 100, f'Enhanced model {action} successfully! R² = {evaluation["r2"]:.3f}', 'step-save')
            logger.info(f"Enhanced salary predictor {action} successfully: R² = {evaluation['r2']:.3f}, MAE = {evaluation['mae']:,.0f}")
            return True
        else:
            fallback_msg = f'Limited training data ({len(training_data)} points) - model will use enhanced fallback predictions'
            set_retraining_progress(session_id, 100, fallback_msg, 'step-save')
            logger.warning(f"Limited training data collected ({len(training_data)} points), but enhanced fallback system available")
            return True  # Still return True since fallback system can handle predictions
            
    except Exception as e:
        error_msg = f"Failed to initialize enhanced salary predictor: {e}"
        set_retraining_progress(session_id, 0, 'Enhanced initialization failed', None, error_msg)
        logger.error(error_msg)
        return False

def retrain_salary_model_with_progress(session_id):
    """Force retrain the salary prediction model with enhanced progress tracking"""
    logger.info("Starting forced model retraining with enhanced adaptive rate limiting...")
    return initialize_salary_predictor_with_progress(session_id, force_retrain=True)

def initialize_salary_predictor(force_retrain=False):
    """Initialize the enhanced salary predictor by training or loading the model"""
    global salary_collector, salary_predictor
    
    try:
        # Initialize enhanced instances
        salary_collector = SalaryDataCollector()
        salary_predictor = SalaryPredictor()
        
        # If force_retrain is False, check if model is already loaded
        if not force_retrain and salary_predictor.is_trained:
            logger.info("Enhanced salary predictor model already loaded and ready")
            return True
        
        # If force_retrain is False, try to load existing model first
        if not force_retrain and os.path.exists(salary_predictor.model_path):
            try:
                salary_predictor.load_model(salary_predictor.model_path)
                logger.info("Enhanced salary predictor model loaded successfully from file")
                return True
            except Exception as e:
                logger.warning(f"Failed to load existing model: {e}")
                logger.info("Will collect data and train new enhanced model...")
        
        # Collect fresh data and train model with enhanced methods
        if force_retrain:
            logger.info("Force retrain requested. Using enhanced data collection...")
        else:
            logger.info("No existing model found. Using enhanced data collection to train new model...")
            
        # Enhanced data collection
        training_data, quality_report = salary_collector.collect_strategic_data(
            max_failures=20,
            target_data_points=25
        )
            
        if len(training_data) >= 5:
            evaluation = salary_predictor.train_model(training_data)
            action = "retrained" if force_retrain else "trained"
            logger.info(f"Enhanced salary predictor {action} successfully: R² = {evaluation['r2']:.3f}")
            return True
        else:
            logger.warning(f"Limited training data collected ({len(training_data)} points), but enhanced fallback system available")
            return True  # Still return True since enhanced fallback system can handle predictions
            
    except Exception as e:
        logger.error(f"Failed to initialize enhanced salary predictor: {e}")
        return False

def retrain_salary_model():
    """Force retrain the salary prediction model with enhanced data collection"""
    logger.info("Starting forced model retraining with enhanced methods...")
    return initialize_salary_predictor(force_retrain=True)

def get_salary_predictor():
    """Get the global enhanced salary predictor instance, initializing if necessary"""
    global salary_predictor
    
    if salary_predictor is None or not salary_predictor.is_trained:
        logger.info("Enhanced salary predictor not initialized. Initializing now...")
        if not initialize_salary_predictor():
            # Even if initialization fails, we can still use fallback predictions
            logger.warning("Enhanced initialization failed, but fallback predictions available")
            salary_predictor = SalaryPredictor()  # Create instance for fallback use
    
    return salary_predictor

def get_competitiveness_analyzer():
    """Get the enhanced global competitiveness analyzer instance"""
    return competitiveness_analyzer

# Enhanced initialization function
def _safe_initialize():
    """Safely initialize enhanced salary predictor without raising exceptions during import"""
    try:
        return initialize_salary_predictor(force_retrain=False)
    except Exception as e:
        logger.warning(f"Failed to initialize enhanced salary predictor on import: {e}")
        logger.info("Enhanced salary predictor will be initialized on first use")
        return False

# Initialize when module is imported, but don't fail if it doesn't work
_initialization_success = _safe_initialize()

# Additional utility functions for monitoring and debugging

def get_rate_limiter_stats():
    """Get current rate limiter statistics"""
    global salary_collector
    if salary_collector and hasattr(salary_collector, 'rate_limiter'):
        return salary_collector.rate_limiter.get_stats()
    return {"status": "Rate limiter not initialized"}

def clear_cache():
    """Clear the API response cache"""
    global salary_collector
    if salary_collector and hasattr(salary_collector, 'cache'):
        cache_size = len(salary_collector.cache)
        salary_collector.cache.clear()
        logger.info(f"Cleared {cache_size} cached responses")
        return {"cleared": cache_size}
    return {"status": "Cache not available"}

def get_model_info():
    """Get detailed information about the current model"""
    global salary_predictor
    if salary_predictor and salary_predictor.is_trained:
        info = {
            "is_trained": salary_predictor.is_trained,
            "feature_columns": salary_predictor.feature_columns,
            # "has_fallback_data": bool(salary_predictor.fallback_data),
            "model_type": type(salary_predictor.model).__name__ if salary_predictor.model else None,
            "model_path": salary_predictor.model_path
        }
        
        # if salary_predictor.fallback_data:
        #     info["fallback_stats"] = {
        #         "overall_median": salary_predictor.fallback_data.get('overall_median'),
        #         "job_categories": len(salary_predictor.fallback_data.get('job_medians', {})),
        #         "locations": len(salary_predictor.fallback_data.get('location_medians', {})),
        #         "experience_levels": len(salary_predictor.fallback_data.get('experience_medians', {}))
        #     }
        
        return info
    return {"status": "Model not trained", "is_trained": False}