import pymongo
import os
import json
from pathlib import Path
import hashlib
import secrets
import time
from datetime import datetime, timedelta
import jwt
from pymongo.errors import ConnectionFailure, OperationFailure

class Database:
    def __init__(self):
        # MongoDB connection settings
        # Using environment variables for security or fallback to localhost
        self.mongo_uri = os.environ.get('MONGO_URI', 'mongodb://localhost:27017/')
        self.db_name = os.environ.get('MONGO_DB_NAME', 'movie_recommender')
        
        # Initialize MongoDB client
        self._initialize_mongo_client()
        
        # JWT secret for token generation
        self.jwt_secret = os.environ.get('JWT_SECRET', secrets.token_hex(32))
        self.token_expiry = 24  # Token expiry in hours
        
        # Rate limiting settings
        self.rate_limit_window = 60  # seconds
        self.max_requests = 100  # max requests per window
        
        # Keep file paths for backward compatibility during transition
        self.users_file = Path('Files/users.json')
        self.ratings_file = Path('Files/user_ratings.json')
        self.preferences_file = Path('Files/user_preferences.json')
        self.metrics_file = Path('Files/recommendation_metrics.json')
        
        # Initialize collections
        self._initialize_collections()
    
    def _initialize_mongo_client(self):
        """Initialize MongoDB client with error handling"""
        try:
            self.client = pymongo.MongoClient(self.mongo_uri, serverSelectionTimeoutMS=5000)
            # Verify connection
            self.client.admin.command('ping')
            print("MongoDB connection successful")
            self.db = self.client[self.db_name]
        except (ConnectionFailure, OperationFailure) as e:
            print(f"MongoDB connection failed: {e}")
            print("Falling back to file-based storage")
            self.client = None
            self.db = None
    
    def _initialize_collections(self):
        """Initialize MongoDB collections if connection is successful"""
        if self.db:
            # Create collections if they don't exist
            if 'users' not in self.db.list_collection_names():
                self.db.create_collection('users')
                # Create unique index on username
                self.db.users.create_index('username', unique=True)
            
            if 'ratings' not in self.db.list_collection_names():
                self.db.create_collection('ratings')
                # Create compound index on username and movie_id
                self.db.ratings.create_index([('username', 1), ('movie_id', 1)], unique=True)
            
            if 'preferences' not in self.db.list_collection_names():
                self.db.create_collection('preferences')
                # Create unique index on username
                self.db.preferences.create_index('username', unique=True)
            
            if 'metrics' not in self.db.list_collection_names():
                self.db.create_collection('metrics')
            
            if 'rate_limits' not in self.db.list_collection_names():
                self.db.create_collection('rate_limits')
                # Create index with TTL for automatic cleanup
                self.db.rate_limits.create_index('timestamp', expireAfterSeconds=self.rate_limit_window)
            
            # Migrate data from files to MongoDB if needed
            self._migrate_data_to_mongodb()
    
    def _migrate_data_to_mongodb(self):
        """Migrate data from JSON files to MongoDB if files exist"""
        if not self.db:
            return
        
        # Migrate users
        if self.users_file.exists():
            try:
                with open(self.users_file, 'r') as f:
                    users_data = json.load(f)
                
                for username, user_data in users_data.items():
                    # Hash password before storing
                    if 'password' in user_data:
                        password = user_data['password']
                        salt = secrets.token_hex(16)
                        hashed_password = self._hash_password(password, salt)
                        user_data['password'] = hashed_password
                        user_data['salt'] = salt
                    
                    user_data['username'] = username
                    # Use update_one with upsert to avoid duplicates
                    self.db.users.update_one(
                        {'username': username},
                        {'$set': user_data},
                        upsert=True
                    )
                print("Users migrated to MongoDB")
            except Exception as e:
                print(f"Error migrating users: {e}")
        
        # Migrate ratings
        if self.ratings_file.exists():
            try:
                with open(self.ratings_file, 'r') as f:
                    ratings_data = json.load(f)
                
                for username, user_ratings in ratings_data.items():
                    for movie_id, rating in user_ratings.items():
                        self.db.ratings.update_one(
                            {'username': username, 'movie_id': movie_id},
                            {'$set': {
                                'username': username,
                                'movie_id': movie_id,
                                'rating': rating,
                                'timestamp': datetime.now()
                            }},
                            upsert=True
                        )
                print("Ratings migrated to MongoDB")
            except Exception as e:
                print(f"Error migrating ratings: {e}")
        
        # Migrate preferences
        if self.preferences_file.exists():
            try:
                with open(self.preferences_file, 'r') as f:
                    preferences_data = json.load(f)
                
                for username, preferences in preferences_data.items():
                    self.db.preferences.update_one(
                        {'username': username},
                        {'$set': {
                            'username': username,
                            'preferences': preferences,
                            'timestamp': datetime.now()
                        }},
                        upsert=True
                    )
                print("Preferences migrated to MongoDB")
            except Exception as e:
                print(f"Error migrating preferences: {e}")
        
        # Migrate metrics
        if self.metrics_file.exists():
            try:
                with open(self.metrics_file, 'r') as f:
                    metrics_data = json.load(f)
                
                self.db.metrics.update_one(
                    {'_id': 'recommendation_metrics'},
                    {'$set': metrics_data},
                    upsert=True
                )
                print("Metrics migrated to MongoDB")
            except Exception as e:
                print(f"Error migrating metrics: {e}")
    
    def _hash_password(self, password, salt):
        """Hash password with salt using SHA-256"""
        return hashlib.sha256((password + salt).encode()).hexdigest()
    
    def _verify_password(self, stored_password, provided_password, salt):
        """Verify password against stored hash"""
        return stored_password == self._hash_password(provided_password, salt)
    
    def _generate_token(self, username):
        """Generate JWT token for user authentication"""
        payload = {
            'username': username,
            'exp': datetime.utcnow() + timedelta(hours=self.token_expiry),
            'iat': datetime.utcnow()
        }
        return jwt.encode(payload, self.jwt_secret, algorithm='HS256')
    
    def _verify_token(self, token):
        """Verify JWT token and return username if valid"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=['HS256'])
            return payload['username']
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
    
    def _check_rate_limit(self, username, action):
        """Check if user has exceeded rate limit for an action"""
        if not self.db:
            return True  # Skip rate limiting if MongoDB is not available
        
        now = datetime.now()
        key = f"{username}:{action}"
        
        # Count requests in the current window
        count = self.db.rate_limits.count_documents({
            'key': key,
            'timestamp': {'$gte': now - timedelta(seconds=self.rate_limit_window)}
        })
        
        if count >= self.max_requests:
            return False
        
        # Record this request
        self.db.rate_limits.insert_one({
            'key': key,
            'timestamp': now
        })
        
        return True
    
    def _fallback_to_file(self, operation, *args, **kwargs):
        """Log fallback to file-based storage"""
        print(f"Falling back to file-based storage for operation: {operation}")
    
    # User Authentication Methods
    def register_user(self, username, password):
        """Register a new user with secure password storage"""
        # Input validation
        if not username or not password:
            return False, 'Username and password are required'
        
        if len(password) < 8:
            return False, 'Password must be at least 8 characters long'
        
        # Rate limiting
        if not self._check_rate_limit('anonymous', 'register'):
            return False, 'Too many registration attempts. Please try again later.'
        
        if self.db:
            try:
                # Check if username exists
                if self.db.users.find_one({'username': username}):
                    return False, 'Username already exists'
                
                # Generate salt and hash password
                salt = secrets.token_hex(16)
                hashed_password = self._hash_password(password, salt)
                
                # Create user document
                user_doc = {
                    'username': username,
                    'password': hashed_password,
                    'salt': salt,
                    'completed_onboarding': False,
                    'created_at': datetime.now(),
                    'last_login': None
                }
                
                # Insert user
                self.db.users.insert_one(user_doc)
                
                # Generate token
                token = self._generate_token(username)
                
                return True, {'message': 'Registration successful', 'token': token}
            except Exception as e:
                print(f"Error registering user: {e}")
                self._fallback_to_file('register_user')
        
        # Fallback to file-based storage
        users = {}
        if self.users_file.exists():
            with open(self.users_file, 'r') as f:
                users = json.load(f)
        
        if username in users:
            return False, 'Username already exists'
        
        users[username] = {
            'password': password,  # Note: Not hashed in file-based storage
            'completed_onboarding': False
        }
        
        with open(self.users_file, 'w') as f:
            json.dump(users, f)
        
        return True, {'message': 'Registration successful'}
    
    def login_user(self, username, password):
        """Authenticate user and return JWT token"""
        # Input validation
        if not username or not password:
            return False, 'Username and password are required'
        
        # Rate limiting
        if not self._check_rate_limit(username, 'login'):
            return False, 'Too many login attempts. Please try again later.'
        
        if self.db:
            try:
                # Find user
                user = self.db.users.find_one({'username': username})
                if not user or not self._verify_password(user['password'], password, user['salt']):
                    return False, 'Invalid username or password'
                
                # Update last login
                self.db.users.update_one(
                    {'username': username},
                    {'$set': {'last_login': datetime.now()}}
                )
                
                # Generate token
                token = self._generate_token(username)
                
                return True, {'message': 'Login successful', 'token': token}
            except Exception as e:
                print(f"Error logging in user: {e}")
                self._fallback_to_file('login_user')
        
        # Fallback to file-based storage
        if self.users_file.exists():
            with open(self.users_file, 'r') as f:
                users = json.load(f)
            
            if username not in users or users[username]['password'] != password:
                return False, 'Invalid username or password'
            
            return True, {'message': 'Login successful'}
        
        return False, 'User database not found'
    
    # User Preferences Methods
    def save_user_preferences(self, username, preferences):
        """Save user preferences to MongoDB"""
        if self.db:
            try:
                # Update preferences
                self.db.preferences.update_one(
                    {'username': username},
                    {'$set': {
                        'username': username,
                        'preferences': preferences,
                        'timestamp': datetime.now()
                    }},
                    upsert=True
                )
                
                # Mark onboarding as completed
                self.db.users.update_one(
                    {'username': username},
                    {'$set': {'completed_onboarding': True}}
                )
                
                return True
            except Exception as e:
                print(f"Error saving preferences: {e}")
                self._fallback_to_file('save_user_preferences')
        
        # Fallback to file-based storage
        # Save preferences
        user_prefs = {}
        if self.preferences_file.exists():
            with open(self.preferences_file, 'r') as f:
                user_prefs = json.load(f)
        
        user_prefs[username] = preferences
        with open(self.preferences_file, 'w') as f:
            json.dump(user_prefs, f)
        
        # Mark onboarding as completed
        users = {}
        if self.users_file.exists():
            with open(self.users_file, 'r') as f:
                users = json.load(f)
        
        if username in users:
            users[username]['completed_onboarding'] = True
            with open(self.users_file, 'w') as f:
                json.dump(users, f)
        
        return True
    
    def get_user(self, username):
        """Get user information from MongoDB or file"""
        if self.db:
            try:
                # Find user
                user = self.db.users.find_one({'username': username})
                if user:
                    # Remove sensitive information
                    if 'password' in user:
                        del user['password']
                    if 'salt' in user:
                        del user['salt']
                    if '_id' in user:
                        del user['_id']
                    return user
                return None
            except Exception as e:
                print(f"Error getting user: {e}")
                self._fallback_to_file('get_user')
        
        # Fallback to file-based storage
        if self.users_file.exists():
            with open(self.users_file, 'r') as f:
                users = json.load(f)
            
            if username in users:
                user_data = users[username].copy()
                # Remove sensitive information
                if 'password' in user_data:
                    del user_data['password']
                user_data['username'] = username
                return user_data
        
        return None
    
    def get_all_users(self):
        """Get all users from MongoDB or file"""
        if self.db:
            try:
                # Find all users
                users_cursor = self.db.users.find({}, {'password': 0, 'salt': 0})
                users = []
                for user in users_cursor:
                    if '_id' in user:
                        del user['_id']
                    users.append(user)
                return users
            except Exception as e:
                print(f"Error getting all users: {e}")
                self._fallback_to_file('get_all_users')
        
        # Fallback to file-based storage
        if self.users_file.exists():
            with open(self.users_file, 'r') as f:
                users_data = json.load(f)
            
            users = []
            for username, user_data in users_data.items():
                user = user_data.copy()
                # Remove sensitive information
                if 'password' in user:
                    del user['password']
                user['username'] = username
                users.append(user)
            
            return users
        
        return []
    
    def update_user(self, username, update_data):
        """Update user information in MongoDB or file"""
        if not username or not update_data:
            return False, 'Username and update data are required'
        
        # Prevent updating sensitive fields
        sensitive_fields = ['password', 'salt']
        for field in sensitive_fields:
            if field in update_data:
                del update_data[field]
        
        if self.db:
            try:
                # Update user
                result = self.db.users.update_one(
                    {'username': username},
                    {'$set': update_data}
                )
                
                if result.matched_count > 0:
                    return True, {'message': 'User updated successfully'}
                return False, 'User not found'
            except Exception as e:
                print(f"Error updating user: {e}")
                self._fallback_to_file('update_user')
        
        # Fallback to file-based storage
        if self.users_file.exists():
            with open(self.users_file, 'r') as f:
                users = json.load(f)
            
            if username in users:
                users[username].update(update_data)
                with open(self.users_file, 'w') as f:
                    json.dump(users, f)
                return True, {'message': 'User updated successfully'}
            
            return False, 'User not found'
        
        return False, 'User database not found'
    
    def get_user_preferences(self, username):
        """Get user preferences from MongoDB or file"""
        if self.db:
            try:
                # Find preferences
                preferences_doc = self.db.preferences.find_one({'username': username})
                if preferences_doc:
                    return preferences_doc.get('preferences', {})
                return {}
            except Exception as e:
                print(f"Error getting preferences: {e}")
                self._fallback_to_file('get_user_preferences')
        
        # Fallback to file-based storage
        if self.preferences_file.exists():
            with open(self.preferences_file, 'r') as f:
                user_prefs = json.load(f)
            
            return user_prefs.get(username, {})
        
        return {}
    
    def get_user(self, username):
        """Get user information from MongoDB or file"""
        if self.db:
            try:
                # Find user
                user = self.db.users.find_one({'username': username})
                if user:
                    # Remove sensitive information
                    if 'password' in user:
                        del user['password']
                    if 'salt' in user:
                        del user['salt']
                    if '_id' in user:
                        del user['_id']
                    return user
                return None
            except Exception as e:
                print(f"Error getting user: {e}")
                self._fallback_to_file('get_user')
        
        # Fallback to file-based storage
        if self.users_file.exists():
            with open(self.users_file, 'r') as f:
                users = json.load(f)
            
            if username in users:
                user_data = users[username].copy()
                # Remove sensitive information
                if 'password' in user_data:
                    del user_data['password']
                user_data['username'] = username
                return user_data
        
        return None
    
    def get_all_users(self):
        """Get all users from MongoDB or file"""
        if self.db:
            try:
                # Find all users
                users_cursor = self.db.users.find({}, {'password': 0, 'salt': 0})
                users = []
                for user in users_cursor:
                    if '_id' in user:
                        del user['_id']
                    users.append(user)
                return users
            except Exception as e:
                print(f"Error getting all users: {e}")
                self._fallback_to_file('get_all_users')
        
        # Fallback to file-based storage
        if self.users_file.exists():
            with open(self.users_file, 'r') as f:
                users_data = json.load(f)
            
            users = []
            for username, user_data in users_data.items():
                user = user_data.copy()
                # Remove sensitive information
                if 'password' in user:
                    del user['password']
                user['username'] = username
                users.append(user)
            
            return users
        
        return []
    
    def update_user(self, username, update_data):
        """Update user information in MongoDB or file"""
        if not username or not update_data:
            return False, 'Username and update data are required'
        
        # Prevent updating sensitive fields
        sensitive_fields = ['password', 'salt']
        for field in sensitive_fields:
            if field in update_data:
                del update_data[field]
        
        if self.db:
            try:
                # Update user
                result = self.db.users.update_one(
                    {'username': username},
                    {'$set': update_data}
                )
                
                if result.matched_count > 0:
                    return True, {'message': 'User updated successfully'}
                return False, 'User not found'
            except Exception as e:
                print(f"Error updating user: {e}")
                self._fallback_to_file('update_user')
        
        # Fallback to file-based storage
        if self.users_file.exists():
            with open(self.users_file, 'r') as f:
                users = json.load(f)
            
            if username in users:
                users[username].update(update_data)
                with open(self.users_file, 'w') as f:
                    json.dump(users, f)
                return True, {'message': 'User updated successfully'}
            
            return False, 'User not found'
        
        return False, 'User database not found'
    
    def get_user_ratings(self, username):
        """Get user ratings from MongoDB or file"""
        if self.db:
            try:
                # Find ratings
                ratings_cursor = self.db.ratings.find({'username': username})
                ratings = {}
                for rating_doc in ratings_cursor:
                    ratings[rating_doc['movie_id']] = rating_doc['rating']
                return ratings
            except Exception as e:
                print(f"Error getting ratings: {e}")
                self._fallback_to_file('get_user_ratings')
        
        # Fallback to file-based storage
        if self.ratings_file.exists():
            with open(self.ratings_file, 'r') as f:
                user_ratings = json.load(f)
            
            return user_ratings.get(username, {})
        
        return {}
    
    def save_user_rating(self, username, movie_id, rating):
        """Save user rating to MongoDB or file"""
        # Input validation
        if not username or not movie_id:
            return False, 'Username and movie ID are required'
        
        try:
            rating = float(rating)
            if rating < 0 or rating > 5:
                return False, 'Rating must be between 0 and 5'
        except ValueError:
            return False, 'Rating must be a number'
        
        # Rate limiting
        if not self._check_rate_limit(username, 'rate_movie'):
            return False, 'Too many rating attempts. Please try again later.'
        
        if self.db:
            try:
                # Update rating
                self.db.ratings.update_one(
                    {'username': username, 'movie_id': movie_id},
                    {'$set': {
                        'username': username,
                        'movie_id': movie_id,
                        'rating': rating,
                        'timestamp': datetime.now()
                    }},
                    upsert=True
                )
                
                return True, {'message': 'Rating saved successfully'}
            except Exception as e:
                print(f"Error saving rating: {e}")
                self._fallback_to_file('save_user_rating')
        
        # Fallback to file-based storage
        user_ratings = {}
        if self.ratings_file.exists():
            with open(self.ratings_file, 'r') as f:
                user_ratings = json.load(f)
        
        if username not in user_ratings:
            user_ratings[username] = {}
        
        user_ratings[username][movie_id] = rating
        
        with open(self.ratings_file, 'w') as f:
            json.dump(user_ratings, f)
        
        return True, {'message': 'Rating saved successfully'}
    
    def delete_user_rating(self, username, movie_id):
        """Delete user rating from MongoDB or file"""
        if not username or not movie_id:
            return False, 'Username and movie ID are required'
        
        if self.db:
            try:
                # Delete rating
                result = self.db.ratings.delete_one({'username': username, 'movie_id': movie_id})
                if result.deleted_count > 0:
                    return True, {'message': 'Rating deleted successfully'}
                return False, 'Rating not found'
            except Exception as e:
                print(f"Error deleting rating: {e}")
                self._fallback_to_file('delete_user_rating')
        
        # Fallback to file-based storage
        if self.ratings_file.exists():
            with open(self.ratings_file, 'r') as f:
                user_ratings = json.load(f)
            
            if username in user_ratings and movie_id in user_ratings[username]:
                del user_ratings[username][movie_id]
                with open(self.ratings_file, 'w') as f:
                    json.dump(user_ratings, f)
                return True, {'message': 'Rating deleted successfully'}
            
            return False, 'Rating not found'
        
        return False, 'User ratings database not found'
    
    def get_recommendation_metrics(self):
        """Get recommendation metrics from MongoDB or file"""
        if self.db:
            try:
                # Find metrics
                metrics_doc = self.db.metrics.find_one({'_id': 'recommendation_metrics'})
                if metrics_doc:
                    # Remove _id field from result
                    if '_id' in metrics_doc:
                        del metrics_doc['_id']
                    return metrics_doc
                return {}
            except Exception as e:
                print(f"Error getting metrics: {e}")
                self._fallback_to_file('get_recommendation_metrics')
        
        # Fallback to file-based storage
        if self.metrics_file.exists():
            with open(self.metrics_file, 'r') as f:
                metrics = json.load(f)
            
            return metrics
        
        return {}
    
    def save_recommendation_metrics(self, metrics):
        """Save recommendation metrics to MongoDB or file"""
        if not isinstance(metrics, dict):
            return False, 'Metrics must be a dictionary'
        
        if self.db:
            try:
                # Update metrics
                self.db.metrics.update_one(
                    {'_id': 'recommendation_metrics'},
                    {'$set': metrics},
                    upsert=True
                )
                
                return True, {'message': 'Metrics saved successfully'}
            except Exception as e:
                print(f"Error saving metrics: {e}")
                self._fallback_to_file('save_recommendation_metrics')
        
        # Fallback to file-based storage
        with open(self.metrics_file, 'w') as f:
            json.dump(metrics, f)
        
        return True, {'message': 'Metrics saved successfully'}
    
    def get_user(self, username):
        """Get user information from MongoDB or file"""
        if self.db:
            try:
                # Find user
                user = self.db.users.find_one({'username': username})
                if user:
                    # Remove sensitive information
                    if 'password' in user:
                        del user['password']
                    if 'salt' in user:
                        del user['salt']
                    if '_id' in user:
                        del user['_id']
                    return user
                return None
            except Exception as e:
                print(f"Error getting user: {e}")
                self._fallback_to_file('get_user')
        
        # Fallback to file-based storage
        if self.users_file.exists():
            with open(self.users_file, 'r') as f:
                users = json.load(f)
            
            if username in users:
                user_data = users[username].copy()
                # Remove sensitive information
                if 'password' in user_data:
                    del user_data['password']
                user_data['username'] = username
                return user_data
        
        return None
    
    def get_all_users(self):
        """Get all users from MongoDB or file"""
        if self.db:
            try:
                # Find all users
                users_cursor = self.db.users.find({}, {'password': 0, 'salt': 0})
                users = []
                for user in users_cursor:
                    if '_id' in user:
                        del user['_id']
                    users.append(user)
                return users
            except Exception as e:
                print(f"Error getting all users: {e}")
                self._fallback_to_file('get_all_users')
        
        # Fallback to file-based storage
        if self.users_file.exists():
            with open(self.users_file, 'r') as f:
                users_data = json.load(f)
            
            users = []
            for username, user_data in users_data.items():
                user = user_data.copy()
                # Remove sensitive information
                if 'password' in user:
                    del user['password']
                user['username'] = username
                users.append(user)
            
            return users
        
        return []
    
    def update_user(self, username, update_data):
        """Update user information in MongoDB or file"""
        if not username or not update_data:
            return False, 'Username and update data are required'
        
        # Prevent updating sensitive fields
        sensitive_fields = ['password', 'salt']
        for field in sensitive_fields:
            if field in update_data:
                del update_data[field]
        
        if self.db:
            try:
                # Update user
                result = self.db.users.update_one(
                    {'username': username},
                    {'$set': update_data}
                )
                
                if result.matched_count > 0:
                    return True, {'message': 'User updated successfully'}
                return False, 'User not found'
            except Exception as e:
                print(f"Error updating user: {e}")
                self._fallback_to_file('update_user')
        
        # Fallback to file-based storage
        if self.users_file.exists():
            with open(self.users_file, 'r') as f:
                users = json.load(f)
            
            if username in users:
                users[username].update(update_data)
                with open(self.users_file, 'w') as f:
                    json.dump(users, f)
                return True, {'message': 'User updated successfully'}
            
            return False, 'User not found'
        
        return False, 'User database not found'