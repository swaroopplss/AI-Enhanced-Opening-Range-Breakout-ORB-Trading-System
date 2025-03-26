import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import joblib
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

class PatternModel:
    def __init__(self, model_type="hybrid"):
        self.scaler = StandardScaler()
        self.model_type = model_type
        
        # Initialize models
        self.nn_model = NearestNeighbors(n_neighbors=5, algorithm='ball_tree')
        self.classifier = None
        
        if model_type == "hybrid" or model_type == "ml":
            self.classifier = GradientBoostingClassifier(
                n_estimators=100, 
                learning_rate=0.1, 
                max_depth=3, 
                random_state=42
            )
        
        self.patterns = None
        self.features = None
        self.labels = None
        
    def fit(self, patterns):
        """Fit model on extracted patterns"""
        logger.info(f"Fitting {self.model_type} model on {len(patterns)} patterns")
        self.patterns = patterns
        
        # Extract feature vectors and labels
        X = np.array([p['features'] for p in patterns])
        y = np.array([1 if p['outcome'] == 1 else 0 for p in patterns])
        
        self.features = X
        self.labels = y
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit nearest neighbors model
        self.nn_model.fit(X_scaled)
        
        # Fit classifier if using hybrid or ml model
        if self.classifier is not None:
            logger.info("Training classifier model")
            self.classifier.fit(X_scaled, y)
        
        return self
    
    def find_similar_patterns(self, features, n_neighbors=5):
        """Find similar patterns to given features"""
        # Scale input features
        features_scaled = self.scaler.transform([features])
# Update these methods in pattern_model.py

def find_similar_patterns(self, features, n_neighbors=5):
    """Find similar patterns to given features, with NaN handling"""
    # Handle any NaN values in features
    features_cleaned = self._handle_nan_values(features)
    
    # Scale input features
    features_scaled = self.scaler.transform([features_cleaned])
    
    # Find nearest neighbors
    distances, indices = self.nn_model.kneighbors(features_scaled)
    
    # Get similar patterns
    similar_patterns = []
    for i, idx in enumerate(indices[0]):
        pattern = self.patterns[idx].copy()
        pattern['similarity_score'] = 1 / (1 + distances[0][i])  # Convert distance to similarity score
        similar_patterns.append(pattern)
    
    return similar_patterns

def _handle_nan_values(self, features):
    """Clean feature vector by replacing NaN values with zeros"""
    import numpy as np
    
    # Convert to numpy array if it's not already
    features_array = np.array(features)
    
    # Replace NaN values with zeros
    features_array = np.nan_to_num(features_array, nan=0.0)
    
    return features_array

def analyze_pattern(self, current_features):
    """Analyze current pattern and predict outcome, with NaN handling"""
    # First check if the features contain NaN values
    if self._contains_nan(current_features):
        import logging
        logger = logging.getLogger(__name__)
        logger.warning("Feature vector contains NaN values - replacing with zeros")
    
    # Find similar patterns with NaN handling
    similar_patterns = self.find_similar_patterns(current_features)
    
    # Calculate confidence based on similar patterns
    win_count = sum(1 for p in similar_patterns if p['outcome'] == 1)
    raw_confidence = win_count / len(similar_patterns)
    
    # Calculate weighted confidence based on similarity
    weighted_confidence = sum(p['similarity_score'] * (1 if p['outcome'] == 1 else 0) 
                            for p in similar_patterns) / sum(p['similarity_score'] for p in similar_patterns)
    
    # Get model prediction if classifier is available
    model_confidence = None
    if self.classifier is not None:
        features_cleaned = self._handle_nan_values(current_features)
        features_scaled = self.scaler.transform([features_cleaned])
        model_confidence = self.classifier.predict_proba(features_scaled)[0][1]
        
        # Final confidence is weighted average
        final_confidence = 0.5 * weighted_confidence + 0.5 * model_confidence
    else:
        final_confidence = weighted_confidence
    
    return {
        'similar_patterns': similar_patterns,
        'raw_confidence': raw_confidence,
        'weighted_confidence': weighted_confidence,
        'model_confidence': model_confidence,
        'final_confidence': final_confidence,
        'recommended_action': 'BUY' if final_confidence > 0.6 else 'NO_TRADE'
    }

def _contains_nan(self, features):
    """Check if feature vector contains any NaN values"""
    import numpy as np
    return np.isnan(np.array(features).astype(float)).any()

# Add this classmethod to your PatternModel class in pattern_model.py:

@classmethod
def load(cls, filepath):
    """Load model from disk"""
    import joblib
    import logging
    
    logger = logging.getLogger(__name__)
    logger.info(f"Loading model from {filepath}")
    
    try:
        data = joblib.load(filepath)
        
        model = cls(model_type=data.get('model_type', "hybrid"))
        model.scaler = data.get('scaler')
        model.nn_model = data.get('model')
        model.classifier = data.get('classifier')
        model.patterns = data.get('patterns', [])
        model.features = data.get('features')
        model.labels = data.get('labels')
        
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        # Create an empty model as fallback
        logger.warning("Creating empty model as fallback")
        return cls()