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
        # Convert to numpy array if it's not already
        features_array = np.array(features)
        
        # Replace NaN values with zeros
        features_array = np.nan_to_num(features_array, nan=0.0)
        
        return features_array

    def analyze_pattern(self, current_features):
        """Analyze current pattern and predict outcome, with NaN handling"""
        # First check if the features contain NaN values
        if self._contains_nan(current_features):
            logger.warning("Feature vector contains NaN values - replacing with zeros")
        
        # Find similar patterns with NaN handling
        similar_patterns = self.find_similar_patterns(current_features)
        
        # Calculate confidence based on similar patterns
        win_count = sum(1 for p in similar_patterns if p['outcome'] == 1)
        raw_confidence = win_count / len(similar_patterns) if similar_patterns else 0.5
        
        # Calculate weighted confidence based on similarity
        if similar_patterns:
            total_similarity = sum(p['similarity_score'] for p in similar_patterns)
            if total_similarity > 0:
                weighted_confidence = sum(p['similarity_score'] * (1 if p['outcome'] == 1 else 0) 
                                        for p in similar_patterns) / total_similarity
            else:
                weighted_confidence = 0.5
        else:
            weighted_confidence = 0.5
        
        # Get model prediction if classifier is available
        model_confidence = None
        if self.classifier is not None:
            features_cleaned = self._handle_nan_values(current_features)
            features_scaled = self.scaler.transform([features_cleaned])
            try:
                model_confidence = self.classifier.predict_proba(features_scaled)[0][1]
            except Exception as e:
                logger.error(f"Error predicting with classifier: {str(e)}")
                model_confidence = 0.5
            
            # Final confidence is weighted average
            final_confidence = 0.5 * weighted_confidence + 0.5 * model_confidence
        else:
            final_confidence = weighted_confidence
        
        # Determine best action based on confidence
        if final_confidence > 0.6:
            # Check direction from similar patterns
            direction_votes = {}
            for p in similar_patterns:
                direction = p.get('direction', 'UNKNOWN')
                direction_votes[direction] = direction_votes.get(direction, 0) + 1
            
            # Get most common direction
            if direction_votes:
                recommended_direction = max(direction_votes.items(), key=lambda x: x[1])[0]
                if 'LONG' in recommended_direction:
                    recommended_action = 'BUY'
                elif 'SHORT' in recommended_direction:
                    recommended_action = 'SELL'
                else:
                    recommended_action = 'NO_TRADE'
            else:
                recommended_action = 'NO_TRADE'
        else:
            recommended_action = 'NO_TRADE'
        
        return {
            'similar_patterns': similar_patterns,
            'raw_confidence': raw_confidence,
            'weighted_confidence': weighted_confidence,
            'model_confidence': model_confidence,
            'final_confidence': final_confidence,
            'recommended_action': recommended_action
        }

    def _contains_nan(self, features):
        """Check if feature vector contains any NaN values"""
        return np.isnan(np.array(features).astype(float)).any()
    
    def save(self, filepath):
        """Save model to disk"""
        data = {
            'model_type': self.model_type,
            'scaler': self.scaler,
            'model': self.nn_model,
            'classifier': self.classifier,
            'patterns': self.patterns,
            'features': self.features,
            'labels': self.labels
        }
        
        joblib.dump(data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """Load model from disk"""
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
            
    def get_feature_importance(self):
        """Get feature importance from classifier"""
        if self.classifier is None or not hasattr(self.classifier, 'feature_importances_'):
            logger.warning("No feature importance available")
            return {}
        
        if self.features is None or self.features.shape[1] == 0:
            logger.warning("No features available")
            return {}
        
        # Create feature names if not available
        feature_names = [f"feature_{i}" for i in range(self.features.shape[1])]
        
        # Get feature importance
        feature_importance = self.classifier.feature_importances_
        
        # Sort by importance
        indices = np.argsort(feature_importance)[::-1]
        
        # Return as dictionary
        importance_dict = {}
        for i in indices:
            importance_dict[feature_names[i]] = feature_importance[i]
        
        return importance_dict
    
    def plot_feature_importance(self, top_n=10):
        """Plot feature importance"""
        if self.classifier is None or not hasattr(self.classifier, 'feature_importances_'):
            logger.warning("No feature importance available to plot")
            return
        
        # Get feature importance
        importance = self.get_feature_importance()
        
        if not importance:
            return
        
        # Convert to dataframe
        importance_df = pd.DataFrame({
            'Feature': list(importance.keys()),
            'Importance': list(importance.values())
        })
        
        # Sort and get top N
        importance_df = importance_df.sort_values('Importance', ascending=False).head(top_n)
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.barh(importance_df['Feature'], importance_df['Importance'])
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title('Feature Importance')
        plt.tight_layout()
        
        return plt.gcf()
    
    def analyze_pattern_with_level_data(self, current_features, level_analyzer, candle_data=None):
        """
        Enhanced pattern analysis that incorporates market level data
        for more accurate trade recommendations
        
        Args:
            current_features: Feature vector of current pattern
            level_analyzer: MarketLevelAnalyzer instance with historical data
            candle_data: Dictionary with current candle properties
            
        Returns:
            Analysis dictionary with recommendations and confidence scores
        """
        # Get base analysis from pattern model
        base_analysis = self.analyze_pattern(current_features)
        
        # If we don't have a level analyzer or candle data, return base analysis
        if level_analyzer is None or candle_data is None:
            return base_analysis
        
        # Determine which level we're near
        level_type = None
        
        # Check distances from key levels
        for key_level in ['or_high', 'or_low', 'or_mid', 'pm_high', 'pm_low', 'pd_high', 'pd_low']:
            distance_key = f'dist_from_{key_level}'
            if distance_key in candle_data and abs(candle_data[distance_key]) < 0.3:
                level_type = key_level
                break
        
        # If we're not near any key level, return base analysis
        if level_type is None:
            return base_analysis
        
        # Get historical performance at this level
        level_stats = level_analyzer.get_level_stats(level_type)
        
        # If no stats available, return base analysis
        if not level_stats or level_stats.get('total_retests', 0) == 0:
            return base_analysis
        
        # Get stats for the current candle pattern
        candle_pattern = 'none'
        for pattern in ['hammer', 'inverted_hammer', 'doji', 'bullish_engulfing', 'bearish_engulfing']:
            if candle_data.get(pattern, False):
                candle_pattern = pattern
                break
        
        # Get pattern-specific win rate
        pattern_win_rate = 0.5  # Default 
        if candle_pattern in level_stats.get('pattern_stats', {}):
            pattern_stats = level_stats['pattern_stats'][candle_pattern]
            if pattern_stats.get('total', 0) >= 3:  # Only use if we have enough samples
                pattern_win_rate = pattern_stats.get('win_rate', 50) / 100
        
        # Get level-specific win rate
        level_win_rate = level_stats.get('win_rate', 50) / 100
        
        # Combine all confidence scores
        combined_confidence = (
            0.4 * base_analysis['final_confidence'] + 
            0.4 * pattern_win_rate + 
            0.2 * level_win_rate
        )
        
        # Update analysis with combined confidence
        enhanced_analysis = base_analysis.copy()
        enhanced_analysis['level_type'] = level_type
        enhanced_analysis['level_win_rate'] = level_win_rate
        enhanced_analysis['pattern_win_rate'] = pattern_win_rate
        enhanced_analysis['final_confidence'] = combined_confidence
        
        # Update recommended action based on combined confidence
        if combined_confidence > 0.6:
            # Determine direction based on level type
            if level_type in ['or_high', 'pm_high', 'pd_high']:
                enhanced_analysis['recommended_action'] = 'BUY'
            elif level_type in ['or_low', 'pm_low', 'pd_low']:
                enhanced_analysis['recommended_action'] = 'SELL'
            else:
                # For mid levels, check which side we're approaching from
                if candle_data.get('close', 0) > candle_data.get(level_type, 0):
                    enhanced_analysis['recommended_action'] = 'SELL'  # Approaching mid from above
                else:
                    enhanced_analysis['recommended_action'] = 'BUY'   # Approaching mid from below
        else:
            enhanced_analysis['recommended_action'] = 'NO_TRADE'
        
        return enhanced_analysis
