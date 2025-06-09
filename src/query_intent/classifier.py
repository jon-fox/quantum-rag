from typing import Dict, Any
import re
import logging

logger = logging.getLogger(__name__)

class QueryIntentClassifier:
    """
    Classifies energy queries by intent to optimize document retrieval.
    
    New feature:
    - time_comparative: for queries asking to compare across multiple months,
      retrieving a balanced set of docs per period.
    """
    
    MONTH_PATTERN = re.compile(r'\b(january|february|march|april|may|june|'
                               r'july|august|september|october|november|december)\b',
                               re.IGNORECASE)
    
    def __init__(self):
        self.intent_patterns = {
            'forecasting': [
                r'\bforecast\b', r'\bpredict\b', r'\bfuture\b', r'\btomorrow\b'
            ],
            'trend_analysis': [
                r'\btrend\b', r'\bover time\b', r'\btime series\b'
            ],
            'outlier_detection': [
                r'\b(maximum|min(imum)?|extreme|peak|outlier)\b'
            ],
            'comparative_analysis': [
                r'\bcompare\b', r'\bversus\b', r'\bvs\b', r'\bdifference\b'
            ]
        }
        
        self.retrieval_strategies = {
            'forecasting': {
                'num_documents': 30,
                'focus': 'recent_similar',
                'description': 'forecasting – recent conditions',
                'rationale': 'Focus on recent patterns for prediction'
            },
            'trend_analysis': {
                'num_documents': 200,
                'focus': 'temporal_sequence', 
                'description': 'trend – extended time series',
                'rationale': 'Comprehensive history for trends'
            },
            'outlier_detection': {
                'num_documents': 50,
                'focus': 'outliers',
                'description': 'outlier detection',
                'rationale': 'Target extreme events'
            },
            'comparative_analysis': {
                'num_documents': 75,
                'focus': 'comparative',
                'description': 'comparative – category comparison',
                'rationale': 'Balanced data for comparison'
            },
            'time_comparative': {
                'num_documents': 50,
                'focus': 'balanced_time_periods',
                'description': 'time comparative – equal docs per month',
                'rationale': 'Ensure each month is represented'
            },
            'general': {
                'num_documents': 100,
                'focus': 'general',
                'description': 'general – broad retrieval',
                'rationale': 'Default strategy'
            }
        }
    
    def classify_and_get_strategy(self, query: str) -> Dict[str, Any]:
        if not isinstance(query, str) or not query.strip():
            logger.warning("Invalid query provided")
            return self.retrieval_strategies['general']
        
        q = query.lower()
        # Detect multi‐month comparison
        months = self.MONTH_PATTERN.findall(q)
        unique_months = {m.lower() for m in months}
        if len(unique_months) >= 2:
            strategy = self.retrieval_strategies['time_comparative'].copy()
            logger.info("Query classified as time_comparative")
            return strategy
        
        # Check other intents
        for intent, patterns in self.intent_patterns.items():
            for pat in patterns:
                if re.search(pat, q):
                    strategy = self.retrieval_strategies[intent].copy()
                    logger.info(f"Query classified as {intent}")
                    return strategy
        
        # Fallback
        strategy = self.retrieval_strategies['general'].copy()
        logger.info("Query classified as general")
        return strategy
    
    def get_intent_summary(self) -> Dict[str, Any]:
        return {
            intent: {
                'patterns': pats,
                'strategy': self.retrieval_strategies[intent]
            }
            for intent, pats in self.intent_patterns.items()
        }
