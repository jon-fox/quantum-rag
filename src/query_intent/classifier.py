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
                'description': 'forecasting - recent conditions',
                'rationale': 'Focus on recent patterns for prediction',
                'filters': {
                    'time_priority': 'recent',
                    'metadata_filters': ['weather_conditions', 'load_patterns'],
                    'exclude_historical': True
                },
                'sort_strategy': 'recency_weighted_similarity'
            },
            'trend_analysis': {
                'num_documents': 200,
                'focus': 'temporal_sequence', 
                'description': 'trend - extended time series',
                'rationale': 'Comprehensive history for trends',
                'filters': {
                    'time_priority': 'chronological_coverage',
                    'metadata_filters': ['timestamp', 'sequential_data'],
                    'require_time_range': True
                },
                'sort_strategy': 'chronological'
            },
            'outlier_detection': {
                'num_documents': 50,
                'focus': 'outliers',
                'description': 'outlier detection',
                'rationale': 'Target extreme events',
                'filters': {
                    'time_priority': 'extreme_values',
                    'metadata_filters': ['deviation_metrics', 'anomaly_indicators'],
                    'threshold_based': True
                },
                'sort_strategy': 'extreme_values_first'
            },
            'comparative_analysis': {
                'num_documents': 75,
                'focus': 'comparative',
                'description': 'comparative - category comparison',
                'rationale': 'Balanced data for comparison',
                'filters': {
                    'time_priority': 'category_balanced',
                    'metadata_filters': ['generation_type', 'operational_mode'],
                    'ensure_diversity': True
                },
                'sort_strategy': 'category_balanced'
            },
            'time_comparative': {
                'num_documents': 50,
                'focus': 'balanced_time_periods',
                'description': 'time comparative - equal docs per month',
                'rationale': 'Ensure each month is represented',
                'filters': {
                    'time_priority': 'multi_period',
                    'metadata_filters': ['month', 'time_period'],
                    'balanced_periods': True
                },
                'sort_strategy': 'temporal_balanced'
            },
            'general': {
                'num_documents': 100,
                'focus': 'general',
                'description': 'general - broad retrieval',
                'rationale': 'Default strategy',
                'filters': {
                    'time_priority': 'mixed',
                    'metadata_filters': [],
                    'no_restrictions': True
                },
                'sort_strategy': 'similarity_only'
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
            strategy['query_filters'] = self._extract_query_filters(query, 'time_comparative')
            logger.info("Query classified as time_comparative")
            return strategy
        
        # Check other intents
        for intent, patterns in self.intent_patterns.items():
            for pat in patterns:
                if re.search(pat, q):
                    strategy = self.retrieval_strategies[intent].copy()
                    strategy['query_filters'] = self._extract_query_filters(query, intent)
                    logger.info(f"Query classified as {intent}")
                    return strategy
        
        # Fallback
        strategy = self.retrieval_strategies['general'].copy()
        strategy['query_filters'] = self._extract_query_filters(query, 'general')
        logger.info("Query classified as general")
        return strategy
    
    def _extract_query_filters(self, query: str, intent: str) -> Dict[str, Any]:
        """Extract specific filters from the query based on intent"""
        filters = {}
        q = query.lower()
        
        # Extract date/time information
        date_patterns = {
            'months': self.MONTH_PATTERN.findall(query),
            'years': re.findall(r'\b(20\d{2})\b', query),
            'days': re.findall(r'\b(\d{1,2}(?:st|nd|rd|th)?)\b', query),
            'time_periods': re.findall(r'\b(morning|afternoon|evening|night|peak|off-peak)\b', q)
        }
        
        # Extract generation types
        generation_types = re.findall(r'\b(renewable|solar|wind|natural gas|coal|nuclear|fossil)\b', q)
        
        # Extract metrics/measurements
        metrics = re.findall(r'\b(generation|demand|load|forecast|temperature|price|mismatch|deviation)\b', q)
        
        # Extract extreme value indicators
        extremes = re.findall(r'\b(highest|lowest|maximum|minimum|peak|extreme|worst|best)\b', q)
        
        # Build intent-specific filters
        if intent == 'forecasting':
            filters['recent_priority'] = True
            filters['include_weather'] = 'weather' in q or 'temperature' in q
            filters['exclude_old_data'] = True
            
        elif intent == 'trend_analysis':
            filters['time_range'] = date_patterns
            filters['sequential_data'] = True
            filters['chronological_order'] = True
            
        elif intent == 'outlier_detection':
            filters['extreme_values'] = extremes
            filters['metrics_focus'] = metrics
            filters['anomaly_priority'] = True
            
        elif intent == 'comparative_analysis':
            filters['generation_types'] = generation_types
            filters['category_balance'] = True
            filters['diverse_scenarios'] = True
            
        elif intent == 'time_comparative':
            filters['time_periods'] = date_patterns
            filters['balanced_coverage'] = True
            
        # Add common filters
        filters['date_info'] = date_patterns
        filters['keywords'] = generation_types + metrics
        
        return filters
    
    def get_intent_summary(self) -> Dict[str, Any]:
        return {
            intent: {
                'patterns': pats,
                'strategy': self.retrieval_strategies[intent]
            }
            for intent, pats in self.intent_patterns.items()
        }
