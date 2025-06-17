from typing import Dict, Any, List
import re
import logging

logger = logging.getLogger(__name__)

def _extract_months(text: str) -> List[str]:
    """Extract month names from text."""
    month_pattern = r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b'
    return re.findall(month_pattern, text, re.IGNORECASE)

def _extract_years(text: str) -> List[str]:
    """Extract years from text."""
    return re.findall(r'\b(20\d{2})\b', text)

class QueryIntentClassifier:
    """
    Clean intent classifier that delegates intelligence to Atomic Agent.
    Handles intent classification and retrieval strategy mapping with minimal fallback logic.
    """
    
    def __init__(self):
        # Initialize the Atomic Agent for intent classification
        try:
            from ..agents.intent_agent_atomic import IntentClassificationAgent
            self.intent_agent = IntentClassificationAgent()
            logger.info("Atomic Agent for intent classification initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize Atomic Agent: {e}. Using simple fallback classification")
            self.intent_agent = None
        
        # Import retrieval strategies from config
        try:
            from ..config.intent_config import get_retrieval_strategy
            self.get_retrieval_strategy = get_retrieval_strategy
            logger.info("Intent configuration loaded successfully")
        except ImportError as e:
            logger.warning(f"Failed to load intent config: {e}. Using fallback strategies")
            self.get_retrieval_strategy = self._fallback_retrieval_strategy
    
    def classify_and_get_strategy(self, query: str) -> Dict[str, Any]:
        """Classify query intent and return appropriate retrieval strategy."""
        if not isinstance(query, str) or not query.strip():
            return self._get_default_strategy()
        
        # Classify intent using Atomic Agent
        predicted_intent = self._classify_intent(query)
        
        # Apply additional heuristics for edge cases
        predicted_intent = self._apply_heuristics(query, predicted_intent)
        
        # Get retrieval strategy for the intent
        strategy = self.get_retrieval_strategy(predicted_intent)
        strategy['query_filters'] = self._extract_basic_filters(query)
        
        logger.info(f"Query classified as: {predicted_intent} - {strategy.get('description', 'No description')}")
        return strategy
    
    def _classify_intent(self, query: str) -> str:
        """Classify query intent using the best available method."""
        if self.intent_agent:
            try:
                return self.intent_agent.classify_query(query)
            except Exception as e:
                logger.error(f"Atomic Agent classification failed: {e}")
                return self._simple_fallback_classify(query)
        else:
            return self._simple_fallback_classify(query)
    
    def _simple_fallback_classify(self, query: str) -> str:
        """Minimal fallback classification when agent is not available."""
        q = query.lower()
        
        # Very basic categorization for critical use cases
        if any(word in q for word in ['predict', 'forecast', 'future', 'tomorrow']):
            return 'forecasting'
        
        if any(word in q for word in ['compare', 'versus', 'vs', 'difference']):
            # Check for multiple months to differentiate comparison types
            months = _extract_months(q)
            if len(set(m.lower() for m in months)) >= 2:
                return 'time_comparative'
            return 'comparative_analysis'
        
        if any(word in q for word in ['trend', 'over time', 'pattern']):
            return 'trend_analysis'
        
        if any(word in q for word in ['highest', 'lowest', 'maximum', 'minimum']):
            return 'outlier_detection'
        
        # Default to direct query for most cases
        return 'direct_query'
    
    def _apply_heuristics(self, query: str, predicted_intent: str) -> str:
        """Apply additional heuristics for edge cases."""
        q = query.lower()
        
        # Check for multiple months even if agent didn't catch it
        months = _extract_months(q)
        unique_months = {m.lower() for m in months}
        if len(unique_months) >= 2 and predicted_intent != 'direct_query':
            logger.info("Overriding classification to time_comparative due to multiple months")
            return 'time_comparative'
        
        return predicted_intent
    
    def _extract_basic_filters(self, query: str) -> Dict[str, Any]:
        """Extract basic filters from query."""
        filters = {}
        q = query.lower()
        
        # Extract months
        months = _extract_months(q)
        if months:
            filters['months'] = [m.lower() for m in months]
        
        # Extract years
        years = _extract_years(query)
        if years:
            filters['years'] = years
        
        return filters
    
    def _fallback_retrieval_strategy(self, intent: str) -> Dict[str, Any]:
        """Fallback retrieval strategies if config is not available."""
        strategies = {
            'direct_query': {'num_documents': 5, 'focus': 'exact_data_lookup'},
            'forecasting': {'num_documents': 30, 'focus': 'recent_similar'},
            'trend_analysis': {'num_documents': 200, 'focus': 'temporal_sequence'},
            'outlier_detection': {'num_documents': 50, 'focus': 'outliers'},
            'comparative_analysis': {'num_documents': 75, 'focus': 'comparative'},
            'time_comparative': {'num_documents': 50, 'focus': 'balanced_time_periods'}
        }
        
        strategy = strategies.get(intent, strategies['direct_query']).copy()
        strategy['intent'] = intent
        strategy['description'] = f"{intent} - fallback strategy"
        return strategy
    
    def _get_default_strategy(self) -> Dict[str, Any]:
        """Get default strategy for invalid queries."""
        return self.get_retrieval_strategy('direct_query')
    
    def get_intent_summary(self) -> Dict[str, Any]:
        """Get summary of classification capabilities."""
        try:
            from ..config.intent_config import get_all_intents
            return {
                'available_intents': list(get_all_intents().keys()),
                'agent_available': self.intent_agent is not None,
                'config_available': True
            }
        except ImportError:
            return {
                'available_intents': ['direct_query', 'forecasting', 'trend_analysis', 
                                    'outlier_detection', 'comparative_analysis', 
                                    'time_comparative'],
                'agent_available': self.intent_agent is not None,
                'config_available': False
            }
