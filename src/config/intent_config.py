"""
Intent classification configuration for the energy RAG system.
Defines intent types, characteristics, and retrieval strategies.
"""
from typing import Dict, Any

# Intent definitions with characteristics and examples
INTENT_DEFINITIONS = {
    'direct_query': {
        'description': 'Request for a specific data point or value for a particular date/time',
        'examples': [
            'What\'s the peak load recorded in ERCOT on June 15, 2025?',
            'What was the average temperature on May 3, 2025?',
            'How much generation was there on April 10, 2025?'
        ],
        'characteristics': 'Asks for specific values, mentions exact dates, uses words like "what", "how much", "peak", "average"'
    },
    'forecasting': {
        'description': 'Request for predictions or future projections',
        'examples': [
            'Predict tomorrow\'s energy demand',
            'What will be the load forecast for next week?',
            'Future renewable generation trends'
        ],
        'characteristics': 'Uses words like "forecast", "predict", "future", "tomorrow", "will be"'
    },
    'trend_analysis': {
        'description': 'Request to analyze patterns or trends over time',
        'examples': [
            'Show load trends over the past 3 months',
            'How has renewable generation changed over time?',
            'Time series analysis of energy demand'
        ],
        'characteristics': 'Mentions time periods, uses words like "trend", "over time", "pattern", "series"'
    },
    'outlier_detection': {
        'description': 'Request to find extreme values, anomalies, or unusual patterns',
        'examples': [
            'Find the highest load days in May 2025',
            'Which days had extreme generation mismatches?',
            'Identify outliers in energy prices'
        ],
        'characteristics': 'Uses words like "maximum", "minimum", "extreme", "outlier", "highest", "lowest", "anomaly"'
    },
    'comparative_analysis': {
        'description': 'Request to compare different periods, sources, or scenarios',
        'examples': [
            'Compare renewable vs fossil fuel generation',
            'Difference between April and June load patterns',
            'DAM forecast vs actual generation comparison'
        ],
        'characteristics': 'Uses words like "compare", "versus", "difference", "vs", "between"'
    },
    'time_comparative': {
        'description': 'Request to compare data across multiple specific months or time periods',
        'examples': [
            'Compare May and June 2025 peak loads',
            'Renewable generation in April vs June 2025',
            'Load patterns across spring months'
        ],
        'characteristics': 'Mentions multiple months/periods explicitly for comparison'
    }
}

# Retrieval strategies mapped to intents
RETRIEVAL_STRATEGIES = {
    'direct_query': {
        'num_documents': 5,
        'focus': 'exact_data_lookup',
        'description': 'direct query - specific data point',
        'rationale': 'Return exact value for specific date/metric',
        'filters': {
            'time_priority': 'exact_date',
            'metadata_filters': ['date_specific'],
            'exact_match': True
        },
        'sort_strategy': 'date_match_first',
        'response_type': 'direct_value'
    },
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
    }
}

def get_intent_definition(intent: str) -> Dict[str, Any]:
    """Get the definition for a specific intent."""
    return INTENT_DEFINITIONS.get(intent, INTENT_DEFINITIONS['direct_query'])

def get_retrieval_strategy(intent: str) -> Dict[str, Any]:
    """Get the retrieval strategy for a specific intent."""
    strategy = RETRIEVAL_STRATEGIES.get(intent, RETRIEVAL_STRATEGIES['direct_query']).copy()
    strategy['intent'] = intent
    return strategy

def get_all_intents() -> Dict[str, Dict[str, Any]]:
    """Get all available intent definitions."""
    return INTENT_DEFINITIONS.copy()
