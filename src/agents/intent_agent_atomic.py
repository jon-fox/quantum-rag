"""
Intent classification agent using Atomic Agents framework for the energy RAG system.
"""
from typing import Dict, Any, Optional
import logging
from atomic_agents.agents.base_agent import BaseAgent, BaseAgentConfig
from atomic_agents.lib.components.system_prompt_generator import SystemPromptGenerator
from atomic_agents.lib.base.base_io_schema import BaseIOSchema
from pydantic import Field
import instructor
from openai import OpenAI
import os

logger = logging.getLogger(__name__)

# Import intent configuration
try:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config.intent_config import get_all_intents
    INTENT_CONFIG_AVAILABLE = True
except ImportError:
    logger.warning("Intent config not available, using basic configuration")
    INTENT_CONFIG_AVAILABLE = False


class IntentClassificationInput(BaseIOSchema):
    """Input schema for intent classification."""
    query: str = Field(..., description="The user query to classify")


class IntentClassificationOutput(BaseIOSchema):
    """Output schema for intent classification."""
    intent: str = Field(..., description="The classified intent category")


class IntentClassificationAgent(BaseAgent):
    """
    Atomic Agent for classifying user queries into energy data analysis intent categories.
    Uses modular configuration for intent definitions and characteristics.
    """
    
    def __init__(self, config: Optional[BaseAgentConfig] = None):
        if config is None:
            # Get intent definitions from config
            if INTENT_CONFIG_AVAILABLE:
                intent_definitions = get_all_intents()
            else:
                # Fallback basic definitions
                intent_definitions = {
                    'direct_query': {'description': 'Specific data point requests'},
                    'forecasting': {'description': 'Future predictions'},
                    'trend_analysis': {'description': 'Pattern analysis over time'},
                    'outlier_detection': {'description': 'Extreme values and anomalies'},
                    'comparative_analysis': {'description': 'Comparisons between entities'},
                    'time_comparative': {'description': 'Multi-period comparisons'}
                }
            
            # Build system prompt with intent knowledge
            intent_knowledge = self._build_intent_knowledge(intent_definitions)
            
            # Create instructor client
            client = instructor.from_openai(OpenAI(api_key=os.environ.get("OPENAI_API_KEY")))
            
            # Create system prompt generator
            system_prompt_gen = SystemPromptGenerator(
                background=[
                    "You are an expert energy data query classification agent.",
                    "You specialize in analyzing user queries about energy data and determining their intent.",
                    "You work with ERCOT energy market data including load, generation, prices, and weather.",
                    f"Available intent categories: {', '.join(intent_definitions.keys())}"
                ],
                steps=[
                    "Analyze the query for specific keywords and patterns",
                    "Consider temporal aspects (specific dates vs time ranges)",
                    "Determine if the user wants specific data points or general analysis",
                    "Look for comparison indicators or forecasting language",
                    "Choose the most specific intent category that matches"
                ],
                output_instructions=[
                    "Return only the intent category name",
                    f"Choose from: {', '.join(intent_definitions.keys())}",
                    "If the query asks for a specific value with a date, use 'direct_query'",
                    "If the query mentions multiple months for comparison, use 'time_comparative'",
                    "Default to 'direct_query' if unsure",
                    f"Intent Knowledge:\n{intent_knowledge}"
                ]
            )
            
            config = BaseAgentConfig(
                client=client,
                model="gpt-4o",
                system_prompt_generator=system_prompt_gen,
                input_schema=IntentClassificationInput,
                output_schema=IntentClassificationOutput,
                temperature=0.1,
                max_tokens=50
            )
        
        super().__init__(config)
        
        # Store intent definitions for validation
        if INTENT_CONFIG_AVAILABLE:
            self.available_intents = set(get_all_intents().keys())
        else:
            self.available_intents = {
                'direct_query', 'forecasting', 'trend_analysis', 'outlier_detection',
                'comparative_analysis', 'time_comparative'
            }
    
    def _build_intent_knowledge(self, intent_definitions: Dict[str, Dict[str, Any]]) -> str:
        """Build formatted intent knowledge for the system prompt."""
        knowledge_parts = []
        
        for intent, definition in intent_definitions.items():
            part = f"**{intent.upper()}**: {definition.get('description', '')}"
            
            if 'characteristics' in definition:
                part += f"\nCharacteristics: {definition['characteristics']}"
            
            if 'examples' in definition and definition['examples']:
                examples = definition['examples'][:2]  # Limit to 2 examples
                part += f"\nExamples: {'; '.join(examples)}"
            
            knowledge_parts.append(part)
        
        return '\n\n'.join(knowledge_parts)
        
    def classify_query(self, query: str) -> str:
        """
        Classify a user query into an intent category.
        
        Args:
            query: The user query to classify
            
        Returns:
            The classified intent category name
        """
        if not query or not query.strip():
            return 'direct_query'
        
        try:
            # Create input schema
            input_data = IntentClassificationInput(query=query)
            
            # Use the agent to process the query
            response = self.run(input_data)
            
            # Extract the intent from the response
            if hasattr(response, 'intent') and response.intent:
                predicted_intent = response.intent.lower()
                
                # Validate that the predicted intent is available
                if predicted_intent in self.available_intents:
                    logger.info(f"Atomic Agent classified query as: {predicted_intent}")
                    return predicted_intent
                else:
                    logger.warning(f"Agent returned invalid intent: {predicted_intent}, defaulting to direct_query")
                    return 'direct_query'
            else:
                logger.warning("Agent response missing intent field")
                return 'direct_query'
                
        except Exception as e:
            logger.error(f"Error in Atomic Agent intent classification: {e}")
            return 'direct_query'
