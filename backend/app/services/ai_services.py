import asyncio
from typing import Dict, List, Optional, Any
import structlog
from openai import AsyncOpenAI
from app.core.config import get_settings
from app.services.agents.literature_agent import LiteratureAgent
from app.services.agents.hypothesis_agent import HypothesisAgent
from app.services.agents.experiment_agent import ExperimentAgent
from app.services.agents.execution_agent import ExecutionAgent
from app.services.agents.analysis_agent import AnalysisAgent
from app.services.agents.writing_agent import WritingAgent

logger = structlog.get_logger()


class AIServicesManager:
    """
    Central manager for all AI services in the co-scientist platform.
    Orchestrates the multi-agent system based on research workflows.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.openai_client = None
        self.agents: Dict[str, Any] = {}
        self.initialized = False
    
    async def initialize(self):
        """Initialize all AI services and agents"""
        try:
            logger.info("Initializing AI Services Manager...")
            
            # Initialize OpenAI client
            self.openai_client = AsyncOpenAI(
                api_key=self.settings.OPENAI_API_KEY
            )
            
            # Initialize specialized agents
            await self._initialize_agents()
            
            # Verify services
            await self._verify_services()
            
            self.initialized = True
            logger.info("AI Services Manager initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize AI Services Manager", error=str(e))
            raise
    
    async def _initialize_agents(self):
        """Initialize all specialized AI agents"""
        # Literature Review Agent
        self.agents['literature'] = LiteratureAgent(
            openai_client=self.openai_client,
            settings=self.settings
        )
        
        # Hypothesis Generation Agent
        self.agents['hypothesis'] = HypothesisAgent(
            openai_client=self.openai_client,
            settings=self.settings
        )
        
        # Experiment Design Agent
        self.agents['experiment'] = ExperimentAgent(
            openai_client=self.openai_client,
            settings=self.settings
        )
        
        # Execution Agent (for computational experiments)
        self.agents['execution'] = ExecutionAgent(
            openai_client=self.openai_client,
            settings=self.settings
        )
        
        # Analysis Agent
        self.agents['analysis'] = AnalysisAgent(
            openai_client=self.openai_client,
            settings=self.settings
        )
        
        # Writing Support Agent
        self.agents['writing'] = WritingAgent(
            openai_client=self.openai_client,
            settings=self.settings
        )
        
        # Initialize all agents
        for agent_name, agent in self.agents.items():
            await agent.initialize()
            logger.info(f"Initialized {agent_name} agent")
    
    async def _verify_services(self):
        """Verify that all services are working"""
        try:
            # Test OpenAI connection
            response = await self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Test connection"}],
                max_tokens=10
            )
            logger.info("OpenAI connection verified")
            
        except Exception as e:
            logger.error("Service verification failed", error=str(e))
            raise
    
    async def conduct_literature_review(
        self, 
        query: str, 
        filters: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Conduct comprehensive literature review"""
        if not self.initialized:
            raise RuntimeError("AI services not initialized")
        
        return await self.agents['literature'].conduct_review(query, filters)
    
    async def generate_hypotheses(
        self, 
        research_goal: str, 
        literature_context: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """Generate research hypotheses"""
        if not self.initialized:
            raise RuntimeError("AI services not initialized")
        
        return await self.agents['hypothesis'].generate_hypotheses(
            research_goal, literature_context
        )
    
    async def design_experiments(
        self, 
        hypothesis: Dict[str, Any], 
        constraints: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """Design experiments to test hypotheses"""
        if not self.initialized:
            raise RuntimeError("AI services not initialized")
        
        return await self.agents['experiment'].design_experiments(
            hypothesis, constraints
        )
    
    async def execute_computational_experiment(
        self, 
        experiment: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute computational experiments"""
        if not self.initialized:
            raise RuntimeError("AI services not initialized")
        
        return await self.agents['execution'].execute_experiment(experiment)
    
    async def analyze_results(
        self, 
        experiment_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze experimental results"""
        if not self.initialized:
            raise RuntimeError("AI services not initialized")
        
        return await self.agents['analysis'].analyze_results(experiment_data)
    
    async def generate_research_update(
        self, 
        user_interests: List[str], 
        timeframe: str = "week"
    ) -> Dict[str, Any]:
        """Generate personalized research updates"""
        if not self.initialized:
            raise RuntimeError("AI services not initialized")
        
        return await self.agents['literature'].generate_research_update(
            user_interests, timeframe
        )
    
    async def assist_with_writing(
        self, 
        content_type: str, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assist with scientific writing"""
        if not self.initialized:
            raise RuntimeError("AI services not initialized")
        
        return await self.agents['writing'].assist_writing(content_type, context)
    
    async def collaborative_workflow(
        self, 
        research_goal: str,
        user_preferences: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Run a collaborative multi-agent workflow for complete research process
        """
        if not self.initialized:
            raise RuntimeError("AI services not initialized")
        
        logger.info("Starting collaborative research workflow", goal=research_goal)
        
        workflow_results = {
            "research_goal": research_goal,
            "workflow_id": f"workflow_{asyncio.current_task().get_name()}",
            "steps": []
        }
        
        try:
            # Step 1: Literature Review
            logger.info("Step 1: Conducting literature review")
            lit_review = await self.conduct_literature_review(research_goal)
            workflow_results["steps"].append({
                "step": "literature_review",
                "status": "completed",
                "results": lit_review
            })
            
            # Step 2: Hypothesis Generation
            logger.info("Step 2: Generating hypotheses")
            hypotheses = await self.generate_hypotheses(
                research_goal, 
                literature_context=lit_review
            )
            workflow_results["steps"].append({
                "step": "hypothesis_generation",
                "status": "completed",
                "results": hypotheses
            })
            
            # Step 3: Experiment Design (for top hypothesis)
            if hypotheses:
                logger.info("Step 3: Designing experiments")
                top_hypothesis = hypotheses[0]  # Assume sorted by score
                experiments = await self.design_experiments(top_hypothesis)
                workflow_results["steps"].append({
                    "step": "experiment_design",
                    "status": "completed",
                    "results": experiments
                })
                
                # Step 4: Analysis and Recommendations
                logger.info("Step 4: Generating analysis and recommendations")
                analysis = await self.agents['analysis'].synthesize_research_plan({
                    "literature_review": lit_review,
                    "hypotheses": hypotheses,
                    "experiments": experiments
                })
                workflow_results["steps"].append({
                    "step": "analysis_synthesis",
                    "status": "completed",
                    "results": analysis
                })
            
            workflow_results["status"] = "completed"
            logger.info("Collaborative workflow completed successfully")
            
        except Exception as e:
            logger.error("Workflow failed", error=str(e))
            workflow_results["status"] = "failed"
            workflow_results["error"] = str(e)
        
        return workflow_results
    
    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up AI Services Manager...")
        
        # Cleanup agents
        for agent_name, agent in self.agents.items():
            try:
                await agent.cleanup()
            except Exception as e:
                logger.warning(f"Error cleaning up {agent_name} agent", error=str(e))
        
        # Close OpenAI client
        if self.openai_client:
            await self.openai_client.close()
        
        self.initialized = False
        logger.info("AI Services Manager cleanup completed") 