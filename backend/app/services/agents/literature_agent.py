from typing import Dict, List, Optional, Any
import aiohttp
import asyncio
from datetime import datetime, timedelta
from .base_agent import BaseAgent


class LiteratureAgent(BaseAgent):
    """
    Specialized agent for conducting literature reviews and staying updated on research.
    Integrates with multiple academic databases and provides AI-powered analysis.
    """
    
    def __init__(self, openai_client, settings):
        super().__init__(openai_client, settings, "LiteratureAgent")
        self.session = None
    
    async def _setup(self):
        """Setup HTTP session for API calls"""
        self.session = aiohttp.ClientSession()
    
    async def conduct_review(self, query: str, filters: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Conduct comprehensive literature review on a research topic
        """
        self.logger.info("Starting literature review", query=query)
        
        # Search multiple databases
        search_results = await self._search_multiple_databases(query, filters)
        
        # Analyze and rank papers
        analyzed_papers = await self._analyze_papers(search_results)
        
        # Generate summary and insights
        summary = await self._generate_review_summary(analyzed_papers, query)
        
        return {
            "query": query,
            "total_papers_found": len(search_results),
            "papers_analyzed": len(analyzed_papers),
            "papers": analyzed_papers[:20],  # Top 20 papers
            "summary": summary,
            "key_findings": summary.get("key_findings", []),
            "research_gaps": summary.get("research_gaps", []),
            "trending_topics": summary.get("trending_topics", []),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _search_multiple_databases(self, query: str, filters: Optional[Dict] = None) -> List[Dict]:
        """Search multiple academic databases"""
        search_tasks = [
            self._search_arxiv(query, filters),
            self._search_pubmed(query, filters),
            self._search_semantic_scholar(query, filters)
        ]
        
        results = await asyncio.gather(*search_tasks, return_exceptions=True)
        
        # Combine and deduplicate results
        all_papers = []
        for result in results:
            if isinstance(result, list):
                all_papers.extend(result)
            else:
                self.logger.warning("Database search failed", error=str(result))
        
        # Deduplicate by DOI/title
        seen = set()
        unique_papers = []
        for paper in all_papers:
            identifier = paper.get('doi') or paper.get('title', '').lower()
            if identifier and identifier not in seen:
                seen.add(identifier)
                unique_papers.append(paper)
        
        return unique_papers
    
    async def _search_arxiv(self, query: str, filters: Optional[Dict] = None) -> List[Dict]:
        """Search arXiv database"""
        try:
            url = f"{self.settings.ARXIV_API_URL}"
            params = {
                'search_query': f'all:{query}',
                'start': 0,
                'max_results': 50,
                'sortBy': 'relevance'
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    # Parse arXiv XML response (simplified)
                    text = await response.text()
                    # In a real implementation, you'd parse the XML properly
                    return await self._parse_arxiv_response(text)
                else:
                    self.logger.warning("arXiv search failed", status=response.status)
                    return []
        except Exception as e:
            self.logger.error("arXiv search error", error=str(e))
            return []
    
    async def _search_pubmed(self, query: str, filters: Optional[Dict] = None) -> List[Dict]:
        """Search PubMed database"""
        try:
            # This is a simplified implementation
            # In practice, you'd use the actual PubMed E-utilities API
            return await self._mock_pubmed_search(query)
        except Exception as e:
            self.logger.error("PubMed search error", error=str(e))
            return []
    
    async def _search_semantic_scholar(self, query: str, filters: Optional[Dict] = None) -> List[Dict]:
        """Search Semantic Scholar API"""
        try:
            url = f"{self.settings.SEMANTIC_SCHOLAR_API_URL}/paper/search"
            params = {
                'query': query,
                'limit': 50,
                'fields': 'title,authors,abstract,citationCount,year,journal'
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('data', [])
                else:
                    self.logger.warning("Semantic Scholar search failed", status=response.status)
                    return []
        except Exception as e:
            self.logger.error("Semantic Scholar search error", error=str(e))
            return []
    
    async def _analyze_papers(self, papers: List[Dict]) -> List[Dict]:
        """Analyze papers using AI to extract insights and rank by relevance"""
        analyzed_papers = []
        
        # Process papers in batches
        batch_size = 5
        for i in range(0, len(papers), batch_size):
            batch = papers[i:i + batch_size]
            batch_results = await asyncio.gather(
                *[self._analyze_single_paper(paper) for paper in batch],
                return_exceptions=True
            )
            
            for result in batch_results:
                if isinstance(result, dict):
                    analyzed_papers.append(result)
        
        # Sort by relevance score
        analyzed_papers.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        return analyzed_papers
    
    async def _analyze_single_paper(self, paper: Dict) -> Dict:
        """Analyze a single paper using AI"""
        try:
            title = paper.get('title', '')
            abstract = paper.get('abstract', '')
            
            analysis_prompt = [
                {
                    "role": "system",
                    "content": self.create_system_prompt(
                        "an expert research analyst specializing in scientific literature review",
                        [
                            "Analyze research papers for relevance, novelty, and impact",
                            "Extract key contributions and methodologies",
                            "Identify connections to broader research themes",
                            "Assess the quality and rigor of the research"
                        ]
                    )
                },
                {
                    "role": "user",
                    "content": f"""Analyze this research paper and provide a structured assessment:

Title: {title}
Abstract: {abstract}

Please provide:
1. Relevance score (0-100)
2. Key contributions (bullet points)
3. Methodology summary
4. Potential impact
5. Connections to related work
6. One-sentence summary

Format as JSON."""
                }
            ]
            
            analysis = await self.generate_structured_completion(
                analysis_prompt,
                schema={}  # Define proper schema in production
            )
            
            # Combine original paper data with analysis
            paper_with_analysis = paper.copy()
            paper_with_analysis.update({
                'ai_analysis': analysis,
                'relevance_score': analysis.get('relevance_score', 50),
                'key_contributions': analysis.get('key_contributions', []),
                'methodology_summary': analysis.get('methodology_summary', ''),
                'ai_summary': analysis.get('summary', '')
            })
            
            return paper_with_analysis
            
        except Exception as e:
            self.logger.error("Paper analysis failed", error=str(e))
            paper['relevance_score'] = 0
            return paper
    
    async def _generate_review_summary(self, papers: List[Dict], query: str) -> Dict[str, Any]:
        """Generate comprehensive review summary"""
        try:
            # Prepare paper summaries for analysis
            paper_summaries = []
            for paper in papers[:10]:  # Analyze top 10 papers
                summary = {
                    'title': paper.get('title', ''),
                    'key_contributions': paper.get('key_contributions', []),
                    'methodology': paper.get('methodology_summary', ''),
                    'relevance_score': paper.get('relevance_score', 0)
                }
                paper_summaries.append(summary)
            
            summary_prompt = [
                {
                    "role": "system",
                    "content": self.create_system_prompt(
                        "a research synthesis expert who creates comprehensive literature reviews",
                        [
                            "Identify key themes and patterns across multiple research papers",
                            "Highlight significant findings and methodological approaches",
                            "Identify research gaps and future directions",
                            "Provide actionable insights for researchers"
                        ]
                    )
                },
                {
                    "role": "user",
                    "content": f"""Based on the following research papers about "{query}", create a comprehensive literature review summary:

{str(paper_summaries)}

Please provide:
1. Key findings (3-5 main insights)
2. Research gaps (areas needing more investigation)
3. Trending topics (emerging themes)
4. Methodological approaches (common methods used)
5. Future research directions
6. Overall assessment of the field

Format as JSON."""
                }
            ]
            
            summary = await self.generate_structured_completion(summary_prompt, schema={})
            return summary
            
        except Exception as e:
            self.logger.error("Review summary generation failed", error=str(e))
            return {
                "key_findings": [],
                "research_gaps": [],
                "trending_topics": [],
                "error": "Failed to generate summary"
            }
    
    async def generate_research_update(
        self, 
        user_interests: List[str], 
        timeframe: str = "week"
    ) -> Dict[str, Any]:
        """Generate personalized research updates for users"""
        self.logger.info("Generating research update", interests=user_interests, timeframe=timeframe)
        
        updates = []
        for interest in user_interests:
            # Search for recent papers in this area
            papers = await self._search_recent_papers(interest, timeframe)
            if papers:
                update = {
                    "topic": interest,
                    "new_papers_count": len(papers),
                    "top_papers": papers[:3],
                    "summary": await self._summarize_recent_developments(papers, interest)
                }
                updates.append(update)
        
        return {
            "timeframe": timeframe,
            "interests": user_interests,
            "updates": updates,
            "generated_at": datetime.utcnow().isoformat()
        }
    
    async def _search_recent_papers(self, topic: str, timeframe: str) -> List[Dict]:
        """Search for recent papers in a specific timeframe"""
        # Calculate date filter based on timeframe
        days_back = {"day": 1, "week": 7, "month": 30}.get(timeframe, 7)
        cutoff_date = datetime.utcnow() - timedelta(days=days_back)
        
        # Search with date filter
        papers = await self._search_multiple_databases(
            topic, 
            filters={"start_date": cutoff_date.isoformat()}
        )
        
        return papers
    
    async def _summarize_recent_developments(self, papers: List[Dict], topic: str) -> str:
        """Summarize recent developments in a research area"""
        try:
            titles_and_abstracts = []
            for paper in papers[:5]:
                titles_and_abstracts.append(f"Title: {paper.get('title', '')}\nAbstract: {paper.get('abstract', '')[:300]}...")
            
            prompt = [
                {
                    "role": "system",
                    "content": "You are a research analyst who summarizes recent developments in scientific fields."
                },
                {
                    "role": "user",
                    "content": f"""Summarize the recent developments in {topic} based on these papers:

{chr(10).join(titles_and_abstracts)}

Provide a concise 2-3 sentence summary of the key developments and trends."""
                }
            ]
            
            summary = await self.generate_completion(prompt)
            return summary
        except Exception as e:
            self.logger.error("Failed to summarize developments", error=str(e))
            return f"Recent papers found in {topic}, but summary generation failed."
    
    # Mock implementations for development
    async def _parse_arxiv_response(self, xml_text: str) -> List[Dict]:
        """Mock arXiv response parser"""
        # In production, implement proper XML parsing
        return [
            {
                "title": f"Mock arXiv Paper on Research Topic",
                "authors": ["Author A", "Author B"],
                "abstract": "This is a mock abstract for development purposes.",
                "arxiv_id": "2024.0001",
                "publication_date": datetime.utcnow().isoformat()
            }
        ]
    
    async def _mock_pubmed_search(self, query: str) -> List[Dict]:
        """Mock PubMed search"""
        return [
            {
                "title": f"Mock PubMed Paper: {query}",
                "authors": ["Researcher C", "Researcher D"],
                "abstract": "Mock abstract from PubMed search for development.",
                "pubmed_id": "12345678",
                "journal": "Mock Journal",
                "publication_date": datetime.utcnow().isoformat()
            }
        ]
    
    async def cleanup(self):
        """Cleanup agent resources"""
        await super().cleanup()
        if self.session:
            await self.session.close() 