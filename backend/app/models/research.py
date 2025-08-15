from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, ForeignKey, JSON, Float, Enum
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from enum import Enum as PyEnum
from app.core.database import Base


class ResearchStatus(PyEnum):
    DRAFT = "draft"
    ACTIVE = "active"
    COMPLETED = "completed"
    ARCHIVED = "archived"


class ExperimentStatus(PyEnum):
    PLANNED = "planned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ResearchProject(Base):
    __tablename__ = "research_projects"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False, index=True)
    description = Column(Text, nullable=True)
    research_goal = Column(Text, nullable=False)
    status = Column(Enum(ResearchStatus), default=ResearchStatus.DRAFT)
    
    # Research domain
    field = Column(String, nullable=True)  # Biology, Chemistry, Physics, etc.
    subfield = Column(String, nullable=True)
    keywords = Column(JSON, nullable=True)  # List of keywords
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    owner_id = Column(Integer, ForeignKey("users.id"))
    owner = relationship("User", back_populates="research_projects")
    
    literature_reviews = relationship("LiteratureReview", back_populates="project")
    hypotheses = relationship("Hypothesis", back_populates="project")
    experiments = relationship("Experiment", back_populates="project")


class LiteratureReview(Base):
    __tablename__ = "literature_reviews"
    
    id = Column(Integer, primary_key=True, index=True)
    query = Column(String, nullable=False)
    summary = Column(Text, nullable=True)
    key_findings = Column(JSON, nullable=True)  # List of key findings
    gaps_identified = Column(JSON, nullable=True)  # Research gaps
    
    # Search parameters
    databases_searched = Column(JSON, nullable=True)  # List of databases
    search_filters = Column(JSON, nullable=True)
    total_papers_found = Column(Integer, default=0)
    papers_reviewed = Column(Integer, default=0)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    user_id = Column(Integer, ForeignKey("users.id"))
    user = relationship("User", back_populates="literature_reviews")
    
    project_id = Column(Integer, ForeignKey("research_projects.id"))
    project = relationship("ResearchProject", back_populates="literature_reviews")
    
    papers = relationship("Paper", back_populates="literature_review")


class Paper(Base):
    __tablename__ = "papers"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    authors = Column(JSON, nullable=True)  # List of authors
    abstract = Column(Text, nullable=True)
    doi = Column(String, nullable=True, index=True)
    arxiv_id = Column(String, nullable=True, index=True)
    pubmed_id = Column(String, nullable=True, index=True)
    
    # Publication details
    journal = Column(String, nullable=True)
    publication_date = Column(DateTime, nullable=True)
    citation_count = Column(Integer, default=0)
    
    # Analysis
    relevance_score = Column(Float, nullable=True)
    ai_summary = Column(Text, nullable=True)
    key_contributions = Column(JSON, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    literature_review_id = Column(Integer, ForeignKey("literature_reviews.id"))
    literature_review = relationship("LiteratureReview", back_populates="papers")


class Hypothesis(Base):
    __tablename__ = "hypotheses"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    statement = Column(Text, nullable=False)
    rationale = Column(Text, nullable=True)
    
    # Hypothesis evaluation
    testability_score = Column(Float, nullable=True)
    novelty_score = Column(Float, nullable=True)
    feasibility_score = Column(Float, nullable=True)
    impact_score = Column(Float, nullable=True)
    overall_score = Column(Float, nullable=True)
    
    # AI-generated content
    suggested_experiments = Column(JSON, nullable=True)
    potential_challenges = Column(JSON, nullable=True)
    related_work = Column(JSON, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    project_id = Column(Integer, ForeignKey("research_projects.id"))
    project = relationship("ResearchProject", back_populates="hypotheses")
    
    experiments = relationship("Experiment", back_populates="hypothesis")


class Experiment(Base):
    __tablename__ = "experiments"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    description = Column(Text, nullable=False)
    methodology = Column(Text, nullable=True)
    
    # Experiment details
    status = Column(Enum(ExperimentStatus), default=ExperimentStatus.PLANNED)
    experiment_type = Column(String, nullable=True)  # computational, wet_lab, simulation, etc.
    estimated_duration = Column(Integer, nullable=True)  # in hours
    required_resources = Column(JSON, nullable=True)
    
    # Results
    results = Column(JSON, nullable=True)
    conclusions = Column(Text, nullable=True)
    statistical_analysis = Column(JSON, nullable=True)
    
    # Automation details (for computational experiments)
    automation_script = Column(Text, nullable=True)
    execution_log = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    user_id = Column(Integer, ForeignKey("users.id"))
    user = relationship("User", back_populates="experiments")
    
    project_id = Column(Integer, ForeignKey("research_projects.id"))
    project = relationship("ResearchProject", back_populates="experiments")
    
    hypothesis_id = Column(Integer, ForeignKey("hypotheses.id"), nullable=True)
    hypothesis = relationship("Hypothesis", back_populates="experiments") 