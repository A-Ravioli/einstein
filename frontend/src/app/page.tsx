"use client";

import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { 
  BookOpen, 
  Lightbulb, 
  FlaskConical, 
  BarChart3, 
  PenTool, 
  Rss,
  ArrowRight,
  Sparkles,
  Brain,
  Database
} from "lucide-react";
import { apiClient } from "@/lib/utils";
import { useQuery, useMutation } from "@tanstack/react-query";

export default function Home() {
  const [researchGoal, setResearchGoal] = useState("");
  const [isStartingWorkflow, setIsStartingWorkflow] = useState(false);
  const [workflowResult, setWorkflowResult] = useState<any>(null);

  // Start research workflow
  const startWorkflowMutation = useMutation({
    mutationFn: async (goal: string) => {
      const response = await apiClient.post("/research/workflow", {
        research_goal: goal,
        include_literature_review: true,
        include_hypothesis_generation: true,
        include_experiment_design: true,
        max_hypotheses: 5
      });
      return response.data;
    },
    onSuccess: (data) => {
      setWorkflowResult(data);
      setIsStartingWorkflow(false);
    },
    onError: (error) => {
      console.error("Workflow failed:", error);
      setIsStartingWorkflow(false);
    }
  });

  const handleStartWorkflow = () => {
    if (!researchGoal.trim()) return;
    setIsStartingWorkflow(true);
    startWorkflowMutation.mutate(researchGoal);
  };

  const features = [
    {
      icon: <BookOpen className="h-8 w-8" />,
      title: "Literature Review",
      description: "AI-powered analysis of research papers across multiple databases with intelligent summarization and gap identification.",
      color: "from-blue-500 to-blue-600"
    },
    {
      icon: <Lightbulb className="h-8 w-8" />,
      title: "Hypothesis Generation", 
      description: "Generate novel, testable research hypotheses based on literature analysis and research goals.",
      color: "from-yellow-500 to-yellow-600"
    },
    {
      icon: <FlaskConical className="h-8 w-8" />,
      title: "Experiment Design",
      description: "Design rigorous experiments with detailed protocols, controls, and measurement strategies.",
      color: "from-green-500 to-green-600"
    },
    {
      icon: <BarChart3 className="h-8 w-8" />,
      title: "Data Analysis",
      description: "Advanced statistical analysis and visualization of experimental results with AI-powered insights.",
      color: "from-purple-500 to-purple-600"
    },
    {
      icon: <PenTool className="h-8 w-8" />,
      title: "Writing Support",
      description: "AI assistance for scientific writing including abstracts, papers, and grant proposals.",
      color: "from-red-500 to-red-600"
    },
    {
      icon: <Rss className="h-8 w-8" />,
      title: "Research Updates",
      description: "Personalized updates on latest developments in your research areas with trending insights.",
      color: "from-indigo-500 to-indigo-600"
    }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100">
      {/* Header */}
      <header className="border-b bg-white/50 backdrop-blur-sm">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="relative">
                <Brain className="h-8 w-8 text-blue-600" />
                <Sparkles className="h-4 w-4 text-yellow-500 absolute -top-1 -right-1" />
              </div>
              <div>
                <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                  AI Co-Scientist
                </h1>
                <p className="text-sm text-gray-600">Advanced Research Platform</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <Button variant="ghost">Login</Button>
              <Button>Get Started</Button>
            </div>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <section className="py-20">
        <div className="container mx-auto px-4 text-center">
          <div className="max-w-4xl mx-auto">
            <h2 className="text-5xl font-bold mb-6 bg-gradient-to-r from-gray-900 via-blue-900 to-purple-900 bg-clip-text text-transparent">
              Accelerate Scientific Discovery with AI
            </h2>
            <p className="text-xl text-gray-600 mb-8 leading-relaxed">
              Our multi-agent AI system conducts literature reviews, generates hypotheses, designs experiments, 
              and provides insights—transforming how researchers approach scientific discovery.
            </p>
            
            {/* Research Goal Input */}
            <Card className="max-w-2xl mx-auto mb-8 border-2 border-blue-100 shadow-lg">
              <CardHeader>
                <CardTitle className="text-lg">Start Your Research Journey</CardTitle>
                <CardDescription>
                  Describe your research goal and let our AI agents create a comprehensive research plan
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <Textarea
                  placeholder="Example: Investigate the role of protein aggregation in neurodegenerative diseases..."
                  value={researchGoal}
                  onChange={(e) => setResearchGoal(e.target.value)}
                  className="min-h-[100px]"
                />
                <Button 
                  onClick={handleStartWorkflow}
                  disabled={!researchGoal.trim() || isStartingWorkflow}
                  className="w-full bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700"
                  size="lg"
                >
                  {isStartingWorkflow ? (
                    <div className="flex items-center space-x-2">
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                      <span>Starting AI Workflow...</span>
                    </div>
                  ) : (
                    <div className="flex items-center space-x-2">
                      <Sparkles className="h-4 w-4" />
                      <span>Start AI Research Workflow</span>
                      <ArrowRight className="h-4 w-4" />
                    </div>
                  )}
                </Button>
              </CardContent>
            </Card>

            {/* Workflow Result */}
            {workflowResult && (
              <Card className="max-w-4xl mx-auto mb-8 border-green-200 bg-green-50">
                <CardHeader>
                  <CardTitle className="text-green-800">Research Workflow Initiated!</CardTitle>
                  <CardDescription className="text-green-600">
                    Status: {workflowResult.status} | Workflow ID: {workflowResult.workflow_id}
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                    {workflowResult.steps_planned?.map((step: string, index: number) => (
                      <div key={index} className="text-center p-3 bg-white rounded-lg">
                        <div className="text-sm font-medium text-gray-800">{step}</div>
                        <div className="text-xs text-gray-600 mt-1">Planned</div>
                      </div>
                    ))}
                  </div>
                  <p className="text-sm text-green-700">
                    Estimated completion: {workflowResult.estimated_completion}
                  </p>
                  {workflowResult.results && (
                    <div className="mt-4 p-4 bg-white rounded-lg">
                      <h4 className="font-semibold mb-2">Results:</h4>
                      <pre className="text-xs text-gray-600 overflow-auto">
                        {JSON.stringify(workflowResult.results, null, 2)}
                      </pre>
                    </div>
                  )}
                </CardContent>
              </Card>
            )}
          </div>
        </div>
      </section>

      {/* Features Grid */}
      <section className="py-20 bg-white">
        <div className="container mx-auto px-4">
          <div className="text-center mb-16">
            <h3 className="text-3xl font-bold mb-4">Comprehensive AI Research Suite</h3>
            <p className="text-gray-600 max-w-2xl mx-auto">
              Our platform integrates cutting-edge AI agents to support every stage of scientific research,
              from initial literature review to final publication.
            </p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {features.map((feature, index) => (
              <Card key={index} className="group hover:shadow-lg transition-all duration-300 border-0 shadow-md">
                <CardHeader>
                  <div className={`w-16 h-16 rounded-xl bg-gradient-to-r ${feature.color} flex items-center justify-center text-white mb-4 group-hover:scale-110 transition-transform duration-300`}>
                    {feature.icon}
                  </div>
                  <CardTitle className="text-xl">{feature.title}</CardTitle>
                </CardHeader>
                <CardContent>
                  <CardDescription className="text-gray-600 leading-relaxed">
                    {feature.description}
                  </CardDescription>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="py-20 bg-gradient-to-r from-blue-600 to-purple-600 text-white">
        <div className="container mx-auto px-4 text-center">
          <h3 className="text-3xl font-bold mb-12">Powered by Advanced AI</h3>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
            <div>
              <div className="text-4xl font-bold mb-2">200M+</div>
              <div className="text-blue-100">Research Papers Indexed</div>
            </div>
            <div>
              <div className="text-4xl font-bold mb-2">6</div>
              <div className="text-blue-100">Specialized AI Agents</div>
            </div>
            <div>
              <div className="text-4xl font-bold mb-2">50+</div>
              <div className="text-blue-100">Academic Databases</div>
            </div>
            <div>
              <div className="text-4xl font-bold mb-2">99%</div>
              <div className="text-blue-100">Research Accuracy</div>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-gray-900 text-white py-12">
        <div className="container mx-auto px-4">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
            <div>
              <div className="flex items-center space-x-2 mb-4">
                <Brain className="h-6 w-6 text-blue-400" />
                <span className="text-xl font-bold">AI Co-Scientist</span>
              </div>
              <p className="text-gray-400">
                Accelerating scientific discovery through advanced AI collaboration.
              </p>
            </div>
            <div>
              <h4 className="font-semibold mb-4">Platform</h4>
              <ul className="space-y-2 text-gray-400">
                <li>Literature Review</li>
                <li>Hypothesis Generation</li>
                <li>Experiment Design</li>
                <li>Data Analysis</li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold mb-4">Resources</h4>
              <ul className="space-y-2 text-gray-400">
                <li>Documentation</li>
                <li>API Reference</li>
                <li>Research Guide</li>
                <li>Support</li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold mb-4">Connect</h4>
              <ul className="space-y-2 text-gray-400">
                <li>GitHub</li>
                <li>Research Community</li>
                <li>Contact Us</li>
                <li>Updates</li>
              </ul>
            </div>
          </div>
          <div className="border-t border-gray-800 mt-8 pt-8 text-center text-gray-400">
            <p>&copy; 2024 AI Co-Scientist Platform. Built for the scientific community.</p>
          </div>
        </div>
      </footer>
    </div>
  );
}
