# ==============================================================================
# 🌐 AI Interview Simulator - Web UI Application
# 
# Flask-based web interface for the CrewAI Interview Simulator
# Features: Modern UI, Real-time progress, File downloads, Results dashboard
# ==============================================================================

import os
import json
import threading
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
from werkzeug.utils import secure_filename
import time

# Import our existing CrewAI simulator
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

# Global variables for progress tracking
current_progress = {"status": "idle", "step": "", "progress": 0, "result": None}
simulation_thread = None

class WebInterviewSimulator:
    """Web-enabled version of the AI Interview Simulator."""
    
    def __init__(self):
        self.llm = None
        self.agents = {}
        self.tasks = []
        self.result = None
        
    def setup_llm(self, api_key):
        """Setup the language model with provided API key."""
        os.environ['OPENAI_API_KEY'] = api_key
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.7,
            max_tokens=2500
        )
        return True
    
    def create_agents(self, job_details):
        """Create the specialized interview agents."""
        global current_progress
        current_progress.update({"step": "Creating AI Agents...", "progress": 20})
        
        # HR Agent
        self.agents['hr'] = Agent(
            role='Senior HR Interview Specialist',
            goal=f'Conduct comprehensive behavioral assessment for {job_details["position"]} role',
            backstory=(
                "You are Sarah Martinez, a Senior HR Business Partner with 15+ years of experience "
                "in talent acquisition. You specialize in behavioral interviewing and cultural fit assessment."
            ),
            llm=self.llm,
            verbose=False,
            memory=True,
            allow_delegation=False
        )
        
        # Technical Agent
        self.agents['tech'] = Agent(
            role='Senior Technical Interview Lead',
            goal=f'Evaluate technical competency and problem-solving for {job_details["position"]} role',
            backstory=(
                f"You are Dr. Alex Chen, a Technical Lead with 12+ years of experience "
                f"building systems using {job_details.get('tech_stack', 'modern technologies')}. "
                "You have conducted over 500 technical interviews."
            ),
            llm=self.llm,
            verbose=False,
            memory=True,
            allow_delegation=False
        )
        
        # Assessment Agent
        self.agents['feedback'] = Agent(
            role='Senior Interview Assessment Director',
            goal='Synthesize multi-perspective feedback into comprehensive hiring recommendations',
            backstory=(
                "You are Dr. Morgan Taylor, Director of Interview Assessment with a Ph.D. in "
                "Organizational Psychology. You excel at creating actionable hiring insights."
            ),
            llm=self.llm,
            verbose=False,
            memory=True,
            allow_delegation=False
        )
        
        current_progress.update({"step": "Agents Created Successfully", "progress": 30})
    
    def create_tasks(self, interview_data):
        """Create comprehensive interview tasks."""
        global current_progress
        current_progress.update({"step": "Creating Interview Tasks...", "progress": 40})
        
        job_details = interview_data['job_details']
        candidate_info = interview_data['candidate_info']
        
        # HR Task
        hr_task = Task(
            description=(
                f"Conduct HR assessment for {candidate_info['name']} applying for {job_details['position']}.\n\n"
                f"Candidate Responses: {interview_data['hr_responses']}\n\n"
                "Rate 1-10 each: Communication, Cultural Fit, Leadership, Adaptability, "
                "Problem-Solving, Growth Mindset, Emotional Intelligence, Conflict Resolution.\n\n"
                "Provide detailed analysis with specific examples and cultural fit assessment."
            ),
            expected_output=(
                "HR ASSESSMENT REPORT with overall score /80, detailed ratings for each area, "
                "key insights, and hire/no-hire recommendation with reasoning."
            ),
            agent=self.agents['hr']
        )
        
        # Technical Task
        tech_task = Task(
            description=(
                f"Conduct technical assessment for {candidate_info['name']} for {job_details['position']}.\n\n"
                f"Required Technologies: {job_details['tech_stack']}\n"
                f"Technical Responses: {interview_data['tech_responses']}\n\n"
                "Rate 1-10 each: Core Knowledge, Problem-Solving, System Design, Code Quality, "
                "Technology Breadth, Practical Experience, Learning Ability, Debugging, "
                "Performance Optimization, Collaboration.\n\n"
                "Evaluate technical depth vs required level and identify growth potential."
            ),
            expected_output=(
                "TECHNICAL ASSESSMENT REPORT with overall score /100, detailed technical ratings, "
                "competency analysis, knowledge gaps, and technical recommendation."
            ),
            agent=self.agents['tech']
        )
        
        # Final Assessment Task
        final_task = Task(
            description=(
                f"Generate comprehensive hiring assessment for {candidate_info['name']}.\n\n"
                "Synthesize HR and technical assessments to provide:\n"
                "- Executive summary with clear recommendation\n"
                "- Overall scoring (HR + Technical = Total /180)\n"
                "- Key strengths and development areas\n"
                "- Hiring decision with confidence level\n"
                "- Compensation and level recommendations\n"
                "- 30-60-90 day onboarding plan\n"
                "- Risk assessment and next steps"
            ),
            expected_output=(
                "COMPREHENSIVE ASSESSMENT REPORT with executive summary, total score interpretation, "
                "detailed recommendations, onboarding plan, and actionable next steps."
            ),
            agent=self.agents['feedback'],
            context=[hr_task, tech_task]
        )
        
        self.tasks = [hr_task, tech_task, final_task]
        current_progress.update({"step": "Tasks Created Successfully", "progress": 50})
    
    def run_simulation(self, interview_data):
        """Execute the complete interview simulation."""
        global current_progress
        
        try:
            current_progress.update({"step": "Starting Interview Simulation...", "progress": 60})
            
            # Create agents and tasks
            self.create_agents(interview_data['job_details'])
            self.create_tasks(interview_data)
            
            # Create crew
            current_progress.update({"step": "Assembling Interview Crew...", "progress": 70})
            
            crew = Crew(
                agents=list(self.agents.values()),
                tasks=self.tasks,
                process=Process.sequential,
                verbose=False,
                memory=True
            )
            
            # Execute simulation
            current_progress.update({"step": "Running Multi-Agent Assessment...", "progress": 80})
            
            result = crew.kickoff()
            
            # Save result
            current_progress.update({"step": "Generating Report...", "progress": 90})
            
            # Prepare final result
            final_result = {
                'assessment': str(result),
                'candidate_info': interview_data['candidate_info'],
                'job_details': interview_data['job_details'],
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'filename': self.generate_filename(interview_data)
            }
            
            self.result = final_result
            current_progress.update({
                "status": "completed", 
                "step": "Assessment Complete!", 
                "progress": 100,
                "result": final_result
            })
            
            return final_result
            
        except Exception as e:
            current_progress.update({
                "status": "error",
                "step": f"Error: {str(e)}",
                "progress": 0,
                "result": None
            })
            return None
    
    def generate_filename(self, interview_data):
        """Generate filename for the report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        candidate_name = interview_data['candidate_info']['name'].replace(' ', '_')
        position = interview_data['job_details']['position'].replace(' ', '_')
        return f"Interview_Assessment_{candidate_name}_{position}_{timestamp}"

# Global simulator instance
simulator = WebInterviewSimulator()

@app.route('/')
def index():
    """Main dashboard page."""
    return render_template('index.html')

@app.route('/setup')
def setup():
    """Setup and configuration page."""
    return render_template('setup.html')

@app.route('/results')
def results():
    """Results and download page."""
    return render_template('results.html')

@app.route('/api/start_simulation', methods=['POST'])
def start_simulation():
    """API endpoint to start the interview simulation."""
    global simulation_thread, current_progress
    
    try:
        data = request.json
        
        # Setup API key
        api_key = data.get('api_key')
        if not api_key:
            return jsonify({"error": "API key is required"}), 400
        
        if not simulator.setup_llm(api_key):
            return jsonify({"error": "Failed to setup language model"}), 400
        
        # Reset progress
        current_progress = {"status": "running", "step": "Initializing...", "progress": 10, "result": None}
        
        # Start simulation in background thread
        simulation_thread = threading.Thread(
            target=simulator.run_simulation,
            args=(data['interview_data'],)
        )
        simulation_thread.start()
        
        return jsonify({"message": "Simulation started successfully"})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/progress')
def get_progress():
    """Get current simulation progress."""
    return jsonify(current_progress)

@app.route('/api/download/<format>')
def download_report(format):
    """Download the assessment report in specified format."""
    global current_progress
    
    if current_progress["status"] != "completed" or not current_progress["result"]:
        return jsonify({"error": "No completed assessment available"}), 404
    
    result = current_progress["result"]
    filename = result['filename']
    
    try:
        if format == 'markdown':
            return download_markdown(result, filename)
        elif format == 'pdf':
            return download_pdf(result, filename)
        elif format == 'json':
            return download_json(result, filename)
        else:
            return jsonify({"error": "Invalid format"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def download_markdown(result, filename):
    """Generate and download markdown report."""
    content = f"""# AI Interview Assessment Report

## Interview Overview
**Generated:** {result['timestamp']}
**System:** AI Interview Simulator v3.0

**Candidate:** {result['candidate_info']['name']}
**Position:** {result['job_details']['position']}
**Company:** {result['job_details']['company']}
**Department:** {result['job_details']['department']}
**Required Skills:** {result['job_details']['tech_stack']}

---

## Comprehensive Assessment

{result['assessment']}

---

*Report generated by AI Interview Simulator Multi-Agent System*
"""
    
    # Save to file
    filepath = f"reports/{filename}.md"
    os.makedirs('reports', exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return send_file(filepath, as_attachment=True, download_name=f"{filename}.md")

def download_pdf(result, filename):
    """Generate and download PDF report (placeholder)."""
    # For now, return markdown with PDF extension
    # In production, you'd use libraries like reportlab or weasyprint
    return download_markdown(result, filename)

def download_json(result, filename):
    """Generate and download JSON report."""
    # Save to file
    filepath = f"reports/{filename}.json"
    os.makedirs('reports', exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    return send_file(filepath, as_attachment=True, download_name=f"{filename}.json")

@app.route('/api/test_connection', methods=['POST'])
def test_connection():
    """Test API key and connection."""
    try:
        api_key = request.json.get('api_key')
        if not api_key:
            return jsonify({"error": "API key is required"}), 400
        
        # Test connection
        os.environ['OPENAI_API_KEY'] = api_key
        test_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, max_tokens=50)
        
        # Simple test
        response = test_llm.invoke("Say 'Connection successful!'")
        
        return jsonify({"message": "Connection successful!", "test_response": str(response.content)})
        
    except Exception as e:
        return jsonify({"error": f"Connection failed: {str(e)}"}), 400

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('reports', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    
    print("🌐 AI Interview Simulator Web UI")
    print("🚀 Starting Flask server...")
    print("📍 Access at: http://localhost:5000")
    
    app.run(debug=True, port=5000)