import os
from typing import Dict, Any
from langchain_groq import ChatGroq
import streamlit as st
from crewai import Agent, Task, Crew, Process
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

@st.cache_resource
def get_llm():
    # Initialize Groq client
    groq_api_key = os.environ.get("GROQ_API_KEY", "")
    if not groq_api_key:
        st.error("GROQ_API_KEY not found in environment variables. Please check your .env file.")
        st.stop()
        
    # Configure the Groq LLM with optimized parameters for educational content
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="groq/llama3-70b-8192",  # Using the most capable model
        temperature=0.3,  # Lower temperature for more consistent, factual responses
        max_tokens=4096,  # Sufficient for detailed explanations
        top_p=0.9,  # Slightly focused but still creative
    )
    return llm

# CrewAI Agents and Tasks
def create_study_tutor_agent():
    return Agent(
        role="Study Tutor",
        goal="Explain complex concepts clearly and help students understand course material with excellent pedagogical techniques",
        backstory="""You are an expert educator with years of experience breaking down difficult concepts into understandable explanations. 
        You excel at adapting your teaching style to match different learning preferences.
        
        CORE PRINCIPLES:
        - Always provide clear, structured explanations with examples
        - Use analogies and real-world connections when helpful
        - Break complex topics into digestible chunks
        - Encourage critical thinking with follow-up questions
        - Adapt explanations based on apparent student level
        - Provide multiple perspectives on difficult concepts
        
        SAFETY GUIDELINES:
        - Only answer questions related to educational content
        - If asked about inappropriate topics, redirect to study material
        - Maintain a professional, encouraging tone
        - Never provide information that could be harmful or dangerous
        - Focus on the uploaded document content when available
        
        RESPONSE FORMAT:
        - Start with a clear, concise answer
        - Provide detailed explanation with examples
        - Include relevant context from the study material
        - End with a question to check understanding or suggest next steps
        
        Remember: Your role is to be a supportive, knowledgeable tutor who helps students learn effectively.""",
        llm=get_llm(),
        verbose=True
    )

def create_note_taker_agent():
    return Agent(
        role="Note-Taker",
        goal="Create organized, comprehensive study notes that enhance learning and retention",
        backstory="""You specialize in creating concise yet comprehensive notes that highlight key concepts, definitions, examples, and connections between ideas. 
        Your notes are well-structured with clear headings and logical flow.
        
        NOTE CREATION PRINCIPLES:
        - Use clear hierarchical structure with headings and subheadings
        - Highlight key terms and definitions prominently
        - Include relevant examples and case studies
        - Show relationships between concepts
        - Use bullet points and numbered lists for clarity
        - Add memory aids and mnemonics where helpful
        
        FORMATTING GUIDELINES:
        - Use markdown formatting for structure
        - Bold important concepts and terms
        - Use bullet points for lists and key points
        - Include numbered steps for processes
        - Add relevant quotes or important statements
        - Create summary sections for major topics
        
        QUALITY STANDARDS:
        - Ensure accuracy and completeness
        - Maintain logical flow and organization
        - Focus on the most important information
        - Make notes scannable and easy to review
        - Include cross-references to related topics""",
        llm=get_llm(),
        verbose=True
    )

def create_assessment_expert_agent():
    return Agent(
        role="Assessment Expert",
        goal="Design comprehensive tests that effectively evaluate student understanding across multiple cognitive levels",
        backstory="""You are a skilled educational assessment expert with years of experience creating varied 
        assessment questions that test different levels of knowledge - from basic recall to complex application and analysis.
        
        ASSESSMENT PRINCIPLES:
        - Create questions that test understanding, not just memorization
        - Use Bloom's Taxonomy to vary cognitive levels (Remember, Understand, Apply, Analyze, Evaluate, Create)
        - Include multiple question types for comprehensive assessment
        - Ensure questions are clear, unambiguous, and fair
        - Provide detailed explanations for all answers
        
        QUESTION TYPES TO INCLUDE:
        - Multiple choice with plausible distractors
        - True/False with explanation requirements
        - Short answer questions
        - Essay/long-form questions
        - Application/scenario-based problems
        - Analysis and comparison questions
        
        DIFFICULTY LEVELS:
        - Easy: Basic recall and simple understanding
        - Medium: Application and analysis of concepts
        - Hard: Synthesis, evaluation, and complex problem-solving
        
        ANSWER KEY REQUIREMENTS:
        - Provide complete, detailed explanations
        - Include reasoning for why answers are correct
        - Explain why incorrect options are wrong (for multiple choice)
        - Offer additional context and learning points
        - Suggest follow-up study areas
        
        QUALITY STANDARDS:
        - Questions must be directly related to the study material
        - Maintain appropriate difficulty progression
        - Ensure cultural sensitivity and accessibility
        - Avoid trick questions or ambiguous wording
        - Include practical applications when relevant""",
        llm=get_llm(),
        verbose=True
    )



def create_learning_coach_agent():
    return Agent(
        role="Learning Coach",
        goal="Analyze performance and suggest learning improvements",
        backstory="You specialize in analyzing learning patterns and progress to provide targeted feedback and improvement strategies. Your coaching helps students identify and overcome knowledge gaps.",
        llm=get_llm(),
        verbose=True
    )

def create_roadmap_planner_agent():
    return Agent(
        role="Study Roadmap Planner",
        goal="Create structured study plans with clear timelines and milestones",
        backstory="You are an expert in educational planning with years of experience creating effective study roadmaps. You excel at breaking down complex materials into manageable learning paths with realistic timeframes.",
        llm=get_llm(),
        verbose=True
    )

def create_flashcard_specialist_agent():
    return Agent(
        role="Flashcard Specialist",
        goal="Create effective memory aids through well-crafted flashcards",
        backstory="You excel at distilling complex information into concise flashcards that facilitate memorization and recall. You know how to balance brevity with clarity to create effective study tools.",
        llm=get_llm(),
        verbose=True
    )

def create_visual_learning_expert_agent():
    return Agent(
        role="Visual Learning Expert",
        goal="Transform topics into visual mind maps that show relationships between concepts",
        backstory="You have expertise in visual learning techniques and can organize information into clear, meaningful visual representations. You excel at identifying key relationships between concepts and presenting them graphically.",
        llm=get_llm(),
        verbose=True
    )



def create_explanation_task(agent, question, context):
    return Task(
        description=f"""As an expert tutor, provide a comprehensive yet accessible explanation for the student's question.
        
        STUDENT QUESTION: {question}
        
        AVAILABLE CONTEXT FROM STUDY MATERIAL:
        {context}
        
        INSTRUCTION GUIDELINES:
        1. CONTENT VALIDATION:
           - Only answer if the question relates to educational/academic content
           - If the question is inappropriate or off-topic, politely redirect to study material
           - If you cannot answer based on the context, clearly state this limitation
        
        2. EXPLANATION STRUCTURE:
           - Start with a direct, clear answer to the question
           - Provide detailed explanation using the context material
           - Use examples, analogies, or real-world connections when helpful
           - Break down complex concepts into simpler parts
           - Connect the explanation to broader concepts when relevant
        
        3. PEDAGOGICAL APPROACH:
           - Adapt language level to match the complexity of the question
           - Use active learning techniques (pose follow-up questions)
           - Encourage deeper thinking about the topic
           - Provide multiple perspectives if applicable
           - Suggest practical applications or next steps
        
        4. RESPONSE FORMAT:
           - Use clear, structured formatting with headings if needed
           - Bold key terms and important concepts
           - Use bullet points for lists and key ideas
           - Include relevant quotes or excerpts from the material
           - End with a question to check understanding or encourage further exploration
        
        5. SAFETY AND APPROPRIATENESS:
           - Maintain a professional, encouraging tone
           - Focus on educational value and learning outcomes
           - If unsure about accuracy, clearly state uncertainty
           - Avoid speculation beyond the provided material
        
        Remember: Your goal is to help the student truly understand the concept, not just provide information.""",
        expected_output="""A well-structured, educational response that includes:
        - Clear, direct answer to the question
        - Detailed explanation with examples
        - Connections to the study material
        - Follow-up question or suggestion for further learning
        - Professional, encouraging tone throughout""",
        agent=agent
    )

def create_notes_generation_task(agent, topic, context):
    return Task(
        description=f"""Create comprehensive, well-structured study notes on the following topic. 
        Include key concepts, definitions, examples, and relationships between ideas. 
        Organize with clear headings and subheadings.
        
        Topic: {topic}
        
        Context information:
        {context}
        """,
        expected_output="Well-structured study notes in markdown format with headings, bullet points, and emphasis on key concepts.",
        agent=agent
    )

def create_test_generation_task(agent, topic, difficulty, context):
    return Task(
        description=f"""Create a comprehensive practice test on the specified topic with proper structure and detailed answer explanations.
        
        TOPIC: {topic}
        DIFFICULTY LEVEL: {difficulty}
        
        CONTEXT FROM STUDY MATERIAL:
        {context}
        
        TEST REQUIREMENTS:
        1. Generate 8-12 questions of varied types:
           - 4-5 Multiple choice questions (with 4 options each)
           - 2-3 True/False questions (with explanation required)
           - 2-3 Short answer questions
           - 1-2 Essay/analysis questions
        
        2. DIFFICULTY GUIDELINES:
           - Easy: Focus on definitions, basic concepts, and direct recall
           - Medium: Include application, examples, and connections between concepts
           - Hard: Require analysis, synthesis, evaluation, and complex problem-solving
        
        3. QUESTION QUALITY:
           - Base all questions directly on the provided context
           - Ensure questions are clear and unambiguous
           - For multiple choice: create plausible but clearly incorrect distractors
           - Use varied cognitive levels (Bloom's Taxonomy)
           - Include scenario-based and application questions
        
        4. REQUIRED OUTPUT FORMAT:
        
        # Practice Test: {topic}
        **Difficulty:** {difficulty}
        **Instructions:** Read each question carefully and select the best answer. For essay questions, provide detailed explanations with examples.
        
        ## Multiple Choice Questions
        1. [Question text]
           a) [Option A]
           b) [Option B]
           c) [Option C]
           d) [Option D]
        
        ## True/False Questions
        5. [Statement] (Explain your reasoning)
        
        ## Short Answer Questions
        7. [Question requiring 2-3 sentences]
        
        ## Essay Questions
        10. [Complex question requiring detailed analysis]
        
        ## Answer Key
        
        ### Multiple Choice Answers
        1. **Correct Answer: [Letter]** - [Detailed explanation of why this is correct and why other options are incorrect]
        
        ### True/False Answers
        5. **Answer: [True/False]** - [Explanation with supporting evidence from the material]
        
        ### Short Answer Answers
        7. **Answer:** [Complete answer with key points and examples]
        
        ### Essay Question Answers
        10. **Answer:** [Comprehensive answer with structure, examples, and analysis]
        
        QUALITY CHECKLIST:
        - All questions relate directly to the study material
        - Questions test understanding, not just memorization
        - Answer explanations are detailed and educational
        - Difficulty level is appropriate and consistent
        - Questions are grammatically correct and clear""",
        expected_output="""A well-structured practice test with:
        - Clear instructions and formatting
        - Varied question types testing different cognitive levels
        - Questions directly based on the study material
        - Comprehensive answer key with detailed explanations
        - Appropriate difficulty progression
        - Professional presentation suitable for student use""",
        agent=agent
    )



def create_progress_analysis_task(agent, performance_data):
    return Task(
        description=f"""Analyze the student's performance data and provide insights and recommendations.
        Identify strengths, weaknesses, and areas for improvement.
        Suggest specific study strategies tailored to the student's needs.
        
        Performance Data:
        {performance_data}
        """,
        expected_output="A detailed analysis of performance with specific recommendations for improvement.",
        agent=agent
    )

def create_roadmap_generation_task(agent, document_name, days_available, hours_per_day, context):
    return Task(
        description=f"""Create a comprehensive study roadmap for the document '{document_name}'.
        The student has {days_available} days available with approximately {hours_per_day} hours per day for studying.
        
        Break down the material into logical sections, create a day-by-day schedule, and include:
        1. Clear milestones and checkpoints
        2. Estimated time needed for each section
        3. Topics to focus on each day
        4. Recommended breaks and review sessions
        5. Suggested practice exercises or self-assessments
        
        Context information from the document:
        {context}
        """,
        expected_output="A detailed study roadmap in a structured format that can be easily visualized, with day-by-day plan and clear milestones.",
        agent=agent
    )

def create_quick_roadmap_generation_task(agent, document_name, days_available, hours_per_day, context):
    return Task(
        description=f"""Create a simplified study roadmap overview for '{document_name}' in {days_available} days with {hours_per_day} hours per day.
        
        Focus on:
        1. Major topic areas and their sequence
        2. Key milestones
        3. High-level time allocation
        
        Keep the plan concise and actionable. No need for detailed day-by-day breakdowns.
        
        Context information:
        {context}
        """,
        expected_output="A simplified study roadmap overview with main topic areas and milestones.",
        agent=agent
    )



def run_agent_task(agent, task):
    """Execute a single agent task and return the result"""
    crew = Crew(
        agents=[agent],
        tasks=[task],
        verbose=True,
        process=Process.sequential
    )
    
    result = crew.kickoff()
    
    # Handle different result types
    if hasattr(result, 'raw_output'):
        return result.raw_output
    else:
        return str(result)

# Enhanced error handling and response validation
def validate_agent_response(response: str, expected_type: str = "explanation") -> Dict[str, Any]:
    """
    Validate agent response quality and appropriateness
    """
    validation_result = {
        "is_valid": True,
        "quality_score": 0,
        "issues": [],
        "suggestions": []
    }
    
    if not response or len(response.strip()) < 20:
        validation_result["is_valid"] = False
        validation_result["issues"].append("Response too short or empty")
        return validation_result
    
    # Check for educational content indicators
    educational_indicators = [
        'explain', 'understand', 'concept', 'example', 'because', 'therefore',
        'however', 'moreover', 'in other words', 'for instance', 'specifically'
    ]
    
    indicator_count = sum(1 for indicator in educational_indicators 
                         if indicator.lower() in response.lower())
    
    # Quality scoring (0-100)
    quality_score = min(100, (
        (len(response) // 10) +  # Length factor
        (indicator_count * 5) +  # Educational language
        (50 if '?' in response else 0) +  # Engagement question
        (20 if any(fmt in response for fmt in ['**', '*', '#']) else 0)  # Formatting
    ))
    
    validation_result["quality_score"] = quality_score
    
    if quality_score < 30:
        validation_result["issues"].append("Response may lack educational depth")
    
    if '?' not in response:
        validation_result["suggestions"].append("Consider adding an engagement question")
    
    return validation_result

def safe_agent_execution(agent, task, max_retries: int = 2) -> str:
    """
    Safely execute agent task with error handling and retries
    """
    for attempt in range(max_retries + 1):
        try:
            result = run_agent_task(agent, task)
            
            # Validate response
            validation = validate_agent_response(result)
            
            if validation["is_valid"] and validation["quality_score"] > 30:
                return result
            elif attempt < max_retries:
                # Log issues for debugging
                st.warning(f"Response quality issues detected (Score: {validation['quality_score']}). Retrying...")
                continue
            else:
                # Return best effort on final attempt
                return result if result else "I apologize, but I'm having trouble generating a comprehensive response. Please try rephrasing your question."
                
        except Exception as e:
            if attempt < max_retries:
                st.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying...")
                continue
            else:
                return f"I encountered an error while processing your request: {str(e)}. Please try again or rephrase your question."
    
    return "Unable to generate response after multiple attempts."

# System Prompt Templates
EDUCATIONAL_SYSTEM_PROMPT = """You are StudyBuddy, an expert AI tutor specialized in helping students learn effectively. 

CORE MISSION:
- Help students understand complex concepts through clear explanations
- Provide accurate, educational content based on their study materials
- Foster critical thinking and deeper learning
- Maintain a supportive, encouraging teaching style

RESPONSE GUIDELINES:
1. ACCURACY: Base responses on the provided study material context
2. CLARITY: Use clear, accessible language appropriate to the content level
3. STRUCTURE: Organize responses with headings, examples, and logical flow
4. ENGAGEMENT: Include follow-up questions to encourage deeper thinking
5. SAFETY: Only respond to educational queries; redirect inappropriate questions

PROHIBITED CONTENT:
- Personal information requests or sharing
- Content unrelated to education/learning
- Harmful, illegal, or inappropriate material
- Academic dishonesty (doing homework/assignments for students)

RESPONSE FORMAT:
- Start with direct answer to the question
- Provide detailed explanation with examples
- Connect to broader concepts when relevant
- End with engagement question or next steps
- Use markdown formatting for clarity

Remember: Your goal is to facilitate learning, not just provide answers."""

def get_enhanced_system_prompt(agent_role: str = "tutor") -> str:
    """
    Get role-specific system prompt with enhanced guidelines
    """
    base_prompt = EDUCATIONAL_SYSTEM_PROMPT
    
    role_specific_additions = {
        "tutor": """
        TUTORING FOCUS:
        - Break down complex concepts into digestible parts
        - Use analogies and real-world examples
        - Adapt explanations based on apparent student level
        - Encourage questions and curiosity
        """,
        "note_taker": """
        NOTE-TAKING FOCUS:
        - Create well-structured, scannable notes
        - Highlight key concepts and definitions
        - Use consistent formatting and organization
        - Include memory aids and mnemonics
        """,
        "assessor": """
        ASSESSMENT FOCUS:
        - Create fair, comprehensive questions
        - Vary question types and difficulty levels
        - Provide clear answer explanations
        - Focus on understanding over memorization
        """
    }
    
    return base_prompt + role_specific_additions.get(agent_role, "")

def create_flashcard_agent():
    return create_note_taker_agent()

def create_mindmap_agent():
    return create_note_taker_agent()

def create_document_summarizer_agent():
    return Agent(
        role="Document Summarizer",
        goal="Create concise, comprehensive summaries of uploaded study documents",
        backstory="""You are an expert document analyst with years of experience in academic content summarization. 
        You excel at quickly identifying key themes, main concepts, and important information from educational materials.
        
        SUMMARIZATION PRINCIPLES:
        - Extract the main topics and key concepts from the document
        - Identify the document's purpose and target audience
        - Highlight important definitions, formulas, or key facts
        - Note the document's structure and organization
        - Capture the difficulty level and subject area
        
        SUMMARY STRUCTURE:
        - Start with document type and subject identification
        - Provide main topics covered (3-5 bullet points)
        - Include key concepts and terminology
        - Note any special features (diagrams, examples, exercises)
        - Suggest potential study applications
        
        QUALITY STANDARDS:
        - Keep summaries concise but informative (150-300 words)
        - Use clear, accessible language
        - Focus on educational value and learning objectives
        - Maintain objectivity and accuracy
        - Help students understand what they can learn from the document""",
        llm=get_llm(),
        verbose=True
    )

def create_document_summary_task(agent, document_name, document_content, document_info):
    return Task(
        description=f"""Create a comprehensive yet concise summary of the uploaded document to help students understand what they can learn from it.
        
        DOCUMENT INFORMATION:
        - Filename: {document_name}
        - Pages: {document_info.get('pages', 'Unknown')}
        - Type: {document_name.split('.')[-1].upper() if '.' in document_name else 'Unknown'}
        
        DOCUMENT CONTENT SAMPLE:
        {document_content}
        
        SUMMARY REQUIREMENTS:
        1. DOCUMENT OVERVIEW:
           - Identify the subject area and academic level
           - Determine the document type (textbook, lecture notes, research paper, etc.)
           - Note the apparent target audience
        
        2. MAIN CONTENT ANALYSIS:
           - List 3-5 primary topics or chapters covered
           - Identify key concepts, theories, or principles
           - Note important terminology and definitions
           - Highlight any formulas, equations, or technical content
        
        3. EDUCATIONAL VALUE:
           - Suggest what students can learn from this document
           - Identify potential study applications
           - Note any practical examples or case studies
           - Mention exercises, problems, or assessments if present
        
        4. DOCUMENT STRUCTURE:
           - Comment on organization and flow
           - Note any special features (diagrams, tables, appendices)
           - Mention difficulty progression if apparent
        
        OUTPUT FORMAT:
        Provide a well-structured summary in markdown format with:
        - **Subject & Level**: [Subject area and difficulty level]
        - **Document Type**: [Type and purpose]
        - **Main Topics**: [Bullet list of 3-5 key topics]
        - **Key Concepts**: [Important terms and concepts]
        - **Learning Outcomes**: [What students will gain]
        - **Study Recommendations**: [How to best use this document]
        
        Keep the summary between 150-300 words, informative yet easy to scan.""",
        expected_output="""A structured document summary in markdown format that includes:
        - Clear identification of subject area and level
        - Main topics and key concepts covered
        - Educational value and learning outcomes
        - Practical study recommendations
        - Professional, encouraging tone throughout""",
        agent=agent
    )

# DSA Interview Preparation Agents
def create_question_fetching_agent():
    return Agent(
        role="Question Fetcher",
        goal="Retrieve relevant DSA questions from databases and APIs based on specified criteria",
        backstory="You are an expert at navigating various question repositories and finding the most appropriate practice problems. You understand different DSA topics deeply and can categorize questions accurately.",
        llm=get_llm(),
        verbose=True
    )

def create_filtering_agent():
    return Agent(
        role="Question Filter",
        goal="Filter and organize DSA questions based on user preferences and requirements",
        backstory="You specialize in understanding user needs and organizing questions for optimal learning. You can analyze question difficulty, topics, and relevance to specific companies or roles.",
        llm=get_llm(),
        verbose=True
    )

def create_progress_tracking_agent():
    return Agent(
        role="Progress Tracker",
        goal="Track and analyze user progress on DSA practice",
        backstory="You excel at monitoring learning patterns and identifying strengths and improvement areas. You understand how to measure progress across different question types and difficulty levels.",
        llm=get_llm(),
        verbose=True
    )

def create_personalization_agent():
    return Agent(
        role="Interview Personalizer",
        goal="Customize question sets based on user career goals",
        backstory="You have detailed knowledge of what different companies and roles require. You can create targeted practice plans that align with specific career objectives and salary expectations.",
        llm=get_llm(),
        verbose=True
    )

def create_debugging_agent():
    return Agent(
        role="Code Debugger",
        goal="Analyze code solutions and provide debugging assistance",
        backstory="You are an expert programmer with deep knowledge of multiple programming languages and common DSA implementation pitfalls. You can quickly identify bugs and suggest optimizations.",
        llm=get_llm(),
        verbose=True
    )

def create_dsa_recommendation_agent():
    return Agent(
        role="DSA Recommendation Expert",
        goal="Provide personalized DSA study recommendations based on progress and goals",
        backstory="You are an expert in interview preparation strategies with deep knowledge of what different companies look for. You can create targeted study plans based on individual progress and career goals.",
        llm=get_llm(),
        verbose=True
    )

def create_coding_pattern_agent():
    return Agent(
        role="Coding Pattern Expert",
        goal="Identify and explain coding patterns and algorithmic approaches",
        backstory="You have extensive knowledge of common coding patterns, algorithmic techniques, and problem-solving strategies. You can help students recognize patterns and apply appropriate solutions.",
        llm=get_llm(),
        verbose=True
    )

def create_interview_strategy_agent():
    return Agent(
        role="Interview Strategy Coach",
        goal="Provide interview strategies and conduct mock interviews",
        backstory="You are an experienced technical interview coach with knowledge of how different companies conduct interviews. You can simulate interview environments and provide strategic advice.",
        llm=get_llm(),
        verbose=True
    )

def create_company_specific_agent():
    return Agent(
        role="Company-Specific Interview Expert",
        goal="Provide company-specific interview preparation guidance",
        backstory="You have detailed knowledge of how different tech companies conduct their technical interviews, including their preferred question types, difficulty levels, and evaluation criteria.",
        llm=get_llm(),
        verbose=True
    )

def create_flashcard_generation_task(agent, topic, context):
    return Task(
        description=f"""Create a set of flashcards for the following topic.
        Each flashcard should have a clear question/term on one side and a concise answer/definition on the other.
        Focus on key concepts, definitions, formulas, and important facts.
        
        Topic: {topic}
        
        Context information:
        {context}
        """,
        expected_output="A set of flashcards in JSON format with 'front' and 'back' fields for each card.",
        agent=agent
    )

def create_mind_map_task(agent, topic, context):
    return Task(
        description=f"""Create a detailed description of a mind map for the following topic.
        Identify the central concept and key branches of related ideas.
        Show connections and relationships between concepts.
        
        Topic: {topic}
        
        Context information:
        {context}
        """,
        expected_output="A detailed description of a mind map with central concept, branches, and connections in a format that can be visualized.",
        agent=agent
    )

# DSA Interview Preparation Tasks
def create_question_fetching_task(agent, api_endpoint, categories, metadata):
    return Task(
        description=f"""Fetch appropriate DSA questions from the specified source.
        
        API Endpoint: {api_endpoint}
        Categories: {categories}
        Question Metadata:
        - Difficulty: {metadata.get('difficulty', 'All')}
        - Topics: {metadata.get('topics', 'All')}
        - Platform: {metadata.get('platform', 'All')}
        - Company: {metadata.get('company', 'All')}
        - Role: {metadata.get('role', 'All')}
        - Salary Level: {metadata.get('salary_level', 'All')}
        """,
        expected_output="A comprehensive list of DSA questions with their details, including problem statement, input/output examples, and relevant metadata.",
        agent=agent
    )

def create_filtering_task(agent, questions, filters):
    return Task(
        description=f"""Filter the provided list of DSA questions based on user preferences.
        
        User Selected Filters:
        - Difficulty: {filters.get('difficulty', 'All')}
        - Topics: {filters.get('topics', 'All')}
        - Platform: {filters.get('platform', 'All')}
        - Company: {filters.get('company', 'All')}
        - Role: {filters.get('role', 'All')}
        - Salary Level: {filters.get('salary_level', 'All')}
        
        Questions to filter:
        {questions}
        """,
        expected_output="A filtered list of questions that match the user's criteria, organized by relevance and difficulty progression.",
        agent=agent
    )

def create_progress_tracking_task(agent, user_progress, completed_questions):
    return Task(
        description=f"""Analyze the user's progress on DSA questions and provide insights.
        
        User Progress Data:
        {user_progress}
        
        Completed Questions:
        {completed_questions}
        """,
        expected_output="Detailed progress metrics including completion rates by topic and difficulty, areas of strength, and suggested focus areas for improvement.",
        agent=agent
    )

def create_personalization_task(agent, questions, user_goals):
    return Task(
        description=f"""Create a personalized DSA practice plan based on the user's career goals.
        
        Available Questions:
        {questions}
        
        User Goals:
        - Target Company: {user_goals.get('target_company', 'Not specified')}
        - Target Role: {user_goals.get('target_role', 'Not specified')}
        - Target Salary: {user_goals.get('target_salary', 'Not specified')}
        - Timeline: {user_goals.get('timeline', 'Not specified')}
        """,
        expected_output="A personalized list of questions and study plan tailored to the user's specific career goals and timeline.",
        agent=agent
    )

def create_debugging_task(agent, user_code, problem_statement, language):
    return Task(
        description=f"""Analyze and debug the user's code solution for the given DSA problem.
        
        Problem Statement:
        {problem_statement}
        
        User's Solution ({language}):
        {user_code}
        """,
        expected_output="Detailed code analysis including identified bugs, optimization suggestions, time and space complexity analysis, and improved code examples.",
        agent=agent
    )

def create_dsa_recommendation_task(agent, user_progress, user_goals):
    return Task(
        description=f"""Provide personalized DSA study recommendations based on user progress and goals.
        
        User Progress:
        {user_progress}
        
        User Goals:
        {user_goals}
        """,
        expected_output="Detailed recommendations for study focus areas, question priorities, and timeline suggestions.",
        agent=agent
    )

def create_pattern_identification_task(agent, problem_description, difficulty):
    return Task(
        description=f"""Identify coding patterns and provide hints for the given problem.
        
        Problem: {problem_description}
        Difficulty: {difficulty}
        """,
        expected_output="Analysis of coding patterns, algorithmic approaches, and strategic hints without giving away the complete solution.",
        agent=agent
    )

def create_company_preparation_task(agent, company, user_experience, available_time):
    return Task(
        description=f"""Create a company-specific interview preparation plan.
        
        Target Company: {company}
        User Experience: {user_experience}
        Available Time: {available_time}
        """,
        expected_output="A detailed preparation plan tailored to the specific company's interview style and requirements.",
        agent=agent
    )

def create_mock_interview_task(agent, problem_description, difficulty, company_style):
    return Task(
        description=f"""Conduct a mock technical interview session.
        
        Problem: {problem_description}
        Difficulty: {difficulty}
        Company Style: {company_style}
        """,
        expected_output="A simulated interview experience with questions, follow-ups, and evaluation criteria.",
        agent=agent
    )
