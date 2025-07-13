import os
from crewai import Agent, Task
from langchain_groq import ChatGroq
import streamlit as st
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize API and models
@st.cache_resource
def get_llm():
    # Initialize Groq client
    groq_api_key = os.environ.get("GROQ_API_KEY", "")
    if not groq_api_key:
        st.error("GROQ_API_KEY not found in environment variables. Please check your .env file.")
        st.stop()
        
    # Configure the Groq LLM
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="groq/llama3-70b-8192",  # Add 'groq/' prefix to the model name
    )
    return llm

# Study Agents
def create_study_tutor_agent():
    return Agent(
        role="Study Tutor",
        goal="Explain complex concepts clearly and help students understand course material",
        backstory="You are an expert educator with years of experience breaking down difficult concepts into understandable explanations. You excel at adapting your teaching style to match different learning preferences.",
        llm=get_llm(),
        verbose=True
    )

def create_note_taker_agent():
    return Agent(
        role="Note-Taker",
        goal="Create organized, comprehensive study notes",
        backstory="You specialize in creating concise yet comprehensive notes that highlight key concepts, definitions, examples, and connections between ideas. Your notes are well-structured with clear headings and logical flow.",
        llm=get_llm(),
        verbose=True
    )

def create_assessment_expert_agent():
    return Agent(
        role="Assessment Expert",
        goal="Design tests to evaluate understanding at different complexity levels",
        backstory="You are skilled at creating varied assessment questions that test different levels of knowledge, from basic recall to complex application. You can generate quizzes ranging from simple to advanced difficulty.",
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

# Study Tasks
def create_explanation_task(agent, question, context):
    return Task(
        description=f"""Explain the following concept using clear, concise language. 
        Break down complex ideas into manageable parts. Use analogies where helpful.
        
        Question: {question}
        
        Context information:
        {context}
        """,
        expected_output="A clear explanation of the concept with examples and analogies if appropriate.",
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
        description=f"""Create a practice test on the following topic with {difficulty} difficulty level.
        Include a mix of question types (multiple choice, short answer, essay questions).
        Provide an answer key with explanations.
        
        Topic: {topic}
        Difficulty: {difficulty}
        
        Context information:
        {context}
        """,
        expected_output="A practice test with varied question types and a comprehensive answer key.",
        agent=agent
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
