# StudyBuddy AI - Enhanced Learning Assistant

A comprehensive AI-powered study companion that helps students learn more effectively through intelligent tutoring, note generation, and assessment tools.

## 🚀 Recent Improvements

### Enhanced Chatbot with Guard Rails
- **Comprehensive Input Validation**: Filters inappropriate content and potential prompt injection attempts
- **Educational Focus**: Validates questions are educational in nature and provides guidance for better interactions
- **Improved Response Quality**: Enhanced system prompts for more structured, educational responses
- **Safety Features**: Built-in content filtering and response validation
- **Better Error Handling**: Graceful error handling with retry logic and user-friendly error messages

### System Prompt Enhancements
- **Role-Specific Prompts**: Tailored system prompts for different agent roles (tutor, note-taker, assessor)
- **Educational Guidelines**: Clear instructions for maintaining educational focus and appropriate content
- **Response Structure**: Standardized formatting for consistent, well-structured responses
- **Engagement Features**: Built-in engagement questions and follow-up suggestions

### User Experience Improvements
- **Interactive Chat Interface**: Enhanced chat UI with feedback buttons and response quality indicators
- **Progress Indicators**: Real-time processing status and detailed progress updates
- **Smart Suggestions**: Context-aware suggestions for follow-up actions (notes, flashcards, mind maps)
- **Input Guidance**: Helpful tips and examples for better user interactions

## 🛡️ Safety & Security Features

### Input Validation
- Content filtering for inappropriate topics
- Prompt injection protection
- Length and format validation
- Educational content verification

### Response Quality Control
- Multi-attempt generation with quality scoring
- Response validation and improvement suggestions
- Error recovery and graceful degradation
- Comprehensive logging for debugging

### Privacy Protection
- Automatic sanitization of personal information
- Secure environment variable handling
- No sensitive data logging
- Safe document processing

## 🎯 Key Features

- **Interactive Chat**: Ask questions about your study materials
- **Smart Note Generation**: AI-powered study notes with proper formatting
- **Flashcard Creation**: Automated flashcard generation for key concepts
- **Mind Maps**: Visual learning through concept mapping
- **Progress Tracking**: Monitor your learning journey
- **DSA Interview Prep**: Specialized tools for coding interview preparation

## 🔧 Setup Instructions

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Setup**:
   - Copy `.env.example` to `.env`
   - Add your Groq API key to the `.env` file:
     ```
     GROQ_API_KEY=your_groq_api_key_here
     ```

3. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

## 📚 Usage Guidelines

### Getting the Best Results
- Ask specific, educational questions about your uploaded documents
- Use clear, descriptive language in your queries
- Take advantage of follow-up suggestions for deeper learning
- Provide feedback on responses to help improve the system

### Example Questions
- "Can you explain [specific concept] in simple terms?"
- "What's the difference between X and Y?"
- "Can you provide an example of [topic]?"
- "How does [concept A] relate to [concept B]?"

## 🧪 Testing

Run the test suite to verify all improvements:
```bash
python test_improvements.py
```

## 🤝 Contributing

We welcome contributions! Please ensure all new features include appropriate tests and documentation.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.