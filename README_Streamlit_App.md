# Streamlit Frontend for Call Center Transcript Evaluator

A comprehensive web-based interface for the Call Center Transcript Evaluator using Mistral 7B. This Streamlit application provides an intuitive way to upload transcript files, run evaluations, and view detailed coaching feedback.

## 🌟 Features

### 📤 **Upload & Evaluation**
- **Drag & Drop Interface**: Easy JSON file upload with validation
- **Real-time Preview**: View transcript metadata and content before evaluation
- **One-Click Evaluation**: Run comprehensive Mistral 7B analysis with a single button
- **Progress Tracking**: Visual progress indicators during evaluation

### 📊 **Comprehensive Results Display**
- **Detailed Feedback**: All evaluation results organized in expandable sections
- **CSR-Specific Analysis**: Feedback tied to specific CSR IDs
- **Sentiment Analysis**: Emotional tone evaluation with coaching
- **Topic Summarization**: Theme identification for each Q&A pair
- **Interactive Tabs**: Organized display of different analysis types

### 📈 **Visual Analytics**
- **Sentiment Distribution**: Pie charts showing sentiment patterns
- **Sentence Length Analysis**: Histograms of communication patterns
- **Performance Metrics**: Key statistics and trends
- **Interactive Charts**: Plotly-powered visualizations

### 📚 **Knowledge Management**
- **Auto-Generated Documents**: Knowledge base creation from transcripts
- **Downloadable Reports**: Comprehensive evaluation reports
- **Export Functionality**: Save results for further analysis

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Mistral 7B model file: `mistral-7b-instruct-v0.2.Q4_K_M.gguf`
- All dependencies from the main evaluator

### Installation

1. **Install Core Dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
   ```

2. **Install Streamlit Dependencies**:
   ```bash
   pip install -r requirements_streamlit.txt
   ```

3. **Download Mistral Model**: Place the model file in your project directory

4. **Launch the Application**:
   ```bash
   streamlit run streamlit_app.py
   ```

### First-Time Setup

1. **Load Model**: Use the sidebar to configure and load your Mistral model
2. **Upload Transcript**: Drag and drop your JSON transcript file
3. **Run Evaluation**: Click "Run Mistral Evaluation" to start analysis
4. **View Results**: Explore the comprehensive feedback and analytics

## 📋 Usage Guide

### 1. **Model Configuration**
- Navigate to the sidebar "Configuration" section
- Enter the path to your Mistral model file
- Click "Load Model" to initialize the evaluator
- Wait for the "Model Ready" confirmation

### 2. **File Upload**
- Go to the "Upload & Evaluate" tab
- Upload a JSON file with the required structure:
  ```json
  {
    "call_ID": "12345",
    "CSR_ID": "JaneDoe123",
    "call_date": "2024-02-01",
    "call_time": "02:16:43",
    "call_transcript": [
      "CSR: Thank you for calling...",
      "Customer: I need help with..."
    ]
  }
  ```

### 3. **Running Evaluation**
- Review the transcript preview
- Click "Run Mistral Evaluation" button
- Monitor the progress bar during analysis
- Wait for completion confirmation

### 4. **Viewing Results**
- Switch to the "Results" tab
- Explore detailed feedback for each Q&A pair:
  - **English Correctness**: Grammar and language evaluation
  - **Sentence Analysis**: Length and clarity recommendations
  - **Word Patterns**: Repetition and crutch word detection
  - **Sentiment Analysis**: Emotional tone with coaching
  - **Topic Summary**: Theme identification

### 5. **Analytics Dashboard**
- View sentiment distribution charts
- Analyze sentence length patterns
- Review performance metrics
- Export visualizations

### 6. **Knowledge Base**
- Access auto-generated knowledge documents
- Review training materials created from transcripts
- Download comprehensive reports

## 🎯 Key Features Explained

### **CSR-Specific Feedback**
The app displays feedback tied to specific CSR IDs, making it easy to:
- Track individual agent performance
- Provide targeted coaching
- Monitor improvement over time
- Generate agent-specific reports

### **Sentiment Analysis Integration**
Each response receives sentiment evaluation with:
- **Polarity Score**: Emotional positivity/negativity (-1 to +1)
- **Subjectivity Score**: Opinion vs. fact-based content (0 to 1)
- **Classification**: Positive, Negative, or Neutral
- **Coaching Feedback**: Actionable advice in natural language

### **Topic & Theme Detection**
Automatic categorization of interactions:
- **Issue Type**: Refund, booking, complaint, etc.
- **Complexity Level**: Simple, moderate, or complex
- **Skills Required**: Knowledge areas needed
- **Training Recommendations**: Suggested improvements

### **Real-Time Evaluation**
The "Run Mistral Evaluation" button executes the complete analysis:
- Extracts Q&A pairs from JSON
- Runs all evaluation modules
- Generates coaching feedback
- Creates knowledge documents
- Provides downloadable reports

## 🔧 Configuration Options

### **Sidebar Settings**
- **Model Path**: Configure Mistral model location
- **Display Options**: Toggle detailed feedback, visualizations, knowledge docs
- **Evaluation Settings**: Customize analysis parameters

### **Advanced Configuration**
Modify `config.py` for advanced settings:
- Sentence length thresholds
- Crutch word lists
- Model parameters
- Output formats

## 📊 Output Examples

### **English Correctness Feedback**
```
Grammar Analysis: Your response shows good overall structure with minor improvements needed.
Spelling: No spelling errors detected.
Suggestions: Consider using more active voice in your explanations.
Score: 8/10
Coaching: Great job maintaining professional language! Try to be more direct in your responses.
```

### **Sentiment Coaching**
```
Your sentiment analysis shows a positive and helpful tone (Polarity: 0.6).
You maintained excellent customer service demeanor throughout the interaction.
Continue this approach to build customer confidence and satisfaction.
```

### **Topic Summary**
```
Main Topic: Flight Cancellation and Refund Request
Issue Type: Service Recovery
Complexity: Moderate
Skills Demonstrated: Empathy, problem-solving, policy knowledge
Recommendations: Continue excellent empathy skills, review refund procedures
```

## 🎨 User Interface

### **Clean, Professional Design**
- Modern Streamlit interface with custom CSS
- Color-coded feedback sections
- Intuitive navigation with tabs
- Responsive layout for different screen sizes

### **Interactive Elements**
- Expandable Q&A pair sections
- Tabbed analysis views
- Interactive charts and graphs
- Progress indicators and status updates

### **Accessibility Features**
- Clear visual hierarchy
- Descriptive labels and help text
- Keyboard navigation support
- Screen reader compatibility

## 🔍 Troubleshooting

### **Common Issues**

1. **Model Loading Fails**
   ```
   Error: Failed to load model
   ```
   - Check model file path and permissions
   - Ensure sufficient system memory
   - Verify llama-cpp-python installation

2. **File Upload Errors**
   ```
   Invalid JSON structure
   ```
   - Verify JSON format and required fields
   - Check file encoding (should be UTF-8)
   - Ensure call_transcript is a list

3. **Evaluation Timeout**
   ```
   Evaluation taking too long
   ```
   - Large transcripts may take several minutes
   - Check CPU usage and available memory
   - Consider processing smaller files first

### **Performance Optimization**

1. **For Large Files**:
   - Process transcripts with fewer Q&A pairs
   - Increase system memory allocation
   - Close other applications during evaluation

2. **For Slow Performance**:
   - Adjust CPU thread count in config
   - Use SSD storage for model file
   - Optimize model parameters

## 📱 Mobile Compatibility

The Streamlit app is responsive and works on:
- Desktop browsers (Chrome, Firefox, Safari, Edge)
- Tablet devices (iPad, Android tablets)
- Mobile phones (with limited functionality)

## 🔒 Security Considerations

- **Local Processing**: All evaluation happens locally
- **No Data Upload**: Transcripts are not sent to external servers
- **Privacy Protection**: Customer data remains on your system
- **Secure File Handling**: Temporary files are automatically cleaned up

## 🚀 Deployment Options

### **Local Development**
```bash
streamlit run streamlit_app.py
```

### **Network Access**
```bash
streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8501
```

### **Production Deployment**
- Docker containerization
- Cloud platform deployment (AWS, GCP, Azure)
- Internal server hosting
- Load balancing for multiple users

## 📈 Future Enhancements

### **Planned Features**
- Batch processing for multiple files
- Historical analysis and trends
- Agent comparison dashboards
- Custom coaching templates
- Integration with CRM systems

### **RAGAS Integration**
When ready to enable RAGAS evaluation:
1. Uncomment RAGAS sections in the main evaluator
2. Install RAGAS dependencies
3. Configure local model integration
4. Enable advanced metrics in the UI

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Test with sample data
5. Submit a pull request

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the configuration guide
3. Test with the provided sample data
4. Create an issue in the repository

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**🎯 Streamlit Call Center Evaluator** | Powered by Mistral 7B | Built with ❤️ and Streamlit

