# AI-Powered Essay Assessment System

## Project Overview

The AI-Powered Essay Assessment System is an artificial intelligence-based application designed to automate essay evaluation and provide consistent scoring and feedback. The system uses Natural Language Processing (NLP), machine learning, and Retrieval-Augmented Generation (RAG) techniques to analyze written responses, predict scores, and generate improvement suggestions.

The aim of this project is to support educational assessment by reducing manual grading effort while providing students with meaningful feedback on their writing quality.

## Features

- Automated essay scoring using a trained machine learning model.
- AI-generated feedback to identify strengths and areas for improvement.
- Retrieval-Augmented Generation (RAG) for improving feedback quality using reference materials.
- Rubric-based evaluation to ensure consistent assessment.
- Web-based interface for submitting essays and viewing results.

## Technologies Used

- Python
- Flask
- Machine Learning
- Natural Language Processing (NLP)
- Scikit-learn
- LangChain
- ChromaDB
- HTML/CSS

## Project Structure

AI-Essay-Grader/
│
├── app.py                     # Main Flask application
├── main.py                    # Main execution script
├── agent_v2.py                # AI agent implementation
├── rag_tool.py                # RAG components
├── 03_rag_setup.py            # RAG setup configuration
├── evaluation_analysis.py     # Model evaluation scripts
├── requirements.txt           # Required Python packages
│
├── models/
│   ├── model.pkl              # Trained scoring model
│   ├── feature_names.json     # Model feature information
│   └── score_meta.json        # Score metadata
│
├── templates/
│   └── index.html             # Web interface
│
├── rubrics/
│   ├── feedback_phrases.txt
│   ├── writing_criteria.txt
│   └── asap_scoring_rubric.docx
│
├── data/                      # Dataset and sample resources
│
└── deploy_files/              # Deployment files


## How to Run
1. Install the required Python packages:
pip install -r requirements.txt

2. Start the application:
python app.py

3. Open the local URL displayed in the terminal (for example, http://127.0.0.1:5000) in your web browser.

Note: Configure any required API keys before running the application if your project uses external AI services.

## Dataset and Model Information

The system uses trained machine learning models for automated essay scoring. 

The repository contains the required model files, feature information, evaluation rubrics, and supporting resources needed to run the application.

## Evaluation

The system performance can be evaluated using standard machine learning metrics, including:

- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- Quadratic Weighted Kappa (QWK)

## Limitations

-The accuracy of the scores depends on the quality and variety of the training data.
-The feedback may not always be perfect and may need to be reviewed by a teacher.
-The system may not perform equally well for every writing style or essay type.

## Future Improvements

Future work will focus on reducing bias in automated essay scoring and strengthening data protection through privacy-preserving techniques, ensuring the system is more fair, transparent, and secure for educational use.

## Acknowledgement

This project was developed as part of Emerging technonologies in AI course in the program of MSc. in Artificial Intelligence.

