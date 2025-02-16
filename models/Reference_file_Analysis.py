import json
from transformers import pipeline
import re


def is_valid_question(question: str) -> bool:
    """Validate if the string is a proper question."""
    if not question or len(question) < 5:
        return False

    # Check for question mark and basic structure
    if not question.endswith('?'):
        return False

    # Check if it starts with a question word
    first_word = question.lower().split()[0]
    question_starters = {'what', 'why', 'how', 'where', 'when', 'which', 'who', 'whose', 'whom'}
    if first_word not in question_starters:
        return False

    return True


def classify_questions_using_bert(input_file):
    # Load a zero-shot classification pipeline
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    try:
        # Read the input file
        with open(input_file, 'r') as file:
            content = file.read()

        # Split content into lines and clean them
        lines = [line.strip() for line in content.split('\n') if line.strip()]

        # Extract the passage (first line)
        info = lines[0]

        # Extract questions
        questions = []
        for line in lines[1:]:
            # Check if line starts with Q or a number
            if line.lower().startswith('q') or line[0].isdigit():
                # Clean the question text
                question = re.sub(r'^[Qq]\d*\)|^\d+\.?\s*', '', line).strip()
                if question:
                    questions.append(question)

        # Clean and validate questions
        cleaned_questions = []
        for q in questions:
            # Capitalize first letter
            q = q[0].upper() + q[1:] if q else q
            # Add question mark if missing
            if not q.endswith('?'):
                q += '?'
            # Validate question
            if is_valid_question(q):
                cleaned_questions.append(q)

        print(f"Found {len(cleaned_questions)} valid questions")

        # Initialize classified questions dictionary
        classified_questions = {}

        # Process each question
        for question in cleaned_questions:
            # Classify the question
            classification = classifier(
                question,
                candidate_labels=["purpose", "detail", "conclusion", "inference", "assumption"]
            )

            # Get the top category
            top_category = classification["labels"][0]

            # Add the question to the appropriate category
            if top_category not in classified_questions:
                classified_questions[top_category] = []

            classified_questions[top_category].append({"question": question})

        # Convert to JSON format
        output_data = {
            "info": info,
            "classified_questions": classified_questions
        }

        # Save the output to a JSON file
        with open("classified_questions.json", "w") as json_file:
            json.dump(output_data, json_file, indent=4)

        print("Classified questions have been successfully saved to classified_questions.json")
        print(f"Questions by category:")
        for category, questions in classified_questions.items():
            print(f"{category}: {len(questions)} questions")

    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found")
    except Exception as e:
        print(f"Error processing questions: {str(e)}")


# Specify the input file path
input_file = "input.txt"
# Call the function to classify and save questions
classify_questions_using_bert(input_file)