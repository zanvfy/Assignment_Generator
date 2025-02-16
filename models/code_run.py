import re

from transformers import T5Tokenizer, T5ForConditionalGeneration
import spacy
import random
from typing import List, Dict
import logging

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class RefinedQuestionGenerator:
    def __init__(self, model_name: str = "t5-base"):
        logger.info(f"Initializing QuestionGenerator with model: {model_name}")
        try:
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(model_name)
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("Successfully loaded all models and tokenizers")
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            raise

    def extract_topics(self, passage: str) -> Dict[str, List[str]]:
        """Extract main topics and subtopics from the passage."""
        logger.info("Starting topic extraction")
        doc = self.nlp(passage)
        topics = {
            'main_topics': [],
            'organizations': [],
            'concepts': []
        }

        logger.debug(f"Processing passage of length: {len(passage)}")

        # Extract organizations and main noun phrases
        logger.info("Extracting named entities")
        for ent in doc.ents:
            if ent.label_ == 'ORG':
                topics['organizations'].append(ent.text)
                logger.debug(f"Found organization: {ent.text}")

        # Extract key concepts using noun phrases
        logger.info("Extracting noun phrases")
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) >= 2 and len(chunk.text.split()) <= 5:
                topics['concepts'].append(chunk.text)
                logger.debug(f"Found concept: {chunk.text}")

        # Extract main topics using subject-verb patterns
        logger.info("Extracting main topics")
        for sent in doc.sents:
            for token in sent:
                if token.dep_ == 'nsubj' and token.head.pos_ == 'VERB':
                    topic = ' '.join([t.text for t in token.subtree])
                    if len(topic.split()) <= 5:
                        topics['main_topics'].append(topic)
                        logger.debug(f"Found main topic: {topic}")

        # Log summary of findings
        for category, items in topics.items():
            logger.info(f"Found {len(items)} {category}: {', '.join(items[:3])}{'...' if len(items) > 3 else ''}")

        return {k: list(set(v)) for k, v in topics.items()}

    def generate_concise_options(self, question: str, passage: str, topics: Dict[str, List[str]]) -> List[str]:
        """Generate shorter, more focused answer options."""
        logger.info(f"Generating options for question: {question[:50]}...")

        try:
            # Generate base answer using T5
            logger.info("Generating base answers using T5")
            input_text = f"answer: {question}\ncontext: {passage}"
            input_ids = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).input_ids

            output_ids = self.model.generate(
                input_ids,
                max_length=50,
                num_return_sequences=2,
                temperature=0.7,
                do_sample=True
            )

            base_answers = [
                self.tokenizer.decode(out, skip_special_tokens=True)
                for out in output_ids
            ]
            logger.debug(f"Generated base answers: {base_answers}")

            # Create additional options using topics
            logger.info("Creating topic-based options")
            topic_based_options = []
            for category, items in topics.items():
                if items:
                    sample_size = min(2 if category == 'main_topics' else 1, len(items))
                    sampled_items = random.sample(items, sample_size)
                    topic_based_options.extend(sampled_items)
                    logger.debug(f"Added {sample_size} options from {category}")

            # Combine and clean options
            logger.info("Cleaning and formatting options")
            all_options = base_answers + topic_based_options

            # Clean and shorten options
            cleaned_options = []
            for opt in all_options:
                if len(opt.split()) > 15:
                    opt = ' '.join(opt.split()[:15]) + '...'
                    logger.debug(f"Truncated long option: {opt}")
                if len(opt.split()) >= 3:
                    cleaned_options.append(opt)

            # Ensure we have 5 unique options
            while len(cleaned_options) < 5:
                if topics['concepts']:
                    new_option = random.choice(topics['concepts'])
                    cleaned_options.append(new_option)
                    logger.debug(f"Added filler option: {new_option}")
                else:
                    filler = f"Option {len(cleaned_options) + 1}"
                    cleaned_options.append(filler)
                    logger.debug(f"Added generic filler: {filler}")

            # Format final options
            unique_options = list(dict.fromkeys(cleaned_options))[:5]
            formatted_options = [f"({chr(65 + i)}) {opt}" for i, opt in enumerate(unique_options)]
            logger.info(f"Generated {len(formatted_options)} final options")

            return formatted_options

        except Exception as e:
            logger.error(f"Error generating options: {str(e)}")
            raise

    # def generate_focused_question(self, question_type: str, topics: Dict[str, List[str]]) -> str:
    #     """Generate more focused questions with proper topic insertion."""
    #     logger.info(f"Generating question of type: {question_type}")
    #
    #     templates = {
    #         "main_idea": [
    #             "What is the main purpose of this passage?",
    #             "What is the central topic being discussed?",
    #             "What is the primary focus of this text?"
    #         ],
    #         "detail": [
    #             "What specific details does the passage provide about {}?",
    #             "How does the passage describe {}?",
    #             "What information is given about {}?"
    #         ],
    #         "conclusion": [
    #             "What can be concluded about {}?",
    #             "What is the main finding regarding {}?",
    #             "What does the passage reveal about {}?"
    #         ],
    #         "purpose": [
    #             "Why does the author discuss {}?",
    #             "What is the significance of {} in the passage?",
    #             "How does {} relate to the main topic?"
    #         ]
    #     }
    #
    #     template = random.choice(templates[question_type])
    #     logger.debug(f"Selected template: {template}")
    #
    #     # Insert topic if needed
    #     if '{}' in template:
    #         logger.info("Template requires topic insertion")
    #         topic_choices = (topics['main_topics'] + topics['organizations'] +
    #                          topics['concepts'])
    #         if topic_choices:
    #             topic = random.choice(topic_choices)
    #             template = template.format(topic)
    #             logger.debug(f"Inserted topic: {topic}")
    #         else:
    #             template = template.format("the subject")
    #             logger.warning("No topics available, using generic subject")
    #
    #     logger.info(f"Generated question: {template}")
    #     return template

    import random
    from typing import Dict, List

    def generate_focused_question(question_type: str, topics: Dict[str, List[str]],
                                  templates: Dict[str, List[str]]) -> str:
        """
        Generate a focused question dynamically based on provided templates.
        Args:
            question_type (str): Type of question to generate (e.g., "main_idea", "detail").
            topics (Dict[str, List[str]]): Dictionary containing extracted topics.
            templates (Dict[str, List[str]]): Dictionary of templates for different question types.
        Returns:
            str: A dynamically generated question.
        """
        # Ensure templates for the given type are available
        if question_type not in templates or not templates[question_type]:
            return f"No template available for question type: {question_type}"

        # Choose a random template for the question type
        template = random.choice(templates[question_type])

        # Check if the template needs topic insertion
        if '{}' in template:
            topic_choices = (topics.get('main_topics', []) +
                             topics.get('organizations', []) +
                             topics.get('concepts', []))
            # Insert a topic or use a fallback
            topic = random.choice(topic_choices) if topic_choices else "the subject"
            template = template.format(topic)

        return template

    def generate_questions(self, passage: str, num_questions: int = 4) -> List[Dict[str, str]]:
        """Generate a set of focused questions with relevant options."""
        logger.info(f"Starting question generation for passage of length {len(passage)}")

        try:
            topics = self.extract_topics(passage)
            questions = []
            question_types = ["main_idea", "detail", "conclusion", "purpose"]

            for q_type in question_types[:num_questions]:
                logger.info(f"Generating question {len(questions) + 1}/{num_questions} of type {q_type}")

                question = self.generate_focused_question(q_type, topics)
                options = self.generate_concise_options(question, passage, topics)

                questions.append({
                    "question": question,
                    "options": options
                })
                logger.info(f"Successfully generated question: {question[:50]}...")

            logger.info(f"Successfully generated {len(questions)} questions")
            return questions

        except Exception as e:
            logger.error(f"Error in question generation process: {str(e)}")
            raise


def main():
    logger.info("Starting main process")
    try:
        generator = RefinedQuestionGenerator()

        # Read passage
        logger.info("Reading input passage")
        with open("../data/passage.txt", "r", encoding="utf-8") as f:
            passage = f.read()

        # Generate questions
        logger.info("Generating questions")
        questions = generator.generate_questions(passage)

        # Save output
        logger.info("Saving generated questions")
        with open("../outputs/output_questions.txt", "w", encoding="utf-8") as f:
            f.write("Based on the Passage, answer the following questions:\n\n")
            for i, q in enumerate(questions, 1):
                f.write(f"{i}. {q['question']}\n")
                for option in q['options']:
                    f.write(f"{option}\n")
                f.write("\n")
        logger.info("Process completed successfully")
    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")
        raise


if __name__ == "__main__":
    main()