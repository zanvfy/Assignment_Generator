import re
import json
from transformers import T5Tokenizer, T5ForConditionalGeneration
import spacy
import random
from typing import List, Dict, Set, Tuple
import logging
from datetime import datetime
from difflib import SequenceMatcher
from nltk.tokenize import sent_tokenize
import numpy as np
from collections import Counter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class EnhancedQuestionGenerator:
    def __init__(self):
        """Initialize the question generator with improved models and processing."""
        logger.info("Initializing EnhancedQuestionGenerator")
        try:
            # Using a more sophisticated T5 model for better question generation
            self.model_name = "iarfmoose/t5-base-question-generator"
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)

            # Load spaCy model for better text processing
            self.nlp = spacy.load("en_core_web_md")

            # Question type templates for validation
            self.question_templates = {
                'what': ['what is', 'what are', 'what was', 'what were', 'what does', 'what did'],
                'why': ['why is', 'why are', 'why was', 'why were', 'why does', 'why did'],
                'how': ['how is', 'how are', 'how was', 'how were', 'how does', 'how did', 'how can', 'how many'],
                'where': ['where is', 'where are', 'where was', 'where were'],
                'when': ['when is', 'when are', 'when was', 'when were', 'when does', 'when did'],
                'which': ['which is', 'which are', 'which was', 'which were'],
                'who': ['who is', 'who are', 'who was', 'who were', 'who does', 'who did']
            }

            logger.info("Successfully loaded all models and components")
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            raise

    def _calculate_sentence_importance(self, sentence: str, doc: spacy.tokens.Doc) -> float:
        """Calculate the importance score of a sentence based on multiple factors."""
        # Convert sentence to spaCy doc
        sent_doc = self.nlp(sentence)

        # Calculate named entity density
        ner_count = len([ent for ent in sent_doc.ents])
        ner_density = ner_count / len(sentence.split())

        # Calculate average word importance using word vectors
        word_importance = np.mean([token.vector_norm for token in sent_doc if not token.is_stop])

        # Calculate sentence position score (earlier sentences get higher scores)
        position_score = 1.0 / (1 + doc.text.index(sentence))

        # Combine scores with weights
        total_score = (0.4 * ner_density +
                       0.3 * word_importance +
                       0.3 * position_score)

        return total_score

    def _extract_key_sentences(self, passage: str) -> List[Tuple[str, float]]:
        """Extract and rank important sentences from the passage."""
        doc = self.nlp(passage)
        sentences_with_scores = []

        for sent in doc.sents:
            sent_text = sent.text.strip()

            # Filter out sentences that are too short or too long
            if 8 <= len(sent_text.split()) <= 50:
                # Calculate importance score
                importance_score = self._calculate_sentence_importance(sent_text, doc)
                sentences_with_scores.append((sent_text, importance_score))

        # Sort by importance score and return top sentences
        return sorted(sentences_with_scores, key=lambda x: x[1], reverse=True)[:15]

    # def _evaluate_question_quality(self, question: str, source_sentence: str) -> float:
    #     """Evaluate the quality of a generated question."""
    #     # Check if question starts with valid question words
    #     starts_with_question_word = any(
    #         question.lower().startswith(template)
    #         for templates in self.question_templates.values()
    #         for template in templates
    #     )
    #     if not starts_with_question_word:
    #         return 0.0
    #
    #     # Check question length
    #     words = question.split()
    #     if len(words) < 6 or len(words) > 20:
    #         return 0.0
    #
    #     # Check for presence of key entities from source
    #     source_doc = self.nlp(source_sentence)
    #     question_doc = self.nlp(question)
    #
    #     print("aaa",source_doc,question_doc)
    #     source_entities = set(ent.text.lower() for ent in source_doc.ents)
    #     question_entities = set(ent.text.lower() for ent in question_doc.ents)
    #
    #     entity_overlap = len(source_entities.intersection(question_entities))
    #     entity_score = entity_overlap / (len(source_entities) + 1e-6)
    #
    #     # Calculate semantic similarity
    #     semantic_similarity = question_doc.similarity(source_doc)
    #
    #     # Combine scores
    #     quality_score = (0.4 * starts_with_question_word +
    #                      0.3 * entity_score +
    #                      0.3 * semantic_similarity)
    #
    #     return quality_score

    def _evaluate_question_quality(self, question: str, source_sentence: str) -> float:
        """Evaluate the quality of a generated question with enhanced criteria."""
        # Check if question starts with a valid question word and ends with a question mark
        valid_start = any(
            question.lower().startswith(template)
            for templates in self.question_templates.values()
            for template in templates
        )
        valid_end = question.strip().endswith('?')
        if not (valid_start and valid_end):
            return 0.0

        # Use NLP tokenization for an accurate word count (ignoring extra spaces)
        question_doc = self.nlp(question)
        word_count = sum(1 for token in question_doc if not token.is_space)
        if word_count < 6 or word_count > 20:
            return 0.0

        # Extract named entities from both question and source sentence
        source_doc = self.nlp(source_sentence)
        source_entities = {ent.text.lower() for ent in source_doc.ents}
        question_entities = {ent.text.lower() for ent in question_doc.ents}
        entity_overlap = len(source_entities.intersection(question_entities))
        entity_score = entity_overlap / (len(source_entities) + 1e-6)

        # Calculate semantic similarity between the question and source sentence
        semantic_similarity = question_doc.similarity(source_doc)

        # Define configurable weights
        weight_start = 0.4
        weight_entity = 0.3
        weight_semantic = 0.3

        # Combine the metrics into a final quality score
        quality_score = (weight_start * 1.0 +  # valid_start check already passed
                         weight_entity * entity_score +
                         weight_semantic * semantic_similarity)

        return quality_score

    def _generate_question_variations(self, sentence: str) -> List[Dict[str, float]]:
        """Generate multiple variations of questions with improved passage relevance."""
        try:
            # Prepare multiple input prompts to get more variety
            input_prompts = [
                f"generate question: {sentence}",
                f"create a question about: {sentence}",
                f"form a question based on this text: {sentence}",
                f"what question can be asked about: {sentence}"
            ]

            questions_with_scores = []

            for prompt in input_prompts:
                input_ids = self.tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)

                # Generate more questions per prompt with diverse parameters
                outputs = self.model.generate(
                    input_ids,
                    max_length=64,
                    num_return_sequences=8,  # Increased for more variety
                    num_beams=8,
                    temperature=0.8,
                    do_sample=True,
                    top_k=50,
                    top_p=0.92,
                    no_repeat_ngram_size=2,
                    length_penalty=1.0,
                    early_stopping=True
                )

                for output_ids in outputs:
                    question = self.tokenizer.decode(output_ids, skip_special_tokens=True)
                    question = self._clean_question(question)

                    # Enhanced question validation
                    if self._is_valid_question(question, sentence):
                        quality_score = self._evaluate_question_quality(question, sentence)

                        if quality_score > 0.2:  # Lowered threshold to get more candidates
                            questions_with_scores.append({
                                "question": question,
                                "quality_score": quality_score
                            })

            return sorted(questions_with_scores, key=lambda x: x["quality_score"], reverse=True)

        except Exception as e:
            logger.error(f"Error generating questions: {str(e)}")
            return []

    def _is_valid_question(self, question: str, source_sentence: str) -> bool:
        """Enhanced validation of generated questions."""
        if not question or not question.endswith('?'):
            return False

        # Basic length check
        words = question.split()
        if len(words) < 5 or len(words) > 25:
            return False

        # Check if question contains key terms from source sentence
        source_words = set(word.lower() for word in source_sentence.split())
        question_words = set(word.lower() for word in words)
        common_words = source_words.intersection(question_words)

        # Require at least 2 meaningful words from source
        meaningful_common_words = [word for word in common_words
                                   if word not in set(['is', 'are', 'was', 'were', 'the', 'a', 'an', 'in', 'on', 'at'])]

        return len(meaningful_common_words) >= 2

    def _clean_question(self, question: str) -> str:
        """Clean and format the generated question with improved processing."""
        # Remove unwanted prefixes and clean whitespace
        question = re.sub(r'^(generate question:|question:|answer:)', '', question, flags=re.IGNORECASE)
        question = ' '.join(question.split())

        # Fix capitalization
        question = question[0].upper() + question[1:] if question else question

        # Ensure proper question mark
        if not question.endswith('?'):
            question += '?'

        # Fix common grammatical issues
        question = re.sub(r'\s+([.,?!])', r'\1', question)  # Fix spacing before punctuation
        question = re.sub(r'([.,?!])([A-Za-z])', r'\1 \2', question)  # Fix spacing after punctuation

        return question

    def _is_similar(self, question: str, existing_questions: Set[str], threshold: float = 0.7) -> bool:
        """Check if a question is too similar to existing ones with improved similarity detection."""
        # Clean and tokenize the question
        question = re.sub(r'[^\w\s]', '', question.lower())
        doc = self.nlp(question)

        # Extract tokens that are not stop words and are of the desired part-of-speech
        new_tokens = set(token.lemma_ for token in doc if not token.is_stop and token.pos_ in ['NOUN', 'PROPN', 'VERB'])
        new_type = question.split()[0].lower()

        for existing in existing_questions:
            existing = re.sub(r'[^\w\s]', '', existing.lower())

            # Check exact matches
            if question == existing:
                return True

            # Check sequence similarity
            if SequenceMatcher(None, question, existing).ratio() > threshold:
                return True

            # Tokenize the existing question
            existing_doc = self.nlp(existing)
            existing_tokens = set(token.lemma_ for token in existing_doc if not token.is_stop and token.pos_ in ['NOUN', 'PROPN', 'VERB'])

            # Check token overlap
            overlap = len(new_tokens.intersection(existing_tokens))
            if overlap >= len(new_tokens) * 0.5:  # 50% content overlap
                return True

        return False

    def generate_questions(self, passage: str, json_data: Dict) -> List[Dict[str, str]]:
        """Generate exact number of high-quality questions based on passage and configuration."""
        try:
            # Extract required number of questions
            info = json_data.get("info", "")
            match = re.search(r'\(\s*\d+\s*X\s*(\d+)\s*=\s*\d+\s*Marks\)', info)
            if not match:
                raise ValueError(f"Could not extract number of questions from info: {info}")

            required_questions = int(match.group(1))
            # required_questions = int(2)
            logger.info(f"Generating {required_questions} questions")

            # Get ranked sentences
            ranked_sentences = self._extract_key_sentences(passage)
            if not ranked_sentences:
                raise ValueError("No valid sentences extracted from passage")

            # Generate and collect questions
            all_questions = []
            seen_questions = set()
            attempts = 0
            max_attempts = len(ranked_sentences) * 3  # Allow multiple attempts per sentence

            while len(all_questions) < required_questions and attempts < max_attempts:
                # Cycle through sentences if needed
                sentence_idx = attempts % len(ranked_sentences)
                sentence, importance = ranked_sentences[sentence_idx]

                # Generate question variations
                question_variations = self._generate_question_variations(sentence)

                for variation in question_variations:
                    question = variation["question"]
                    quality_score = variation["quality_score"]

                    # Skip if question is too similar to existing ones
                    if not self._is_similar(question, seen_questions, threshold=0.75):  # Reduced threshold
                        all_questions.append({
                            "question": question,
                            "quality_score": quality_score,
                            "importance_score": importance
                        })
                        seen_questions.add(question)

                        if len(all_questions) >= required_questions:
                            break

                attempts += 1

            # If we still don't have enough questions, lower the similarity threshold
            if len(all_questions) < required_questions:
                logger.warning(f"Only generated {len(all_questions)} questions. Lowering similarity threshold...")
                attempts = 0
                while len(all_questions) < required_questions and attempts < max_attempts:
                    sentence_idx = attempts % len(ranked_sentences)
                    sentence, importance = ranked_sentences[sentence_idx]

                    question_variations = self._generate_question_variations(sentence)

                    for variation in question_variations:
                        if len(all_questions) >= required_questions:
                            break

                        question = variation["question"]
                        quality_score = variation["quality_score"]

                        if not self._is_similar(question, seen_questions, threshold=0.6):  # Much lower threshold
                            all_questions.append({
                                "question": question,
                                "quality_score": quality_score,
                                "importance_score": importance
                            })
                            seen_questions.add(question)

                    attempts += 1

            # Ensure we return exactly the required number of questions
            final_questions = sorted(
                all_questions,
                key=lambda x: (x["quality_score"] * 0.7 + x["importance_score"] * 0.3),
                reverse=True
            )[:required_questions]

            print("-------------------------------------------------------------------------------------")
            print("seen",seen_questions)
            print("-------------------------------------------------------------------------------------")
            print("all", all_questions)
            print("-------------------------------------------------------------------------------------")

            logger.info(f"Successfully generated {len(final_questions)} questions")
            return [{"question": q["question"]} for q in final_questions]



        except Exception as e:
            logger.error(f"Error generating questions: {str(e)}")
            raise


def main():
    """Main function to run the enhanced question generator."""
    try:
        # Initialize generator
        generator = EnhancedQuestionGenerator()

        # Read passage
        with open("../data/passage.txt", "r", encoding="utf-8") as f:
            passage = f.read().strip()
            if not passage:
                raise ValueError("Empty passage file")

        # Read JSON configuration
        with open("classified_questions.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("Invalid JSON format: expected dictionary")

        # Generate questions
        questions = generator.generate_questions(passage, data)

        # Save output
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"../outputs/output_questions.txt"

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(data.get("info", "") + "\n\n")
            for i, q in enumerate(questions, 1):
                f.write(f"{i}. {q['question']}\n\n")

        logger.info(f"Questions generated and saved successfully to {output_file}")

    except FileNotFoundError as e:
        logger.error(f"File not found: {str(e)}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")
        raise


if __name__ == "__main__":
    main()