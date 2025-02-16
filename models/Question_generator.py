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
            # Relaxed length constraints to get more candidate sentences
            if 8 <= len(sent_text.split()) <= 60: # Changed from 8-50 to 6-60
                # Calculate importance score
                importance_score = self._calculate_sentence_importance(sent_text, doc)
                sentences_with_scores.append((sent_text, importance_score))

        # Sort by importance score and return top sentences
        return sorted(sentences_with_scores, key=lambda x: x[1], reverse=True)[:30]  # Changed from 15 to 30

    def _evaluate_contextual_relevance(self, question: str, source_sentence: str, full_passage: str) -> float:
        """Evaluate if the question makes sense in the context of both the source sentence and full passage."""
        question_doc = self.nlp(question)
        source_doc = self.nlp(source_sentence)
        passage_doc = self.nlp(full_passage)

        # 1. Entity Validation
        source_entities = {(ent.text.lower(), ent.label_) for ent in source_doc.ents}
        question_entities = {(ent.text.lower(), ent.label_) for ent in question_doc.ents}

        # Check if entities in question exist in source or passage
        entity_validity = 0.0
        if question_entities:
            passage_entities = {(ent.text.lower(), ent.label_) for ent in passage_doc.ents}
            valid_entities = question_entities.intersection(source_entities.union(passage_entities))
            entity_validity = len(valid_entities) / len(question_entities)

        # 2. Subject-Verb Agreement Check
        def get_subject_verb_pairs(doc):
            pairs = []
            for token in doc:
                if "subj" in token.dep_:
                    for ancestor in token.ancestors:
                        if ancestor.pos_ == "VERB":
                            pairs.append((token, ancestor))
                            break
            return pairs

        question_pairs = get_subject_verb_pairs(question_doc)
        source_pairs = get_subject_verb_pairs(source_doc)

        # Check if subject-verb pairs in question are valid
        verb_agreement = 0.0
        if question_pairs:
            matching_pairs = 0
            for q_subj, q_verb in question_pairs:
                for s_subj, s_verb in source_pairs:
                    if (q_subj.lemma_ == s_subj.lemma_ or
                            q_subj.similarity(s_subj) > 0.7):
                        matching_pairs += 1
                        break
            verb_agreement = matching_pairs / len(question_pairs)

        # 3. Temporal Consistency
        def get_temporal_info(doc):
            return {token.text.lower() for token in doc
                    if token.ent_type_ in ['DATE', 'TIME']
                    or token.lemma_ in ['now', 'then', 'before', 'after', 'during']}

        question_temporals = get_temporal_info(question_doc)
        source_temporals = get_temporal_info(source_doc)
        passage_temporals = get_temporal_info(passage_doc)

        temporal_consistency = 1.0
        if question_temporals:
            valid_temporals = question_temporals.intersection(
                source_temporals.union(passage_temporals))
            temporal_consistency = len(valid_temporals) / len(question_temporals)

        # 4. Semantic Role Validation
        def extract_semantic_roles(doc):
            roles = []
            for token in doc:
                if token.dep_ in ['nsubj', 'dobj', 'iobj']:
                    roles.append((token.dep_, token.lemma_))
            return roles

        question_roles = extract_semantic_roles(question_doc)
        source_roles = extract_semantic_roles(source_doc)

        role_validity = 0.0
        if question_roles:
            matching_roles = sum(1 for q_role in question_roles
                                 if any(q_role[1] == s_role[1]
                                        for s_role in source_roles))
            role_validity = matching_roles / len(question_roles)

        # Calculate final contextual relevance score
        weights = {
            'entity_validity': 0.3,
            'verb_agreement': 0.25,
            'temporal_consistency': 0.2,
            'role_validity': 0.25
        }

        contextual_score = (
                weights['entity_validity'] * entity_validity +
                weights['verb_agreement'] * verb_agreement +
                weights['temporal_consistency'] * temporal_consistency +
                weights['role_validity'] * role_validity
        )

        return contextual_score

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

        # Use NLP tokenization for accurate word count
        question_doc = self.nlp(question)
        word_count = sum(1 for token in question_doc if not token.is_space)
        if word_count < 6 or word_count > 20:
            return 0.0

        # Extract named entities
        source_doc = self.nlp(source_sentence)
        source_entities = {ent.text.lower() for ent in source_doc.ents}
        question_entities = {ent.text.lower() for ent in question_doc.ents}
        entity_overlap = len(source_entities.intersection(question_entities))
        entity_score = entity_overlap / (len(source_entities) + 1e-6)

        # Calculate semantic similarity
        semantic_similarity = question_doc.similarity(source_doc)

        # Combine scores with weights
        weight_start = 0.4
        weight_entity = 0.3
        weight_semantic = 0.3

        quality_score = (weight_start * 1.0 +
                         weight_entity * entity_score +
                         weight_semantic * semantic_similarity)

        return quality_score

    def _generate_question_variations(self, sentence: str, full_passage: str) -> List[Dict[str, float]]:
        """Generate multiple variations of questions with improved context validation."""
        try:
            # Added more input prompts for variety
            input_prompts = [
                f"generate question: {sentence}",
                f"create a question about: {sentence}",
                f"form a question based on this text: {sentence}",
                f"what question can be asked about: {sentence}",
                f"ask a question about: {sentence}",
                f"generate a question to test understanding of: {sentence}",
                f"create a question to check knowledge of: {sentence}",
                f"what would you ask about: {sentence}"
            ]

            questions_with_scores = []

            for prompt in input_prompts:
                input_ids = self.tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)

                # Increased number of generated questions per prompt
                outputs = self.model.generate(
                    input_ids,
                    max_length=64,
                    num_return_sequences=12,  # Increased from 8 to 12
                    num_beams=12,  # Increased from 8 to 12
                    temperature=0.9,  # Slightly increased from 0.8 to 0.9 for more variety
                    do_sample=True,
                    top_k=100,  # Increased from 50 to 100
                    top_p=0.95,  # Slightly increased from 0.92 to 0.95
                    no_repeat_ngram_size=2,
                    length_penalty=1.0,
                    early_stopping=True
                )

                for output_ids in outputs:
                    question = self.tokenizer.decode(output_ids, skip_special_tokens=True)
                    question = self._clean_question(question)

                    if self._is_valid_question(question, sentence):
                        quality_score = self._evaluate_question_quality(question, sentence)
                        contextual_score = self._evaluate_contextual_relevance(
                            question, sentence, full_passage)

                        # Slightly relaxed thresholds
                        if quality_score > 0.15 and contextual_score > 0.25:  # Reduced from 0.2 and 0.3
                            combined_score = (quality_score * 0.6 + contextual_score * 0.4)
                            questions_with_scores.append({
                                "question": question,
                                "quality_score": combined_score
                            })

            return sorted(questions_with_scores,
                          key=lambda x: x["quality_score"],
                          reverse=True)

        except Exception as e:
            logger.error(f"Error generating questions: {str(e)}")
            return []

    def _is_valid_question(self, question: str, source_sentence: str) -> bool:
        """Enhanced validation of generated questions."""
        if not question or not question.endswith('?'):
            return False

        # Length check
        words = question.split()
        if len(words) < 5 or len(words) > 25:
            return False

        # Check key terms from source
        source_words = set(word.lower() for word in source_sentence.split())
        question_words = set(word.lower() for word in words)
        common_words = source_words.intersection(question_words)

        meaningful_common_words = [word for word in common_words
                                   if word not in set(['is', 'are', 'was', 'were',
                                                       'the', 'a', 'an', 'in', 'on', 'at'])]

        return len(meaningful_common_words) >= 2

    def _clean_question(self, question: str) -> str:
        """Clean and format the generated question."""
        # Remove unwanted prefixes
        question = re.sub(r'^(generate question:|question:|answer:)', '', question, flags=re.IGNORECASE)
        question = ' '.join(question.split())

        # Fix capitalization
        question = question[0].upper() + question[1:] if question else question

        # Ensure question mark
        if not question.endswith('?'):
            question += '?'

        # Fix spacing around punctuation
        question = re.sub(r'\s+([.,?!])', r'\1', question)
        question = re.sub(r'([.,?!])([A-Za-z])', r'\1 \2', question)

        return question

    def _is_similar(self, question: str, existing_questions: Set[str], threshold: float = 0.7) -> bool:
        """Check if a question is too similar to existing ones."""
        question = re.sub(r'[^\w\s]', '', question.lower())
        doc = self.nlp(question)

        new_tokens = set(token.lemma_ for token in doc
                         if not token.is_stop and token.pos_ in ['NOUN', 'PROPN', 'VERB'])
        new_type = question.split()[0].lower()

        for existing in existing_questions:
            existing = re.sub(r'[^\w\s]', '', existing.lower())

            # Check exact matches
            if question == existing:
                return True

            # Check sequence similarity
            if SequenceMatcher(None, question, existing).ratio() > threshold:
                return True

            # Check token overlap
            existing_doc = self.nlp(existing)
            existing_tokens = set(token.lemma_ for token in existing_doc
                                  if not token.is_stop and token.pos_ in ['NOUN', 'PROPN', 'VERB'])

            overlap = len(new_tokens.intersection(existing_tokens))
            if overlap >= len(new_tokens) * 0.5:
                return True

        return False

    def generate_questions(self, passage: str, json_data: Dict) -> List[Dict[str, str]]:
        """Generate questions with enhanced context validation."""
        try:
            # Extract required number of questions
            info = json_data.get("info", "")
            match = re.search(r'\(\s*\d+\s*X\s*(\d+)\s*=\s*\d+\s*Marks\)', info)
            if not match:
                raise ValueError(f"Could not extract number of questions from info: {info}")

            # Force specific number of questions regardless of JSON config
            # required_questions = 100  # Or any number you want
            required_questions = int(match.group(1))
            logger.info(f"Generating {required_questions} questions")

            # Get ranked sentences
            ranked_sentences = self._extract_key_sentences(passage)
            if not ranked_sentences:
                raise ValueError("No valid sentences extracted from passage")

            # Generate and collect questions
            all_questions = []
            seen_questions = set()
            attempts = 0
            max_attempts = len(ranked_sentences) * 5  # Increased from 3 to 5

            while len(all_questions) < required_questions and attempts < max_attempts:
                sentence_idx = attempts % len(ranked_sentences)
                sentence, importance = ranked_sentences[sentence_idx]

                # Generate questions with context validation
                question_variations = self._generate_question_variations(sentence, passage)

                for variation in question_variations:
                    question = variation["question"]
                    quality_score = variation["quality_score"]

                    if not self._is_similar(question, seen_questions, threshold=0.75):
                        all_questions.append({
                            "question": question,
                            "quality_score": quality_score,
                            "importance_score": importance
                        })
                        seen_questions.add(question)

                        if len(all_questions) >= required_questions:
                            break

                attempts += 1

            # Try with lower similarity threshold if needed
            if len(all_questions) < required_questions:
                logger.warning(f"Only generated {len(all_questions)} questions. Lowering similarity threshold...")
                attempts = 0
                while len(all_questions) < required_questions and attempts < max_attempts:
                    sentence_idx = attempts % len(ranked_sentences)
                    sentence, importance = ranked_sentences[sentence_idx]

                    question_variations = self._generate_question_variations(sentence, passage)

                    for variation in question_variations:
                        if len(all_questions) >= required_questions:
                            break

                        question = variation["question"]
                        quality_score = variation["quality_score"]

                        if not self._is_similar(question, seen_questions, threshold=0.6):
                            all_questions.append({
                                "question": question,
                                "quality_score": quality_score,
                                "importance_score": importance
                            })
                            seen_questions.add(question)

                    attempts += 1

            # Sort and select final questions
            final_questions = sorted(
                all_questions,
                key=lambda x: (x["quality_score"] * 0.7 + x["importance_score"] * 0.3),
                reverse=True
            )[:required_questions]

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
        with open("../data/passage1.txt", "r", encoding="utf-8") as f:
            passage = f.read().strip()
            if not passage:
                raise ValueError("Empty passage file")

        # Read JSON configuration
        with open("../data/classified_questions.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("Invalid JSON format: expected dictionary")

        # Generate questions
        questions = generator.generate_questions(passage, data)

        # Save output
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"../outputs/generated_questions_{timestamp}.txt"

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