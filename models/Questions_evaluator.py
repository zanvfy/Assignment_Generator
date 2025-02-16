import re
import json
import spacy
import logging
from typing import List, Dict, Set, Tuple
from difflib import SequenceMatcher
import numpy as np
from pathlib import Path
from datetime import datetime
import pandas as pd
from collections import defaultdict
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from sklearn.metrics import f1_score, accuracy_score
from textblob import TextBlob

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('stopwords', quiet=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class GrammarChecker:
    """Custom grammar checker using NLTK and spaCy."""

    def __init__(self, nlp):
        self.nlp = nlp
        self.common_errors = {
            r'\b(a)\s+[aeiou]': 'Consider using "an" before vowel sounds',
            r'\b(is|are|was|were)\s+\w+ing\b': 'Check verb tense agreement',
            r'\b(their|there|they\'re|your|you\'re|its|it\'s)\b': 'Check proper usage of homophones',
            r'\s+,': 'Remove space before comma',
            r'\s+\.': 'Remove space before period',
            r'\b(less)\s+\w+s\b': 'Consider using "fewer" for countable nouns',
            r'\b(amount)\s+of\s+\w+s\b': 'Consider using "number" for countable nouns',
            r'\b(between)\s+\w+\s+and\s+\w+\s+and\s+\w+\b': 'Use "among" for more than two items'
        }

    def check_grammar(self, text: str) -> List[Dict]:
        """Check for common grammar issues."""
        errors = []
        doc = self.nlp(text)

        # Check for basic sentence structure
        if not text[0].isupper():
            errors.append({
                'message': 'Sentence should start with a capital letter',
                'category': 'Capitalization'
            })

        # Check common error patterns
        for pattern, message in self.common_errors.items():
            if re.search(pattern, text, re.IGNORECASE):
                errors.append({
                    'message': message,
                    'category': 'Usage'
                })

        # Check for subject-verb agreement
        for sent in doc.sents:
            subj = None
            verb = None
            for token in sent:
                if token.dep_ == 'nsubj':
                    subj = token
                elif token.pos_ == 'VERB':
                    verb = token
                if subj and verb:
                    if not self._check_agreement(subj, verb):
                        errors.append({
                            'message': f'Possible subject-verb agreement error with "{subj}" and "{verb}"',
                            'category': 'Grammar'
                        })

        return errors

    def _check_agreement(self, subject, verb):
        """Check subject-verb agreement."""
        if subject.pos_ == 'PRON':
            if subject.text.lower() in ['i', 'you', 'we', 'they']:
                return True
            if subject.text.lower() in ['he', 'she', 'it']:
                return verb.tag_ in ['VBZ', 'VBP']
        return True


class AdvancedQuestionEvaluator:
    def __init__(self):
        """Initialize the question evaluator with required models and tools."""
        logger.info("Initializing AdvancedQuestionEvaluator")
        try:
            # Load NLP models and tools
            self.nlp = spacy.load("en_core_web_md")
            self.grammar_checker = GrammarChecker(self.nlp)
            self.stopwords = set(stopwords.words('english'))

            # Question difficulty patterns
            self.difficulty_patterns = {
                'easy': {
                    'patterns': r'\b(what|where|when|who|which)\b',
                    'max_length': 15,
                    'technical_terms': 1
                },
                'medium': {
                    'patterns': r'\b(how|why|explain|describe)\b',
                    'max_length': 25,
                    'technical_terms': 3
                },
                'hard': {
                    'patterns': r'\b(analyze|evaluate|compare|contrast|synthesize|hypothesize)\b',
                    'max_length': 35,
                    'technical_terms': 5
                }
            }

            # Cognitive level patterns for Bloom's Taxonomy
            self.cognitive_patterns = {
                'remember': r'\b(define|list|recall|name|identify)\b',
                'understand': r'\b(explain|describe|discuss|interpret)\b',
                'apply': r'\b(solve|apply|demonstrate|use|implement)\b',
                'analyze': r'\b(analyze|compare|contrast|examine|investigate)\b',
                'evaluate': r'\b(evaluate|assess|judge|critique|recommend)\b',
                'create': r'\b(create|design|develop|formulate|propose)\b'
            }

            logger.info("Successfully initialized evaluation components")
        except Exception as e:
            logger.error(f"Error initializing evaluator: {str(e)}")
            raise

    def evaluate_grammar(self, question: str) -> Dict:
        """Evaluate grammar and language quality."""
        # Use custom grammar checker
        grammar_errors = self.grammar_checker.check_grammar(question)

        # Use TextBlob for additional language analysis
        blob = TextBlob(question)

        # Calculate language score based on error count
        error_count = len(grammar_errors)
        language_score = max(0, 1 - (error_count * 0.1))

        return {
            'error_count': error_count,
            'errors': grammar_errors,
            'language_score': language_score,
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity
        }

    # [Rest of the class methods remain exactly the same as in your original code]
    def load_questions(self, file_path: str) -> List[str]:
        """Load and validate questions from a file."""
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                logger.error(f"File not found: {file_path}")
                return []

            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()

            # Extract questions using multiple patterns
            questions = []

            # Try numbered format
            numbered = re.findall(r'\d+\.\s*([^.?]+\?)', content)
            if numbered:
                questions.extend(numbered)

            # Try line-by-line format
            if not questions:
                lines = content.split('\n')
                questions.extend(line.strip() for line in lines if line.strip() and '?' in line)

            # Clean and validate questions
            cleaned_questions = []
            for q in questions:
                q = re.sub(r'^\d+\.\s*', '', q.strip())
                q = q[0].upper() + q[1:] if q else q
                if not q.endswith('?'):
                    q += '?'
                if self._is_valid_question(q):
                    cleaned_questions.append(q)

            return cleaned_questions

        except Exception as e:
            logger.error(f"Error loading questions: {str(e)}")
            return []

    def _is_valid_question(self, question: str) -> bool:
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

    def evaluate_question_difficulty(self, question: str) -> Dict:
        """Evaluate the difficulty level of a question."""
        question_lower = question.lower()
        doc = self.nlp(question)

        # Count technical terms
        technical_terms = len([token for token in doc if token.pos_ in ['NOUN', 'PROPN']
                               and token.text.lower() not in self.stopwords])

        # Calculate complexity metrics
        word_count = len([token for token in doc if not token.is_space])

        # Determine difficulty level
        difficulty_scores = {}
        for level, criteria in self.difficulty_patterns.items():
            score = 0
            if re.search(criteria['patterns'], question_lower):
                score += 0.4
            if word_count <= criteria['max_length']:
                score += 0.3
            if technical_terms <= criteria['technical_terms']:
                score += 0.3
            difficulty_scores[level] = score

        # Get the highest scoring difficulty level
        difficulty_level = max(difficulty_scores.items(), key=lambda x: x[1])

        return {
            'level': difficulty_level[0],
            'confidence': difficulty_level[1],
            'metrics': {
                'word_count': word_count,
                'technical_terms': technical_terms
            }
        }

    def _difficulty_to_numeric(self, difficulty: str) -> float:
        """Convert difficulty level to numeric value for comparison."""
        difficulty_map = {
            'easy': 1.0,
            'medium': 2.0,
            'hard': 3.0
        }
        return difficulty_map.get(difficulty.lower(), 1.0)

    def calculate_similarity_metrics(self, generated_questions: List[str],
                                     reference_questions: List[str]) -> Dict:
        """
        Calculate detailed similarity metrics based on question complexity factors.
        """
        # Initialize metrics containers
        gen_metrics = {
            'word_counts': [],
            'technical_terms': [],
            'difficulty_levels': [],
            'grammar_scores': [],
            'polarity_scores': [],
            'subjectivity_scores': []
        }

        ref_metrics = {
            'word_counts': [],
            'technical_terms': [],
            'difficulty_levels': [],
            'grammar_scores': [],
            'polarity_scores': [],
            'subjectivity_scores': []
        }

        # Calculate metrics for both sets
        for question in generated_questions:
            doc = self.nlp(question)
            grammar_eval = self.evaluate_grammar(question)
            difficulty_eval = self.evaluate_question_difficulty(question)

            technical_terms = len([token for token in doc if token.pos_ in ['NOUN', 'PROPN']
                                   and token.text.lower() not in self.stopwords])

            gen_metrics['word_counts'].append(len([token for token in doc if not token.is_space]))
            gen_metrics['technical_terms'].append(technical_terms)
            gen_metrics['difficulty_levels'].append(self._difficulty_to_numeric(difficulty_eval['level']))
            gen_metrics['grammar_scores'].append(grammar_eval['language_score'])
            gen_metrics['polarity_scores'].append(grammar_eval['polarity'])
            gen_metrics['subjectivity_scores'].append(grammar_eval['subjectivity'])

        for question in reference_questions:
            doc = self.nlp(question)
            grammar_eval = self.evaluate_grammar(question)
            difficulty_eval = self.evaluate_question_difficulty(question)

            technical_terms = len([token for token in doc if token.pos_ in ['NOUN', 'PROPN']
                                   and token.text.lower() not in self.stopwords])

            ref_metrics['word_counts'].append(len([token for token in doc if not token.is_space]))
            ref_metrics['technical_terms'].append(technical_terms)
            ref_metrics['difficulty_levels'].append(self._difficulty_to_numeric(difficulty_eval['level']))
            ref_metrics['grammar_scores'].append(grammar_eval['language_score'])
            ref_metrics['polarity_scores'].append(grammar_eval['polarity'])
            ref_metrics['subjectivity_scores'].append(grammar_eval['subjectivity'])

        # Calculate similarity scores
        similarity_scores = {
            'word_count_similarity': self._calculate_distribution_similarity(
                gen_metrics['word_counts'], ref_metrics['word_counts']),
            'technical_terms_similarity': self._calculate_distribution_similarity(
                gen_metrics['technical_terms'], ref_metrics['technical_terms']),
            'difficulty_similarity': self._calculate_distribution_similarity(
                gen_metrics['difficulty_levels'], ref_metrics['difficulty_levels']),
            'grammar_similarity': self._calculate_distribution_similarity(
                gen_metrics['grammar_scores'], ref_metrics['grammar_scores']),
            'polarity_similarity': self._calculate_distribution_similarity(
                gen_metrics['polarity_scores'], ref_metrics['polarity_scores']),
            'subjectivity_similarity': self._calculate_distribution_similarity(
                gen_metrics['subjectivity_scores'], ref_metrics['subjectivity_scores'])
        }

        # Calculate aggregate statistics
        aggregate_stats = {
            'generated': {
                'avg_word_count': np.mean(gen_metrics['word_counts']),
                'avg_technical_terms': np.mean(gen_metrics['technical_terms']),
                'avg_difficulty': np.mean(gen_metrics['difficulty_levels']),
                'avg_grammar_score': np.mean(gen_metrics['grammar_scores']),
                'avg_polarity': np.mean(gen_metrics['polarity_scores']),
                'avg_subjectivity': np.mean(gen_metrics['subjectivity_scores'])
            },
            'reference': {
                'avg_word_count': np.mean(ref_metrics['word_counts']),
                'avg_technical_terms': np.mean(ref_metrics['technical_terms']),
                'avg_difficulty': np.mean(ref_metrics['difficulty_levels']),
                'avg_grammar_score': np.mean(ref_metrics['grammar_scores']),
                'avg_polarity': np.mean(ref_metrics['polarity_scores']),
                'avg_subjectivity': np.mean(ref_metrics['subjectivity_scores'])
            }
        }

        overall_similarity = np.mean(list(similarity_scores.values()))

        return {
            'similarity_scores': similarity_scores,
            'aggregate_stats': aggregate_stats,
            'overall_similarity': overall_similarity,
            'metrics_distribution': {
                'generated': gen_metrics,
                'reference': ref_metrics
            }
        }

    def _calculate_distribution_similarity(self, dist1: List[float], dist2: List[float]) -> float:
        """Calculate similarity between two distributions."""
        dist1_norm = np.array(dist1) / np.max(dist1) if len(dist1) > 0 and np.max(dist1) > 0 else np.array(dist1)
        dist2_norm = np.array(dist2) / np.max(dist2) if len(dist2) > 0 and np.max(dist2) > 0 else np.array(dist2)

        mean_sim = 1 - abs(np.mean(dist1_norm) - np.mean(dist2_norm))

        hist1, bins = np.histogram(dist1_norm, bins=10)
        hist2, _ = np.histogram(dist2_norm, bins=bins)

        hist1_norm = hist1 / np.sum(hist1) if np.sum(hist1) > 0 else hist1
        hist2_norm = hist2 / np.sum(hist2) if np.sum(hist2) > 0 else hist2

        overlap = 1 - np.sum(np.abs(hist1_norm - hist2_norm)) / 2

        return float((mean_sim + overlap) / 2)

    # def calculate_similarity_metrics(self, generated_questions: List[str],
    #                                  reference_questions: List[str]) -> Dict:
    #     """
    #     Calculate detailed similarity metrics between generated and reference questions.
    #     Shows exactly how each question matches against reference questions.
    #     """
    #     similarities = []
    #     detailed_comparisons = []
    #
    #     for gen_q in generated_questions:
    #         gen_doc = self.nlp(gen_q)
    #         question_sims = []
    #
    #         # Track detailed comparisons for this question
    #         question_comparison = {
    #             'generated_question': gen_q,
    #             'comparisons': []
    #         }
    #
    #         for ref_q in reference_questions:
    #             ref_doc = self.nlp(ref_q)
    #
    #             # Calculate semantic similarity using spaCy
    #             semantic_sim = gen_doc.similarity(ref_doc)
    #
    #             # Calculate sequence similarity using SequenceMatcher
    #             sequence_sim = SequenceMatcher(None, gen_q.lower(), ref_q.lower()).ratio()
    #
    #             # Calculate word overlap
    #             gen_words = set(token.text.lower() for token in gen_doc
    #                             if not token.is_stop and token.is_alpha)
    #             ref_words = set(token.text.lower() for token in ref_doc
    #                             if not token.is_stop and token.is_alpha)
    #             word_overlap = len(gen_words.intersection(ref_words)) / len(
    #                 gen_words.union(ref_words)) if gen_words.union(ref_words) else 0
    #
    #             # Calculate common words and unique words
    #             common_words = gen_words.intersection(ref_words)
    #             gen_unique = gen_words - ref_words
    #             ref_unique = ref_words - gen_words
    #
    #             # Track detailed comparison for this pair
    #             comparison = {
    #                 'reference_question': ref_q,
    #                 'semantic_similarity': semantic_sim,
    #                 'sequence_similarity': sequence_sim,
    #                 'word_overlap': word_overlap,
    #                 'common_words': list(common_words),
    #                 'generated_unique': list(gen_unique),
    #                 'reference_unique': list(ref_unique),
    #                 'average_similarity': (semantic_sim + sequence_sim + word_overlap) / 3
    #             }
    #
    #             question_sims.append(comparison)
    #             question_comparison['comparisons'].append(comparison)
    #
    #         # Get the best matching reference question
    #         best_match = max(question_sims, key=lambda x: x['average_similarity'])
    #         similarities.append(best_match)
    #         detailed_comparisons.append(question_comparison)
    #
    #     return {
    #         'average_semantic_similarity': np.mean([s['semantic_similarity'] for s in similarities]),
    #         'average_sequence_similarity': np.mean([s['sequence_similarity'] for s in similarities]),
    #         'average_word_overlap': np.mean([s['word_overlap'] for s in similarities]),
    #         'matches': similarities,
    #         'detailed_comparisons': detailed_comparisons
    #     }

    def evaluate_questions(self, generated_file: str, reference_file: str) -> Dict:
        """Perform comprehensive evaluation of generated questions against references."""
        # Load questions
        generated_questions = self.load_questions(generated_file)
        reference_questions = self.load_questions(reference_file)

        if not generated_questions or not reference_questions:
            raise ValueError("No questions found in one or both input files")

        # Evaluate each generated question
        evaluations = []
        for question in generated_questions:
            # Basic evaluation metrics
            grammar_eval = self.evaluate_grammar(question)
            difficulty_eval = self.evaluate_question_difficulty(question)

            evaluations.append({
                'question': question,
                'grammar_evaluation': grammar_eval,
                'difficulty_evaluation': difficulty_eval
            })

        # Calculate similarity metrics
        similarity_metrics = self.calculate_similarity_metrics(generated_questions, reference_questions)

        # Calculate overall metrics
        overall_metrics = {
            'total_generated': len(generated_questions),
            'total_reference': len(reference_questions),
            'average_grammar_score': np.mean([e['grammar_evaluation']['language_score'] for e in evaluations]),
            'difficulty_distribution': self._calculate_difficulty_distribution(evaluations),
            'similarity_metrics': similarity_metrics
        }

        return {
            'evaluations': evaluations,
            'overall_metrics': overall_metrics
        }

    def _calculate_difficulty_distribution(self, evaluations: List[Dict]) -> Dict:
        """Calculate the distribution of question difficulties."""
        distribution = defaultdict(int)
        total = len(evaluations)

        for eval_data in evaluations:
            difficulty = eval_data['difficulty_evaluation']['level']
            distribution[difficulty] += 1

        return {level: (count / total) * 100 for level, count in distribution.items()}

    def generate_report(self, evaluation_results: Dict, output_dir: str):
        """Generate a detailed evaluation report with enhanced explanations."""
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = output_dir / f"evaluation_report.txt"
            excel_path = output_dir / f"evaluation_metrics.xlsx"

            # Generate main report
            with open(report_path, 'w', encoding='utf-8') as f:
                # Header
                f.write("=" * 80 + "\n")
                f.write("QUESTION EVALUATION REPORT "+timestamp+" \n")
                f.write("=" * 80 + "\n\n")

                # Report Overview
                f.write("REPORT OVERVIEW\n")
                f.write("-" * 50 + "\n")
                f.write("This report analyzes the quality and characteristics of generated questions ")
                f.write("compared to reference questions. It examines various aspects including:\n")
                f.write("- Question difficulty and complexity\n")
                f.write("- Grammar and language quality\n")
                f.write("- Technical content density\n")
                f.write("- Similarity to reference questions\n\n")

                # Executive Summary
                f.write("EXECUTIVE SUMMARY\n")
                f.write("-" * 50 + "\n")
                metrics = evaluation_results['overall_metrics']
                f.write(f"Total Generated Questions: {metrics['total_generated']}\n")
                f.write(f"Total Reference Questions: {metrics['total_reference']}\n")
                f.write(f"Average Grammar Score: {metrics['average_grammar_score']:.2f}")
                f.write(" (Scale: 0-1, where 1 indicates perfect grammar)\n\n")

                # Overall Similarity Analysis
                f.write("SIMILARITY ANALYSIS\n")
                f.write("-" * 50 + "\n")
                f.write("The following scores indicate how well the generated questions match ")
                f.write("the reference questions in various aspects:\n\n")

                sim_metrics = metrics['similarity_metrics']

                # Average similarity scores
                # f.write("Average Semantic Similarity: ")
                # f.write(f"{sim_metrics['average_semantic_similarity']:.2f}")
                # f.write(" (Measures meaning/concept alignment)\n")
                #
                # f.write("Average Sequence Similarity: ")
                # f.write(f"{sim_metrics['average_sequence_similarity']:.2f}")
                # f.write(" (Measures structural/pattern matching)\n")

                # f.write("Average Word Overlap: ")
                # f.write(f"{sim_metrics['average_word_overlap']:.2f}")
                # f.write(" (Measures vocabulary consistency)\n\n")

                # Difficulty Distribution
                f.write("DIFFICULTY DISTRIBUTION\n")
                f.write("-" * 50 + "\n")
                f.write("Question difficulty breakdown (percentage of total):\n")
                for level, percentage in metrics['difficulty_distribution'].items():
                    f.write(f"{level.capitalize()}: {percentage:.1f}%")
                    if level == 'easy':
                        f.write(" (Basic recall and simple concept questions)\n")
                    elif level == 'medium':
                        f.write(" (Understanding and application questions)\n")
                    elif level == 'hard':
                        f.write(" (Analysis and evaluation questions)\n")
                f.write("\n")

                # Detailed Question Analysis
                f.write("DETAILED QUESTION ANALYSIS\n")
                f.write("=" * 80 + "\n\n")
                f.write("Analysis of individual questions, showing key metrics and quality indicators:\n\n")

                for eval_data in evaluation_results['evaluations']:
                    f.write("-" * 80 + "\n")
                    f.write(f"Question: {eval_data['question']}\n")
                    f.write(f"Difficulty Level: {eval_data['difficulty_evaluation']['level']}\n")
                    f.write(f"Confidence Score: {eval_data['difficulty_evaluation']['confidence']:.2f}")
                    f.write(" (Higher scores indicate more reliable difficulty assessment)\n")
                    f.write(f"Grammar Score: {eval_data['grammar_evaluation']['language_score']:.2f}")
                    f.write(" (1.0 = perfect grammar)\n\n")

                    # Technical Metrics with explanations
                    f.write("Technical Analysis:\n")
                    metrics = eval_data['difficulty_evaluation']['metrics']
                    f.write(f"- Word Count: {metrics['word_count']} ")
                    f.write("(Indicates question length and potential complexity)\n")
                    f.write(f"- Technical Terms: {metrics['technical_terms']} ")
                    f.write("(Number of domain-specific or advanced vocabulary words)\n")

                    # Grammar Analysis with context
                    if eval_data['grammar_evaluation']['errors']:
                        f.write("\nGrammar Issues:\n")
                        for error in eval_data['grammar_evaluation']['errors']:
                            f.write(f"- {error['message']} ({error['category']})\n")
                    else:
                        f.write("\nNo grammar issues detected\n")

                    # Sentiment Analysis with explanation
                    f.write("\nTone Analysis:\n")
                    polarity = eval_data['grammar_evaluation']['polarity']
                    subjectivity = eval_data['grammar_evaluation']['subjectivity']
                    f.write(f"- Polarity: {polarity:.2f} ")
                    f.write("(-1 = negative, 0 = neutral, 1 = positive tone)\n")
                    f.write(f"- Subjectivity: {subjectivity:.2f} ")
                    f.write("(0 = objective, 1 = subjective)\n\n")

            # Generate Excel report with detailed metrics
            df = pd.DataFrame([{
                'Question': eval_data['question'],
                'Difficulty Level': eval_data['difficulty_evaluation']['level'],
                'Grammar Score': eval_data['grammar_evaluation']['language_score'],
                'Error Count': eval_data['grammar_evaluation']['error_count'],
                'Technical Terms': eval_data['difficulty_evaluation']['metrics']['technical_terms'],
                'Word Count': eval_data['difficulty_evaluation']['metrics']['word_count'],
                'Polarity': eval_data['grammar_evaluation']['polarity'],
                'Subjectivity': eval_data['grammar_evaluation']['subjectivity']
            } for eval_data in evaluation_results['evaluations']])

            df.to_excel(excel_path, index=False)

            logger.info(f"Evaluation report generated: {report_path}")
            logger.info(f"Excel metrics saved: {excel_path}")

        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            raise


def main():
    """Main function to run the question evaluator."""
    try:
        # Initialize evaluator
        evaluator = AdvancedQuestionEvaluator()

        # Setup file paths
        current_dir = Path(__file__).parent
        generated_file = "../outputs/output_questions.txt"
        reference_file = current_dir / "input.txt"
        output_dir = "../outputs"

        # Perform evaluation
        evaluation_results = evaluator.evaluate_questions(str(generated_file), str(reference_file))

        # Generate report
        evaluator.generate_report(evaluation_results, str(output_dir))

    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")
        raise


if __name__ == "__main__":
    main()