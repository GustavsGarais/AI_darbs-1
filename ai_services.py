from huggingface_hub import InferenceClient
from openai import OpenAI
from typing import List
import re
import json
from config import Config
from models import Keyword, Question

class AIService:
    """Service class for AI operations using Hugging Face and OpenAI models"""
    
    def __init__(self):
        """Initialize AI service with Hugging Face and OpenAI clients"""
        Config.validate()
        self.hf_client = InferenceClient(token=Config.HUGGINGFACE_API_TOKEN)
        self.openai_client = OpenAI(api_key=Config.OPENAI_API_KEY)
        
    def extract_text_summary(self, text: str) -> str:
        """
        Extract and summarize text content using Hugging Face model
        
        Args:
            text: Input text to summarize
            
        Returns:
            Summarized text
        """
        try:
            # Truncate text if too long (max ~1024 tokens for most models)
            max_chars = 3000
            text_to_summarize = text[:max_chars] if len(text) > max_chars else text
            
            # Try summarization with Hugging Face
            try:
                response = self.hf_client.summarization(
                    text_to_summarize,
                    model=Config.TEXT_EXTRACTION_MODEL
                )
                
                if isinstance(response, str):
                    summary = response
                elif isinstance(response, list) and len(response) > 0:
                    if isinstance(response[0], dict) and 'summary_text' in response[0]:
                        summary = response[0]['summary_text']
                    elif isinstance(response[0], str):
                        summary = response[0]
                    else:
                        summary = str(response[0])
                else:
                    summary = str(response)
                
                if summary and len(summary) > 20:
                    return summary
            except Exception as e:
                print(f"⚠ Hugging Face summarization failed: {str(e)}")
            
            print("  Using alternative summarization model as fallback...")
            # Try with a different, more accessible model
            response = self.hf_client.summarization(
                text_to_summarize,
                model="Falconsai/text_summarization"  # Alternative summarization model
            )
            
            summary = response.strip()
            if "Summary:" in summary:
                summary = summary.split("Summary:")[-1].strip()
            
            return summary if summary else text[:500]
            
        except Exception as e:
            print(f"✗ Error in text extraction: {str(e)}")
            # Final fallback: return first few sentences
            sentences = text.split('.')[:3]
            return '. '.join(sentences) + '.'
    
    def generate_keywords(self, text: str, num_keywords: int = 5) -> List[Keyword]:
        """
        Generate important keywords using OpenAI API
        
        Args:
            text: Input text
            num_keywords: Number of keywords to generate
            
        Returns:
            List of Keyword objects
        """
        try:
            # Use OpenAI to generate keywords
            response = self.openai_client.chat.completions.create(
                model=Config.OPENAI_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that extracts important keywords from text for creating educational test questions. Focus on specific names, dates, numbers, places, and key concepts."
                    },
                    {
                        "role": "user",
                        "content": f"""From the following text, extract exactly {num_keywords} important keywords that would be good for creating test questions. Focus on:
- Specific names (people, places, organizations)
- Important numbers, dates, or measurements  
- Key technical terms or concepts
- Specific facts or events

Return ONLY the keywords as a numbered list, one per line.

Text:
{text[:2500]}

Keywords:"""
                    }
                ],
                temperature=0.3,
                max_tokens=200
            )
            
            keywords_text = response.choices[0].message.content.strip()
            
            # Parse keywords from response
            keywords_list = []
            for line in keywords_text.split('\n'):
                line = line.strip()
                if not line:
                    continue
                # Remove numbering, bullets, dashes
                cleaned = re.sub(r'^[\d\.\-\*\)\]\:]+\s*', '', line)
                cleaned = cleaned.strip('.,;:"\'"')
                if cleaned and len(cleaned) > 1 and len(cleaned) < 100:
                    keywords_list.append(cleaned)
            
            # Convert to Keyword objects
            keywords = [Keyword(word=kw) for kw in keywords_list[:num_keywords]]
            
            # If not enough keywords, extract from text
            if len(keywords) < num_keywords:
                print(f"  ⚠ Only got {len(keywords)} keywords, extracting more from text...")
                # Extract numbers with context
                number_matches = re.findall(r'\b(\d+(?:\.\d+)?)\s+([a-zA-Z]+)\b', text)
                for num, unit in number_matches[:num_keywords - len(keywords)]:
                    keyword_text = f"{num} {unit}"
                    if keyword_text not in [k.word for k in keywords]:
                        keywords.append(Keyword(word=keyword_text))
                
                # Extract capitalized phrases
                capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
                for cap in capitalized:
                    if len(keywords) >= num_keywords:
                        break
                    if len(cap) > 3 and cap not in [k.word for k in keywords]:
                        keywords.append(Keyword(word=cap))
            
            return keywords[:num_keywords]
            
        except Exception as e:
            print(f"✗ Error generating keywords with OpenAI: {str(e)}")
            # Fallback: extract from text directly
            print("  Using fallback keyword extraction...")
            keywords = []
            
            # Initialize empty keywords list
            keywords = []
            
            # Get dates and years
            date_matches = re.findall(r'\b(\d{4})\b|\b(\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)(?:\s+\d{4})?)\b', text)
            for match in date_matches:
                date = next(d for d in match if d)  # Get non-empty group
                if date not in [k.word for k in keywords]:
                    keywords.append(Keyword(word=date))
            
            # Get measurements with units
            measurement_matches = re.findall(r'\b(\d+(?:,\d{3})*(?:\.\d+)?)\s+(?:meters?|feet|kilometers?|miles|tons?|kg|pounds?|euros?|dollars?)\b', text, re.IGNORECASE)
            for match in measurement_matches:
                if match not in [k.word for k in keywords]:
                    keywords.append(Keyword(word=match))
            
            # Get proper nouns (names, places, etc.)
            proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
            for noun in proper_nouns:
                if len(noun) > 3 and noun not in [k.word for k in keywords]:
                    keywords.append(Keyword(word=noun))
            
            # Get technical or specific terms (words followed by explanations)
            technical_terms = re.findall(r'\b([A-Za-z]+(?:-[A-Za-z]+)*)\s+(?:is|are|was|were|refers to|means|defined as)\b', text)
            for term in technical_terms:
                if len(term) > 3 and term not in [k.word for k in keywords]:
                    keywords.append(Keyword(word=term))
            
            return keywords[:num_keywords]
    
    def generate_questions(self, text: str, keywords: List[Keyword], num_questions: int = 5) -> List[Question]:
        """
        Generate test questions using OpenAI API based on extracted keywords
        
        Args:
            text: Input text for context
            keywords: List of keywords to create questions about
            num_questions: Number of questions to generate
            
        Returns:
            List of Question objects
        """
        questions = []
        keywords_to_use = keywords[:min(num_questions, len(keywords))]
        
        for i, keyword in enumerate(keywords_to_use):
            try:
                # Get context around the keyword
                context = self._get_keyword_context(text, keyword.word)
                
                # Use OpenAI to generate question
                response = self.openai_client.chat.completions.create(
                    model=Config.OPENAI_MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert educational test creator. Create challenging multiple-choice questions that test real comprehension. All 4 options must be plausible and relate to the topic. Avoid obvious wrong answers like 'not mentioned in text'."
                        },
                        {
                            "role": "user",
                            "content": f"""Based on the following text, create ONE challenging multiple-choice test question about "{keyword.word}".

Context from text:
{context}

Requirements:
- Question must specifically test knowledge about "{keyword.word}"
- All 4 answer options must be plausible and related to the topic
- Only ONE option should be correct based on the text
- Wrong answers should be reasonable but incorrect facts
- Avoid obvious wrong answers like "not mentioned" or "unrelated"

Format EXACTLY as follows:

Question: [clear, specific question about {keyword.word}]
A) [plausible answer option]
B) [plausible answer option]
C) [plausible answer option]
D) [plausible answer option]
Correct: [A, B, C, or D]

Generate the question:"""
                        }
                    ],
                    temperature=0.7,
                    max_tokens=350
                )
                
                question_text = response.choices[0].message.content.strip()
                question = self._parse_question(question_text)
                
                if question and question.is_valid():
                    questions.append(question)
                    print(f"  ✓ Generated question {i+1} about '{keyword.word}'")
                else:
                    print(f"  ⚠ Failed to parse question for '{keyword.word}', creating fallback...")
                    fallback = self._create_keyword_question(keyword.word, context)
                    if fallback:
                        questions.append(fallback)
                    
            except Exception as e:
                print(f"✗ Error generating question for '{keyword.word}': {str(e)}")
                # Create a fallback question
                context = self._get_keyword_context(text, keyword.word)
                fallback = self._create_keyword_question(keyword.word, context)
                if fallback:
                    questions.append(fallback)
        
        # If we still need more questions, create general ones
        while len(questions) < num_questions:
            try:
                print(f"  Generating additional question {len(questions) + 1}...")
                response = self.openai_client.chat.completions.create(
                    model=Config.OPENAI_MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful assistant that creates educational multiple-choice test questions."
                        },
                        {
                            "role": "user",
                            "content": f"""Based on this text, create ONE multiple-choice question about an important fact:

{text[:1500]}

Format:
Question: [question text]
A) [first option]
B) [second option]
C) [third option]
D) [fourth option]
Correct: [A, B, C, or D]"""
                        }
                    ],
                    temperature=0.8,
                    max_tokens=250
                )
                
                question_text = response.choices[0].message.content.strip()
                question = self._parse_question(question_text)
                
                if question and question.is_valid():
                    questions.append(question)
                else:
                    break
                    
            except Exception as e:
                print(f"✗ Error generating additional question: {str(e)}")
                break
        
        return questions[:num_questions]
    
    def _get_keyword_context(self, text: str, keyword: str) -> str:
        """Extract relevant context around a keyword from text"""
        keyword_lower = keyword.lower()
        text_lower = text.lower()
        
        if keyword_lower in text_lower:
            pos = text_lower.index(keyword_lower)
            start = max(0, pos - 200)
            end = min(len(text), pos + len(keyword) + 200)
            context = text[start:end]
            return context
        
        return text[:500]
    
    def _create_keyword_question(self, keyword: str, context: str) -> Question:
        """Create a more sophisticated fallback question about a specific keyword"""
        # Find the sentence containing the keyword
        keyword_lower = keyword.lower()
        sentences = [s.strip() for s in context.split('.') if s.strip()]
        relevant_sentence = next((s for s in sentences if keyword_lower in s.lower()), '')

        if not relevant_sentence:
            return self._create_fallback_question(keyword)

        # Determine question type based on keyword characteristics
        if re.search(r'\b\d{4}\b', keyword):  # Year
            question_text = f"What significant event occurred in {keyword}?"
            correct_option = relevant_sentence
            wrong_options = [
                f"Planning phase began in {keyword}",
                f"Initial designs were completed in {keyword}",
                f"Major modifications were made in {keyword}"
            ]
        elif re.search(r'\b\d+\s*(?:meters?|m|feet|ft)\b', keyword, re.I):  # Measurement
            question_text = f"What does the measurement {keyword} refer to?"
            correct_option = relevant_sentence
            wrong_options = [
                "It was the initial planned size",
                "It was the size after modifications",
                "It represents a different dimension"
            ]
        elif keyword[0].isupper():  # Proper noun
            question_text = f"What is stated about {keyword} in the text?"
            correct_option = relevant_sentence
            wrong_options = [
                "It was involved in a different way",
                "It had a minor role in the events",
                "It was referenced differently"
            ]
        else:  # Other keywords
            question_text = f"What does the text say about {keyword}?"
            correct_option = relevant_sentence
            wrong_options = [
                "It had a different significance",
                "It was mentioned in another context",
                "It played a different role"
            ]

        return Question(
            question_text=question_text,
            options=[correct_option] + wrong_options,
            correct_answer=0,
            explanation=f"This is directly stated in the text: {relevant_sentence}"
        )

    def _create_fallback_question(self, keyword: str) -> Question:
        """Create a basic question when no context is found"""
        return Question(
            question_text=f"What is mentioned about {keyword} in the text?",
            options=[
                "It was an important element discussed",
                "It was not directly relevant",
                "It had a minor significance",
                "It was only briefly mentioned"
            ],
            correct_answer=0,
            explanation=f"The text mentions {keyword} as an important element"
        )
    
    def _parse_question(self, response: str) -> Question:
        """Parse question from model response"""
        try:
            lines = [line.strip() for line in response.strip().split('\n') if line.strip()]
            
            question_text = ""
            options = []
            correct_answer = 0
            
            for line in lines:
                # Find question
                if 'Question:' in line:
                    question_text = line.split('Question:', 1)[-1].strip()
                    question_text = re.sub(r'^\d+[\.\)]\s*', '', question_text)
                elif not question_text and not re.match(r'^[A-D][\)\.]', line):
                    question_text = line
                
                # Find options
                if re.match(r'^[A-D][\)\.\:]', line):
                    option_text = re.sub(r'^[A-D][\)\.\:]\s*', '', line).strip()
                    if option_text:
                        options.append(option_text)
                
                # Find correct answer
                if 'Correct:' in line or 'Answer:' in line:
                    answer_part = re.split(r'(Correct|Answer):', line, flags=re.IGNORECASE)[-1].strip()
                    match = re.search(r'[A-D]', answer_part.upper())
                    if match:
                        correct_answer = ord(match.group()) - ord('A')
            
            # Ensure we have 4 options
            while len(options) < 4:
                options.append(f"Option {chr(65 + len(options))}")
            
            if question_text and len(options) >= 4:
                return Question(
                    question_text=question_text,
                    options=options[:4],
                    correct_answer=min(correct_answer, 3)
                )
            
        except Exception as e:
            print(f"✗ Error parsing question: {str(e)}")
        
        return None