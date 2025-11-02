from dataclasses import dataclass
from typing import List, Dict

@dataclass
class TextContent:
    """Data class for storing text content from file"""
    raw_text: str
    file_path: str
    word_count: int
    
    @classmethod
    def from_file(cls, file_path: str) -> 'TextContent':
        """Create TextContent instance from a file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            word_count = len(text.split())
            return cls(raw_text=text, file_path=file_path, word_count=word_count)
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")
        except Exception as e:
            raise Exception(f"Error reading file: {str(e)}")

@dataclass
class Keyword:
    """Data class for storing a keyword"""
    word: str
    relevance: float = 1.0

@dataclass
class Question:
    """Data class for storing a test question with multiple choice answers"""
    question_text: str
    options: List[str]
    correct_answer: int  # Index of correct answer (0-3)
    explanation: str = ""
    
    def to_dict(self) -> Dict:
        """Convert question to dictionary"""
        return {
            'question': self.question_text,
            'options': self.options,
            'correct_answer': self.correct_answer,
            'explanation': self.explanation
        }
    
    def is_valid(self) -> bool:
        """Validate question has proper structure"""
        return (
            len(self.options) == 4 and
            0 <= self.correct_answer < 4 and
            len(self.question_text) > 0
        )

@dataclass
class ProcessedContent:
    """Data class for storing all processed content"""
    original_text: TextContent
    summary: str
    keywords: List[Keyword]
    questions: List[Question]
    
    def get_valid_questions(self) -> List[Question]:
        """Return only valid questions"""
        return [q for q in self.questions if q.is_valid()]
    
    def to_dict(self) -> Dict:
        """Convert processed content to dictionary"""
        return {
            'file_path': self.original_text.file_path,
            'word_count': self.original_text.word_count,
            'summary': self.summary,
            'keywords': [kw.word for kw in self.keywords],
            'questions': [q.to_dict() for q in self.get_valid_questions()]
        }