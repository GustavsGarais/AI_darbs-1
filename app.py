from models import TextContent, ProcessedContent
from ai_services import AIService
from typing import Optional

class TextProcessingApp:
    """Main application class for text processing"""
    
    def __init__(self):
        """Initialize the application"""
        self.ai_service = AIService()
        self.current_content: Optional[ProcessedContent] = None
    
    def load_text_file(self, file_path: str) -> TextContent:
        """
        Load text from file
        
        Args:
            file_path: Path to text file
            
        Returns:
            TextContent object
        """
        try:
            text_content = TextContent.from_file(file_path)
            print(f"✓ Loaded file: {file_path}")
            print(f"  Word count: {text_content.word_count}")
            return text_content
        except Exception as e:
            raise Exception(f"Failed to load file: {str(e)}")
    
    def process_text(self, file_path: str, num_keywords: int = 5, num_questions: int = 5) -> ProcessedContent:
        """
        Process text file through all AI services
        
        Args:
            file_path: Path to text file
            num_keywords: Number of keywords to extract
            num_questions: Number of questions to generate
            
        Returns:
            ProcessedContent object with all results
        """
        print("\n" + "="*60)
        print("STARTING TEXT PROCESSING")
        print("="*60)
        
        # Step 1: Load text
        print("\n[1/4] Loading text file...")
        text_content = self.load_text_file(file_path)
        
        # Step 2: Extract/summarize text
        print("\n[2/4] Extracting and summarizing text...")
        summary = self.ai_service.extract_text_summary(text_content.raw_text)
        print(f"✓ Summary generated ({len(summary)} characters)")
        
        # Step 3: Generate keywords
        print(f"\n[3/4] Generating {num_keywords} keywords...")
        keywords = self.ai_service.generate_keywords(text_content.raw_text, num_keywords)
        print(f"✓ Generated {len(keywords)} keywords")
        
        # Step 4: Generate questions based on keywords
        print(f"\n[4/4] Generating {num_questions} test questions based on keywords...")
        print(f"  Using keywords: {[kw.word for kw in keywords]}")
        questions = self.ai_service.generate_questions(
            text=text_content.raw_text,
            keywords=keywords,
            num_questions=num_questions
        )
        print(f"✓ Generated {len(questions)} questions")
        
        # Create processed content
        self.current_content = ProcessedContent(
            original_text=text_content,
            summary=summary,
            keywords=keywords,
            questions=questions
        )
        
        print("\n" + "="*60)
        print("PROCESSING COMPLETE")
        print("="*60 + "\n")
        
        return self.current_content
    
    def get_results(self) -> Optional[ProcessedContent]:
        """Get current processed content"""
        return self.current_content
    
    def export_results(self, output_file: str = "results.txt"):
        """
        Export results to a text file
        
        Args:
            output_file: Output file path
        """
        if not self.current_content:
            print("No content to export")
            return
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("TEXT PROCESSING RESULTS\n")
                f.write("="*60 + "\n\n")
                
                f.write(f"Source File: {self.current_content.original_text.file_path}\n")
                f.write(f"Word Count: {self.current_content.original_text.word_count}\n\n")
                
                f.write("SUMMARY\n")
                f.write("-"*60 + "\n")
                # Wrap summary text
                summary_words = self.current_content.summary.split()
                line = ""
                for word in summary_words:
                    if len(line) + len(word) + 1 <= 60:
                        line += word + " "
                    else:
                        f.write(line.strip() + "\n")
                        line = word + " "
                if line:
                    f.write(line.strip() + "\n")
                f.write("\n")
                
                f.write("KEYWORDS\n")
                f.write("-"*60 + "\n")
                for i, kw in enumerate(self.current_content.keywords, 1):
                    f.write(f"{i}. {kw.word}\n")
                f.write("\n")
                
                f.write("TEST QUESTIONS\n")
                f.write("-"*60 + "\n")
                for i, q in enumerate(self.current_content.get_valid_questions(), 1):
                    f.write(f"\nQuestion {i}: {q.question_text}\n")
                    for j, option in enumerate(q.options):
                        marker = "→" if j == q.correct_answer else " "
                        f.write(f"{marker} {chr(65+j)}) {option}\n")
                    f.write(f"Correct Answer: {chr(65+q.correct_answer)}\n")
            
            print(f"✓ Results exported to: {output_file}")
            
        except Exception as e:
            print(f"Error exporting results: {str(e)}")